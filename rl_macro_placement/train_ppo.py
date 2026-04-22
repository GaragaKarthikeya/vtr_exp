#!/usr/bin/env python3

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.distributions import Normal
except ImportError as exc:
    raise SystemExit(
        "PyTorch is required for PPO training. Activate the VTR virtualenv first:\n"
        "  source ./activate-vtr.sh"
    ) from exc

from rl_macro_placement.config import default_config
from rl_macro_placement.env import MacroPlacementEnv
from rl_macro_placement.layout_space import LayoutAction
from rl_macro_placement.vtr_metrics import VTRMetrics


@dataclass
class Transition:
    state: torch.Tensor
    action: torch.Tensor
    logprob: torch.Tensor
    reward: float
    value: torch.Tensor


class PolicyValueNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone(state)
        mean = self.actor_mean(feats)
        value = self.critic(feats).squeeze(-1)
        return mean, value

    def set_log_std(self, value: float) -> None:
        with torch.no_grad():
            self.log_std.fill_(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple PPO trainer for VTR macro placement")
    parser.add_argument("--epochs", type=int, default=12, help="Number of PPO outer epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Number of VTR evaluations per epoch")
    parser.add_argument("--ppo-steps", type=int, default=4, help="Optimization passes per epoch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="Entropy bonus coefficient")
    parser.add_argument(
        "--entropy-coef-final",
        type=float,
        default=0.001,
        help="Final entropy bonus coefficient after linear decay",
    )
    parser.add_argument("--value-coef", type=float, default=0.5, help="Critic loss coefficient")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer width")
    parser.add_argument(
        "--log-std-initial",
        type=float,
        default=0.0,
        help="Initial Gaussian log standard deviation",
    )
    parser.add_argument(
        "--log-std-final",
        type=float,
        default=-1.25,
        help="Final Gaussian log standard deviation after linear decay",
    )
    parser.add_argument("--save-dir", type=Path, default=None, help="Output directory for PPO artifacts")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=0,
        help="Number of concurrent VTR evaluations per batch (0 uses min(batch_size, CPU count))",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def action_to_feature_vector(action: LayoutAction | None, state_dict: dict) -> list[float]:
    if action is None:
        return [0.0, 0.0, 0.0]

    aspect_lo, aspect_hi = state_dict["search_space"]["aspect_ratio"]
    dsp_lo, dsp_hi = state_dict["search_space"]["dsp_startx"]
    bram_lo, bram_hi = state_dict["search_space"]["bram_startx"]

    def normalize(value: float, lo: float, hi: float) -> float:
        if hi == lo:
            return 0.0
        return ((value - lo) / (hi - lo)) * 2.0 - 1.0

    return [
        normalize(float(action.aspect_ratio), float(aspect_lo), float(aspect_hi)),
        normalize(float(action.dsp_startx), float(dsp_lo), float(dsp_hi)),
        normalize(float(action.bram_startx), float(bram_lo), float(bram_hi)),
    ]


def build_state_vector(
    state_dict: dict,
    previous_action: LayoutAction | None,
    previous_reward: float,
    best_action: LayoutAction | None,
    best_reward: float,
) -> torch.Tensor:
    cp = float(state_dict["baseline_critical_path_ns"])
    wire = float(state_dict["baseline_total_wirelength"])
    aspect_lo, aspect_hi = state_dict["search_space"]["aspect_ratio"]
    dsp_lo, dsp_hi = state_dict["search_space"]["dsp_startx"]
    bram_lo, bram_hi = state_dict["search_space"]["bram_startx"]
    prev_action_features = action_to_feature_vector(previous_action, state_dict)
    best_action_features = action_to_feature_vector(best_action, state_dict)

    vec = torch.tensor(
        [
            cp / 100.0,
            wire / 100000.0,
            float(aspect_lo),
            float(aspect_hi),
            float(dsp_lo) / 20.0,
            float(dsp_hi) / 20.0,
            float(bram_lo) / 20.0,
            float(bram_hi) / 20.0,
            *prev_action_features,
            previous_reward,
            *best_action_features,
            best_reward,
        ],
        dtype=torch.float32,
    )
    return vec


def raw_action_to_layout(raw_action: torch.Tensor, state_dict: dict) -> LayoutAction:
    aspect_lo, aspect_hi = state_dict["search_space"]["aspect_ratio"]
    dsp_lo, dsp_hi = state_dict["search_space"]["dsp_startx"]
    bram_lo, bram_hi = state_dict["search_space"]["bram_startx"]

    scaled = torch.tanh(raw_action)

    def scale_component(value: float, lo: float, hi: float) -> float:
        return lo + ((value + 1.0) * 0.5 * (hi - lo))

    aspect = scale_component(float(scaled[0]), float(aspect_lo), float(aspect_hi))
    dsp = int(round(scale_component(float(scaled[1]), float(dsp_lo), float(dsp_hi))))
    bram = int(round(scale_component(float(scaled[2]), float(bram_lo), float(bram_hi))))

    if dsp == bram:
        if bram < bram_hi:
            bram += 1
        elif bram > bram_lo:
            bram -= 1

    return LayoutAction(
        aspect_ratio=aspect,
        dsp_startx=dsp,
        bram_startx=bram,
    )


def select_action(model: PolicyValueNet, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean, value = model(state.unsqueeze(0))
    std = torch.exp(model.log_std).unsqueeze(0)
    dist = Normal(mean, std)
    action = dist.sample()
    logprob = dist.log_prob(action).sum(dim=-1)
    return action.squeeze(0), logprob.squeeze(0), value.squeeze(0)


def ppo_update(
    model: PolicyValueNet,
    optimizer: optim.Optimizer,
    transitions: list[Transition],
    clip_eps: float,
    entropy_coef: float,
    value_coef: float,
    ppo_steps: int,
) -> dict[str, float]:
    states = torch.stack([t.state for t in transitions], dim=0)
    actions = torch.stack([t.action for t in transitions], dim=0)
    old_logprobs = torch.stack([t.logprob for t in transitions], dim=0).detach()
    rewards = torch.tensor([t.reward for t in transitions], dtype=torch.float32)
    old_values = torch.stack([t.value for t in transitions], dim=0).detach()

    advantages = rewards - old_values
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    last_loss = 0.0
    last_actor = 0.0
    last_critic = 0.0
    last_entropy = 0.0

    for _ in range(ppo_steps):
        mean, values = model(states)
        std = torch.exp(model.log_std).unsqueeze(0).expand_as(mean)
        dist = Normal(mean, std)
        new_logprobs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1).mean()

        ratios = torch.exp(new_logprobs - old_logprobs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = torch.mean((rewards - values) ** 2)
        loss = actor_loss + (value_coef * critic_loss) - (entropy_coef * entropy)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        last_loss = float(loss.item())
        last_actor = float(actor_loss.item())
        last_critic = float(critic_loss.item())
        last_entropy = float(entropy.item())

    return {
        "loss": last_loss,
        "actor_loss": last_actor,
        "critic_loss": last_critic,
        "entropy": last_entropy,
    }


def scheduled_value(epoch: int, total_epochs: int, start: float, end: float) -> float:
    if total_epochs <= 1:
        return end
    alpha = epoch / float(total_epochs - 1)
    return ((1.0 - alpha) * start) + (alpha * end)


def evaluate_layout(cfg, baseline_metrics: VTRMetrics, layout_action: LayoutAction, eval_name: str) -> dict:
    env = MacroPlacementEnv(cfg)
    env.set_baseline_metrics(baseline_metrics)
    step_result = env.evaluate_action(layout_action, eval_name=eval_name)
    return {
        "reward": float(step_result.reward),
        "info": step_result.info,
    }


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    cfg = default_config()
    save_dir = args.save_dir.expanduser().resolve() if args.save_dir else (cfg.runs_root / "ppo")
    save_dir.mkdir(parents=True, exist_ok=True)

    env = MacroPlacementEnv(cfg)
    initial_state = env.reset()
    baseline_metrics = env.baseline_metrics
    if baseline_metrics is None:
        raise RuntimeError("Baseline metrics were not initialized after env.reset()")
    previous_action: LayoutAction | None = None
    previous_reward = 0.0
    best_action: LayoutAction | None = None
    baseline_cp = initial_state["baseline_critical_path_ns"]
    baseline_wire = initial_state["baseline_total_wirelength"]
    state_dim = build_state_vector(
        initial_state,
        previous_action=previous_action,
        previous_reward=previous_reward,
        best_action=best_action,
        best_reward=0.0,
    ).numel()
    action_dim = 3
    parallel_workers = args.parallel_workers if args.parallel_workers > 0 else min(args.batch_size, os.cpu_count() or 1)

    model = PolicyValueNet(state_dim=state_dim, action_dim=action_dim, hidden_dim=args.hidden_dim)
    model.set_log_std(args.log_std_initial)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_reward = -math.inf
    history: list[dict] = []

    for epoch in range(args.epochs):
        entropy_coef_epoch = scheduled_value(
            epoch=epoch,
            total_epochs=args.epochs,
            start=args.entropy_coef,
            end=args.entropy_coef_final,
        )
        log_std_epoch = scheduled_value(
            epoch=epoch,
            total_epochs=args.epochs,
            start=args.log_std_initial,
            end=args.log_std_final,
        )
        model.set_log_std(log_std_epoch)

        transitions: list[Transition] = []
        epoch_rewards: list[float] = []
        best_epoch_info: dict | None = None
        pending_samples: list[dict] = []

        for batch_index in range(args.batch_size):
            state_dict = env.reset()
            state_vec = build_state_vector(
                state_dict,
                previous_action=previous_action,
                previous_reward=previous_reward,
                best_action=best_action,
                best_reward=(0.0 if best_reward == -math.inf else best_reward),
            )
            raw_action, logprob, value = select_action(model, state_vec)
            layout_action = raw_action_to_layout(raw_action, state_dict)
            eval_index = env.eval_index
            env.eval_index += 1
            eval_name = f"eval_{eval_index:04d}_{layout_action.tag()}"
            pending_samples.append(
                {
                    "transition": Transition(
                        state=state_vec,
                        action=raw_action.detach(),
                        logprob=logprob.detach(),
                        reward=0.0,
                        value=value.detach(),
                    ),
                    "layout_action": layout_action,
                    "eval_name": eval_name,
                    "batch_index": batch_index,
                }
            )

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_layout,
                    cfg,
                    baseline_metrics,
                    sample["layout_action"],
                    sample["eval_name"],
                ): sample
                for sample in pending_samples
            }
            completed_samples: list[dict] = []
            for future in concurrent.futures.as_completed(future_map):
                sample = future_map[future]
                result = future.result()
                sample["transition"].reward = result["reward"]
                sample["info"] = result["info"]
                completed_samples.append(sample)

        completed_samples.sort(key=lambda sample: sample["batch_index"])

        for sample in completed_samples:
            transition = sample["transition"]
            info = sample["info"]
            transitions.append(transition)
            epoch_rewards.append(float(transition.reward))

            if best_epoch_info is None or transition.reward > float(best_epoch_info["reward"]):
                best_epoch_info = info

            if transition.reward > best_reward:
                best_reward = float(transition.reward)
                best_action = LayoutAction(
                    aspect_ratio=float(info["action"]["aspect_ratio"]),
                    dsp_startx=int(info["action"]["dsp_startx"]),
                    bram_startx=int(info["action"]["bram_startx"]),
                )
                torch.save(model.state_dict(), save_dir / "best_policy.pt")
                (save_dir / "best_result.json").write_text(json.dumps(info, indent=2), encoding="utf-8")

        last_sample = completed_samples[-1]
        previous_action = last_sample["layout_action"]
        previous_reward = float(last_sample["transition"].reward)

        metrics = ppo_update(
            model=model,
            optimizer=optimizer,
            transitions=transitions,
            clip_eps=args.clip_eps,
            entropy_coef=entropy_coef_epoch,
            value_coef=args.value_coef,
            ppo_steps=args.ppo_steps,
        )

        summary = {
            "epoch": epoch,
            "mean_reward": sum(epoch_rewards) / len(epoch_rewards),
            "best_reward": max(epoch_rewards),
            "batch_size": len(epoch_rewards),
            "parallel_workers": parallel_workers,
            "baseline_critical_path_ns": baseline_cp,
            "baseline_total_wirelength": baseline_wire,
            "previous_reward_for_next_epoch": previous_reward,
            "best_reward_so_far": (None if best_reward == -math.inf else best_reward),
            "entropy_coef_epoch": entropy_coef_epoch,
            "log_std_epoch": log_std_epoch,
            **metrics,
            "best_epoch_info": best_epoch_info,
        }
        history.append(summary)
        print(json.dumps(summary, indent=2))

        torch.save(model.state_dict(), save_dir / "latest_policy.pt")
        (save_dir / "training_history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
