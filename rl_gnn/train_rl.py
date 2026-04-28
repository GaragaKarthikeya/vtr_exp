#!/usr/bin/env python3

from __future__ import annotations

import json
import shutil
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from env import VTRPlacementEnv


class BestResultWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, save_dir: str = "rl_best"):
        super().__init__(env)

        self.script_dir = Path(__file__).resolve().parent
        self.save_dir = self.script_dir / save_dir
        self.best_reward = -float("inf")

        self.save_dir.mkdir(parents=True, exist_ok=True)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward > self.best_reward:
            self.best_reward = float(reward)

            best_data = {
                "best_reward": self.best_reward,
                "action": np.asarray(action).tolist(),
                "info": info,
            }

            (self.save_dir / "best_result.json").write_text(
                json.dumps(best_data, indent=4)
            )

            arch_file = self.script_dir / "vpr_arch_run.xml"
            if arch_file.exists():
                shutil.copy2(arch_file, self.save_dir / "raygentop_best_vpr_arch_run.xml")

            metric_file = self.script_dir / "custom_run" / "custom_metric.txt"
            if metric_file.exists():
                shutil.copy2(metric_file, self.save_dir / "raygentop_best_custom_metric.txt")

            run_dir = self.script_dir / "custom_run"
            best_run_dir = self.save_dir / "raygentop_best_custom_run"

            if run_dir.exists():
                if best_run_dir.exists():
                    shutil.rmtree(best_run_dir)
                shutil.copytree(run_dir, best_run_dir)

            print(f"New best reward: {self.best_reward:.4f}")
            print(f"Best result saved in: {self.save_dir}")

        return obs, reward, terminated, truncated, info


def main() -> None:
    env = VTRPlacementEnv(
        req_file_path="resources.txt",
        baseline_file_path="raygentop_traditional_metric.txt",
    )

    env = BestResultWrapper(env, save_dir="rl_best")
    env = Monitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=8,
        batch_size=8,
        gamma=1.0,
    )

    model.learn(total_timesteps=50)

    model.save("vtr_placement_ppo")

    obs, _ = env.reset()
    action, _ = model.predict(obs, deterministic=True)

    print("Testing final trained action:")
    print(action)

    obs, reward, terminated, truncated, info = env.step(action)

    print("Final trained reward:", reward)
    print("Final trained info:")
    print(json.dumps(info, indent=4))


if __name__ == "__main__":
    main()
