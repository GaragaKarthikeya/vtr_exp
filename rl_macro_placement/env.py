from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import ExperimentConfig
from .layout_space import LayoutAction, clamp_action, write_arch_variant
from .vtr_metrics import VTRMetrics, compute_reward, run_vtr


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    info: dict


class MacroPlacementEnv:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.cfg.runs_root.mkdir(parents=True, exist_ok=True)
        self.variants_dir = self.cfg.runs_root / "arch_variants"
        self.evals_dir = self.cfg.runs_root / "evaluations"
        self.variants_dir.mkdir(parents=True, exist_ok=True)
        self.evals_dir.mkdir(parents=True, exist_ok=True)

        self.baseline_metrics: VTRMetrics | None = None
        self.eval_index = 0

    def set_baseline_metrics(self, baseline_metrics: VTRMetrics) -> None:
        self.baseline_metrics = baseline_metrics

    def reset(self) -> dict:
        if self.baseline_metrics is None:
            self.baseline_metrics = self._ensure_baseline()

        return {
            "baseline_critical_path_ns": self.baseline_metrics.critical_path_ns,
            "baseline_total_wirelength": self.baseline_metrics.total_wirelength,
            "search_space": {
                "aspect_ratio": [self.cfg.aspect_ratio_min, self.cfg.aspect_ratio_max],
                "dsp_startx": [self.cfg.dsp_startx_min, self.cfg.dsp_startx_max],
                "bram_startx": [self.cfg.bram_startx_min, self.cfg.bram_startx_max],
            },
        }

    def step(self, action: LayoutAction) -> StepResult:
        if self.baseline_metrics is None:
            self.baseline_metrics = self._ensure_baseline()

        legal_action = clamp_action(action, self.cfg)
        eval_name = f"eval_{self.eval_index:04d}_{legal_action.tag()}"
        self.eval_index += 1
        return self.evaluate_action(legal_action, eval_name=eval_name)

    def evaluate_action(self, action: LayoutAction, eval_name: str) -> StepResult:
        if self.baseline_metrics is None:
            self.baseline_metrics = self._ensure_baseline()

        legal_action = clamp_action(action, self.cfg)
        out_dir = self.evals_dir / eval_name
        arch_variant = self.variants_dir / f"{eval_name}.xml"

        write_arch_variant(self.cfg.arch_file, legal_action, arch_variant)
        result = run_vtr(
            repo_root=self.cfg.repo_root,
            arch_file=arch_variant,
            circuit_file=self.cfg.circuit_file,
            out_dir=out_dir,
            extra_args=list(self.cfg.vtr_extra_args),
            timeout_sec=self.cfg.timeout_sec,
        )

        if not result.ok or result.metrics is None:
            info = {
                "success": False,
                "action": asdict(legal_action),
                "arch_variant": str(arch_variant),
                "out_dir": str(out_dir),
                "reason": result.reason,
                "returncode": result.returncode,
            }
            self._write_summary(out_dir / "summary.json", info)
            return StepResult(reward=self.cfg.failure_reward, done=True, info=info)

        reward = compute_reward(
            baseline=self.baseline_metrics,
            current=result.metrics,
            cp_weight=self.cfg.cp_weight,
            wire_weight=self.cfg.wire_weight,
            cp_regression_penalty=self.cfg.cp_regression_penalty,
        )

        info = {
            "success": True,
            "action": asdict(legal_action),
            "arch_variant": str(arch_variant),
            "out_dir": str(out_dir),
            "critical_path_ns": result.metrics.critical_path_ns,
            "total_wirelength": result.metrics.total_wirelength,
            "baseline_critical_path_ns": self.baseline_metrics.critical_path_ns,
            "baseline_total_wirelength": self.baseline_metrics.total_wirelength,
            "reward": reward,
        }
        self._write_summary(out_dir / "summary.json", info)
        return StepResult(reward=reward, done=True, info=info)

    def _ensure_baseline(self) -> VTRMetrics:
        out_dir = self.cfg.runs_root / "baseline"
        result = run_vtr(
            repo_root=self.cfg.repo_root,
            arch_file=self.cfg.arch_file,
            circuit_file=self.cfg.circuit_file,
            out_dir=out_dir,
            extra_args=list(self.cfg.vtr_extra_args),
            timeout_sec=self.cfg.timeout_sec,
        )
        if not result.ok or result.metrics is None:
            raise RuntimeError(f"Baseline run failed: {result.reason or result.returncode}")

        self._write_summary(
            out_dir / "summary.json",
            {
                "success": True,
                "arch_file": str(self.cfg.arch_file),
                "circuit_file": str(self.cfg.circuit_file),
                "critical_path_ns": result.metrics.critical_path_ns,
                "total_wirelength": result.metrics.total_wirelength,
            },
        )
        return result.metrics

    @staticmethod
    def _write_summary(path: Path, payload: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
