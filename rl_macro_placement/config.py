from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentConfig:
    repo_root: Path
    arch_file: Path
    circuit_file: Path
    runs_root: Path
    aspect_ratio_min: float = 0.7
    aspect_ratio_max: float = 1.8
    dsp_startx_min: int = 2
    dsp_startx_max: int = 14
    bram_startx_min: int = 2
    bram_startx_max: int = 14
    cp_weight: float = 0.8
    wire_weight: float = 0.2
    cp_regression_penalty: float = 1.0
    failure_reward: float = -5.0
    timeout_sec: int = 0
    vtr_extra_args: tuple[str, ...] = ()


def default_config() -> ExperimentConfig:
    repo_root = Path(__file__).resolve().parent.parent
    return ExperimentConfig(
        repo_root=repo_root,
        arch_file=repo_root / "arch" / "k6_frac_N10_frac_chain_depop50_mem32K_40nm.xml",
        circuit_file=repo_root / "tests" / "diffeq1.v",
        runs_root=repo_root / "rl_macro_placement" / "runs",
    )
