from __future__ import annotations

import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


CP_RE = re.compile(r"Final critical path delay \(least slack\):\s*([0-9.eE+-]+)\s*ns")
WIRE_RE = re.compile(r"Total wirelength:\s*([0-9.eE+-]+)")


@dataclass(frozen=True)
class VTRMetrics:
    critical_path_ns: float
    total_wirelength: float


@dataclass(frozen=True)
class VTRRunResult:
    ok: bool
    out_dir: Path
    metrics: VTRMetrics | None
    returncode: int
    reason: str | None = None


def run_vtr(repo_root: Path, arch_file: Path, circuit_file: Path, out_dir: Path, extra_args: list[str], timeout_sec: int) -> VTRRunResult:
    cmd = [
        sys.executable,
        str(repo_root / "run-vtr.py"),
        str(arch_file),
        str(circuit_file),
        str(out_dir),
        *extra_args,
    ]

    try:
        completed = subprocess.run(
            cmd,
            check=False,
            timeout=(None if timeout_sec <= 0 else timeout_sec),
        )
    except subprocess.TimeoutExpired:
        return VTRRunResult(
            ok=False,
            out_dir=out_dir,
            metrics=None,
            returncode=124,
            reason="timeout",
        )

    if completed.returncode != 0:
        return VTRRunResult(
            ok=False,
            out_dir=out_dir,
            metrics=None,
            returncode=completed.returncode,
            reason="vtr_failed",
        )

    try:
        metrics = parse_vpr_metrics(out_dir / "vpr.out")
    except Exception as exc:
        return VTRRunResult(
            ok=False,
            out_dir=out_dir,
            metrics=None,
            returncode=completed.returncode,
            reason=f"parse_failed:{exc}",
        )

    return VTRRunResult(
        ok=True,
        out_dir=out_dir,
        metrics=metrics,
        returncode=completed.returncode,
        reason=None,
    )


def parse_vpr_metrics(vpr_out_file: Path) -> VTRMetrics:
    if not vpr_out_file.is_file():
        raise FileNotFoundError(f"Missing VPR log: {vpr_out_file}")

    critical_path_ns = None
    total_wirelength = None

    for raw_line in vpr_out_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()

        if total_wirelength is None:
            wire_match = WIRE_RE.search(line)
            if wire_match:
                total_wirelength = float(wire_match.group(1))

        if critical_path_ns is None:
            cp_match = CP_RE.search(line)
            if cp_match:
                critical_path_ns = float(cp_match.group(1))

        if critical_path_ns is not None and total_wirelength is not None:
            break

    if critical_path_ns is None:
        raise ValueError("Could not parse final critical path delay from vpr.out")
    if total_wirelength is None:
        raise ValueError("Could not parse total wirelength from vpr.out")

    return VTRMetrics(
        critical_path_ns=critical_path_ns,
        total_wirelength=total_wirelength,
    )


def compute_reward(
    baseline: VTRMetrics,
    current: VTRMetrics,
    cp_weight: float,
    wire_weight: float,
    cp_regression_penalty: float,
) -> float:
    cp_improvement = (baseline.critical_path_ns - current.critical_path_ns) / baseline.critical_path_ns
    wire_improvement = (baseline.total_wirelength - current.total_wirelength) / baseline.total_wirelength
    cp_regression = max(0.0, -cp_improvement)
    return (
        (cp_weight * cp_improvement)
        + (wire_weight * wire_improvement)
        - (cp_regression_penalty * cp_regression)
    )
