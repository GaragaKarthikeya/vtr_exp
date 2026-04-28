#!/usr/bin/env python3

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


def run_vtr(script_dir: Path) -> int:
    arch_file = script_dir / "vpr_arch_run.xml"
    circuit_file = script_dir / "raygentop.v"
    output_dir = script_dir / "custom_run"

    vtr_root = Path("/root/vtr-verilog-to-routing")
    vtr_python = vtr_root / ".venv" / "bin" / "python"
    vtr_flow_script = vtr_root / "vtr_flow" / "scripts" / "run_vtr_flow.py"
    cmos_tech_file = vtr_root / "vtr_flow" / "tech" / "PTM_45nm" / "45nm.xml"

    if not arch_file.is_file():
        print(f"Missing architecture file: {arch_file}", file=sys.stderr)
        return 1

    if not circuit_file.is_file():
        print(f"Missing circuit file: {circuit_file}", file=sys.stderr)
        return 1

    if not vtr_python.is_file():
        print(f"Missing VTR Python: {vtr_python}", file=sys.stderr)
        return 1

    if not vtr_flow_script.is_file():
        print(f"Missing run_vtr_flow.py: {vtr_flow_script}", file=sys.stderr)
        return 1

    if output_dir.exists():
        shutil.rmtree(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(vtr_python),
        str(vtr_flow_script),
        str(circuit_file),
        str(arch_file),
        "-temp_dir",
        str(output_dir),
    ]

    if cmos_tech_file.is_file():
        cmd.extend(["-cmos_tech", str(cmos_tech_file)])

    print("Running VTR:")
    print(" ".join(cmd))

    return subprocess.run(cmd, check=False).returncode


def extract_metrics(script_dir: Path) -> dict[str, float]:
    run_dir = script_dir / "custom_run"
    circuit_name = "raygentop"

    vpr_out_file = run_dir / "vpr.out"
    crit_path_file = run_dir / "vpr.crit_path.out"
    power_file = run_dir / f"{circuit_name}.power"
    output_file = run_dir / "custom_metric.txt"

    metrics = {
        "delay_ns": float("inf"),
        "wirelength": float("inf"),
        "power_w": float("inf"),
    }

    # Parse wirelength
    if vpr_out_file.exists():
        content = vpr_out_file.read_text(errors="ignore")

        wl_pattern = (
            r"Wire length results.*?"
            r"Total wirelength:\s+([0-9]+),\s+average net length:\s+([0-9.]+)"
        )

        match = re.search(wl_pattern, content, re.DOTALL)

        if match:
            metrics["wirelength"] = float(match.group(1))
        else:
            print(f"Warning: Could not extract wirelength stats from {vpr_out_file}")
    else:
        print(f"Warning: Could not find {vpr_out_file}")

    # Parse critical path delay
    if crit_path_file.exists():
        content = crit_path_file.read_text(errors="ignore")

        match = re.search(
            r"Final critical path delay \(least slack\):\s+([0-9.]+)\s+ns",
            content,
        )

        if match:
            metrics["delay_ns"] = float(match.group(1))
        else:
            print(f"Warning: Could not extract delay from {crit_path_file}")
    else:
        print(f"Warning: Could not find {crit_path_file}")

    # Parse total power
    if power_file.exists():
        content = power_file.read_text(errors="ignore")

        match = re.search(
            r"^Total\s+([0-9\.eE\-]+)",
            content,
            re.MULTILINE,
        )

        if match:
            metrics["power_w"] = float(match.group(1))
        else:
            print(f"Warning: Could not extract power from {power_file}")
    else:
        print(f"Warning: Could not find {power_file}")

    output_file.write_text(json.dumps(metrics, indent=4))

    print(f"Extraction complete. Data saved to {output_file}:")
    print(json.dumps(metrics, indent=4))

    return metrics


def main() -> int:
    script_dir = Path(__file__).resolve().parent

    return_code = run_vtr(script_dir)

    if return_code != 0:
        print(f"VTR failed with return code {return_code}", file=sys.stderr)
        return return_code

    extract_metrics(script_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
