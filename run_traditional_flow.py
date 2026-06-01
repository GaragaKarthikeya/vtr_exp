#!/usr/bin/env python3

import os
import re
import sys
import json
import argparse
import shutil
import subprocess
from pathlib import Path

def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        raise FileNotFoundError(f"Missing .env file at: {env_file}")

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = os.path.expandvars(value.strip().strip("'\""))

def run_vtr_flow(benchmark_file: Path, arch_file: Path, output_dir: Path) -> int:
    vtr_python = Path(os.environ.get("VTR_VENV_PATH", "/home/digital-2/.venv")) / "bin" / "python"
    if not vtr_python.is_file():
        vtr_python = Path(sys.executable)

    vtr_flow_script = Path(os.environ.get("VTR_FLOW_SCRIPT", "/home/digital-2/vtr/vtr_flow/scripts/run_vtr_flow.py"))
    power_tech_file = Path(os.environ.get("VTR_POWER_TECH_FILE", "/home/digital-2/vtr/vtr_flow/tech/PTM_45nm/45nm.xml"))

    if not vtr_flow_script.is_file():
        print(f"Error: run_vtr_flow.py not found at: {vtr_flow_script}", file=sys.stderr)
        return 1

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(vtr_python),
        str(vtr_flow_script),
        str(benchmark_file),
        str(arch_file),
        "-temp_dir",
        str(output_dir),
    ]

    if power_tech_file.is_file():
        cmd.extend(["-cmos_tech", str(power_tech_file)])

    print(f"Executing VTR flow command:")
    print(" ".join(cmd))
    print("-" * 60)

    # Forward stdout/stderr to the console while running
    return subprocess.run(cmd, check=False).returncode

def parse_metrics(vpr_out_file: Path, crit_path_file: Path, power_file: Path, output_file: Path) -> dict:
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
    print(f"Stored metrics in {output_file}")
    return metrics

def parse_resources(vpr_out_file: Path, output_file: Path) -> dict:
    resources = {
        "fpga_size": [0, 0],
        "requirements": {  # What the circuit NEEDS (Netlist)
            "io": 0, "clb": 0, "dsp": 0, "bram": 0
        },
        "limits": {        # What the architecture HAS (Architecture)
            "io": 0, "clb": 0, "dsp": 0, "bram": 0
        }
    }

    if not vpr_out_file.exists():
        print(f"Error: {vpr_out_file} not found.")
        return resources

    content = vpr_out_file.read_text(errors="ignore")

    # 1. Extract FPGA Size (Limits)
    # Matches: FPGA sized to 16 x 16
    size_match = re.search(r"FPGA sized to\s+([0-9]+)\s+x\s+([0-9]+)", content)
    if size_match:
        resources["fpga_size"] = [int(size_match.group(1))-2, int(size_match.group(2))-2]
    
    # 2. Extract Netlist Requirements & Architecture Limits
    resource_map = {
        "io": "io",
        "clb": "clb",
        "mult_36": "dsp",
        "memory": "bram"
    }

    for vpr_label, key in resource_map.items():
        req_pattern = rf"Netlist\s+([0-9]+)\s+blocks of type: {vpr_label}"
        lim_pattern = rf"Architecture\s+([0-9]+)\s+blocks of type: {vpr_label}"
        
        req_match = re.search(req_pattern, content)
        lim_match = re.search(lim_pattern, content)
        
        if req_match:
            resources["requirements"][key] = int(req_match.group(1))
        if lim_match:
            resources["limits"][key] = int(lim_match.group(1))

    output_file.write_text(json.dumps(resources, indent=4))
    print(f"Stored resources in {output_file}")
    return resources

def main() -> int:
    parser = argparse.ArgumentParser(description="Run VTR Flow on Traditional Architecture & Extract Metrics")
    parser.add_argument("--benchmark", type=str, required=True, help="Name of the benchmark (e.g. diffeq1)")
    parser.add_argument("--arch", type=str, default="arch/k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml", 
                        help="Path to the architecture XML file (default: arch/k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml)")
    
    args = parser.parse_args()
    
    script_dir = Path(__file__).resolve().parent
    env_file = script_dir / ".env"
    
    try:
        load_env_file(env_file)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    benchmark_name = args.benchmark
    # Support both bare name (diffeq1) and full name (diffeq1.v)
    if benchmark_name.endswith(".v"):
        benchmark_name = benchmark_name[:-2]

    benchmark_file = script_dir / "benchmarks" / f"{benchmark_name}.v"
    arch_file = Path(args.arch).expanduser().resolve()
    output_dir = script_dir / "runs" / f"{benchmark_name}_traditional"

    if not benchmark_file.is_file():
        print(f"Error: Benchmark Verilog file not found at: {benchmark_file}", file=sys.stderr)
        return 1
    
    if not arch_file.is_file():
        print(f"Error: Architecture XML file not found at: {arch_file}", file=sys.stderr)
        return 1

    # Run the VTR flow
    return_code = run_vtr_flow(benchmark_file, arch_file, output_dir)
    
    print("-" * 60)
    if return_code != 0:
        print(f"VTR Flow failed with return code: {return_code}. Attempting to parse whatever was generated...", file=sys.stderr)
    
    # Define parsing source paths
    vpr_out_file = output_dir / "vpr.out"
    crit_path_file = output_dir / "vpr.crit_path.out"
    power_file = output_dir / f"{benchmark_name}.power"
    
    # Define destination paths
    metric_dest = script_dir / f"{benchmark_name}_traditional_metric.txt"
    resource_dest = script_dir / f"{benchmark_name}_traditional_resources.txt"

    # Parse and save
    metrics = parse_metrics(vpr_out_file, crit_path_file, power_file, metric_dest)
    resources = parse_resources(vpr_out_file, resource_dest)

    print("-" * 60)
    print("Parsed Traditional Performance Metrics:")
    print(json.dumps(metrics, indent=4))
    print("\nParsed Traditional Resources and Limits:")
    print(json.dumps(resources, indent=4))

    return return_code

if __name__ == "__main__":
    raise SystemExit(main())
