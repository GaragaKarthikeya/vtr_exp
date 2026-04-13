#!/usr/bin/env python3

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        raise FileNotFoundError(f"Missing .env at: {env_file}")

    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = os.path.expandvars(value.strip().strip("'\""))


def validate_inputs(arch_file: Path, circuit_file: Path) -> None:
    if arch_file.suffix != ".xml":
        raise ValueError(f"Expected an architecture XML as the first argument, got: {arch_file}")

    if circuit_file.suffix not in {".v", ".sv", ".eblif"}:
        raise ValueError(
            f"Expected a circuit file (.v, .sv, or .eblif) as the second argument, got: {circuit_file}"
        )


def main(argv: list[str]) -> int:
    script_dir = Path(__file__).resolve().parent
    env_file = script_dir / ".env"

    try:
        load_env_file(env_file)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    vtr_root = Path(os.environ.get("VTR_ROOT", "/home/karthikeya/vtr-verilog-to-routing"))
    vtr_venv_path = Path(os.environ.get("VTR_VENV_PATH", str(vtr_root / ".venv")))
    vtr_flow_script = Path(os.environ.get("VTR_FLOW_SCRIPT", ""))

    if not vtr_flow_script.is_file():
        print("VTR_FLOW_SCRIPT is missing or invalid in .env", file=sys.stderr)
        return 1

    venv_python = vtr_venv_path / "bin" / "python"
    if not venv_python.is_file():
        venv_python = vtr_root / ".venv" / "bin" / "python"

    if not venv_python.is_file():
        print(
            f"Virtualenv Python not found at {vtr_venv_path / 'bin' / 'python'} or {vtr_root / '.venv' / 'bin' / 'python'}",
            file=sys.stderr,
        )
        return 1

    if len(argv) < 2:
        print(
            "Usage:\n"
            "  ./run-vtr.py <arch.xml> <circuit.v|circuit.eblif> [output-dir] [extra run_vtr_flow.py args...]\n\n"
            "Examples:\n"
            "  ./run-vtr.py vtr_arch/timing/<arch-file>.xml vtr_tests/<design>.v\n"
            "  ./run-vtr.py vtr_arch/timing/<arch-file>.xml vtr_tests/<design>.v runs/diffeq1\n",
            file=sys.stderr,
        )
        return 1

    arch_file = Path(argv[0])
    circuit_file = Path(argv[1])
    remaining = argv[2:]

    try:
        validate_inputs(arch_file, circuit_file)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    output_dir = script_dir / "temp"
    extra_args = remaining

    if remaining and not remaining[0].startswith("-"):
        output_dir = Path(remaining[0]).expanduser().resolve()
        extra_args = remaining[1:]

    if output_dir == script_dir / "temp" and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(venv_python),
        str(vtr_flow_script),
        str(circuit_file),
        str(arch_file),
        "-temp_dir",
        str(output_dir),
        *extra_args,
    ]

    return subprocess.run(cmd, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
