#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def find_single_architecture(arch_dir: Path) -> Path:
    arch_files = sorted(arch_dir.glob("*.xml"))
    if not arch_files:
        raise FileNotFoundError(f"No .xml architecture files found in: {arch_dir}")
    if len(arch_files) > 1:
        names = "\n  ".join(p.name for p in arch_files)
        raise ValueError(
            "Multiple architecture XML files found. Use --arch-file to choose one:\n"
            f"  {names}"
        )
    return arch_files[0]


def find_tests(test_dir: Path) -> list[Path]:
    tests = sorted(test_dir.glob("*.v"))
    if not tests:
        raise FileNotFoundError(f"No .v test files found in: {test_dir}")
    return tests


def run_one(run_vtr_script: Path, arch_file: Path, test_file: Path, out_dir: Path, extra_args: list[str]) -> int:
    cmd = [
        sys.executable,
        str(run_vtr_script),
        str(arch_file),
        str(test_file),
        str(out_dir),
        *extra_args,
    ]
    return subprocess.run(cmd, check=False).returncode


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Run all test Verilog files against an architecture via run-vtr.py"
    )
    parser.add_argument(
        "--arch-dir",
        type=Path,
        default=root / "arch",
        help="Directory containing architecture XML files (default: ./arch)",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=root / "tests",
        help="Directory containing test .v files (default: ./tests)",
    )
    parser.add_argument(
        "--arch-file",
        type=Path,
        default=None,
        help="Path to a specific architecture XML file (overrides --arch-dir scan)",
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=root / "runs",
        help="Base output directory for run outputs (default: ./runs)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running remaining tests even if one fails",
    )
    parser.add_argument(
        "vtr_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to run-vtr.py (prefix with --)",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    run_vtr_script = root / "run-vtr.py"

    if not run_vtr_script.is_file():
        print(f"Missing glue script: {run_vtr_script}", file=sys.stderr)
        return 1

    arch_dir = args.arch_dir.expanduser().resolve()
    test_dir = args.test_dir.expanduser().resolve()
    runs_dir = args.runs_dir.expanduser().resolve()

    if args.arch_file is not None:
        arch_file = args.arch_file.expanduser().resolve()
        if not arch_file.is_file():
            print(f"Architecture file not found: {arch_file}", file=sys.stderr)
            return 1
    else:
        try:
            arch_file = find_single_architecture(arch_dir)
        except (FileNotFoundError, ValueError) as exc:
            print(str(exc), file=sys.stderr)
            return 1

    try:
        tests = find_tests(test_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    runs_dir.mkdir(parents=True, exist_ok=True)

    forwarded_args = [arg for arg in args.vtr_args if arg != "--"]

    failures: list[tuple[str, int]] = []

    print(f"Architecture: {arch_file}")
    print(f"Tests: {len(tests)}")
    print(f"Output root: {runs_dir}\n")

    for index, test_file in enumerate(tests, start=1):
        out_dir = runs_dir / test_file.stem
        print(f"[{index}/{len(tests)}] Running {test_file.name} -> {out_dir}")
        code = run_one(run_vtr_script, arch_file, test_file, out_dir, forwarded_args)

        if code == 0:
            print("  PASS\n")
            continue

        print(f"  FAIL (exit code {code})\n")
        failures.append((test_file.name, code))
        if not args.continue_on_error:
            break

    if failures:
        print("Failed tests:", file=sys.stderr)
        for name, code in failures:
            print(f"  - {name}: {code}", file=sys.stderr)
        return 1

    print("All tests completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
