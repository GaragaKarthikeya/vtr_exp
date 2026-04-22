#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import re
import subprocess
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    aspect_ratio: float
    dsp_startx: int
    bram_startx: int


@dataclass
class EvalResult:
    ok: bool
    reward: float
    power_w: float | None
    critical_path_s: float | None
    out_dir: Path
    reason: str | None = None


CSV_FIELDS = [
    "iteration",
    "aspect_ratio",
    "dsp_startx",
    "bram_startx",
    "ok",
    "reward",
    "power_w",
    "critical_path_s",
    "out_dir",
    "reason",
]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Simple RL (CEM) search over VTR layout knobs for one benchmark and one architecture"
    )
    parser.add_argument(
        "--arch",
        type=Path,
        default=root / "vtr_arch" / "power" / "k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml",
        help="Base architecture XML",
    )
    parser.add_argument(
        "--circuit",
        type=Path,
        default=root / "vtr_tests" / "verilog" / "diffeq1.v",
        help="Benchmark circuit",
    )
    parser.add_argument("--iterations", type=int, default=8, help="CEM iterations")
    parser.add_argument("--population", type=int, default=6, help="Candidates per iteration")
    parser.add_argument("--elite-frac", type=float, default=0.34, help="Elite fraction for CEM update")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint if available")
    parser.add_argument("--checkpoint-file", type=Path, default=None, help="Checkpoint file path")
    parser.add_argument("--log-file", type=Path, default=None, help="Search log file path")
    parser.add_argument(
        "--console-log",
        action="store_true",
        help="Also stream logs to terminal (default: file-only logging)",
    )
    parser.add_argument("--timeout-sec", type=int, default=0, help="Per VTR run timeout in seconds (0 disables)")
    parser.add_argument("--max-retries", type=int, default=1, help="Retries per failed VTR run")
    parser.add_argument(
        "--plot-every-eval",
        action="store_true",
        help="Update progress plots after each evaluation (default: per iteration)",
    )

    parser.add_argument("--aspect-min", type=float, default=0.7)
    parser.add_argument("--aspect-max", type=float, default=1.8)
    parser.add_argument("--dsp-x-min", type=int, default=2)
    parser.add_argument("--dsp-x-max", type=int, default=14)
    parser.add_argument("--bram-x-min", type=int, default=2)
    parser.add_argument("--bram-x-max", type=int, default=14)

    parser.add_argument(
        "--penalty-weight",
        type=float,
        default=6.0e5,
        help="Penalty multiplier for critical-path regression over baseline",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=root / "runs" / "simple_rl",
        help="Output root",
    )
    parser.add_argument(
        "--vtr-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args forwarded to run-vtr.py (prefix with --)",
    )
    return parser.parse_args()


def sanitize_extra_args(extra_args: list[str]) -> list[str]:
    return [a for a in extra_args if a != "--"]


def setup_logger(log_file: Path, console_log: bool) -> logging.Logger:
    logger = logging.getLogger("simple_rl_layout_search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    if console_log:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    return logger


def write_arch_variant(base_arch: Path, candidate: Candidate, out_path: Path) -> None:
    tree = ET.parse(base_arch)
    root = tree.getroot()

    auto_layout = root.find("./layout/auto_layout")
    if auto_layout is None:
        raise ValueError("Expected <layout><auto_layout ...> in architecture file")

    auto_layout.set("aspect_ratio", f"{candidate.aspect_ratio:.4f}")

    dsp_cols = [c for c in auto_layout.findall("col") if c.get("type") == "mult_36"]
    mem_cols = [c for c in auto_layout.findall("col") if c.get("type") == "memory"]

    if not dsp_cols:
        raise ValueError("No <col type=\"mult_36\"> found")
    if not mem_cols:
        raise ValueError("No <col type=\"memory\"> found")

    # Keep existing repeat/priority/starty settings, only mutate startx.
    dsp_cols[0].set("startx", str(candidate.dsp_startx))
    mem_cols[0].set("startx", str(candidate.bram_startx))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_path, encoding="utf-8", xml_declaration=False)


def parse_power_and_cp(power_file: Path) -> tuple[float, float]:
    if not power_file.is_file():
        raise FileNotFoundError(f"Power report not found: {power_file}")

    power_val = None
    cp_val = None

    total_re = re.compile(r"^Total\s+([0-9.eE+-]+)")
    cp_re = re.compile(r"^Critical Path:\s*([0-9.eE+-]+)")

    for line in power_file.read_text(encoding="utf-8", errors="replace").splitlines():
        if cp_val is None:
            m = cp_re.match(line.strip())
            if m:
                cp_val = float(m.group(1))
        if power_val is None:
            m = total_re.match(line.strip())
            if m:
                power_val = float(m.group(1))
        if power_val is not None and cp_val is not None:
            break

    if power_val is None or cp_val is None:
        raise ValueError(f"Failed to parse power/cp from {power_file}")

    return power_val, cp_val


def run_vtr(
    arch_file: Path,
    circuit_file: Path,
    out_dir: Path,
    extra_args: list[str],
    timeout_sec: int,
) -> tuple[int, str | None]:
    root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(root / "run-vtr.py"),
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
        return completed.returncode, None
    except subprocess.TimeoutExpired:
        return 124, "timeout"


def candidate_key(c: Candidate) -> tuple[float, int, int]:
    return (round(c.aspect_ratio, 4), c.dsp_startx, c.bram_startx)


def key_to_str(key: tuple[float, int, int]) -> str:
    return f"{key[0]:.4f}|{key[1]}|{key[2]}"


def result_to_dict(res: EvalResult) -> dict[str, object]:
    return {
        "ok": res.ok,
        "reward": res.reward,
        "power_w": res.power_w,
        "critical_path_s": res.critical_path_s,
        "out_dir": str(res.out_dir),
        "reason": res.reason,
    }


def result_from_dict(data: dict[str, object]) -> EvalResult:
    return EvalResult(
        ok=bool(data["ok"]),
        reward=float(data["reward"]),
        power_w=(None if data["power_w"] is None else float(data["power_w"])),
        critical_path_s=(None if data["critical_path_s"] is None else float(data["critical_path_s"])),
        out_dir=Path(str(data["out_dir"])),
        reason=(None if data["reason"] is None else str(data["reason"])),
    )


def candidate_to_dict(c: Candidate | None) -> dict[str, object] | None:
    if c is None:
        return None
    return {
        "aspect_ratio": c.aspect_ratio,
        "dsp_startx": c.dsp_startx,
        "bram_startx": c.bram_startx,
    }


def candidate_from_dict(data: dict[str, object] | None) -> Candidate | None:
    if data is None:
        return None
    return Candidate(
        aspect_ratio=float(data["aspect_ratio"]),
        dsp_startx=int(data["dsp_startx"]),
        bram_startx=int(data["bram_startx"]),
    )


def sample_candidate(
    rng: random.Random,
    mu: tuple[float, float, float],
    sigma: tuple[float, float, float],
    args: argparse.Namespace,
) -> Candidate:
    a = min(args.aspect_max, max(args.aspect_min, rng.gauss(mu[0], sigma[0])))
    d = int(round(min(args.dsp_x_max, max(args.dsp_x_min, rng.gauss(mu[1], sigma[1])))))
    b = int(round(min(args.bram_x_max, max(args.bram_x_min, rng.gauss(mu[2], sigma[2])))))
    return Candidate(aspect_ratio=a, dsp_startx=d, bram_startx=b)


def update_distribution(elites: list[Candidate], args: argparse.Namespace) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    a_vals = [c.aspect_ratio for c in elites]
    d_vals = [float(c.dsp_startx) for c in elites]
    b_vals = [float(c.bram_startx) for c in elites]

    mu = (
        sum(a_vals) / len(a_vals),
        sum(d_vals) / len(d_vals),
        sum(b_vals) / len(b_vals),
    )

    def std(vals: list[float], floor: float) -> float:
        if len(vals) < 2:
            return floor
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        return max(floor, math.sqrt(var))

    sigma = (
        std(a_vals, 0.05),
        std(d_vals, 0.8),
        std(b_vals, 0.8),
    )

    # Clamp means within legal bounds.
    mu = (
        min(args.aspect_max, max(args.aspect_min, mu[0])),
        min(args.dsp_x_max, max(args.dsp_x_min, mu[1])),
        min(args.bram_x_max, max(args.bram_x_min, mu[2])),
    )

    return mu, sigma


def write_csv(log_rows: list[dict[str, object]], csv_path: Path) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(log_rows)


def update_live_plots(log_rows: list[dict[str, object]], runs_root: Path, logger: logging.Logger) -> None:
    if not log_rows:
        return

    successful = [r for r in log_rows if int(r["ok"]) == 1 and r["reward"] is not None]
    if not successful:
        return

    latest = successful[-1]
    best = max(successful, key=lambda r: float(r["reward"]))
    latest_json = {
        "latest": latest,
        "best": best,
        "evaluations": len(log_rows),
        "successful_evaluations": len(successful),
    }
    (runs_root / "progress_latest.json").write_text(json.dumps(latest_json, indent=2), encoding="utf-8")

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        logger.info("matplotlib not installed; skipping plot update")
        return

    eval_idx = list(range(1, len(successful) + 1))
    rewards = [float(r["reward"]) for r in successful]
    best_so_far = []
    cur = -1.0e30
    for r in rewards:
        cur = max(cur, r)
        best_so_far.append(cur)

    fig = plt.figure(figsize=(9, 5))
    plt.plot(eval_idx, rewards, label="reward", linewidth=1.2)
    plt.plot(eval_idx, best_so_far, label="best_so_far", linewidth=1.8)
    plt.xlabel("Successful evaluation index")
    plt.ylabel("Reward")
    plt.title("Simple RL Search Progress")
    plt.grid(True, alpha=0.35)
    plt.legend()
    fig.tight_layout()
    fig.savefig(runs_root / "progress_reward.png", dpi=150)
    plt.close(fig)

    powers = [float(r["power_w"]) for r in successful if r["power_w"] is not None]
    cps = [float(r["critical_path_s"]) for r in successful if r["critical_path_s"] is not None]
    if len(powers) == len(cps) and powers:
        fig = plt.figure(figsize=(9, 5))
        ax1 = plt.gca()
        ax1.plot(eval_idx, powers, color="tab:blue", label="power_w")
        ax1.set_xlabel("Successful evaluation index")
        ax1.set_ylabel("Power (W)", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.grid(True, alpha=0.35)

        ax2 = ax1.twinx()
        ax2.plot(eval_idx, cps, color="tab:red", label="critical_path_s")
        ax2.set_ylabel("Critical path (s)", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")

        plt.title("Power and Timing Trend")
        fig.tight_layout()
        fig.savefig(runs_root / "progress_power_timing.png", dpi=150)
        plt.close(fig)


def save_checkpoint(
    checkpoint_file: Path,
    next_iteration: int,
    mu: tuple[float, float, float],
    sigma: tuple[float, float, float],
    baseline_power: float,
    baseline_cp: float,
    cache: dict[tuple[float, int, int], EvalResult],
    log_rows: list[dict[str, object]],
    best_candidate: Candidate | None,
    best_result: EvalResult | None,
) -> None:
    payload = {
        "version": 1,
        "next_iteration": next_iteration,
        "mu": list(mu),
        "sigma": list(sigma),
        "baseline_power": baseline_power,
        "baseline_cp": baseline_cp,
        "cache": {key_to_str(k): result_to_dict(v) for k, v in cache.items()},
        "log_rows": log_rows,
        "best_candidate": candidate_to_dict(best_candidate),
        "best_result": (None if best_result is None else result_to_dict(best_result)),
    }
    checkpoint_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_checkpoint(
    checkpoint_file: Path,
) -> tuple[
    int,
    tuple[float, float, float],
    tuple[float, float, float],
    float,
    float,
    dict[tuple[float, int, int], EvalResult],
    list[dict[str, object]],
    Candidate | None,
    EvalResult | None,
]:
    data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
    next_iteration = int(data["next_iteration"])
    mu = tuple(float(x) for x in data["mu"])
    sigma = tuple(float(x) for x in data["sigma"])
    baseline_power = float(data["baseline_power"])
    baseline_cp = float(data["baseline_cp"])

    cache: dict[tuple[float, int, int], EvalResult] = {}
    for k, v in data["cache"].items():
        a_str, d_str, b_str = str(k).split("|")
        key = (float(a_str), int(d_str), int(b_str))
        cache[key] = result_from_dict(v)

    log_rows = list(data["log_rows"])
    best_candidate = candidate_from_dict(data.get("best_candidate"))
    best_result = None if data.get("best_result") is None else result_from_dict(data["best_result"])
    return next_iteration, mu, sigma, baseline_power, baseline_cp, cache, log_rows, best_candidate, best_result


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    arch = args.arch.expanduser().resolve()
    circuit = args.circuit.expanduser().resolve()
    runs_root = args.runs_root.expanduser().resolve()
    extra_args = sanitize_extra_args(args.vtr_extra_args)

    if not arch.is_file():
        raise FileNotFoundError(f"Architecture file not found: {arch}")
    if not circuit.is_file():
        raise FileNotFoundError(f"Circuit file not found: {circuit}")

    runs_root.mkdir(parents=True, exist_ok=True)
    variants_dir = runs_root / "arch_variants"
    evals_dir = runs_root / "evals"
    variants_dir.mkdir(parents=True, exist_ok=True)
    evals_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = (
        args.checkpoint_file.expanduser().resolve()
        if args.checkpoint_file
        else (runs_root / "checkpoint.json")
    )
    log_file = args.log_file.expanduser().resolve() if args.log_file else (runs_root / "search.log")
    logger = setup_logger(log_file, args.console_log)
    logger.info("Starting simple RL layout search")

    stem = circuit.stem

    # State init/resume.
    if args.resume and checkpoint_file.is_file():
        (
            start_iteration,
            mu,
            sigma,
            baseline_power,
            baseline_cp,
            cache,
            log_rows,
            best_candidate,
            best_result,
        ) = load_checkpoint(checkpoint_file)
        logger.info("Resumed from checkpoint %s at iteration %d", checkpoint_file, start_iteration)
    else:
        baseline_out = evals_dir / "baseline"
        baseline_code, baseline_reason = run_vtr(arch, circuit, baseline_out, extra_args, args.timeout_sec)
        if baseline_code != 0:
            raise RuntimeError(f"Baseline VTR run failed ({baseline_reason or baseline_code}). Fix this before RL search.")
        baseline_power, baseline_cp = parse_power_and_cp(baseline_out / f"{stem}.power")

        logger.info("Baseline power=%g W, cp=%g s", baseline_power, baseline_cp)

        start_iteration = 1
        mu = (
            (args.aspect_min + args.aspect_max) / 2.0,
            (args.dsp_x_min + args.dsp_x_max) / 2.0,
            (args.bram_x_min + args.bram_x_max) / 2.0,
        )
        sigma = (
            max(0.1, (args.aspect_max - args.aspect_min) / 4.0),
            max(1.0, (args.dsp_x_max - args.dsp_x_min) / 4.0),
            max(1.0, (args.bram_x_max - args.bram_x_min) / 4.0),
        )
        cache = {}
        log_rows = []
        best_candidate = None
        best_result = None
        save_checkpoint(
            checkpoint_file,
            start_iteration,
            mu,
            sigma,
            baseline_power,
            baseline_cp,
            cache,
            log_rows,
            best_candidate,
            best_result,
        )

    log_csv = runs_root / "search_log.csv"

    try:
        for it in range(start_iteration, args.iterations + 1):
            logger.info("Iteration %02d started | mu=(%.4f, %.2f, %.2f)", it, mu[0], mu[1], mu[2])
        batch: list[Candidate] = [sample_candidate(rng, mu, sigma, args) for _ in range(args.population)]

        eval_pairs: list[tuple[Candidate, EvalResult]] = []
        for idx, cand in enumerate(batch, start=1):
            key = candidate_key(cand)
            if key in cache:
                result = cache[key]
                logger.info(
                    "it=%02d cand=%02d cache hit | a=%.4f d=%d b=%d reward=%g",
                    it,
                    idx,
                    key[0],
                    key[1],
                    key[2],
                    result.reward,
                )
            else:
                name = f"it{it:02d}_c{idx:02d}_a{key[0]:.4f}_d{key[1]}_b{key[2]}"
                arch_variant = variants_dir / f"{name}.xml"
                out_dir = evals_dir / name

                write_arch_variant(arch, cand, arch_variant)
                code = 1
                run_reason: str | None = None
                for attempt in range(1, args.max_retries + 1):
                    code, run_reason = run_vtr(arch_variant, circuit, out_dir, extra_args, args.timeout_sec)
                    if code == 0:
                        break
                    logger.warning(
                        "it=%02d cand=%02d attempt=%d failed code=%s reason=%s",
                        it,
                        idx,
                        attempt,
                        code,
                        run_reason,
                    )

                if code != 0:
                    result = EvalResult(
                        ok=False,
                        reward=-1.0e9,
                        power_w=None,
                        critical_path_s=None,
                        out_dir=out_dir,
                        reason=("vtr_failed" if run_reason is None else f"vtr_failed:{run_reason}"),
                    )
                else:
                    try:
                        power_w, cp_s = parse_power_and_cp(out_dir / f"{stem}.power")
                        penalty = max(0.0, cp_s - baseline_cp)
                        reward = -(power_w + args.penalty_weight * penalty)
                        result = EvalResult(
                            ok=True,
                            reward=reward,
                            power_w=power_w,
                            critical_path_s=cp_s,
                            out_dir=out_dir,
                        )
                    except Exception as exc:  # pragma: no cover
                        result = EvalResult(
                            ok=False,
                            reward=-1.0e9,
                            power_w=None,
                            critical_path_s=None,
                            out_dir=out_dir,
                            reason=f"parse_error:{exc}",
                        )

                cache[key] = result

            eval_pairs.append((cand, result))

            log_rows.append(
                {
                    "iteration": it,
                    "aspect_ratio": key[0],
                    "dsp_startx": key[1],
                    "bram_startx": key[2],
                    "ok": int(result.ok),
                    "reward": result.reward,
                    "power_w": result.power_w,
                    "critical_path_s": result.critical_path_s,
                    "out_dir": str(result.out_dir),
                    "reason": result.reason,
                }
            )

            if best_result is None or result.reward > best_result.reward:
                best_result = result
                best_candidate = cand

            write_csv(log_rows, log_csv)
            if args.plot_every_eval:
                update_live_plots(log_rows, runs_root, logger)

            save_checkpoint(
                checkpoint_file,
                it,
                mu,
                sigma,
                baseline_power,
                baseline_cp,
                cache,
                log_rows,
                best_candidate,
                best_result,
            )

        eval_pairs.sort(key=lambda p: p[1].reward, reverse=True)
        elite_count = max(1, int(math.ceil(args.population * args.elite_frac)))
        elites = [cand for cand, result in eval_pairs[:elite_count] if result.ok]
        if not elites:
            elites = [cand for cand, _result in eval_pairs[:elite_count]]

        mu, sigma = update_distribution(elites, args)

        top = eval_pairs[0]
        logger.info(
            "iter=%02d top_reward=%g a=%.4f d=%d b=%d mu=(%.4f,%.2f,%.2f)",
            it,
            top[1].reward,
            top[0].aspect_ratio,
            top[0].dsp_startx,
            top[0].bram_startx,
            mu[0],
            mu[1],
            mu[2],
        )

        write_csv(log_rows, log_csv)
        update_live_plots(log_rows, runs_root, logger)
        save_checkpoint(
            checkpoint_file,
            it + 1,
            mu,
            sigma,
            baseline_power,
            baseline_cp,
            cache,
            log_rows,
            best_candidate,
            best_result,
        )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Checkpoint saved at %s", checkpoint_file)
        save_checkpoint(
            checkpoint_file,
            it,
            mu,
            sigma,
            baseline_power,
            baseline_cp,
            cache,
            log_rows,
            best_candidate,
            best_result,
        )
        return 130
    except Exception as exc:
        logger.error("Unhandled exception: %s", exc)
        logger.error(traceback.format_exc())
        save_checkpoint(
            checkpoint_file,
            it,
            mu,
            sigma,
            baseline_power,
            baseline_cp,
            cache,
            log_rows,
            best_candidate,
            best_result,
        )
        raise

    if best_candidate is None or best_result is None:
        raise RuntimeError("No candidates were evaluated")

    write_csv(log_rows, log_csv)
    update_live_plots(log_rows, runs_root, logger)

    best_summary = {
        "best_candidate": {
            "aspect_ratio": round(best_candidate.aspect_ratio, 4),
            "dsp_startx": best_candidate.dsp_startx,
            "bram_startx": best_candidate.bram_startx,
        },
        "best_result": {
            "reward": best_result.reward,
            "power_w": best_result.power_w,
            "critical_path_s": best_result.critical_path_s,
            "ok": best_result.ok,
            "out_dir": str(best_result.out_dir),
            "reason": best_result.reason,
        },
        "baseline": {
            "power_w": baseline_power,
            "critical_path_s": baseline_cp,
        },
        "search": {
            "iterations": args.iterations,
            "population": args.population,
            "elite_frac": args.elite_frac,
            "seed": args.seed,
        },
    }

    with (runs_root / "best_summary.json").open("w", encoding="utf-8") as f:
        json.dump(best_summary, f, indent=2)

    logger.info("Best candidate summary: %s", json.dumps(best_summary, indent=2))
    logger.info("Wrote CSV log: %s", log_csv)
    logger.info("Checkpoint file: %s", checkpoint_file)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
