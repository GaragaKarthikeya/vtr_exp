#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from rl_macro_placement.config import default_config


@dataclass(frozen=True)
class EvalRecord:
    summary_path: Path
    reward: float
    critical_path_ns: float
    total_wirelength: float
    cp_improvement_pct: float
    wire_improvement_pct: float
    aspect_ratio: float
    dsp_startx: int
    bram_startx: int

    @property
    def layout_key(self) -> tuple[float, int, int]:
        return (round(self.aspect_ratio, 4), self.dsp_startx, self.bram_startx)


def parse_args() -> argparse.Namespace:
    cfg = default_config()
    parser = argparse.ArgumentParser(description="Summarize RL macro placement runs")
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=cfg.runs_root,
        help="Root directory containing baseline/, evaluations/, and ppo/",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top unique layouts to print",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def pct_improvement(baseline: float, value: float) -> float:
    return ((baseline - value) / baseline) * 100.0


def build_record(summary_path: Path) -> EvalRecord | None:
    payload = load_json(summary_path)
    if not payload.get("success", False):
        return None

    action = payload["action"]
    baseline_cp = float(payload["baseline_critical_path_ns"])
    baseline_wire = float(payload["baseline_total_wirelength"])
    critical_path_ns = float(payload["critical_path_ns"])
    total_wirelength = float(payload["total_wirelength"])

    return EvalRecord(
        summary_path=summary_path,
        reward=float(payload["reward"]),
        critical_path_ns=critical_path_ns,
        total_wirelength=total_wirelength,
        cp_improvement_pct=pct_improvement(baseline_cp, critical_path_ns),
        wire_improvement_pct=pct_improvement(baseline_wire, total_wirelength),
        aspect_ratio=float(action["aspect_ratio"]),
        dsp_startx=int(action["dsp_startx"]),
        bram_startx=int(action["bram_startx"]),
    )


def collect_records(runs_root: Path) -> list[EvalRecord]:
    records: list[EvalRecord] = []
    for summary_path in sorted((runs_root / "evaluations").glob("*/summary.json")):
        record = build_record(summary_path)
        if record is not None:
            records.append(record)
    return records


def top_unique_records(records: list[EvalRecord], top_k: int) -> list[EvalRecord]:
    best_by_key: dict[tuple[float, int, int], EvalRecord] = {}
    for record in records:
        key = record.layout_key
        current = best_by_key.get(key)
        if current is None or record.reward > current.reward:
            best_by_key[key] = record

    unique_records = sorted(
        best_by_key.values(),
        key=lambda record: (record.reward, record.cp_improvement_pct, record.wire_improvement_pct),
        reverse=True,
    )
    return unique_records[:top_k]


def format_record(index: int, record: EvalRecord) -> str:
    return (
        f"{index}. reward={record.reward:.6f} | "
        f"cp={record.critical_path_ns:.4f} ns ({record.cp_improvement_pct:+.2f}%) | "
        f"wire={record.total_wirelength:.0f} ({record.wire_improvement_pct:+.2f}%) | "
        f"a={record.aspect_ratio:.4f} d={record.dsp_startx} b={record.bram_startx} | "
        f"{record.summary_path.parent.name}"
    )


def main() -> int:
    args = parse_args()
    runs_root = args.runs_root.expanduser().resolve()

    baseline_summary = load_json(runs_root / "baseline" / "summary.json")
    baseline_cp = float(baseline_summary["critical_path_ns"])
    baseline_wire = float(baseline_summary["total_wirelength"])

    records = collect_records(runs_root)
    if not records:
        raise SystemExit(f"No successful evaluation summaries found under {runs_root / 'evaluations'}")

    best_overall = max(records, key=lambda record: record.reward)
    top_unique = top_unique_records(records, args.top_k)

    print("RL Macro Placement Report")
    print(f"runs_root: {runs_root}")
    print(f"baseline: cp={baseline_cp:.4f} ns, wire={baseline_wire:.0f}")
    print(f"successful_evaluations: {len(records)}")
    print()
    print("Best Overall")
    print(format_record(1, best_overall))

    best_result_json = runs_root / "ppo" / "best_result.json"
    if best_result_json.is_file():
        best_result = load_json(best_result_json)
        print()
        print("Best Result JSON")
        print(
            "reward={reward:.6f} | cp={cp:.4f} ns | wire={wire:.0f} | a={a:.4f} d={d} b={b}".format(
                reward=float(best_result["reward"]),
                cp=float(best_result["critical_path_ns"]),
                wire=float(best_result["total_wirelength"]),
                a=float(best_result["action"]["aspect_ratio"]),
                d=int(best_result["action"]["dsp_startx"]),
                b=int(best_result["action"]["bram_startx"]),
            )
        )

    print()
    print(f"Top {len(top_unique)} Unique Layouts")
    for idx, record in enumerate(top_unique, start=1):
        print(format_record(idx, record))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
