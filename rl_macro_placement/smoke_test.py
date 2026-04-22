#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rl_macro_placement.config import default_config
from rl_macro_placement.env import MacroPlacementEnv
from rl_macro_placement.layout_space import LayoutAction


def main() -> int:
    cfg = default_config()
    env = MacroPlacementEnv(cfg)

    state = env.reset()
    print("Baseline state:")
    print(json.dumps(state, indent=2))

    probe_action = LayoutAction(
        aspect_ratio=1.0,
        dsp_startx=8,
        bram_startx=4,
    )
    result = env.step(probe_action)

    print("\nProbe evaluation:")
    print(json.dumps(result.info, indent=2))
    print(f"\nReward: {result.reward:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
