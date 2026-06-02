#!/usr/bin/env python3

import os
import sys
import json
import time
import shutil
import sqlite3
from pathlib import Path
import numpy as np

# Import environment class
from fpga_env import FPGAEnv

SCRIPT_DIR = Path(__file__).resolve().parent

def run_tests():
    print("="*60)
    print("RUNNING FPGA RL ENVIRONMENT VERIFICATION TESTS")
    print("="*60)

    # 1. Load baseline baseline info
    res_file = SCRIPT_DIR / "diffeq1_traditional_resources.txt"
    metric_file = SCRIPT_DIR / "diffeq1_traditional_metric.txt"
    res_data = json.loads(res_file.read_text())
    metric_data = json.loads(metric_file.read_text())
    
    fpga_size = res_data["fpga_size"]
    width = int(fpga_size[0])
    height = int(fpga_size[1])
    reqs = res_data["requirements"]
    req_dsp = reqs["dsp"]
    req_bram = reqs["bram"]

    db_path = str(SCRIPT_DIR / "runs" / "test_vtr_layout_cache.db")
    # Clean up old test db if it exists
    if Path(db_path).is_file():
        Path(db_path).unlink()

    # Instantiate Env
    env = FPGAEnv(
        benchmark_name="diffeq1",
        width=width,
        height=height,
        req_dsp=req_dsp,
        req_bram=req_bram,
        traditional_metrics=metric_data,
        cache_db_path=db_path
    )

    print(f"Instantiated environment for '{env.benchmark_name}' (grid size {width}x{height})")
    print(f"Blocks to place: {env.blocks_to_place}")

    # TEST 1: Reset environment
    print("\n--- TEST 1: Reset Environment ---")
    obs, info = env.reset()
    assert obs.shape == (width, height, 2), "Observation shape is incorrect!"
    assert np.all(obs[:, :, 0] == 0), "Grid is not empty on reset!"
    assert np.all(obs[:, :, 1] == 1), "Block indicator in Channel 1 is incorrect (should be 1 for DSP)!"
    print("TEST 1 PASSED: Environment reset successfully.")

    # TEST 2: Action masking
    print("\n--- TEST 2: Action Masking ---")
    mask = env.get_action_mask()
    assert mask.shape == (width * height,), "Action mask shape is incorrect!"
    assert np.sum(mask) > 0, "Action mask has no valid positions!"
    print(f"TEST 2 PASSED: Action mask contains {np.sum(mask)} valid placements out of {len(mask)}.")

    # TEST 3: Out-of-bounds check
    print("\n--- TEST 3: Out of Bounds Constraints ---")
    # Place a DSP where it goes off-grid vertically (e.g. y = 13, height = 4, 13 + 4 - 1 = 16 > 14)
    # Action = x * height + y = (x - 1) * 14 + (y - 1)
    # Let's try x = 2, y = 13 -> action = 1 * 14 + 12 = 26
    invalid_action = 1 * 14 + 12
    obs, reward, terminated, truncated, info = env.step(invalid_action)
    assert terminated == True, "Episode did not terminate on out of bounds!"
    assert reward == -10.0, f"Incorrect reward for out of bounds: {reward}!"
    assert info["status"] == "out_of_bounds", f"Incorrect status for out of bounds: {info['status']}"
    print("TEST 3 PASSED: Vertically out-of-bounds placement successfully caught.")

    # Reset
    env.reset()

    # TEST 4: Overlap and block heights
    print("\n--- TEST 4: Overlap Collision Check ---")
    # Place a DSP at (2, 1) -> action = (2 - 1)*14 + (1 - 1) = 14
    obs, reward, terminated, truncated, info = env.step(14)
    assert terminated == False, "Episode terminated prematurely!"
    assert reward == 0.0, "Intermediate step reward should be 0.0!"
    assert env.grid[1, 0] == 1, "DSP start coordinate not marked as 1!"
    assert np.all(env.grid[1, 1:4] == -1), "DSP height cells not marked as -1!"
    
    # Try to place another DSP overlapping the first one (e.g. at (2, 3) -> action = 1 * 14 + 2 = 16)
    obs, reward, terminated, truncated, info = env.step(16)
    assert terminated == True, "Episode did not terminate on overlap!"
    assert reward == -10.0, "Incorrect reward for overlap placement!"
    assert info["status"] == "overlap", f"Incorrect status for overlap: {info['status']}"
    print("TEST 4 PASSED: Overlap check and multi-cell vertical block height reservation work.")

    # Reset
    env.reset()

    # TEST 5: Complete valid placement and VTR run
    print("\n--- TEST 5: Valid Placement and VTR Execution ---")
    # Let's place 5 DSPs at non-overlapping valid coordinates
    # DSP 1: (2, 1)  -> action = (2-1)*14 + (1-1) = 14
    # DSP 2: (4, 1)  -> action = (4-1)*14 + (1-1) = 42
    # DSP 3: (6, 1)  -> action = (6-1)*14 + (1-1) = 70
    # DSP 4: (8, 1)  -> action = (8-1)*14 + (1-1) = 98
    # DSP 5: (10, 1) -> action = (10-1)*14 + (1-1) = 126
    
    start_time = time.time()
    actions = [14, 42, 70, 98, 126]
    for i, act in enumerate(actions[:-1]):
        obs, reward, terminated, truncated, info = env.step(act)
        assert not terminated, f"Episode terminated prematurely at step {i}!"

    print("Placing final DSP to trigger VTR execution (this should take ~10 seconds)...")
    obs, reward, terminated, truncated, info = env.step(actions[-1])
    eval_time = time.time() - start_time
    
    print(f"VTR Execution completed in {eval_time:.2f}s")
    assert terminated == True, "Episode did not terminate on final block placement!"
    assert info["success"] == True, f"VTR evaluation failed: {info.get('error')}"
    assert info["cached"] == False, "VTR run should NOT be cached for the first run!"
    assert reward > -5.0, f"Reward seems too low: {reward}"
    print(f"TEST 5 PASSED: Placement complete, VTR executed, metrics successfully parsed.")
    print(f"  Parsed Metrics: Wirelength={info['wirelength']}, Delay={info['delay_ns']}ns, Power={info['power_w']}W")
    print(f"  Final Calculated Reward: {reward:.5f}")

    # TEST 6: SQLite Caching validation
    print("\n--- TEST 6: SQLite Layout Caching ---")
    env.reset()
    for i, act in enumerate(actions[:-1]):
        env.step(act)

    print("Placing final DSP with IDENTICAL coordinates to trigger SQLite cache read...")
    start_time = time.time()
    obs, reward_cache, terminated, truncated, info_cache = env.step(actions[-1])
    cache_time = time.time() - start_time

    print(f"Cache retrieval completed in {cache_time*1000:.2f}ms")
    assert terminated == True, "Episode did not terminate on final cache block placement!"
    assert info_cache["success"] == True, "Cached VTR evaluation failed!"
    assert info_cache["cached"] == True, "VTR run was not read from the cache!"
    assert reward_cache == reward, f"Reward from cache {reward_cache} does not match original {reward}!"
    assert cache_time < 0.05, f"Cache retrieval took too long: {cache_time:.4f}s"
    print("TEST 6 PASSED: SQLite caching successfully returned metrics in 0 ms and matched original reward.")

    # Clean up test database
    if Path(db_path).is_file():
        Path(db_path).unlink()

    print("\n" + "="*60)
    print("ALL VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_tests()
