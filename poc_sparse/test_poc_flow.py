#!/usr/bin/env python3

import sys
import json
from pathlib import Path

# Setup pathing for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PARENT_DIR))

from fpga_env_poc import FPGAEnvPoC

def run_mock_placement_test():
    print("=" * 60)
    print("RUNNING PROOF-OF-CONCEPT SPARSE PLACEMENT MOCK FLOW")
    print("=" * 60)
    
    # Instantiate Gym environment
    # places exactly 5 DSPs and 73 CLBs
    env = FPGAEnvPoC(
        benchmark_name="diffeq1",
        virtual_width=18,
        virtual_height=18,
        req_dsp=5,
        req_clb=73,
    )
    
    env.reset()

    # Pre-calculated dense mock placement (96.8% core utilization)
    # places DSPs at columns 1-5, row 1 (occupying rows 1-4)
    # places 12 CLBs per column for columns 1-5 (rows 5-16), totaling 60 CLBs
    # places the remaining 13 CLBs in column 6 (rows 1-13)
    # Bounding Box: 6x16 core (W_core=6, H_core=16) -> Area = 96 tiles!
    mock_placements = []

    # 1. Place 5 DSPs (1..5)
    for x in range(1, 6):
        mock_placements.append((x, 1))

    # 2. Place 60 CLBs (columns 1..5, rows 5..16)
    for x in range(1, 6):
        for y in range(5, 17):
            mock_placements.append((x, y))

    # 3. Place remaining 13 CLBs (column 6, rows 1..13)
    for y in range(1, 14):
        mock_placements.append((6, y))

    # Assert total blocks is 78
    assert len(mock_placements) == 78, f"Incorrect mock placement size: {len(mock_placements)}"

    print(f"Executing step sequence inside Gym environment...")
    for idx, (x, y) in enumerate(mock_placements):
        # Encode coordinates to discrete action space index
        # x_env = x, y_env = y
        act = (x - 1) * env.core_h + (y - 1)
        
        # Verify action is in active action mask
        mask = env.get_action_mask()
        if mask[act] == 0:
            print(f"Error: Proposed action index {act} (coords: {x}, {y}) is masked out by environment!")
            env.close()
            return 1
            
        obs, reward, terminated, truncated, info = env.step(act)
        
        # Monitor dynamic status on complete
        if terminated or truncated:
            print(f"\nEpisode finished! Steps: {env.current_step}/{env.total_blocks}")
            print(f"Status: {info.get('status')}")
            if info.get("error"):
                print(f"VTR Compiler Error: {info.get('error')}")
                env.close()
                return 1
            
            print(f"Evaluated Metrics:")
            print(f"  - Dynamic BBox Size: {info.get('width') - 2} x {info.get('height') - 2} (Fabric: {info.get('width')}x{info.get('height')})")
            print(f"  - Wirelength:       {info.get('wirelength')} units")
            print(f"  - Delay:            {info.get('delay_ns')} ns")
            print(f"  - Power:            {info.get('power_w')} W")
            print(f"  - Reward Score:     {reward:.5f}")
            print(f"  - Cached lookup:    {info.get('cached')}")
            break
            
    print("=" * 60)
    print("MOCK TEST FLOW EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    env.close()
    return 0

if __name__ == "__main__":
    sys.exit(run_mock_placement_test())
