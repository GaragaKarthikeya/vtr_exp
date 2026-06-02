#!/usr/bin/env python3

import os
import sys
import json
import argparse
import time
from pathlib import Path
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

# Setup pathing for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PARENT_DIR))

# Import PoC environment and visualizer
from fpga_env_poc import FPGAEnvPoC
from visualize_layout_poc import visualize_poc

class BestSparseLayoutCallback(BaseCallback):
    """
    Tracks, bakes, and visualizes the absolute best sparse eFPGA placement 
    discovered during training.
    """
    def __init__(self, benchmark_name: str, verbose: int = 0):
        super().__init__(verbose)
        self.benchmark_name = benchmark_name
        self.best_reward = -float("inf")
        self.best_info = {}
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        
        for idx, info in enumerate(infos):
            # Check if a complete placement was evaluated
            if "success" in info:
                reward = rewards[idx]
                if info["success"] and reward > self.best_reward:
                    self.best_reward = reward
                    self.best_info = info
                    
                    print(f"\n" + "="*60)
                    print(f"[🔥 NEW BEST SPARSE FPGA CORE DISCOVERED!]")
                    print(f"Reward:        {reward:.5f}")
                    print(f"BBox Dimension: {info['width'] - 2} x {info['height'] - 2} (Fabric: {info['width']}x{info['height']})")
                    print(f"Wirelength:    {info['wirelength']}")
                    print(f"Delay (ns):    {info['delay_ns']}")
                    print(f"Power (W):     {info['power_w']}")
                    print(f"Cached:        {info['cached']}")
                    print(f"Time elapsed:  {time.time() - self.start_time:.1f}s")
                    print("="*60 + "\n")
                    
                    self._save_best_layout(info)
                    
        return True

    def _save_best_layout(self, info):
        try:
            from bake_layout_poc import bake_layout_poc
            
            best_arch_path = SCRIPT_DIR / "best_sparse_baked_layout.xml"
            bake_layout_poc(
                benchmark_name=self.benchmark_name,
                clbs=info["shifted_clbs"],
                dsps=info["shifted_dsps"],
                mems=[],
                width=info["width"],
                height=info["height"],
                output_path=str(best_arch_path)
            )
            
            # Save coordinate metadata
            coords_path = SCRIPT_DIR / "best_sparse_layout_coordinates.txt"
            coords_data = {
                "reward": self.best_reward,
                "width": info["width"],
                "height": info["height"],
                "wirelength": info["wirelength"],
                "delay_ns": info["delay_ns"],
                "power_w": info["power_w"],
                "dsps": info["shifted_dsps"],
                "clbs": info["shifted_clbs"],
                "timestamp_s": time.time() - self.start_time
            }
            coords_path.write_text(json.dumps(coords_data, indent=4))
            
            # Programmatically render the ANSI core layout directly to terminal logs!
            print("\n[GRID VISUALIZATION OF BEST OPTIMIZED PLACEMENT]")
            visualize_poc(str(coords_path))
            
        except Exception as e:
            print(f"Error saving best layout: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="PPO Trainer for Sparse eFPGA Layout Placement")
    parser.add_argument("--benchmark", type=str, default="diffeq1")
    parser.add_argument("--n_envs", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    benchmark_name = args.benchmark
    if benchmark_name.endswith(".v"):
        benchmark_name = benchmark_name[:-2]

    # Resolve required CPU workers
    n_envs = args.n_envs
    if n_envs is None:
        cpu_count = os.cpu_count() or 1
        n_envs = min(8, cpu_count)

    print("="*60)
    print(f"SPARSE eFPGA PLACEMENT OPTIMIZATION WORKFLOW: {benchmark_name.upper()}")
    print(f"Placement Budget: 5 DSPs, 73 CLBs explicitly placed")
    print(f"Grid Canvas: 18x18 Virtual Core (16x16 available space)")
    print(f"Parallel Workers: {n_envs}")
    print("="*60)

    # Initialize Environment Parameters
    # places exactly 5 DSPs and 73 CLBs
    env_kwargs = {
        "benchmark_name": benchmark_name,
        "virtual_width": 18,
        "virtual_height": 18,
        "req_dsp": 5,
        "req_clb": 73,
        "cache_db_path": str(PARENT_DIR / "runs" / "vtr_layout_cache_poc.db")
    }

    # Vectorize environments
    env = make_vec_env(
        FPGAEnvPoC,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )

    callback = BestSparseLayoutCallback(benchmark_name=benchmark_name)

    tb_log = None
    try:
        import tensorboard
        tb_log = str(PARENT_DIR / "runs" / "tb_logs_poc")
    except ImportError:
        pass

    # SB3 PPO model configuration
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=8,
        gamma=0.99,
        seed=args.seed,
        verbose=1,
        tensorboard_log=tb_log
    )

    print("\nStarting PPO training on Sparse Core grid...")
    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        print("\nSparse layout PPO training completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Gracefully shutting down...")
    finally:
        env.close()

if __name__ == "__main__":
    main()
