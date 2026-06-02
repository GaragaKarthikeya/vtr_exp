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

# Import environment class
from fpga_env import FPGAEnv

SCRIPT_DIR = Path(__file__).resolve().parent

class BestLayoutCallback(BaseCallback):
    """
    Custom Stable Baselines3 callback to track, log, and permanently save 
    the best-performing FPGA architecture found during training.
    """
    def __init__(self, benchmark_name: str, width: int, height: int, verbose: int = 0):
        super().__init__(verbose)
        self.benchmark_name = benchmark_name
        self.width = width
        self.height = height
        self.best_reward = -float("inf")
        self.best_info = {}
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        
        for idx, info in enumerate(infos):
            # Check if a complete episode was evaluated
            if "success" in info:
                reward = rewards[idx]
                if info["success"] and reward > self.best_reward:
                    self.best_reward = reward
                    self.best_info = info
                    
                    print(f"\n" + "="*50)
                    print(f"[NEW BEST FPGA LAYOUT FOUND!]")
                    print(f"Reward: {reward:.5f}")
                    print(f"Wirelength: {info['wirelength']}")
                    print(f"Delay (ns): {info['delay_ns']}")
                    print(f"Power (W): {info['power_w']}")
                    print(f"Placed DSPs: {info['placed_dsps']}")
                    print(f"Placed BRAMs: {info['placed_brams']}")
                    print(f"Cached: {info['cached']}")
                    print(f"Time elapsed: {time.time() - self.start_time:.1f}s")
                    print("="*50 + "\n")
                    
                    self._save_best_layout(info)
                    
        return True

    def _save_best_layout(self, info):
        try:
            # Dynamically import bake_layout to avoid circular import issues
            from bake_layout import bake_layout
            
            best_arch_path = SCRIPT_DIR / f"best_baked_layout_{self.benchmark_name}.xml"
            bake_layout(
                benchmark_name=self.benchmark_name,
                dsps=info["placed_dsps"],
                mems=info["placed_brams"],
                width=self.width + 2,
                height=self.height + 2,
                output_path=str(best_arch_path)
            )
            
            # Save metadata and coordinates
            coords_path = SCRIPT_DIR / f"best_layout_coordinates_{self.benchmark_name}.txt"
            coords_data = {
                "reward": self.best_reward,
                "wirelength": info["wirelength"],
                "delay_ns": info["delay_ns"],
                "power_w": info["power_w"],
                "dsps": info["placed_dsps"],
                "brams": info["placed_brams"],
                "timestamp_s": time.time() - self.start_time
            }
            coords_path.write_text(json.dumps(coords_data, indent=4))
            print(f"Successfully wrote best arch to: {best_arch_path.name}")
            print(f"Successfully wrote coordinates to: {coords_path.name}")
            
        except Exception as e:
            print(f"Error saving best layout: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Parallel RL Pipeline for FPGA Placement Optimization")
    parser.add_argument("--benchmark", type=str, default="diffeq1", help="Name of Verilog benchmark (e.g. diffeq1)")
    parser.add_argument("--n_envs", type=int, default=None, help="Number of parallel environment workers")
    parser.add_argument("--timesteps", type=int, default=10000, help="Total training timesteps")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate for PPO agent")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size for PPO updates")
    parser.add_argument("--n_steps", type=int, default=80, help="Number of steps to run per environment per rollout")
    args = parser.parse_args()

    benchmark_name = args.benchmark
    if benchmark_name.endswith(".v"):
        benchmark_name = benchmark_name[:-2]

    # Load baseline resource constraints and limits
    res_file = SCRIPT_DIR / f"{benchmark_name}_traditional_resources.txt"
    metric_file = SCRIPT_DIR / f"{benchmark_name}_traditional_metric.txt"

    if not res_file.is_file() or not metric_file.is_file():
        print(f"Error: Traditional baseline files not found for '{benchmark_name}' in {SCRIPT_DIR}", file=sys.stderr)
        sys.exit(1)

    try:
        res_data = json.loads(res_file.read_text())
        metric_data = json.loads(metric_file.read_text())
    except Exception as e:
        print(f"Error parsing traditional baseline logs: {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve core grid dimension
    fpga_size = res_data.get("fpga_size")
    if not fpga_size or len(fpga_size) != 2:
        print(f"Error: Invalid 'fpga_size' in resource baseline log: {res_file}", file=sys.stderr)
        sys.exit(1)
        
    width = int(fpga_size[0])
    height = int(fpga_size[1])

    # Resolve required DSPs and BRAMs from netlist requirements
    reqs = res_data.get("requirements", {})
    req_dsp = reqs.get("dsp", 0)
    req_bram = reqs.get("bram", 0)

    print("="*60)
    print(f"Loaded traditional architecture baseline for: {benchmark_name}")
    print(f"Grid size (Core): {width}x{height} (Passes {width+2}x{height+2} with IOs to bake)")
    print(f"Netlist Block Requirements: DSP={req_dsp}, BRAM={req_bram}")
    print(f"Normalization Baselines:")
    print(f"  - Wirelength: {metric_data.get('wirelength')}")
    print(f"  - Delay (ns): {metric_data.get('delay_ns')}")
    print(f"  - Power (W):  {metric_data.get('power_w')}")
    print("="*60)

    # Determine CPU core count and parallel workers
    n_envs = args.n_envs
    if n_envs is None:
        cpu_count = os.cpu_count() or 1
        # Set n_envs to min(8, cpu_count) to be extremely fast but conservative with memory
        n_envs = min(8, cpu_count)
    
    print(f"Configuring {n_envs} parallel workers using SubprocVecEnv...")

    # Define environment arguments
    env_kwargs = {
        "benchmark_name": benchmark_name,
        "width": width,
        "height": height,
        "req_dsp": req_dsp,
        "req_bram": req_bram,
        "traditional_metrics": metric_data,
        "cache_db_path": str(SCRIPT_DIR / "runs" / f"vtr_layout_cache_{benchmark_name}.db")
    }

    # Create vectorized environment
    env = make_vec_env(
        FPGAEnv,
        n_envs=n_envs,
        seed=42,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs
    )

    # Set up best-layout tracking callback
    callback = BestLayoutCallback(benchmark_name=benchmark_name, width=width, height=height)

    # Check if tensorboard is installed
    try:
        import tensorboard
        tb_log = str(SCRIPT_DIR / "runs" / "tb_logs")
    except ImportError:
        tb_log = None
        print("Warning: TensorBoard is not installed. Disabling TensorBoard logging.")

    # Instantiate Stable Baselines3 PPO agent
    # We use a Multi-Layer Perceptron (MlpPolicy) since grid state is small, but PPO handles it extremely well.
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=args.lr,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        tensorboard_log=tb_log
    )

    print("\nStarting reinforcement learning training pipeline...")
    print(f"Total target timesteps: {args.timesteps}")
    print(f"Rollout buffer: {n_envs * args.n_steps} steps per policy update.")
    print("="*60)

    try:
        model.learn(total_timesteps=args.timesteps, callback=callback)
        print("\nTraining completed successfully!")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting gracefully...")
    finally:
        # Close environment processes
        env.close()

if __name__ == "__main__":
    main()
