#!/usr/bin/env python3

import os
import sys
import time
import sqlite3
import subprocess
from pathlib import Path

# ANSI colors for premium terminal display
COLOR_RESET = "\033[0m"
COLOR_HEADER = "\033[95m"
COLOR_SUCCESS = "\033[92m"
COLOR_WARNING = "\033[93m"
COLOR_INFO = "\033[96m"

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR / "runs" / "vtr_layout_cache_diffeq1.db"

def get_db_counts() -> tuple[int, int]:
    """Queries SQLite database to get total records and successful records."""
    if not DB_PATH.is_file():
        return 0, 0
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        cursor.execute("SELECT count(*), sum(success) FROM layout_cache")
        row = cursor.fetchone()
        conn.close()
        total = row[0] if row[0] is not None else 0
        success = row[1] if row[1] is not None else 0
        return total, success
    except Exception as e:
        print(f"Error querying database: {e}", file=sys.stderr)
        return 0, 0

def run_seed_pipeline():
    print("=" * 60)
    print(f"{COLOR_HEADER}FPGA RL GNN DATASET GENERATOR: 5-SEED PIPELINE{COLOR_RESET}")
    print("=" * 60)
    
    seeds = [42, 100, 2026, 777, 999]
    layouts_per_seed = 200
    
    python_bin = Path(os.environ.get("VTR_VENV_PATH", "/home/digital-2/.venv")) / "bin" / "python"
    if not python_bin.is_file():
        python_bin = Path(sys.executable)
        
    train_script = SCRIPT_DIR / "train.py"
    export_script = SCRIPT_DIR / "export_dataset.py"

    if not train_script.is_file():
        print(f"Error: train.py not found at: {train_script}", file=sys.stderr)
        return 1

    # Check initial counts
    init_total, init_success = get_db_counts()
    print(f"{COLOR_INFO}Initial Layout Cache Status:{COLOR_RESET}")
    print(f"  - Total Placements Evaluated: {init_total}")
    print(f"  - Successful Placements:      {init_success}")
    print("-" * 60)

    for i, seed in enumerate(seeds):
        # We start counting from the current success count in the database
        total_before, success_before = get_db_counts()
        target_success = success_before + layouts_per_seed
        
        print(f"\n{COLOR_INFO}[SEED {i+1}/5] Starting RL Seed Run: {seed}{COLOR_RESET}")
        print(f"  - Current Successful Layouts: {success_before}")
        print(f"  - Target Successful Layouts:  {target_success}")
        
        # Build command with n_steps=5 to update after exactly 8 placements
        cmd = [
            str(python_bin),
            str(train_script),
            "--n_envs", "8",
            "--n_steps", "5",         # Updates every 8 placements
            "--timesteps", "2500",      # Generates up to 500 complete layouts
            "--seed", str(seed),
            "--benchmark", "diffeq1"
        ]
        
        print(f"  - Command: {' '.join(cmd)}")
        
        # Start training subprocess and redirect stdout/stderr to a dedicated log file
        log_file_path = SCRIPT_DIR / "runs" / f"ppo_training_seed_{seed}.log"
        log_file = open(log_file_path, "w")
        
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        
        # Non-blocking log monitor and DB poll loop
        try:
            start_time = time.time()
            last_db_check = time.time()
            
            while True:
                # Check if process finished on its own
                ret_code = process.poll()
                if ret_code is not None:
                    print(f"\n{COLOR_WARNING}Seed {seed} process exited on its own with return code {ret_code}.{COLOR_RESET}")
                    break

                # Poll database metrics every 5 seconds
                current_time = time.time()
                if current_time - last_db_check >= 5.0:
                    last_db_check = current_time
                    _, current_success = get_db_counts()
                    new_successful = current_success - success_before
                    elapsed_s = current_time - start_time
                    
                    print(f"  [{elapsed_s:.1f}s] Successful generated: {new_successful}/{layouts_per_seed} (Db Total: {current_success})", end="\r")
                    
                    if current_success >= target_success:
                        print(f"\n\n{COLOR_SUCCESS}[SUCCESS] Target of {layouts_per_seed} layouts reached for Seed {seed}!{COLOR_RESET}")
                        print("  Gracefully terminating seed subprocess...")
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        break
                
                # Sleep a moment to avoid heavy CPU spin
                time.sleep(1.0)
                
        except KeyboardInterrupt:
            print(f"\n{COLOR_WARNING}Pipeline execution interrupted by user. Stopping seed {seed}...{COLOR_RESET}")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            return 1
        finally:
            log_file.close()
            
        # Quick recovery sleep between seeds
        time.sleep(2.0)

    # All seeds completed! Launch exporter
    final_total, final_success = get_db_counts()
    print("\n" + "=" * 60)
    print(f"{COLOR_SUCCESS}5-SEED DATASET GENERATION WORKFLOW COMPLETED SUCCESSFULLY!{COLOR_RESET}")
    print("=" * 60)
    print(f"Final Layout Cache Stats:")
    print(f"  - Total unique layouts evaluated: {final_total}")
    print(f"  - Total successful layouts:       {final_success}")
    print(f"  - Total invalid layouts:          {final_total - final_success}")
    print("-" * 60)
    
    if export_script.is_file():
        print(f"{COLOR_INFO}Triggering JSON dataset export...{COLOR_RESET}")
        export_cmd = [str(python_bin), str(export_script)]
        subprocess.run(export_cmd)
    else:
        print(f"Warning: export_dataset.py not found at {export_script}", file=sys.stderr)
        
    return 0

if __name__ == "__main__":
    sys.exit(run_seed_pipeline())
