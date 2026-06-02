import os
import re
import sys
import uuid
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Load environment variables from .env
def load_env_file(env_file: Path) -> None:
    if not env_file.is_file():
        return
    for raw_line in env_file.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = os.path.expandvars(value.strip().strip("'\""))

# Resolve script directory and load .env
SCRIPT_DIR = Path(__file__).resolve().parent
load_env_file(SCRIPT_DIR / ".env")

class FPGAEnv(gym.Env):
    """
    Custom Gymnasium Environment for placing DSP and BRAM blocks onto a VTR FPGA core grid.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark_name: str = "diffeq1",
        width: int = 14,
        height: int = 14,
        req_dsp: int = 5,
        req_bram: int = 0,
        traditional_metrics: dict = None,
        cache_db_path: str = None,
    ):
        super(FPGAEnv, self).__init__()
        
        self.benchmark_name = benchmark_name
        self.width = width
        self.height = height
        self.req_dsp = req_dsp
        self.req_bram = req_bram
        
        # Default baseline traditional metrics (diffeq1 baseline)
        if traditional_metrics is None:
            self.traditional_metrics = {
                "delay_ns": 27.1504,
                "wirelength": 24301.0,
                "power_w": 0.007658,
            }
        else:
            self.traditional_metrics = traditional_metrics

        # Setup SQLite layout cache db
        if cache_db_path is None:
            self.cache_db_path = str(SCRIPT_DIR / "runs" / "vtr_layout_cache.db")
        else:
            self.cache_db_path = cache_db_path
        
        self._init_cache_db()

        # Build sequence of blocks to place: place all DSPs first, then all BRAMs
        # 1 = DSP, 2 = BRAM
        self.blocks_to_place = [1] * self.req_dsp + [2] * self.req_bram
        self.total_blocks = len(self.blocks_to_place)

        # Action Space: Discrete mapping to grid coordinates A -> (x, y)
        self.action_space = spaces.Discrete(self.width * self.height)

        # Observation Space: (width, height, 2)
        # Channel 0: Occupancy Grid (-1 = occupied by block height, 0 = empty/CLB, 1 = DSP start, 2 = BRAM start)
        # Channel 1: Current block being placed indicator (all cells filled with 1 for DSP, 2 for BRAM)
        self.observation_space = spaces.Box(
            low=-1,
            high=2,
            shape=(self.width, self.height, 2),
            dtype=np.int8
        )

        # Environment State Variables
        self.grid = np.zeros((self.width, self.height), dtype=np.int8)
        self.placed_dsps = []
        self.placed_brams = []
        self.current_step = 0

    def _init_cache_db(self):
        """Initializes the process-safe SQLite database for layout caching."""
        db_file = Path(self.cache_db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layout_cache (
                cache_key TEXT PRIMARY KEY,
                delay_ns REAL,
                wirelength REAL,
                power_w REAL,
                success INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def get_action_mask(self) -> np.ndarray:
        """
        Computes a binary action mask representing valid placement coordinates.
        1 = Valid, 0 = Invalid.
        """
        mask = np.zeros(self.width * self.height, dtype=np.int8)
        if self.current_step >= self.total_blocks:
            return mask
        
        block_type = self.blocks_to_place[self.current_step]
        block_height = 4 if block_type == 1 else 6

        for act in range(self.width * self.height):
            x = 1 + (act // self.height)
            y = 1 + (act % self.height)
            
            # Boundary check
            if x < 1 or x > self.width:
                continue
            if y < 1 or (y + block_height - 1) > self.height:
                continue
            
            # bake_layout specific constraints
            # height parameter inside bake_layout is core_height + 2
            h_baked = self.height + 2
            if block_type == 1 and x > (h_baked - 4):  # DSP check
                continue
            if block_type == 2 and x > (h_baked - 6):  # BRAM check
                continue
                
            # Overlap check
            overlap = False
            for dy in range(block_height):
                if self.grid[x - 1, y - 1 + dy] != 0:
                    overlap = True
                    break
            
            if not overlap:
                mask[act] = 1
                
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.width, self.height), dtype=np.int8)
        self.placed_dsps = []
        self.placed_brams = []
        self.current_step = 0
        
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.width, self.height, 2), dtype=np.int8)
        obs[:, :, 0] = self.grid
        
        if self.current_step < self.total_blocks:
            # Fill Channel 1 with indicator of block type being placed (1 or 2)
            obs[:, :, 1] = self.blocks_to_place[self.current_step]
        else:
            obs[:, :, 1] = 0
            
        return obs

    def step(self, action: int):
        info = {"status": "normal", "error": None}
        
        # Check if already completed
        if self.current_step >= self.total_blocks:
            obs = self._get_obs()
            return obs, 0.0, True, False, {"status": "already_completed"}

        block_type = self.blocks_to_place[self.current_step]
        block_height = 4 if block_type == 1 else 6

        # Decode coordinates (1-indexed for VTR)
        x = int(1 + (action // self.height))
        y = int(1 + (action % self.height))

        # 1. Bounds Verification
        if x < 1 or x > self.width or y < 1 or (y + block_height - 1) > self.height:
            info["status"] = "out_of_bounds"
            info["error"] = f"Block {block_type} at ({x}, {y}) out of bounds (height={block_height})"
            # Return invalid placement penalty and terminate
            return self._get_obs(), -10.0, True, False, info

        # 2. bake_layout specific constraints
        # height argument inside bake_layout is core_height + 2
        h_baked = self.height + 2
        if block_type == 1 and x > (h_baked - 4):
            info["status"] = "invalid_coordinate"
            info["error"] = f"DSP x={x} exceeds max allowed {h_baked - 4}"
            return self._get_obs(), -10.0, True, False, info
        if block_type == 2 and x > (h_baked - 6):
            info["status"] = "invalid_coordinate"
            info["error"] = f"BRAM x={x} exceeds max allowed {h_baked - 6}"
            return self._get_obs(), -10.0, True, False, info

        # 3. Overlap Verification
        overlap = False
        for dy in range(block_height):
            if self.grid[x - 1, y - 1 + dy] != 0:
                overlap = True
                break
        
        if overlap:
            info["status"] = "overlap"
            info["error"] = f"Overlap detected at ({x}, {y}) for block {block_type}"
            return self._get_obs(), -10.0, True, False, info

        # Apply block to grid
        for dy in range(block_height):
            self.grid[x - 1, y - 1 + dy] = -1
        # Set start position
        self.grid[x - 1, y - 1] = block_type

        # Add to coordinates list
        if block_type == 1:
            self.placed_dsps.append((x, y))
        else:
            self.placed_brams.append((x, y))

        self.current_step += 1

        # Check if placement sequence is complete
        if self.current_step == self.total_blocks:
            # We are done placing all blocks. Evaluate layout!
            reward, eval_info = self._evaluate_layout()
            info.update(eval_info)
            return self._get_obs(), reward, True, False, info
        
        # Intermediate placement step
        return self._get_obs(), 0.0, False, False, info

    def _evaluate_layout(self) -> tuple[float, dict]:
        """Runs the bake-layout and VTR flow, parses performance metrics, and returns reward."""
        eval_info = {
            "success": False, 
            "cached": False,
            "placed_dsps": list(self.placed_dsps),
            "placed_brams": list(self.placed_brams)
        }
        
        # Generate unique cache key
        sorted_dsps = sorted(self.placed_dsps)
        sorted_brams = sorted(self.placed_brams)
        cache_key = f"dsps:{sorted_dsps}|brams:{sorted_brams}"
        
        # 1. Query SQLite Cache
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            delay_ns, wirelength, power_w, success = cached_result
            eval_info["cached"] = True
            eval_info["success"] = bool(success)
            eval_info["delay_ns"] = delay_ns
            eval_info["wirelength"] = wirelength
            eval_info["power_w"] = power_w
            
            if success:
                reward = self._compute_reward(wirelength, power_w, delay_ns)
                return reward, eval_info
            else:
                return -10.0, eval_info

        # 2. Bake and Run VTR Flow (No cache found)
        # Create temp output file and temp VTR workspace to avoid multi-worker conflicts
        worker_id = uuid.uuid4().hex[:8]
        temp_arch_file = SCRIPT_DIR / f"temp_arch_{worker_id}.xml"
        temp_run_dir = SCRIPT_DIR / "runs" / f"temp_run_{worker_id}"

        # Import bake_layout function
        sys.path.append(str(SCRIPT_DIR))
        try:
            from bake_layout import bake_layout
            bake_res = bake_layout(
                benchmark_name=self.benchmark_name,
                dsps=self.placed_dsps,
                mems=self.placed_brams,
                width=self.width + 2,
                height=self.height + 2,
                output_path=str(temp_arch_file)
            )
            if bake_res == -1:
                # Invalid coordinates check failed in bake_layout
                self._write_cache(cache_key, float("inf"), float("inf"), float("inf"), 0)
                eval_info["error"] = "bake_layout returned invalid placement"
                return -10.0, eval_info
        except Exception as e:
            eval_info["error"] = f"bake_layout error: {str(e)}"
            return -10.0, eval_info

        # Run VTR flow subprocess
        benchmark_file = SCRIPT_DIR / "benchmarks" / f"{self.benchmark_name}.v"
        
        vtr_python = Path(os.environ.get("VTR_VENV_PATH", "/home/digital-2/.venv")) / "bin" / "python"
        if not vtr_python.is_file():
            vtr_python = Path(sys.executable)

        vtr_flow_script = Path(os.environ.get("VTR_FLOW_SCRIPT", "/home/digital-2/vtr/vtr_flow/scripts/run_vtr_flow.py"))
        power_tech_file = Path(os.environ.get("VTR_POWER_TECH_FILE", "/home/digital-2/vtr/vtr_flow/tech/PTM_45nm/45nm.xml"))

        if not vtr_flow_script.is_file():
            eval_info["error"] = f"run_vtr_flow.py not found at: {vtr_flow_script}"
            self._cleanup_temp_files(temp_arch_file, temp_run_dir)
            return -10.0, eval_info

        if temp_run_dir.exists():
            shutil.rmtree(temp_run_dir)
        temp_run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            str(vtr_python),
            str(vtr_flow_script),
            str(benchmark_file),
            str(temp_arch_file),
            "-temp_dir",
            str(temp_run_dir),
        ]
        if power_tech_file.is_file():
            cmd.extend(["-cmos_tech", str(power_tech_file)])

        try:
            # Run VTR flow
            vtr_res = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=120 # 2 minute timeout limit per run
            )
            
            # Import parse functions from run_traditional_flow.py
            from run_traditional_flow import parse_metrics
            
            vpr_out_file = temp_run_dir / "vpr.out"
            crit_path_file = temp_run_dir / "vpr.crit_path.out"
            power_file = temp_run_dir / f"{self.benchmark_name}.power"
            metric_dest = temp_run_dir / "metrics.txt"

            if vtr_res.returncode == 0 and vpr_out_file.is_file():
                metrics = parse_metrics(vpr_out_file, crit_path_file, power_file, metric_dest)
                
                # Check parsed successfully
                delay_ns = metrics.get("delay_ns", float("inf"))
                wirelength = metrics.get("wirelength", float("inf"))
                power_w = metrics.get("power_w", float("inf"))
                
                if delay_ns != float("inf") and wirelength != float("inf") and power_w != float("inf"):
                    # Success! Cache result
                    self._write_cache(cache_key, delay_ns, wirelength, power_w, 1)
                    eval_info["success"] = True
                    eval_info["delay_ns"] = delay_ns
                    eval_info["wirelength"] = wirelength
                    eval_info["power_w"] = power_w
                    
                    reward = self._compute_reward(wirelength, power_w, delay_ns)
                    self._cleanup_temp_files(temp_arch_file, temp_run_dir)
                    return reward, eval_info
            
            # Run failed or metrics unparseable
            self._write_cache(cache_key, float("inf"), float("inf"), float("inf"), 0)
            eval_info["error"] = f"VTR flow failed (returncode={vtr_res.returncode})"
            
        except subprocess.TimeoutExpired:
            self._write_cache(cache_key, float("inf"), float("inf"), float("inf"), 0)
            eval_info["error"] = "VTR flow timeout expired"
        except Exception as e:
            self._write_cache(cache_key, float("inf"), float("inf"), float("inf"), 0)
            eval_info["error"] = f"VTR run execution error: {str(e)}"
        
        self._cleanup_temp_files(temp_arch_file, temp_run_dir)
        return -10.0, eval_info

    def _compute_reward(self, wl: float, pw: float, dl: float) -> float:
        """Applies the negative performance metric sum-based reward formula."""
        trad_wl = self.traditional_metrics["wirelength"]
        trad_pw = self.traditional_metrics["power_w"]
        trad_dl = self.traditional_metrics["delay_ns"]
        
        # reward = -0.33 * (wire/trad_wl + power/trad_pw + delay/trad_dl) / 3
        # Which simplifies to: -0.11 * (wire/trad_wl + power/trad_pw + delay/trad_dl)
        reward = -0.11 * ((wl / trad_wl) + (pw / trad_pw) + (dl / trad_dl))
        return reward

    def _cleanup_temp_files(self, arch_file: Path, run_dir: Path):
        """Cleans up temp architecture XML and temp build directory to save disk space."""
        try:
            if arch_file.is_file():
                arch_file.unlink()
            if run_dir.exists():
                shutil.rmtree(run_dir)
        except Exception:
            pass

    def _check_cache(self, key: str) -> tuple | None:
        """Reads database entry for layout caching."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT delay_ns, wirelength, power_w, success FROM layout_cache WHERE cache_key = ?",
                (key,)
            )
            row = cursor.fetchone()
            conn.close()
            return row
        except Exception:
            return None

    def _write_cache(self, key: str, delay: float, wl: float, power: float, success: int):
        """Writes database entry for layout caching (process-safe)."""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO layout_cache VALUES (?, ?, ?, ?, ?)",
                (key, delay, wl, power, success)
            )
            conn.commit()
            conn.close()
        except Exception:
            pass
