import os
import re
import sys
import uuid
import sqlite3
import shutil
import subprocess
from pathlib import Path
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Setup pathing for imports
SCRIPT_DIR = Path(__file__).resolve().parent
PARENT_DIR = SCRIPT_DIR.parent
sys.path.append(str(SCRIPT_DIR))
sys.path.append(str(PARENT_DIR))

# Import layout baker dynamically
from bake_layout_poc import bake_layout_poc
from run_traditional_flow import parse_metrics

class FPGAEnvPoC(gym.Env):
    """
    Gymnasium Environment for Proof-of-Concept Sparse eFPGA Layout Placement.
    Sequentially places exactly 5 DSPs and 73 CLBs inside a 20x20 virtual canvas.
    Dynamically shrink-wraps perimeter IOs around the custom-placed bounding box.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark_name: str = "diffeq1",
        virtual_width: int = 18,
        virtual_height: int = 18,
        req_dsp: int = 5,
        req_clb: int = 73,
        cache_db_path: str = None,
    ):
        super(FPGAEnvPoC, self).__init__()
        
        self.benchmark_name = benchmark_name
        self.virtual_width = virtual_width
        self.virtual_height = virtual_height
        self.req_dsp = req_dsp
        self.req_clb = req_clb
        
        # Hardcoded diffeq1 traditional baselines for normalization
        self.traditional_metrics = {
            "delay_ns": 27.1504,
            "wirelength": 24301.0,
            "power_w": 0.007658,
        }

        # Cache db setup
        if cache_db_path is None:
            self.cache_db_path = str(PARENT_DIR / "runs" / "vtr_layout_cache_poc.db")
        else:
            self.cache_db_path = cache_db_path
        self._init_cache_db()

        # Build block budget queue: 5 DSPs (1), then 73 CLBs (3)
        # Type indicator: 1 = DSP, 3 = CLB
        self.blocks_to_place = [1] * self.req_dsp + [3] * self.req_clb
        self.total_blocks = len(self.blocks_to_place)

        # Virtual interior core dimension (excluding IO boundary of the virtual canvas)
        # Inside the virtual 18x18 grid, coordinates index from x=1..16, y=1..16
        self.core_w = self.virtual_width - 2
        self.core_h = self.virtual_height - 2
        self.total_cells = self.core_w * self.core_h

        # Action Space: Discrete mapping to virtual core coordinates (0..255)
        self.action_space = spaces.Discrete(self.total_cells)

        # Observation Space: (virtual_width, virtual_height, 2)
        self.observation_space = spaces.Box(
            low=-1,
            high=3,
            shape=(self.virtual_width, self.virtual_height, 2),
            dtype=np.int8
        )

        # State Variables
        self.grid = np.zeros((self.virtual_width, self.virtual_height), dtype=np.int8)
        self.placed_dsps = []
        self.placed_clbs = []
        self.current_step = 0

    def _init_cache_db(self):
        """Initializes SQLite database for translation-invariant caching."""
        db_file = Path(self.cache_db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS layout_cache_poc (
                cache_key TEXT PRIMARY KEY,
                width INTEGER,
                height INTEGER,
                delay_ns REAL,
                wirelength REAL,
                power_w REAL,
                success INTEGER
            )
        """)
        conn.commit()
        conn.close()

    def get_action_mask(self) -> np.ndarray:
        """Computes a binary action mask representing valid coordinates on the virtual grid."""
        mask = np.zeros(self.total_cells, dtype=np.int8)
        if self.current_step >= self.total_blocks:
            return mask
        
        block_type = self.blocks_to_place[self.current_step]
        block_height = 4 if block_type == 1 else 1

        for act in range(self.total_cells):
            x = 1 + (act // self.core_h)
            y = 1 + (act % self.core_h)
            
            # Boundary checks
            if x < 1 or x > self.core_w:
                continue
            if y < 1 or (y + block_height - 1) > self.core_h:
                continue
            
            # Overlap check
            overlap = False
            for dy in range(block_height):
                if self.grid[x, y + dy] != 0:
                    overlap = True
                    break
            
            if not overlap:
                mask[act] = 1
                
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = np.zeros((self.virtual_width, self.virtual_height), dtype=np.int8)
        self.placed_dsps = []
        self.placed_clbs = []
        self.current_step = 0
        
        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        obs = np.zeros((self.virtual_width, self.virtual_height, 2), dtype=np.int8)
        obs[:, :, 0] = self.grid
        
        if self.current_step < self.total_blocks:
            obs[:, :, 1] = self.blocks_to_place[self.current_step]
        else:
            obs[:, :, 1] = 0
            
        return obs

    def step(self, action: int):
        info = {"status": "normal", "error": None}
        
        if self.current_step >= self.total_blocks:
            return self._get_obs(), 0.0, True, False, {"status": "already_completed"}

        block_type = self.blocks_to_place[self.current_step]
        block_height = 4 if block_type == 1 else 1

        # Decode coordinates relative to virtual core (1-indexed)
        x = int(1 + (action // self.core_h))
        y = int(1 + (action % self.core_h))

        # 1. Bounds check
        if x < 1 or x > self.core_w or y < 1 or (y + block_height - 1) > self.core_h:
            info["status"] = "out_of_bounds"
            return self._get_obs(), -20.0, True, False, info

        # 2. Overlap check
        overlap = False
        for dy in range(block_height):
            if self.grid[x, y + dy] != 0:
                overlap = True
                break
        
        if overlap:
            info["status"] = "overlap"
            return self._get_obs(), -20.0, True, False, info

        # Apply block occupancy to virtual grid
        for dy in range(block_height):
            self.grid[x, y + dy] = -1
        self.grid[x, y] = block_type

        # Track block coordinates
        if block_type == 1:
            self.placed_dsps.append((x, y))
        else:
            self.placed_clbs.append((x, y))

        self.current_step += 1

        # Episode completion triggers dynamic bounding box co-optimization
        if self.current_step == self.total_blocks:
            reward, eval_info = self._evaluate_dynamic_layout()
            info.update(eval_info)
            return self._get_obs(), reward, True, False, info
        
        return self._get_obs(), 0.0, False, False, info

    def _evaluate_dynamic_layout(self) -> tuple[float, dict]:
        eval_info = {
            "success": False,
            "cached": False,
            "placed_dsps": list(self.placed_dsps),
            "placed_clbs": list(self.placed_clbs)
        }

        # 1. Extract dynamic bounding box of placed blocks
        all_xs = [x for (x, y) in self.placed_dsps] + [x for (x, y) in self.placed_clbs]
        all_ys = [y for (x, y) in self.placed_clbs] + [y for (x, y) in self.placed_dsps]
        
        # account for vertical height of DSP blocks in calculating bounding max y
        all_dsp_ys_span = [y + 3 for (x, y) in self.placed_dsps]
        all_ys_extended = all_ys + all_dsp_ys_span

        x_min, x_max = min(all_xs), max(all_xs)
        y_min, y_max = min(all_ys_extended), max(all_ys_extended)

        # Compute dynamic core dimensions
        W_core = x_max - x_min + 1
        H_core = y_max - y_min + 1

        # Core Shift to (1, 1) to eliminate absolute coordinate translation bias
        shifted_dsps = [(x - x_min + 1, y - y_min + 1) for (x, y) in self.placed_dsps]
        shifted_clbs = [(x - x_min + 1, y - y_min + 1) for (x, y) in self.placed_clbs]

        # Sizing with boundary IOs shrink-wrapped
        width = W_core + 2
        height = H_core + 2

        eval_info["width"] = width
        eval_info["height"] = height
        eval_info["shifted_dsps"] = shifted_dsps
        eval_info["shifted_clbs"] = shifted_clbs

        # Create translation-invariant cache key
        sorted_dsps = sorted(shifted_dsps)
        sorted_clbs = sorted(shifted_clbs)
        cache_key = f"dsps:{sorted_dsps}|clbs:{sorted_clbs}"

        # 2. Query SQLite Cache
        cached_result = self._check_cache(cache_key)
        if cached_result is not None:
            c_width, c_height, delay_ns, wirelength, power_w, success = cached_result
            eval_info["cached"] = True
            eval_info["success"] = bool(success)
            eval_info["delay_ns"] = delay_ns
            eval_info["wirelength"] = wirelength
            eval_info["power_w"] = power_w
            
            if success:
                reward = self._compute_reward(W_core, H_core, wirelength, power_w, delay_ns)
                return reward, eval_info
            else:
                return -15.0, eval_info

        # 3. Execute VTR Flow (Bake architecture XML & place-and-route)
        worker_id = uuid.uuid4().hex[:8]
        temp_arch_file = SCRIPT_DIR / f"temp_arch_{worker_id}.xml"
        temp_run_dir = SCRIPT_DIR / "runs" / f"temp_run_{worker_id}"

        try:
            bake_layout_poc(
                benchmark_name=self.benchmark_name,
                clbs=shifted_clbs,
                dsps=shifted_dsps,
                mems=[],
                width=width,
                height=height,
                output_path=str(temp_arch_file)
            )
        except Exception as e:
            eval_info["error"] = f"Baker failed: {e}"
            return -15.0, eval_info

        benchmark_file = PARENT_DIR / "benchmarks" / f"{self.benchmark_name}.v"
        vtr_python = Path(os.environ.get("VTR_VENV_PATH", "/home/digital-2/.venv")) / "bin" / "python"
        if not vtr_python.is_file():
            vtr_python = Path(sys.executable)

        vtr_flow_script = Path(os.environ.get("VTR_FLOW_SCRIPT", "/home/digital-2/vtr/vtr_flow/scripts/run_vtr_flow.py"))
        power_tech_file = Path(os.environ.get("VTR_POWER_TECH_FILE", "/home/digital-2/vtr/vtr_flow/tech/PTM_45nm/45nm.xml"))

        if not vtr_flow_script.is_file():
            eval_info["error"] = f"run_vtr_flow.py script missing at {vtr_flow_script}"
            self._cleanup(temp_arch_file, temp_run_dir)
            return -15.0, eval_info

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
            # Execute VTR flow subprocess (max 2 minute routing timeout)
            vtr_res = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=120
            )
            
            vpr_out = temp_run_dir / "vpr.out"
            crit_path = temp_run_dir / "vpr.crit_path.out"
            power = temp_run_dir / f"{self.benchmark_name}.power"
            metrics_txt = temp_run_dir / "metrics.txt"

            if vtr_res.returncode == 0 and vpr_out.is_file():
                metrics = parse_metrics(vpr_out, crit_path, power, metrics_txt)
                delay_ns = metrics.get("delay_ns", float("inf"))
                wirelength = metrics.get("wirelength", float("inf"))
                power_w = metrics.get("power_w", float("inf"))

                if delay_ns != float("inf") and wirelength != float("inf") and power_w != float("inf"):
                    # Success: write to cache and return reward
                    self._write_cache(cache_key, width, height, delay_ns, wirelength, power_w, 1)
                    eval_info["success"] = True
                    eval_info["delay_ns"] = delay_ns
                    eval_info["wirelength"] = wirelength
                    eval_info["power_w"] = power_w

                    reward = self._compute_reward(W_core, H_core, wirelength, power_w, delay_ns)
                    self._cleanup(temp_arch_file, temp_run_dir)
                    return reward, eval_info

            # Failed routing
            self._write_cache(cache_key, width, height, float("inf"), float("inf"), float("inf"), 0)
            eval_info["error"] = f"VTR failed with code {vtr_res.returncode}"
            
        except subprocess.TimeoutExpired:
            self._write_cache(cache_key, width, height, float("inf"), float("inf"), float("inf"), 0)
            eval_info["error"] = "VTR timeout expired"
        except Exception as e:
            eval_info["error"] = f"Subprocess error: {e}"

        self._cleanup(temp_arch_file, temp_run_dir)
        return -15.0, eval_info

    def _compute_reward(self, W_core: int, H_core: int, wl: float, power: float, delay: float) -> float:
        """
        Computes dynamic multi-objective PPA reward.
        Primary Objective: Minimize core area footprint (W_core * H_core).
        Secondary Objective: Maintain/minimize dynamic timing, wirelength, and power delays.
        """
        area = W_core * H_core
        
        # Normalize routing metrics
        wl_norm = wl / self.traditional_metrics["wirelength"]
        power_norm = power / self.traditional_metrics["power_w"]
        delay_norm = delay / self.traditional_metrics["delay_ns"]

        # Reward emphasizes footprint compression (area weight = 0.2 per tile)
        # with secondary timing/power constraints.
        reward = - 0.2 * area - 0.1 * (wl_norm + power_norm + delay_norm)
        return float(reward)

    def _check_cache(self, cache_key: str) -> tuple | None:
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT width, height, delay_ns, wirelength, power_w, success FROM layout_cache_poc WHERE cache_key=?",
                (cache_key,)
            )
            row = cursor.fetchone()
            conn.close()
            return row
        except Exception:
            return None

    def _write_cache(self, cache_key: str, w: int, h: int, delay: float, wl: float, power: float, success: int):
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO layout_cache_poc VALUES (?, ?, ?, ?, ?, ?, ?)",
                (cache_key, w, h, delay, wl, power, success)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"[CACHE ERROR] Failed writing to SQLite: {e}", file=sys.stderr)

    def _cleanup(self, arch_file: Path, run_dir: Path):
        try:
            if arch_file.is_file():
                arch_file.unlink()
            if run_dir.exists():
                shutil.rmtree(run_dir)
        except Exception:
            pass
