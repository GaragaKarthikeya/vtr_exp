import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np

try:
    import torch
    from torch_geometric.data import Data
except ImportError:
    pass

from blif_to_gnn import parse_blif, build_features

class VTREnv:
    def __init__(self, 
                 base_arch: Path, 
                 circuit_list: list[Path], 
                 runs_root: Path,
                 vtr_extra_args: list[str] = None,
                 penalty_weight: float = 6.0e5):
        self.base_arch = base_arch
        self.circuit_list = circuit_list
        self.runs_root = runs_root
        self.vtr_extra_args = vtr_extra_args or []
        self.penalty_weight = penalty_weight
        
        self.current_circuit_idx = -1
        self.baseline_stats = {} # circuit_path -> (power, cp)
        
        # Cache for circuit graphs
        self.graph_cache = {} # circuit_path -> PyG Data
        
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self.evals_dir = self.runs_root / "evals"
        self.evals_dir.mkdir(parents=True, exist_ok=True)

    def _get_blif_path(self, circuit_path: Path):
        baseline_dir = self.evals_dir / f"baseline_{circuit_path.stem}"
        blif_path = baseline_dir / f"{circuit_path.stem}.parmys.blif"
        if not blif_path.exists():
            print(f"Generating baseline and BLIF for {circuit_path.name}...")
            self._run_vtr_raw(self.base_arch, circuit_path, baseline_dir)
            
        return blif_path

    def _run_vtr_raw(self, arch: Path, circuit: Path, out_dir: Path):
        root = Path(__file__).resolve().parent.parent
        cmd = [
            "python3",
            str(root / "run-vtr.py"),
            str(arch),
            str(circuit),
            str(out_dir),
            *self.vtr_extra_args
        ]
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "run_error.log", "w") as f:
                f.write(result.stderr)

    def _parse_power_and_cp(self, power_file: Path):
        total_re = re.compile(r"^Total\s+([0-9.eE+-]+)")
        cp_re = re.compile(r"^Critical Path:\s*([0-9.eE+-]+)")
        power_val, cp_val = None, None
        if not power_file.is_file():
            return None, None
        for line in power_file.read_text().splitlines():
            if cp_val is None:
                m = cp_re.match(line.strip())
                if m: cp_val = float(m.group(1))
            if power_val is None:
                m = total_re.match(line.strip())
                if m: power_val = float(m.group(1))
        return power_val, cp_val

    def reset(self, circuit_idx=None):
        if circuit_idx is None:
            self.current_circuit_idx = (self.current_circuit_idx + 1) % len(self.circuit_list)
        else:
            self.current_circuit_idx = circuit_idx
            
        circuit_path = self.circuit_list[self.current_circuit_idx]
        
        if circuit_path not in self.baseline_stats:
            blif = self._get_blif_path(circuit_path)
            power, cp = self._parse_power_and_cp(blif.parent / f"{circuit_path.stem}.power")
            self.baseline_stats[circuit_path] = (power, cp)
            
        if circuit_path not in self.graph_cache:
            blif = self._get_blif_path(circuit_path)
            nodes, edge_index = parse_blif(blif)
            x = torch.from_numpy(build_features(nodes))
            edge_index_t = torch.tensor(edge_index, dtype=torch.long).t().contiguous() if edge_index else torch.zeros((2, 0), dtype=torch.long)
            self.graph_cache[circuit_path] = Data(x=x, edge_index=edge_index_t)
            
        return self.graph_cache[circuit_path]

    def step(self, action):
        circuit_path = self.circuit_list[self.current_circuit_idx]
        baseline_power, baseline_cp = self.baseline_stats[circuit_path]
        
        aspect_ratio, dsp_x, bram_x = action
        name = f"eval_{circuit_path.stem}_a{aspect_ratio:.2f}_d{int(round(dsp_x))}_b{int(round(bram_x))}"
        out_dir = self.evals_dir / name
        arch_variant = out_dir / "arch.xml"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        self._write_arch_variant(self.base_arch, action, arch_variant)
        self._run_vtr_raw(arch_variant, circuit_path, out_dir)
        
        power, cp = self._parse_power_and_cp(out_dir / f"{circuit_path.stem}.power")
        
        if power is None or cp is None:
            reward = -10.0
            success = False
        else:
            penalty = max(0.0, cp - baseline_cp)
            reward = -(power + self.penalty_weight * penalty)
            reward = reward / (baseline_power or 1.0)
            success = True
            
        return self.graph_cache[circuit_path], reward, True, {"success": success, "power": power, "cp": cp}

    def _write_arch_variant(self, base_arch, action, out_path):
        aspect_ratio, dsp_x, bram_x = action
        tree = ET.parse(base_arch)
        root = tree.getroot()
        auto_layout = root.find("./layout/auto_layout")
        auto_layout.set("aspect_ratio", f"{aspect_ratio:.4f}")
        
        dsp_startx = int(round(dsp_x))
        bram_startx = int(round(bram_x))

        for col in auto_layout.findall("col"):
            ctype = col.get("type")
            if ctype == "mult_36":
                col.set("startx", str(dsp_startx))
            elif ctype == "memory":
                col.set("startx", str(bram_startx))
            elif ctype == "EMPTY":
                orig_startx = col.get("startx")
                if orig_startx == "6":
                    col.set("startx", str(dsp_startx))
                elif orig_startx == "2":
                    col.set("startx", str(bram_startx))
                
        tree.write(out_path)
