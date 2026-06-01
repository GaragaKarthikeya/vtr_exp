#!/usr/bin/env python3

from __future__ import annotations

import json
from pathlib import Path

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from architecture_generator import generate_layout_data, bake_architecture
from vtr_run import run_vtr, extract_metrics


class VTRPlacementEnv(gym.Env):
    def __init__(
        self,
        req_file_path: str = "resources.txt",
        baseline_file_path: str = "raygentop_traditional_metric.txt",
    ):
        super().__init__()

        self.script_dir = Path(__file__).resolve().parent

        req_path = self.script_dir / req_file_path
        baseline_path = self.script_dir / baseline_file_path

        with req_path.open("r") as f:
            reqs = json.load(f)

        self.num_dsps = reqs["requirements"]["dsp"]
        self.num_brams = reqs["requirements"]["bram"]
        self.req_io = reqs["requirements"]["io"]
        self.req_clb = reqs["requirements"]["clb"]
        self.initial_width = reqs["fpga_size"][0]

        with baseline_path.open("r") as f:
            self.baseline = json.load(f)

        total_hard_blocks = self.num_dsps + self.num_brams

        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(total_hard_blocks,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32,
        )

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)

        

        dsp_actions = action[: self.num_dsps]
        bram_actions = action[self.num_dsps :]

        min_x = 1
        max_x = self.initial_width - 2

        dsp_x_coords = []
        for a in dsp_actions:
            x_float = min_x + float(a) * (max_x - min_x)
            x = int(round(x_float))

            if x % 2 == 0:
                lower = x - 1
                upper = x + 1

                candidates = []

                if min_x <= lower <= max_x:
                    candidates.append(lower)

                if min_x <= upper <= max_x:
                    candidates.append(upper)

                x = min(candidates, key=lambda c: abs(x_float - c))

            dsp_x_coords.append(x)


        bram_x_coords = []
        for a in bram_actions:
            x_float = min_x + float(a) * (max_x - min_x)
            x = int(round(x_float))

            if x % 2 != 0:
                lower = x - 1
                upper = x + 1

                candidates = []

                if min_x <= lower <= max_x:
                    candidates.append(lower)

                if min_x <= upper <= max_x:
                    candidates.append(upper)

                x = min(candidates, key=lambda c: abs(x_float - c))

            bram_x_coords.append(x)


      

        width, height, dsps, mems = generate_layout_data(
            dsp_x=dsp_x_coords,
            bram_x=bram_x_coords,
            req_io=self.req_io,
            req_clb=self.req_clb,
            io_cap=8,
        )

        template_path = (
            self.script_dir
            / "../arch/k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml.j2"
        ).resolve()

        output_path = self.script_dir / "vpr_arch_run.xml"

        bake_architecture(
            template_path=template_path,
            output_path=output_path,
            width=width,
            height=height,
            dsps=dsps,
            mems=mems,
        )

        print(f"Executing VTR... DSPs: {dsps}, BRAMs: {mems}")

        return_code = run_vtr(self.script_dir)

        if return_code != 0:
            print("VTR run failed. Penalizing agent.")

            metrics = {
                "delay_ns": float("inf"),
                "wirelength": float("inf"),
                "power_w": float("inf"),
            }

            reward = -10.0

        else:
            metrics = extract_metrics(self.script_dir)

            values = np.array(
                [
                    metrics["delay_ns"],
                    metrics["wirelength"],
                    metrics["power_w"],
                ],
                dtype=np.float64,
            )

            if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
                reward = -5.0
            else:
                delay_score = 1.0 - (metrics["delay_ns"] / self.baseline["delay_ns"])
                wire_score = 1.0 - (metrics["wirelength"] / self.baseline["wirelength"])
                power_score = 1.0 - (metrics["power_w"] / self.baseline["power_w"])

                

                reward = (
                    delay_score * 0.34
                    + wire_score * 0.33
                    + power_score * 0.33
                )

                reward = float(reward)

        info = {
            "delay_ns": metrics["delay_ns"],
            "wirelength": metrics["wirelength"],
            "power_w": metrics["power_w"],
            "width": width,
            "height": height,
            "dsps": dsps,
            "mems": mems,
            "vtr_return_code": return_code,
        }

        print(f"Step complete -> Reward: {reward:.3f}\n")

        observation = np.zeros(1, dtype=np.float32)
        terminated = True
        truncated = False

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}


if __name__ == "__main__":
    env = VTRPlacementEnv()

    sample_action = env.action_space.sample()
    print(f"Testing environment with action: {sample_action}")

    obs, reward, terminated, truncated, info = env.step(sample_action)

    print(f"Final reward: {reward}")
    print(f"Final info: {info}")
