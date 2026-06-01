#!/usr/bin/env python3

import sys
import json
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

def bake_layout(benchmark_name: str, dsps: list[tuple[int, int]], mems: list[tuple[int, int]], output_path: str = None) -> Path:
    """
    Bakes custom coordinates of DSPs and BRAMs onto a VTR architecture XML template.
    
    Args:
        benchmark_name (str): Name of the benchmark (e.g. 'diffeq1').
        dsps (list of tuple): List of (x, y) coordinates for DSPs.
        mems (list of tuple): List of (x, y) coordinates for BRAMs.
        output_path (str, optional): Target output path. Defaults to 'baked_layout_arch.xml'.
        
    Returns:
        Path: Path to the generated baked XML file.
    """
    if benchmark_name.endswith(".v"):
        benchmark_name = benchmark_name[:-2]

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    template_path = script_dir / "template" / "k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml.j2"
    
    if output_path is None:
        output_path = script_dir / "baked_layout_arch.xml"
    else:
        output_path = Path(output_path).resolve()

    # Load traditional resource log to extract core layout size
    res_file = script_dir / f"{benchmark_name}_traditional_resources.txt"
    if not res_file.is_file():
        # Fallback to rl_gnn subdirectory if needed
        res_file = script_dir / "rl_gnn" / f"{benchmark_name}_traditional_resources.txt"

    if not res_file.is_file():
        raise FileNotFoundError(f"Traditional resources log not found for '{benchmark_name}' (expected: {res_file.name})")

    res_data = json.loads(res_file.read_text())
    fpga_size = res_data.get("fpga_size")
    if not fpga_size or len(fpga_size) != 2:
        raise ValueError(f"Invalid or missing 'fpga_size' in resource log: {res_file}")

    # Map core size directly (NO +2 offset!)
    width = int(fpga_size[0])
    height = int(fpga_size[1])

    # Render Jinja template
    if not template_path.is_file():
        raise FileNotFoundError(f"Jinja template file not found at: {template_path}")

    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(template_path.name)

    rendered_xml = template.render(
        layout_name="my_layout",
        width=width,
        height=height,
        dsps=dsps,
        mems=mems,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered_xml)

    print(f"Successfully baked architecture to: {output_path}")
    print(f"Baked Layout Size: width={width}, height={height}")
    print(f"Placed DSPs ({len(dsps)}): {dsps}")
    print(f"Placed BRAMs ({len(mems)}): {mems}")
    
    return output_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Bake Custom Layout Placements onto VTR Architecture XML File")
    parser.add_argument("benchmark_name", type=str, help="Name of the benchmark (e.g. diffeq1)")
    parser.add_argument("--dsps", type=str, nargs="+", default=[], help="DSP coordinate pairs (e.g. --dsps 1,1 1,5)")
    parser.add_argument("--brams", type=str, nargs="+", default=[], help="BRAM coordinate pairs (e.g. --brams 2,1 8,1)")
    parser.add_argument("--output", type=str, default=None, help="Path to write the output baked XML file")
    
    args = parser.parse_args()
    
    def parse_coords(coord_strings):
        coords = []
        for s in coord_strings:
            parts = s.split(",")
            if len(parts) == 2:
                coords.append((int(parts[0]), int(parts[1])))
            else:
                raise ValueError(f"Invalid coordinate format: '{s}'. Expected format is 'x,y'.")
        return coords

    try:
        dsps = parse_coords(args.dsps)
        mems = parse_coords(args.brams)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    try:
        bake_layout(args.benchmark_name, dsps, mems, args.output)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
