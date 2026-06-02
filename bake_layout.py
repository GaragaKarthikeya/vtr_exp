#!/usr/bin/env python3

import sys
import json
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

#the size of the fpga size received is stripped by 2 ,so we add 2 while writing the size of the board again.

def bake_layout(
    benchmark_name: str,
    dsps: list[tuple[int, int]],
    mems: list[tuple[int, int]],
    width: int,
    height: int,
    output_path: str | None = None
) -> Path | int:
    """
    Bakes custom coordinates of DSPs and BRAMs onto a VTR architecture XML template.
    
    Args:
        benchmark_name (str): Name of the benchmark (e.g. 'diffeq1').
        dsps (list of tuple): List of (x, y) coordinates for DSPs.
        mems (list of tuple): List of (x, y) coordinates for BRAMs.
        width (int): FPGA layout width.
        height (int): FPGA layout height.
        output_path (str, optional): Target output path. Defaults to 'baked_layout_arch.xml' in cwd.
        
    Returns:
        Path | int: Path to the generated baked XML file, or -1 if the placement is invalid.
    """
    # Check invalid placement condition:
    # "if the x coordinate of the dsp is greater than the fpga height -4 and the x coordinate of the bram is greater than fpga_height-6"
    has_invalid_dsp = any(x > (height - 4) for x, y in dsps)
    has_invalid_bram = any(x > (height - 6) for x, y in mems)
    if has_invalid_dsp and has_invalid_bram:
        print("Invalid placement: DSP x > height - 4 and BRAM x > height - 6. Aborting generation.")
        return -1

    if benchmark_name.endswith(".v"):
        benchmark_name = benchmark_name[:-2]

    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    template_path = script_dir / "template" / "k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml.j2"
    
    if output_path is None:
        resolved_path = Path.cwd() / "baked_layout_arch.xml"
    else:
        resolved_path = Path(output_path).resolve()

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

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(rendered_xml)

    print(f"Successfully baked architecture to: {resolved_path}")
    print(f"Baked Layout Size: width={width}, height={height}")
    print(f"Placed DSPs ({len(dsps)}): {dsps}")
    print(f"Placed BRAMs ({len(mems)}): {mems}")
    
    return resolved_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Bake Custom Layout Placements onto VTR Architecture XML File")
    parser.add_argument("benchmark_name", type=str, help="Name of the benchmark (e.g. diffeq1)")
    parser.add_argument("--dsps", type=str, nargs="+", default=[], help="DSP coordinate pairs (e.g. --dsps 1,1 1,5)")
    parser.add_argument("--brams", type=str, nargs="+", default=[], help="BRAM coordinate pairs (e.g. --brams 2,1 8,1)")
    parser.add_argument("--width", type=int, default=None, help="FPGA width size")
    parser.add_argument("--height", type=int, default=None, help="FPGA height size")
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

    # Fallback/resolve width and height
    width = args.width
    height = args.height
    if width is None or height is None:
        benchmark_name = args.benchmark_name
        if benchmark_name.endswith(".v"):
            benchmark_name = benchmark_name[:-2]
        script_dir = Path(__file__).resolve().parent
        res_file = script_dir / f"{benchmark_name}_traditional_resources.txt"
        # if not res_file.is_file():
        #     res_file = script_dir / "rl_gnn" / f"{benchmark_name}_traditional_resources.txt"

        if not res_file.is_file():
            print(f"Error: Traditional resources log not found for '{benchmark_name}' (expected: {res_file.name}) and custom dimensions were not fully specified.", file=sys.stderr)
            return 1

        try:
            res_data = json.loads(res_file.read_text())
            fpga_size = res_data.get("fpga_size")
            if not fpga_size or len(fpga_size) != 2:
                raise ValueError(f"Invalid or missing 'fpga_size' in resource log: {res_file}")
            if width is None:
                width = int(fpga_size[0])
            if height is None:
                height = int(fpga_size[1])
        except Exception as e:
            print(f"Error parsing resource log: {e}", file=sys.stderr)
            return 1

    try:
        result = bake_layout(args.benchmark_name, dsps, mems, width+2, height+2, args.output)
        if result == -1:
            print("Error: Invalid placement coordinates specified.", file=sys.stderr)
            return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
