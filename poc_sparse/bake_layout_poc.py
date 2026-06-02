#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, StrictUndefined

def bake_layout_poc(
    benchmark_name: str,
    clbs: list[tuple[int, int]],
    dsps: list[tuple[int, int]],
    mems: list[tuple[int, int]],
    width: int,
    height: int,
    output_path: str | None = None
) -> Path:
    """
    Bakes dynamically computed coordinates of CLBs, DSPs, and BRAMs onto a VTR architecture template.
    """
    # Resolve paths
    script_dir = Path(__file__).resolve().parent
    template_path = script_dir / "template" / "k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm_poc.xml.j2"
    
    if output_path is None:
        resolved_path = Path.cwd() / "baked_layout_poc.xml"
    else:
        resolved_path = Path(output_path).resolve()

    if not template_path.is_file():
        raise FileNotFoundError(f"PoC Jinja template file not found at: {template_path}")

    # Initialize Jinja environment
    env = Environment(
        loader=FileSystemLoader(template_path.parent),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )

    template = env.get_template(template_path.name)

    # Render template with active coordinates
    rendered_xml = template.render(
        width=width,
        height=height,
        clbs=clbs,
        dsps=dsps,
        mems=mems,
    )

    # Write output XML file
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(rendered_xml)

    print(f"[BAKER] Successfully baked sparse architecture to: {resolved_path}")
    print(f"[BAKER] Dimensions: width={width}, height={height}")
    print(f"[BAKER] Resources: CLBs={len(clbs)}, DSPs={len(dsps)}, BRAMs={len(mems)}")
    
    return resolved_path

def main() -> int:
    parser = argparse.ArgumentParser(description="Bake Sparse Coordinates onto VTR XML")
    parser.add_argument("benchmark_name", type=str)
    parser.add_argument("--clbs", type=str, nargs="+", default=[], help="CLB coordinates x,y")
    parser.add_argument("--dsps", type=str, nargs="+", default=[], help="DSP coordinates x,y")
    parser.add_argument("--brams", type=str, nargs="+", default=[], help="BRAM coordinates x,y")
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--output", type=str, default=None)
    
    args = parser.parse_args()
    
    def parse_coords(coord_strings):
        coords = []
        for s in coord_strings:
            parts = s.split(",")
            if len(parts) == 2:
                coords.append((int(parts[0]), int(parts[1])))
            else:
                raise ValueError(f"Invalid coordinate format: '{s}'. Expected 'x,y'")
        return coords

    try:
        clbs = parse_coords(args.clbs)
        dsps = parse_coords(args.dsps)
        brams = parse_coords(args.brams)
    except ValueError as e:
        print(f"Error parsing coordinates: {e}", file=sys.stderr)
        return 1

    try:
        bake_layout_poc(
            args.benchmark_name,
            clbs,
            dsps,
            brams,
            args.width,
            args.height,
            args.output
        )
    except Exception as e:
        print(f"Error baking: {e}", file=sys.stderr)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
