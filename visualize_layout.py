#!/usr/bin/env python3

import sys
import json
from pathlib import Path

# ANSI color codes for premium console visuals
COLOR_RESET = "\033[0m"
COLOR_IO = "\033[93m"       # Bright Yellow
COLOR_DSP = "\033[91m"      # Bright Red
COLOR_BRAM = "\033[94m"     # Bright Blue
COLOR_CLB = "\033[90m"      # Dark Gray
COLOR_CORNER = "\033[37m"   # Light Gray
COLOR_METRIC = "\033[96m"   # Cyan

def visualize(coordinates_file: str | None = None):
    script_dir = Path(__file__).resolve().parent
    
    if coordinates_file is None:
        # Default to best layout of diffeq1
        coords_path = script_dir / "best_layout_coordinates_diffeq1.txt"
    else:
        coords_path = Path(coordinates_file).resolve()

    if not coords_path.is_file():
        print(f"Error: Coordinates file not found at: {coords_path}")
        print("Please make sure the training has completed at least one successful placement episode.")
        return 1

    try:
        data = json.loads(coords_path.read_text())
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return 1

    # Load resources to get core size
    benchmark_name = coords_path.name.replace("best_layout_coordinates_", "").replace(".txt", "")
    res_path = script_dir / f"{benchmark_name}_traditional_resources.txt"
    
    width, height = 14, 14  # Defaults
    if res_path.is_file():
        try:
            res_data = json.loads(res_path.read_text())
            fpga_size = res_data.get("fpga_size")
            if fpga_size and len(fpga_size) == 2:
                width = int(fpga_size[0])
                height = int(fpga_size[1])
        except Exception:
            pass

    # Grid size with perimeter IOs included
    full_width = width + 2
    full_height = height + 2

    # Initialize 2D grid matrix of size (full_width, full_height)
    # Using cell dictionary to represent states
    grid = {}
    for x in range(full_width):
        for y in range(full_height):
            # Corners are EMPTY
            if (x == 0 or x == full_width - 1) and (y == 0 or y == full_height - 1):
                grid[(x, y)] = (" ", COLOR_CORNER)
            # Perimeter is IO
            elif x == 0 or x == full_width - 1 or y == 0 or y == full_height - 1:
                grid[(x, y)] = ("IO", COLOR_IO)
            # Core cells are CLBs
            else:
                grid[(x, y)] = ("· ", COLOR_CLB)

    # Place DSPs (height = 4)
    placed_dsps = data.get("dsps", [])
    for idx, (x, y) in enumerate(placed_dsps):
        for dy in range(4):
            if y + dy < full_height:
                if dy == 0:
                    label = f"D{idx}"
                else:
                    label = "d|"
                grid[(x, y + dy)] = (label, COLOR_DSP)

    # Place BRAMs (height = 6)
    placed_brams = data.get("brams", [])
    for idx, (x, y) in enumerate(placed_brams):
        for dy in range(6):
            if y + dy < full_height:
                if dy == 0:
                    label = f"B{idx}"
                else:
                    label = "b|"
                grid[(x, y + dy)] = (label, COLOR_BRAM)

    # Print Header and Performance Metrics
    print("=" * 60)
    print(f"FPGA ARCHITECTURE GRID VISUALIZATION ({benchmark_name.upper()})")
    print("=" * 60)
    print(f"{COLOR_METRIC}Performance Metrics:{COLOR_RESET}")
    print(f"  - Reward:     {data.get('reward'):.5f}")
    print(f"  - Wirelength: {data.get('wirelength')} units")
    print(f"  - Delay:      {data.get('delay_ns')} ns")
    print(f"  - Power:      {data.get('power_w')} W")
    print(f"  - Placed:     {len(placed_dsps)} DSPs, {len(placed_brams)} BRAMs")
    print(f"  - Core Size:  {width} x {height} (Full grid: {full_width} x {full_height})")
    print("-" * 60)

    # Print the grid (reversing y coordinate because y=0 is bottom and y=full_height-1 is top)
    for y in reversed(range(full_height)):
        row_str = []
        for x in range(full_width):
            label, color = grid[(x, y)]
            row_str.append(f"{color}{label}{COLOR_RESET}")
        
        # Add row coordinate for easy reading
        y_coord_str = f"{y:02d} "
        print(y_coord_str + " ".join(row_str))

    # Print column coordinates footer
    footer_str = "    " + " ".join(f"{x:02d}" for x in range(full_width))
    print(footer_str)
    
    print("-" * 60)
    print(f"Legend: {COLOR_IO}IO{COLOR_RESET}=IO Block  "
          f"{COLOR_CLB}· {COLOR_RESET}=CLB/Logic  "
          f"{COLOR_DSP}D0-D4{COLOR_RESET}=DSP Start  "
          f"{COLOR_DSP}d|{COLOR_RESET}=DSP Body  "
          f"{COLOR_BRAM}B0-B4{COLOR_RESET}=BRAM Start  "
          f"{COLOR_BRAM}b|{COLOR_RESET}=BRAM Body")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    coords_file = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(visualize(coords_file))
