#!/usr/bin/env python3

import sys
import json
from pathlib import Path

# ANSI color codes for premium console visuals
COLOR_RESET = "\033[0m"
COLOR_IO = "\033[93m"       # Bright Yellow
COLOR_DSP = "\033[91m"      # Bright Red
COLOR_CLB = "\033[92m"      # Bright Green (placed CLB)
COLOR_EMPTY = "\033[90m"    # Dark Gray (empty slot)
COLOR_CORNER = "\033[37m"   # Light Gray
COLOR_METRIC = "\033[96m"   # Cyan

def visualize_poc(coordinates_file: str | None = None):
    script_dir = Path(__file__).resolve().parent
    
    if coordinates_file is None:
        coords_path = script_dir / "best_sparse_layout_coordinates.txt"
    else:
        coords_path = Path(coordinates_file).resolve()

    if not coords_path.is_file():
        print(f"Error: Coordinates file not found at: {coords_path}")
        return 1

    try:
        data = json.loads(coords_path.read_text())
    except Exception as e:
        print(f"Error reading JSON file: {e}")
        return 1

    # Read dynamic layout dimensions
    width = data.get("width", 14)
    height = data.get("height", 14)
    
    # Initialize 2D grid matrix of size (width, height)
    grid = {}
    for x in range(width):
        for y in range(height):
            # Corners are EMPTY
            if (x == 0 or x == width - 1) and (y == 0 or y == height - 1):
                grid[(x, y)] = ("  ", COLOR_CORNER)
            # Perimeter is IO
            elif x == 0 or x == width - 1 or y == 0 or y == height - 1:
                grid[(x, y)] = ("IO", COLOR_IO)
            # Core cells default to EMPTY in our sparse core
            else:
                grid[(x, y)] = (" .", COLOR_EMPTY)

    # Place CLBs (exactly 73 CLBs, height = 1)
    placed_clbs = data.get("clbs", [])
    for idx, (x, y) in enumerate(placed_clbs):
        if 0 <= x < width and 0 <= y < height:
            grid[(x, y)] = ("CL", COLOR_CLB)

    # Place DSPs (exactly 5 DSPs, height = 4)
    placed_dsps = data.get("dsps", [])
    for idx, (x, y) in enumerate(placed_dsps):
        for dy in range(4):
            if y + dy < height:
                if dy == 0:
                    label = f"D{idx}"
                else:
                    label = "d|"
                grid[(x, y + dy)] = (label, COLOR_DSP)

    # Print Header and Performance Metrics
    print("=" * 60)
    print(f"SPARSE eFPGA DYNAMIC CORE GRID VISUALIZATION")
    print("=" * 60)
    print(f"{COLOR_METRIC}Physical Design & Routing Metrics:{COLOR_RESET}")
    print(f"  - Reward:        {data.get('reward', 0.0):.5f}")
    print(f"  - Wirelength:    {data.get('wirelength', 'N/A')} units")
    print(f"  - Delay:         {data.get('delay_ns', 'N/A')} ns")
    print(f"  - Power:         {data.get('power_w', 'N/A')} W")
    print(f"  - Placed:        {len(placed_clbs)} CLBs, {len(placed_dsps)} DSPs")
    print(f"  - Core BBox:     {width - 2} x {height - 2} (Area: {(width-2)*(height-2)} tiles)")
    print(f"  - Fabric Size:   {width} x {height} (Total Footprint: {width*height} tiles)")
    
    active_tiles = len(placed_clbs) + len(placed_dsps) * 4
    total_interior = (width - 2) * (height - 2)
    empty_tiles = total_interior - active_tiles
    print(f"  - Core Density:  {active_tiles}/{total_interior} active tiles ({active_tiles/total_interior*100:.1f}% utilization)")
    print(f"  - Empty Wasted:  {empty_tiles} interior tiles (Obfuscation Mask)")
    print("-" * 60)

    # Print the grid (y=0 is bottom, y=height-1 is top)
    for y in reversed(range(height)):
        row_str = []
        for x in range(width):
            label, color = grid[(x, y)]
            row_str.append(f"{color}{label}{COLOR_RESET}")
        
        # Add row coordinate for easy reading
        y_coord_str = f"{y:02d} "
        print(y_coord_str + " ".join(row_str))

    # Print column coordinates footer
    footer_str = "    " + " ".join(f"{x:02d}" for x in range(width))
    print(footer_str)
    
    print("-" * 60)
    print(f"Legend: {COLOR_IO}IO{COLOR_RESET}=IO Boundary  "
          f"{COLOR_CLB}CL{COLOR_RESET}=Active CLB  "
          f"{COLOR_DSP}D0-D4{COLOR_RESET}=DSP Start  "
          f"{COLOR_DSP}d|{COLOR_RESET}=DSP Body  "
          f"{COLOR_EMPTY} .{COLOR_RESET}=Empty Silicon (No Overhead)")
    print("=" * 60)

    return 0

if __name__ == "__main__":
    coords_file = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(visualize_poc(coords_file))
