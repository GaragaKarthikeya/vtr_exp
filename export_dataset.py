#!/usr/bin/env python3

import os
import re
import sys
import json
import sqlite3
from pathlib import Path

# ANSI coloring
COLOR_RESET = "\033[0m"
COLOR_SUCCESS = "\033[92m"
COLOR_INFO = "\033[96m"

def export_dataset(db_file: str | None = None, output_file: str | None = None):
    script_dir = Path(__file__).resolve().parent
    
    if db_file is None:
        db_path = script_dir / "runs" / "vtr_layout_cache_diffeq1.db"
    else:
        db_path = Path(db_file).resolve()

    if output_file is None:
        out_path = script_dir / "diffeq1_gnn_dataset.json"
    else:
        out_path = Path(output_file).resolve()

    if not db_path.is_file():
        print(f"Error: Cache database not found at: {db_path}")
        return 1

    print(f"{COLOR_INFO}Connecting to SQLite Cache Database: {db_path.name}{COLOR_RESET}")
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    try:
        cursor.execute("SELECT cache_key, delay_ns, wirelength, power_w, success FROM layout_cache")
        rows = cursor.fetchall()
    except Exception as e:
        print(f"Error reading database: {e}")
        conn.close()
        return 1
        
    conn.close()

    print(f"Found {len(rows)} total records in layout cache.")
    
    dataset = []
    success_count = 0
    
    # Helper to parse coordinates list from cache key string:
    # Key format: "dsps:[(9, 10), (5, 8)]|brams:[]"
    for cache_key, delay, wl, power, success in rows:
        try:
            # Parse dsps
            dsp_match = re.search(r"dsps:(\[.*?\])", cache_key)
            # Parse brams
            bram_match = re.search(r"brams:(\[.*?\])", cache_key)
            
            if dsp_match:
                # Convert string tuple format [(1, 2)] to valid JSON list of lists [[1, 2]]
                dsp_str = dsp_match.group(1).replace("(", "[").replace(")", "]")
                dsps = json.loads(dsp_str)
            else:
                dsps = []

            if bram_match:
                bram_str = bram_match.group(1).replace("(", "[").replace(")", "]")
                brams = json.loads(bram_str)
            else:
                brams = []

            # Compute reward for successful ones:
            # Load baseline traditional metrics (diffeq1 baseline)
            trad_wl = 24301.0
            trad_pw = 0.007658
            trad_dl = 27.1504
            
            if success == 1:
                # Additive reward formula: -0.11 * (wire/trad_wl + power/trad_pw + delay/trad_dl)
                reward = -0.11 * ((wl / trad_wl) + (power / trad_pw) + (delay / trad_dl))
                success_count += 1
            else:
                reward = -10.0

            dataset.append({
                "dsps": dsps,
                "brams": brams,
                "reward": reward,
                "wirelength": wl,
                "delay_ns": delay,
                "power_w": power,
                "success": int(success)
            })
            
        except Exception as e:
            print(f"Warning: Failed to parse record '{cache_key}': {e}")
            continue

    # Write out to JSON dataset file
    try:
        out_path.write_text(json.dumps(dataset, indent=4))
        print(f"\n{COLOR_SUCCESS}Dataset successfully exported to: {out_path.name}{COLOR_RESET}")
        print(f"  - Total Layouts:  {len(dataset)}")
        print(f"  - Successful:     {success_count}")
        print(f"  - Failed/Routing: {len(dataset) - success_count}")
        print(f"  - Format:         JSON Array of objects with layout coordinates and performance metrics")
    except Exception as e:
        print(f"Error writing output dataset: {e}")
        return 1

    return 0

if __name__ == "__main__":
    db_file = sys.argv[1] if len(sys.argv) > 1 else None
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    sys.exit(export_dataset(db_file, output_file))
