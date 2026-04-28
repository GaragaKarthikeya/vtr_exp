import os
import re
import json

def extract_baseline_only(run_dir, circuit_name, output_file="raygentop_traditional_metric.txt"):
    """
    Reads existing VPR output files and extracts the metrics to a JSON text file.
    No execution, just parsing.
    """
    # Define exact file paths based on the directory you provide
    vpr_out_file = os.path.join(run_dir, "vpr.out")
    crit_path_file = os.path.join(run_dir, "vpr.crit_path.out")
    power_file = os.path.join(run_dir, f"{circuit_name}.power")

    metrics = {
        "delay_ns": float('inf'),
        "wirelength": float('inf'),
        "power_w": float('inf')
    }

    # 1. Parse Wirelength
    if os.path.exists(vpr_out_file):
       with open(vpr_out_file, 'r') as f:
            content = f.read()
            
            # This regex finds the block and captures:
            # Group 1: Total wirelength (Integer)
            # Group 2: Average net length (Float)
            wl_pattern = r"Wire length results.*?Total wirelength:\s+([0-9]+),\s+average net length:\s+([0-9.]+)"
            match = re.search(wl_pattern, content, re.DOTALL)
            
            if match:
                metrics["wirelength"] = float(match.group(1))
             
            else:
                print(f"Warning: Could not extract wirelength stats from {vpr_out_file}")
    else:
        print(f"Warning: Could not find {vpr_out_file}")

    # 2. Parse Critical Path Delay
    if os.path.exists(crit_path_file):
        with open(crit_path_file, 'r') as f:
            match = re.search(r"Final critical path delay \(least slack\):\s+([0-9.]+)\s+ns", f.read())
            if match:
                metrics["delay_ns"] = float(match.group(1))
    else:
        print(f"Warning: Could not find {crit_path_file}")

    # 3. Parse Total Power
    if os.path.exists(power_file):
        with open(power_file, 'r') as f:
            match = re.search(r"^Total\s+([0-9\.eE\-]+)", f.read(), re.MULTILINE)
            if match:
                metrics["power_w"] = float(match.group(1))
    else:
        print(f"Warning: Could not find {power_file}")

    # Save cleanly to your text file
    with open(output_file, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Extraction complete. Data saved to {output_file}:")
    print(json.dumps(metrics, indent=4))

# --- RUN IT ---
if __name__ == "__main__":
    # Point this strictly to the folder where your baseline files already sit
    # Example: if your files are in "runs/diffeq1/"
    TARGET_DIR = "/root/desktop/vtr_exp/runs/raygentop"
    CIRCUIT = "raygentop"
    
    extract_baseline_only(TARGET_DIR, CIRCUIT)