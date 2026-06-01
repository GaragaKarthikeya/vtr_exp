import os
import re
import json

def extract_resources(run_dir, output_file="resources.txt"):
    """
    Parses VPR resource usage and grid size to establish RL environment limits.
    """
    vpr_out_file = os.path.join(run_dir, "vpr.out")
    
    resources = {
        "fpga_size": [0, 0],
        "requirements": {  # What the circuit NEEDS (Netlist)
            "io": 0, "clb": 0, "dsp": 0, "bram": 0
        },
        "limits": {        # What the architecture HAS (Architecture)
            "io": 0, "clb": 0, "dsp": 0, "bram": 0
        }
    }

    if not os.path.exists(vpr_out_file):
        print(f"Error: {vpr_out_file} not found.")
        return

    with open(vpr_out_file, 'r') as f:
        content = f.read()

        # 1. Extract FPGA Size (Limits)
        # Matches: FPGA sized to 16 x 16
        size_match = re.search(r"FPGA sized to\s+([0-9]+)\s+x\s+([0-9]+)", content)
        if size_match:
            resources["fpga_size"] = [int(size_match.group(1))-2, int(size_match.group(2))-2]
        # 2. Extract Netlist Requirements & Architecture Limits
        # We look for the "Resource usage..." block and capture the counts
        # Mapping: mult_36 -> dsp, memory -> bram
        resource_map = {
            "io": "io",
            "clb": "clb",
            "mult_36": "dsp",
            "memory": "bram"
        }

        for vpr_label, key in resource_map.items():
            # Pattern logic: Find "Netlist" or "Architecture", skip a newline/tabs, find the count, then the label
            # This handles the specific structure of the VPR Resource Usage table
            req_pattern = rf"Netlist\s+([0-9]+)\s+blocks of type: {vpr_label}"
            lim_pattern = rf"Architecture\s+([0-9]+)\s+blocks of type: {vpr_label}"
            
            req_match = re.search(req_pattern, content)
            lim_match = re.search(lim_pattern, content)
            
            if req_match:
                resources["requirements"][key] = int(req_match.group(1))
            if lim_match:
                resources["limits"][key] = int(lim_match.group(1))

    # Save to resources.txt
    with open(output_file, 'w') as f:
        json.dump(resources, f, indent=4)
    
    print(f"Resources saved to {output_file}:")
    print(json.dumps(resources, indent=4))

if __name__ == "__main__":
    # Point this to your baseline directory
    extract_resources("/root/desktop/vtr_exp/runs/raygentop")