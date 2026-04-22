# VTR RL-GNN Architecture Optimization

## Project Overview
This repository contains an experimental framework for optimizing FPGA architectures using Reinforcement Learning (RL) and Graph Neural Networks (GNN), built around the **Verilog-to-Routing (VTR)** toolchain. 

The primary goal is to perform sample-efficient architectural space exploration (e.g., finding optimal aspect ratios, DSP/BRAM placements, and layout dimensions) that minimizes power consumption and critical path delay while maintaining routability.

### Main Technologies
*   **VTR (Verilog-to-Routing):** The underlying open-source academic CAD flow for FPGA research.
*   **Python 3:** The primary language for orchestration scripts, parsing tool outputs, and implementing RL algorithms (e.g., Cross-Entropy Method).
*   **XML:** Used extensively by VTR for describing FPGA architectures.
*   **Verilog:** Hardware description language used for the benchmark circuits.

## Directory Structure & Key Files
*   **`run-vtr.py`**: A wrapper script that simplifies running the VTR flow for a given architecture and circuit. It creates a temporary run directory, loads necessary environment variables, and calls VTR.
*   **`run-vtr-batch.py`**: Likely used for iterating the VTR flow over a batch of benchmark circuits.
*   **`activate-vtr.sh`**: Helper script to activate the Python virtual environment associated with the local VTR installation.
*   **`tools/simple_rl_layout_search.py`**: A reinforcement learning implementation (specifically using the Cross-Entropy Method) that searches for optimal architecture variations (aspect ratio, DSP/BRAM column X-coordinates) to minimize a combined penalty of power and critical path delay.
*   **`tools/rrg_to_gnn_data.py`**: A script to extract and convert the Routing Resource Graph (RRG) from VTR into a format suitable for Graph Neural Network training.
*   **`tests/`**: Contains the standard Verilog benchmark circuits (e.g., `diffeq1.v`, `sha.v`).
*   **`*.xml`**: Various VTR architecture definition files used as baselines for exploration.
*   **`plan-rlGnnVtrArchitectureOptimization.prompt.md`**: A detailed document outlining the phased plan for building the complete RL-GNN loop.
*   **`docs/vtr-overview.md`**: Essential reading that explains the basic VTR stages (Synthesis, Packing, Placement, Routing) and what to look for in the tool's output artifacts.

## Building and Running

### Prerequisites
The scripts expect a local installation of VTR. Paths are typically configured via a `.env` file in the root directory. Required environment variables (handled in `run-vtr.py`):
*   `VTR_ROOT`: Path to the VTR installation.
*   `VTR_FLOW_SCRIPT`: Path to the `run_vtr_flow.py` script.
*   `VTR_POWER_TECH_FILE` / `VTR_CMOS_TECH_FILE`: Path to the power/tech XML file.

Activate the environment before running tools:
```bash
source activate-vtr.sh
```

### Running a Single VTR Flow
Use the wrapper script to run a specific architecture and benchmark:
```bash
./run-vtr.py <arch.xml> <circuit.v|circuit.eblif> [output-dir] [extra args...]

# Example:
./run-vtr.py k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml tests/diffeq1.v runs/my_experiment
```
*Note: If `[output-dir]` is omitted, it defaults to a disposable `./temp` directory which is wiped on every run.*

### Running the Simple RL Layout Search
Execute the Cross-Entropy Method layout search on a specific architecture and benchmark:
```bash
python3 tools/simple_rl_layout_search.py --arch <base_arch.xml> --circuit tests/<benchmark>.v
```
This script handles mutating the architecture XML, running VTR, parsing the `*.power` files for metrics, evaluating the reward, and saving checkpoints.

## Development Conventions
*   **Python:** Scripts strictly use Python 3 with type hints (e.g., `from __future__ import annotations`, modern standard library typing).
*   **Robustness:** Wrapper scripts and RL loops are designed to gracefully handle tool failures (e.g., unroutable architectures, timeouts) by applying heavy penalties rather than crashing.
*   **Metric Parsing:** Extracting data relies heavily on parsing VTR log files (like `output.txt` or `*.power`) using regular expressions, as seen in `simple_rl_layout_search.py`.
*   **Reproducibility:** The ML experiments generate JSON checkpoints (`checkpoint.json`), CSV logs, and auto-generated matplotlib graphs to track progression accurately.
