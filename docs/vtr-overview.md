# VTR Overview

This repository wraps the VTR flow in two small helper scripts:

- `activate-vtr.sh` activates the Python virtual environment used by VTR.
- `run-vtr.py` runs the actual VTR flow on a design and architecture.

VTR stands for Verilog-to-Routing. It takes a Verilog design and an FPGA architecture description, then turns the design into a routed implementation that fits that architecture.

## Main Stages

The flow usually looks like this:

1. **Input setup**
   - Read the architecture XML.
   - Read the Verilog or EBLIF design.
   - Create a working directory for the run.

2. **Synthesis**
   - Convert Verilog into a lower-level logic netlist.
   - Tools such as Yosys and ABC help here.
   - The output is usually a BLIF-style netlist.

3. **Packing**
   - Group logic blocks into FPGA logic clusters.
   - This step decides how the synthesized logic fits into the FPGA primitives.

4. **Placement**
   - Assign packed logic blocks to physical locations on the FPGA.
   - This is where the design gets a concrete chip layout.

5. **Routing**
   - Connect the placed blocks using the FPGA routing network.
   - This produces the final routed design.

6. **Analysis and reporting**
   - Measure timing, utilization, critical path, and memory/runtime.
   - Write reports and logs for debugging and performance checking.

## What `temp/` Usually Contains

The `temp/` folder is the working directory for a run. It typically contains:

### Input copies and run metadata

- The architecture XML used for the run.
- The input Verilog or EBLIF design.
- Small metadata files such as `arch.info` and `output.txt`.

### Synthesis artifacts

- Intermediate and final `.blif` files.
- Logs from Yosys, ABC, and synthesis helper scripts.
- Files like `synthesis.tcl`, `abc.rc`, and `abc.history`.

### VPR artifacts

- `*.net` for the packed netlist.
- `*.place` for placement.
- `*.route` for routing.
- Timing and utilization reports in `*.rpt` files.

### Logs

- `vpr.out`, `vpr_stdout.log`, and other `*.out` files capture tool output.
- These are the first files to inspect when something goes wrong.

## How To Read The Folder

If you are trying to understand one run, a good order is:

1. Start with `temp/output.txt` for the summary.
2. Check `temp/vpr.out` for the main VPR log.
3. Check `temp/parmys.out` and `temp/synthesis.tcl` for synthesis details.
4. Inspect `temp/diffeq1.pre-vpr.blif`, `temp/diffeq1.net`, `temp/diffeq1.place`, and `temp/diffeq1.route` if you want the actual design artifacts.

## Important Detail About This Repo

`run-vtr.py` deletes `./temp` before each run, so the folder is meant to be disposable and recreated every time.

That means:

- Do not treat `temp/` as a permanent archive.
- Copy out any report or artifact you want to keep.
- If you want separate runs preserved, create a run-specific output directory instead of reusing `temp/`.
