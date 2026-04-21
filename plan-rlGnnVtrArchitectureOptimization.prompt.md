## Plan: RL-GNN VTR Architecture Optimization

Build a constrained RL loop that generates a fixed grid layout from algorithm-selected DSP/BRAM/CLB placements, computes exact grid width and height, evaluates candidates through the existing VTR glue flow, and learns a policy using a GNN embedding of each circuit netlist. Use a small-budget, sample-efficient strategy: warm-start with random/LHS sampling + surrogate-assisted policy updates, then fine-tune RL on true VTR rewards.

**Steps**
1. Phase 1: Define search space and hard constraints (*blocks all later steps*)
2. Lock editable knobs to a generated `layout/fixed_layout` block: `aspect_ratio`, explicit DSP/BRAM/CLB coordinates, and layout dimensions.
3. Encode actions as: continuous `aspect_ratio` plus variable-length coordinate sets for DSP and BRAM placements, plus CLB fill strategy or explicit CLB coordinates.
4. Enforce legality constraints during action decoding: no duplicates per type, deterministic ordering, legal positive x/y coordinates, respect block heights via valid row alignment, and avoid illegal overlaps/collisions between mutually exclusive tile assignments.
5. Compute exact `width` and `height` from chosen placements plus required CLB capacity under the selected `aspect_ratio`, then emit `<fixed_layout name="custom_layout" width="{calculated_width}" height="{calculated_height}">`.
6. Generate explicit placement tags inside `fixed_layout` using `<single>` and/or non-repeating `<col>` tags for arbitrary algorithm-selected coordinates. Do not emit `auto_layout`.

6. Phase 2: Build observation pipeline with GNN circuit encoder (*depends on 1; parallel with step 10*)
7. Parse circuit graph from synthesized netlist artifacts (or pre-synthesis Verilog-to-graph path), with node types (LUT/FF/adder/mult/ram/io), edge direction, fanout/fanin, and optional toggle/activity features when available.
8. Train or precompute a compact circuit embedding via a lightweight message-passing GNN (GraphSAGE/GAT with shallow depth) and cache per circuit to avoid recomputation during RL episodes.
9. Compose RL state as: [circuit embedding] + [current architecture action vector] + [recent metric history/statistics], enabling cross-circuit generalization under subset-train/holdout evaluation.

10. Phase 3: Create evaluation harness around existing flow (*depends on 1; parallel with 2*)
11. Reuse `run-vtr.py` as the execution backend; create an experiment driver that materializes candidate XML, launches VTR per training circuit, and parses metrics from `output.txt`, timing reports, and power reports.
12. Extract reward inputs per run: total power, critical path delay, and total wirelength (from VPR outputs; fall back to routed-wire proxies if explicit aggregate wirelength field is absent).
13. Add robust failure handling: unroutable/timeout/flow-error candidates receive penalty + structured failure tags for replay analysis.

14. Phase 4: Reward design and optimization strategy (*depends on 2 and 3*)
15. Use normalized multi-objective reward with feasibility-aware shaping:
16. Primary scalar reward: weighted negative normalized metrics across training circuits: power, critical path, wirelength.
17. Constraint shaping: strong penalty for failed runs, soft penalty for timing regressions beyond baseline, optional reward bonus for Pareto improvements.
18. Small-budget approach (~200 VTR runs):
19. Stage A (exploration, ~60 runs): random/LHS over constrained actions to fit metric normalizers and train a surrogate regressor.
20. Stage B (guided search, ~100 runs): policy optimization using surrogate proposals filtered by uncertainty + periodic true-eval correction.
21. Stage C (exploit, ~40 runs): RL fine-tune on true evaluations near best Pareto frontier.

22. Phase 5: Train/validation protocol with holdout circuits (*depends on 4*)
23. Split tests into train subset and holdout subset; optimize only on train reward; report final Pareto set on both train and holdout to detect overfitting.
24. Track experiment artifacts: candidate XML, seed, action vector, per-circuit metrics, scalar reward, and pass/fail status.

25. Phase 6: Deliverables and operating workflow (*depends on all prior phases*)
26. Produce a reproducible CLI pipeline: prepare embeddings, run search, evaluate top-K candidates, export best architecture XML variants.
27. Output ranked and Pareto summaries with deltas vs baseline architecture for power, delay, and wirelength.

**Relevant files**
- `/home/karthikeya/vtr_exp/run-vtr.py` — reuse as canonical VTR invocation glue and environment setup.
- `/home/karthikeya/vtr_exp/run-vtr-batch.py` — reuse patterns for iterating circuits and run directory management.
- `/home/karthikeya/vtr_exp/vtr_arch/timing/k6_N10_mem32K_40nm.xml` — baseline template to transform from `auto_layout` into generated `fixed_layout` with explicit placement tags.
- `/home/karthikeya/vtr_exp/vtr_arch/custom_grid/non_column_wide_aspect_ratio.xml` — reference for alternate layout variants.
- `/home/karthikeya/vtr_exp/vtr_arch/custom_grid/non_column_tall_aspect_ratio.xml` — reference for alternate layout variants.
- `/home/karthikeya/vtr_exp/tests/` — circuit set for train/holdout split.
- `/home/karthikeya/vtr_exp/runs/` — metric parsing targets (`*.power`, timing reports, route/place outputs).

**Verification**
1. Unit-check action decoder: generated XML contains only legal, non-repeating column coordinates with expected `<col>` expansion and valid priorities.
2. Smoke-test one episode on a single circuit: candidate XML generates, VTR runs successfully, parser extracts all three metrics.
3. Budget test: run exactly ~200 evaluations with checkpoint/resume and deterministic seeds.
4. Quality test: compare best candidate vs baseline on train subset and holdout subset for power, critical path, and wirelength.
5. Robustness test: intentionally invalid action triggers graceful penalty path without crashing the search loop.

**Decisions**
- Objective: jointly optimize power, critical path delay, and wirelength.
- Budget: small (~200 VTR evaluations), requiring sample-efficient optimization.
- Data split: subset training with holdout validation.
- Search constraints: algorithm computes exact grid dimensions and generates `fixed_layout` with explicit DSP/BRAM/CLB placement tags (`single`/`col`), no `auto_layout` emission.
- Scope excluded for now: routing transistor sizing/Fc/channel-width edits; keep optimization focused on layout-level knobs.

**Further Considerations**
1. Reward weighting recommendation: start equal after z-score normalization, then adjust weights from baseline sensitivity (Option A equal, Option B power-biased, Option C timing-constrained lexicographic).
2. RL algorithm recommendation under low budget: PPO is simple but data-hungry; prefer SAC + surrogate or Bayesian-RL hybrid for better sample efficiency.
3. Wirelength source recommendation: use direct VPR reported total wirelength when available; otherwise define a single consistent proxy and keep it fixed across all experiments.
