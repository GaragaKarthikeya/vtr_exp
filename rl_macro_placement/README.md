# RL Macro Placement Sandbox

This directory is a clean starting point for reinforcement-learning experiments on one fixed VTR benchmark and one fixed architecture.

Current target:

- Architecture: `arch/k6_frac_N10_frac_chain_depop50_mem32K_40nm.xml`
- Circuit: `tests/diffeq1.v`
- Objectives: minimize `critical path delay` and `total wirelength`

The environment is intentionally simple:

- one benchmark
- one architecture template
- one-step episodes
- layout knobs exposed as actions
- VTR is the evaluator

That makes it a good fit for a first proof-of-concept where we want to show RL can guide macro-block placement.

## Search Space

The current action vector controls:

- `aspect_ratio`
- `dsp_startx` for the `mult_36` column
- `bram_startx` for the `memory` column

These are the macro-placement knobs visible in the architecture's `<layout><auto_layout ...>` section.

## Files

- `config.py`: default experiment configuration
- `layout_space.py`: action/candidate definition and XML mutation
- `vtr_metrics.py`: VTR invocation and metric parsing
- `env.py`: RL-style environment wrapper
- `smoke_test.py`: baseline run plus one candidate evaluation

## Run the Smoke Test

```bash
source ./activate-vtr.sh
python3 -m rl_macro_placement.smoke_test
```

Artifacts are written under:

```text
rl_macro_placement/runs/
```

## Environment Behavior

Each `step()` call:

1. writes an architecture variant XML
2. runs `run-vtr.py`
3. parses `runs/.../vpr.out`
4. extracts:
   - `Final critical path delay`
   - `Total wirelength`
5. computes a reward relative to the baseline run

Reward is:

```text
cp_weight * cp_improvement
+ wire_weight * wire_improvement
- cp_regression_penalty * max(0, -cp_improvement)
```

where each improvement is normalized against the baseline:

```text
(baseline - current) / baseline
```

So:

- positive reward means improvement
- negative reward means regression
- critical-path regressions are penalized explicitly, even if wirelength improves

## Run PPO

The PPO trainer uses the VTR Python virtual environment, which already has `torch` installed.

```bash
source ./activate-vtr.sh
python3 -m rl_macro_placement.train_ppo --epochs 10 --batch-size 4
```

Trainer outputs are written under:

```text
rl_macro_placement/runs/ppo/
```

## Summarize Results

```bash
source ./activate-vtr.sh
python3 -m rl_macro_placement.report --top-k 10
```

## Notes

- Episodes are one-step on purpose. This is effectively a contextual bandit over architecture variants, which is often enough for a first architecture-search result.
- You can later extend the search space to multiple DSP/BRAM columns, priorities, or explicit fixed-layout coordinates.
