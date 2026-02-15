# CE-RJSSP Experiment Framework

This repository implements a constraint-enhanced re-entrant semiconductor scheduling benchmark and a lightweight **G4DQN-style** reinforcement learning scheduler.

## Implemented capabilities
- CE-RJSSP instance generation with re-entry, flexible machine assignment, setup times, maintenance windows, breakdowns, batch-machine behavior, and due dates.
- Graph-style MDP state (operation + machine features), operation-machine action space, and feasibility masking.
- Multi-objective shaping reward over makespan, tardiness, energy, and WIP.
- Baseline suite: FIFO, SPT, LPT, MWKR, ATC, GA, CP-SAT proxy.
- RL agent with graph encoder, dueling double-Q style learning, prioritized replay, candidate action set, and imitation warm start.
- Experiment runner and SVG plotting for training curves and metric comparison.
- Ablation runner for key modules.

## Run full experiment
```bash
python experiments/run_experiment.py --episodes 40 --out outputs/main
```

Outputs:
- `summary.json`
- `training_curve.svg`
- `compare_makespan.svg`
- `compare_tardiness.svg`
- `compare_energy.svg`
- `compare_wip.svg`

## Run ablation
```bash
python experiments/run_ablation.py
```

Outputs:
- `outputs/ablation/ablation_summary.json`
- `outputs/ablation/ablation_makespan.svg`
