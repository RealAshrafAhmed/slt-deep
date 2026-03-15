# Single-Chain Sub-Experiment (Sub-Experiment 1)

This experiment isolates training on a **single Markov chain** and focuses on:

1. Learning the transition operator from next-token prediction.
2. Measuring how strongly predictions depend on the most recent token.
3. Visualizing convergence toward the stationary distribution at long horizons.

## Architecture Rule

- Task orchestration: mise
- Notebook execution backend: uv + papermill

## Notebook Flow

1. `01_chain.ipynb` - chain construction and stationary distribution
2. `02_dataset.ipynb` - synthetic sequence generation
3. `03_model.ipynb` - model definition
4. `04_training.ipynb` - training and checkpointing
5. `05_analysis.ipynb` - baseline model analysis
6. `06_single_chain_focus.ipynb` - recent-token concentration + limiting-distribution analyses

## Commands

Run from repository root (`slt-deep/`).

## Task Definitions

- Experiment tasks for this folder live in `projects/markov-chain-learning/experiments/single-chain/tasks.mise.toml`.
- Root task loading is configured in `slt-deep/.mise.toml` via `task_config.includes`.
- Add new experiment-specific tasks in this local `tasks.mise.toml` file, not in the root mise file.

### Smoke run

```bash
mise run mcl-sc:pipeline -- --variant full --epochs 5
```

### Full run

```bash
mise run mcl-sc:pipeline -- --variant full --epochs 50 --n-sequences 10000
```

### Focus notebook only

```bash
mise run mcl-sc:focus -- --variant full
```

## Outputs

Artifacts are written under:

- `data/single/sequences.pt`
- `data/single/checkpoint_<variant>.pt`
- `data/single/executed/02_dataset.ipynb`
- `data/single/executed/04_training_<variant>.ipynb`
- `data/single/executed/05_analysis_<variant>.ipynb`
- `data/single/executed/06_single_chain_focus_<variant>.ipynb`
