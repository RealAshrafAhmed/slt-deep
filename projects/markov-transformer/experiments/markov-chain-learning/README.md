# Markov Chain Learning Experiment

This experiment is orchestrated with mise tasks and executed with uv + papermill.

## Architecture Rule

- Task orchestration: mise
- Notebook execution backend: uv + papermill

## Commands

Run from repository root.

### Inspect available tasks

```bash
mise tasks
```

### Smoke run (single regime, full model)

```bash
mise run mcl:pipeline -- --regime single --variant full --epochs 5
```

### One training notebook

```bash
mise run mcl:training -- --regime single --variant no_token --epochs 25
```

### One analysis notebook

```bash
mise run mcl:analysis -- --regime single --variant full
```

### Full matrix

```bash
mise run mcl:matrix
```

### Restricted matrix

```bash
mise run mcl:matrix -- --regimes single,two_far --variants full,no_pos --epochs 20
```

## Task Arguments

Preferred usage is passing flags after `--`:

- `--regime`: `single | two_far | two_close`
- `--variant`: `full | no_token | no_pos | no_embeddings`
- `--dataset-seed`: default `0`
- `--training-seed`: default `42`
- `--n-sequences`: default `10000`
- `--max-seq-len`: default `32`
- `--epochs`: default `50`
- `--lr`: default `0.003`
- `--batch-size`: default `256`
- `--regimes`: CSV list for matrix runs
- `--variants`: CSV list for matrix runs

## Environment Variable Fallback

Tasks still accept environment variables for CI/scripted usage.

- `REGIME`: `single | two_far | two_close`
- `VARIANT`: `full | no_token | no_pos | no_embeddings`
- `DATASET_SEED`: default `0`
- `TRAINING_SEED`: default `42`
- `N_SEQUENCES`: default `10000`
- `MAX_SEQ_LEN`: default `32`
- `EPOCHS`: default `50`
- `LR`: default `0.003`
- `BATCH_SIZE`: default `256`
- `REGIMES`: CSV list for matrix runs
- `VARIANTS`: CSV list for matrix runs

## Outputs

Artifacts are written under:

- `data/<regime>/sequences.pt`
- `data/<regime>/checkpoint_<variant>.pt`
- `data/<regime>/executed/02_dataset.ipynb`
- `data/<regime>/executed/04_training_<variant>.ipynb`
- `data/<regime>/executed/05_analysis_<variant>.ipynb`
