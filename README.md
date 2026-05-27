# PRISM: command-line reproduction code

This repository contains a compact command-line version of the code used for the PRISM experiments.  The layout keeps the familiar `LLM-Adapters/` data and output directories, but puts the training and evaluation entry point in one script.

## Directory layout

```text
PRISM/
  train_eval.py                 # single command-line entry point
  src/prism_cli/                # training, evaluation, and optimizer code
  LLM-Adapters/
    ft-training_set/            # Math-10K and GLUE8 instruction data
    dataset/                    # Math-10K evaluation sets
    trained_models/             # generated adapters
    experiment/                 # generated evaluation JSON/CSV files
```

No notebooks are required.

## Methods

The command line supports exactly three methods:

```text
adam    AdamW LoRA baseline
rite    LoRA-RITE baseline
prism   final PRISM implementation
```

The `prism` method uses the final horizontal-lift configuration:

```text
spectral residual initialization
rank-space adaptive update
floor_factor = 0.5
floor_mode = scalar
lift_gauge_fix = both
second_moment_debias = false
condition_strategy = raise_small
```

## Default hyperparameters

The dataset defaults match the paper-style settings:

| dataset | steps | learning rate | cutoff | train_on_inputs |
|---|---:|---:|---:|---|
| Math-10K | 300 | 3e-4 | 256 | true |
| GLUE8 | 500 | 2e-4 | 384 | false |

Shared defaults are `batch_size=64`, `micro_batch_size=4`, `seed=42`, `lora_r=16`, `lora_alpha=16`, `lora_dropout=0.05`, and target modules `q_proj,k_proj,v_proj,up_proj,down_proj`.

For DP runs, the defaults are `epsilon=6`, `delta=1e-5`, and `max_grad_norm=1.0`.  Non-private runs use the same setup with clipping/noise disabled.

## Example commands

Math-10K, PRISM, DP at epsilon 6:

```bash
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6
```

Math-10K, AdamW baseline, DP at epsilon 6:

```bash
python train_eval.py --dataset math10k --method adam --privacy dp --epsilon 6
```

GLUE8, PRISM, DP at epsilon 6:

```bash
python train_eval.py --dataset glue8 --method prism --privacy dp --epsilon 6
```

Non-private PRISM on both datasets:

```bash
python train_eval.py --dataset math10k --method prism --privacy nondp
python train_eval.py --dataset glue8   --method prism --privacy nondp
```

Rank and backbone sweeps:

```bash
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --lora_r 8
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --lora_r 32
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --base_model google/gemma-2-9b
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --base_model google/gemma-3-12b-pt
```

Repeat a run with the same seed but a separate output directory:

```bash
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --repeat_id 1
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --repeat_id 2
```

## Resume and caching

By default, completed adapters and completed evaluation files are reused.  For a clean rerun:

```bash
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --force_train --force_eval
```

During manual PRISM and DP baseline loops, `_resume_checkpoint.pt` is written periodically.  If a job is interrupted, rerun the same command to continue from the checkpoint.  Use `--no_resume` to disable this behavior.

## Outputs

Adapters are saved under:

```text
LLM-Adapters/trained_models/<run-tag>/
```

Evaluation summaries are saved under:

```text
LLM-Adapters/experiment/<run-tag>/summary.csv
LLM-Adapters/experiment/<run-tag>/details.csv
```

## Notes

Math-10K evaluation uses the included `LLM-Adapters/evaluate.py` protocol and writes per-task JSON files before summarizing exact-answer accuracy.  GLUE8 evaluation uses official GLUE validation splits and standard GLUE metrics through the Hugging Face `evaluate` package.

### Math-10K PRISM notebook-equivalence note

For `--dataset math10k --method prism`, the command-line trainer intentionally uses the same DataLoader sampler behavior as the Math-10K PRISM notebook artifact: `shuffle=True` without an explicit `torch.Generator`.  GLUE8 keeps the explicit seeded DataLoader generator used by the GLUE8 notebook artifact.  This is not a hyperparameter change; it only preserves the minibatch-order behavior of the corresponding notebook implementations.

If you previously ran an older command-line package, force a clean rerun to avoid reusing a cached low-score adapter:

```bash
python train_eval.py --dataset math10k --method prism --privacy dp --epsilon 6 --force_train --force_eval --no_resume
```
