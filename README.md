# PRISM-DP-LoRA

Code for **PRISM: Gauge-Invariant Tangent-Space Differentially Private LoRA**.

This repository provides training and evaluation scripts for PRISM and baseline LoRA fine-tuning on GLUE8 and Math-10K.

## Setup

```bash
pip install -r requirements.txt
```

The project follows the `LLM-Adapters` directory layout:

```text
LLM-Adapters/
  ft-training_set/
  dataset/
  trained_models/
  experiment/
```

## Usage

### PRISM on Math-10K

```bash
python train_eval.py \
  --dataset math10k \
  --method prism \
  --privacy dp \
  --epsilon 6
```

### PRISM on GLUE8

```bash
python train_eval.py \
  --dataset glue8 \
  --method prism \
  --privacy dp \
  --epsilon 6
```

### AdamW baseline

```bash
python train_eval.py \
  --dataset math10k \
  --method adam \
  --privacy dp \
  --epsilon 6
```

For non-private training:

```bash
python train_eval.py \
  --dataset math10k \
  --method prism \
  --privacy nondp
```

## Common options

```text
--dataset {math10k,glue8}
--method {prism,adam}
--privacy {dp,nondp}
--epsilon 6
--lora_r 16
--base_model google/gemma-3-4b-pt
--seed 42
```

## Outputs

Trained adapters are saved to:

```text
LLM-Adapters/trained_models/
```

Evaluation results are saved to:

```text
LLM-Adapters/experiment/
```

## Acknowledgements

This repository builds on the directory structure and parts of the training/evaluation pipeline from [LLM-Adapters](https://github.com/AGI-Edgerunners/LLM-Adapters). We thank the LLM-Adapters authors for releasing their code and datasets.

Portions adapted from LLM-Adapters are distributed under the Apache-2.0 license; see `licenses/Apache-2.0.txt`.

