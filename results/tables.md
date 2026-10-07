# Generated tables

*Generated 2026-09-29 02:03 UTC from `results/runs/`. Do not edit by hand — rerun the notebook instead.*

## Table 1. Experiment results

| ID | Model | RQ | Val macro-F1 | Test macro-F1 | 95% CI | Test acc. | Params |
|---|---|---|---|---|---|---|---|
| E4 | AfroXLMR base (fine-tuned) | RQ3 | 0.6526 | **0.6143** | 0.598–0.631 | 0.7569 | 278,054,414 |
| E1 | TF-IDF char 2-5 grams + LR | RQ1 | 0.6995 | **0.5870** | 0.571–0.603 | 0.7261 | 1,359,736 |
| A2 | E4 — max_length 256 | ablation | 0.6105 | **0.5750** | 0.559–0.591 | 0.7287 | 278,054,414 |
| E2b | BiLSTM (fastText embeddings) | RQ2 | 0.6707 | **0.5262** | 0.508–0.543 | 0.6911 | 22,136,618 |
| B1 | TF-IDF word 1-2 grams + LR | baseline | 0.6391 | **0.5115** | 0.494–0.528 | 0.6925 | 2,800,014 |
| E2a | BiLSTM (random embeddings) | RQ2 | 0.5730 | **0.4419** | 0.424–0.458 | 0.5907 | 22,136,618 |
| A1 | E4 — title only | ablation | 0.4494 | **0.4213** | 0.409–0.433 | 0.6439 | 278,054,414 |
| E3 | XLM-R base (fine-tuned) | RQ3 | 0.3671 | **0.3603** | 0.350–0.371 | 0.6140 | 278,054,414 |
| B0 | Majority class | floor | 0.0274 | **0.0134** | 0.012–0.015 | 0.1039 | 1 |

## Table 2. Per-class test metrics — best model (AfroXLMR base (fine-tuned))

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| politics | 0.749 | 0.566 | 0.644 | 442 |
| sport | 0.948 | 0.959 | 0.953 | 812 |
| economy | 0.578 | 0.703 | 0.634 | 222 |
| health | 0.658 | 0.753 | 0.703 | 320 |
| entertainment | 0.866 | 0.831 | 0.848 | 848 |
| history | 0.059 | 0.917 | 0.111 | 12 |
| technology | 0.671 | 0.928 | 0.779 | 152 |
| tourism | 0.736 | 0.605 | 0.664 | 152 |
| culture | 0.642 | 0.335 | 0.440 | 203 |
| fashion | 0.651 | 0.656 | 0.653 | 151 |
| religion | 0.791 | 0.881 | 0.833 | 335 |
| environment | 0.932 | 0.234 | 0.374 | 175 |
| education | 0.156 | 0.071 | 0.098 | 70 |
| relationship | 0.802 | 0.936 | 0.864 | 360 |
