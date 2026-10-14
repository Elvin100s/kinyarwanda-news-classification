# Generated tables

*Generated 2026-10-14 23:46 UTC from `results/runs/`. Do not edit by hand — rerun the notebook instead.*

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

## Table 2. Per-class test metrics — selected model (TF-IDF char 2-5 grams + LR)

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| politics | 0.699 | 0.584 | 0.636 | 442 |
| sport | 0.955 | 0.925 | 0.940 | 812 |
| economy | 0.488 | 0.662 | 0.562 | 222 |
| health | 0.619 | 0.741 | 0.674 | 320 |
| entertainment | 0.828 | 0.824 | 0.826 | 848 |
| history | 0.048 | 0.833 | 0.091 | 12 |
| technology | 0.539 | 0.822 | 0.651 | 152 |
| tourism | 0.827 | 0.533 | 0.648 | 152 |
| culture | 0.656 | 0.300 | 0.412 | 203 |
| fashion | 0.852 | 0.497 | 0.628 | 151 |
| religion | 0.835 | 0.740 | 0.785 | 335 |
| environment | 0.927 | 0.291 | 0.443 | 175 |
| education | 0.179 | 0.071 | 0.102 | 70 |
| relationship | 0.722 | 0.947 | 0.820 | 360 |
