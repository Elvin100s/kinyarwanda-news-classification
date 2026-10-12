# Kinyarwanda News Topic Classification

Classifying Kinyarwanda news articles into 14 topics on the **KINNEWS** benchmark
(Niyongabo et al., 2020), comparing sparse linear models, a BiLSTM written from scratch, and
general versus African-centric Transformers — with a data audit, seed-variance study, error
analysis and a Gradio web app.

## Links

| | |
|---|---|
| Live app | `[[TODO: deployed web application URL]]` |
| Demo video | `[[TODO: video URL]]` |
| Report | [`report/Kinyarwanda_News_Classification_Report.docx`](report/) (PDF preview alongside) |
| Final executed notebook | [`notebooks/kinyarwanda_news_classification_executed_kaggle.ipynb`](notebooks/) |

## Problem and users

Kinyarwanda is an official language of Rwanda, spoken by an estimated 30 million people
across Rwanda, Uganda, DR Congo and Tanzania (figure cited by Niyongabo et al., 2020), but it
is under-resourced in NLP (Joshi et al., 2020). Rwandan outlets publish news in Kinyarwanda
daily; automatic topic tagging helps organise, search and route that content.

**Intended users:** newsroom and archive staff tagging articles, aggregators routing content
by topic, and researchers or developers who need a documented Kinyarwanda classifier.

## Research questions

| RQ | Question |
|---|---|
| RQ1 | How far can a sparse linear model go, and do character n-grams beat word n-grams? |
| RQ2 | Do in-domain fastText embeddings help a BiLSTM over random initialisation? |
| RQ3 | Does African-centric pretraining (AfroXLMR) beat general multilingual pretraining (XLM-R)? |
| RQ4 | What does the best model still get wrong, and why? |

## Dataset

**KINNEWS** (Niyongabo et al., 2020): Kinyarwanda news from fifteen news websites and five
newspapers, 14 topics, **MIT licence**. The raw version is used; the publishers' 48 original
categories were merged into 14 by the dataset's annotators. All published counts are
confirmed by the notebook: 17,014 training and 4,254 test articles.

**What the audit found:**

1. **55.2% of the training file is duplicated** (7,629 distinct texts in 17,014 rows; one
   article appears 80 times). Duplicates are removed *before* the validation split, so copies
   cannot sit on both sides of it. Training set: 17,014 → **7,626** articles.
2. **Label noise:** 156 groups of identical articles carry conflicting labels (1,446 rows).
3. **Train/test shift:** the official splits have different topic mixes — environment is ~19×
   more frequent in test than in the raw training set (~8× after deduplication).
4. **Long articles:** 74.5% exceed 512 subword tokens; 61.6% of word types occur only once.

| Split | Articles | Use |
|---|---|---|
| train | 6,863 | deduplicated, stratified 90% |
| validation | 763 | all tuning, early stopping, model selection |
| test | 4,254 | official split, untouched |

## Method

| ID | Model | RQ |
|---|---|---|
| B0 | Majority class | floor |
| B1 | TF-IDF word 1–2 grams + logistic regression | baseline |
| E1 | TF-IDF character 2–5 grams (`char_wb`) + logistic regression | RQ1 |
| E2a | BiLSTM, random embeddings | RQ2 |
| E2b | BiLSTM, fastText embeddings trained on the training split only | RQ2 |
| E3 | XLM-R base, fine-tuned (class-weighted loss) | RQ3 |
| E4 | AfroXLMR base, fine-tuned (class-weighted loss) | RQ3 |
| A1 | E4, title only | ablation |
| A2 | E4, `max_length` 256 | ablation |

Every trained model uses balanced class weights. The primary metric is **macro-F1**, with 95%
bootstrap confidence intervals and McNemar's test; E2a, E2b, E3 and E4 are also re-trained with
three seeds. **The final model is selected on validation macro-F1, never on test.**

## Results

From the final Kaggle run (Version 7, one T4 GPU, 1 h 42 min) — `results/experiments.csv`,
`results/runs/*.json`.

| ID | Model | Val macro-F1 | Test macro-F1 | 95% CI | Test acc. | Params |
|---|---|---|---|---|---|---|
| **E1** | TF-IDF char 2–5 grams + LR | **0.699** | 0.587 | 0.571–0.603 | 0.726 | 1.36 M |
| E2b | BiLSTM + fastText | 0.671 | 0.526 | 0.508–0.543 | 0.691 | 22.1 M |
| E4 | AfroXLMR base | 0.653 | **0.614** | 0.598–0.631 | 0.757 | 278 M |
| B1 | TF-IDF word 1–2 grams + LR | 0.639 | 0.512 | 0.494–0.528 | 0.693 | 2.80 M |
| A2 | E4, max_length 256 | 0.610 | 0.575 | 0.559–0.591 | 0.729 | 278 M |
| E2a | BiLSTM, random | 0.573 | 0.442 | 0.424–0.458 | 0.591 | 22.1 M |
| A1 | E4, title only | 0.449 | 0.421 | 0.409–0.433 | 0.644 | 278 M |
| E3 | XLM-R base | 0.367 | 0.360 | 0.350–0.371 | 0.614 | 278 M |
| B0 | Majority class | 0.027 | 0.013 | 0.012–0.015 | 0.104 | 1 |

**Seed variance** (test macro-F1, seeds 42/43/44 — `results/seed_variance.json`):
E2a 0.425 ± 0.016 · E2b 0.506 ± 0.018 · E3 0.298 ± 0.128 · E4 0.616 ± 0.003.
E2b beats E2a and E4 beats E3 on every seed.

## Key findings

- **RQ1 — yes.** Character n-grams beat word n-grams by **+7.6** macro-F1 with half the
  parameters (McNemar p = 2.8 × 10⁻¹²); the gain is concentrated in the smallest topics
  (environment +0.30, tourism +0.23).
- **RQ2 — yes.** fastText initialisation improves the BiLSTM by **+8.4** (p = 2.7 × 10⁻⁴⁸), on all
  three seeds — but the BiLSTM still trails E1 and overfits on <7,000 articles.
- **RQ3 — yes, strongly.** AfroXLMR beats XLM-R by **+25.4** (p = 1.5 × 10⁻⁸⁰) with an identical
  tokenizer; XLM-R, which never saw Kinyarwanda, scores zero F1 on tourism, fashion, culture
  and environment and is unstable across seeds.
- **Selection vs. shift.** E1 is selected on validation, but AfroXLMR scores higher on the
  shifted test set (0.614 vs 0.587): E1 loses 11.3 points from validation to test, AfroXLMR
  only 3.8. The selection was not changed after seeing test results.
- **RQ4.** E1's errors come mainly from overlapping topics (politics/culture → history,
  education → relationship), classes with ~30 training articles, and label noise — its most
  confident mistake is a football draw labelled *entertainment*.

## Reproducing this work

The notebook source of truth is `notebooks/kinyarwanda_news_classification.py` (jupytext);
`notebooks/kinyarwanda_news_classification.ipynb` is generated from it.

**On Kaggle or Colab (GPU, full run):** open the notebook, choose a T4 GPU (and Internet on, on
Kaggle), set `CONFIG["fast_mode"] = False` in Section 0.2, and run all cells — about 1 h 45 min.
`notebooks/kinyarwanda_news_classification_colab.ipynb` is a ready-made Colab copy that saves to
Google Drive so a dropped session resumes. Every experiment skips itself if its result JSON
exists.

**Locally (CPU smoke test):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
jupytext --to ipynb notebooks/kinyarwanda_news_classification.py
jupyter lab notebooks/kinyarwanda_news_classification.ipynb
```

With `fast_mode = True` (the default) the notebook subsamples the data and runs in about ten
minutes on CPU; its results go to `results/runs/_fast/` and are never reported.

## The web app

`app/` is a Gradio app serving **E1**, the model selected on validation: paste a Kinyarwanda
article or headline and get the top three topics with probabilities, plus a warning when the
top probability is below the threshold chosen on validation (0.46). It ships example inputs
from real test headlines, including a known failure and an English out-of-domain example.

The app loads `app/model/e1_pipeline.joblib` (TF-IDF vectoriser + logistic regression, 10.8 MB)
and applies the training preprocessing from `app/model/preprocess.json`. Re-scoring the full
official test set through the app's own code reproduces the notebook's E1 result exactly
(macro-F1 0.5870, accuracy 0.7261). It needs no GPU.

```bash
pip install -r app/requirements.txt   # Python 3.12+
python app/app.py
```

## Repository structure

```
README.md  requirements.txt  references.bib
notebooks/  kinyarwanda_news_classification.py              source of truth (jupytext)
            kinyarwanda_news_classification.ipynb           generated
            kinyarwanda_news_classification_colab.ipynb     Colab full-run copy
            kinyarwanda_news_classification_executed_kaggle.ipynb   final run, with outputs
data/       splits.json, audit/                              split indices, audit evidence
figures/    01–08 *.png                                      figures from the final run
results/    runs/*.json (+ runs/seeds/), experiments.csv, tables.md, seed_variance.json,
            significance_tests.json, confidence_threshold.json, latency.json, model_card.md
app/        app.py, requirements.txt, examples.json, model_meta.json,
            model/e1_pipeline.joblib, model/preprocess.json
report/     report (.docx + PDF preview) and the script that builds it
docs/       notebook_explanations.md — full rationale for every step
scripts/    summarise_manual_review.py — summarises the manual label review
```

Per-article prediction files are written to `results/predictions/` by the notebook; they are
part of the Kaggle run's output.

## Attribution

Built on **KINNEWS** (Niyongabo et al., 2020, MIT licence) and the pretrained checkpoints
`xlm-roberta-base` and `Davlan/afro-xlmr-base`, using PyTorch, Hugging Face Transformers,
scikit-learn, gensim and Gradio (versions pinned in `requirements.txt`). The data audit,
preprocessing, BiLSTM, fastText training, class-weighted fine-tuning, evaluation, seed-variance
study, error analysis and app were written for this project.

## Limitations

1. **Train/test shift** — test scores describe a reweighted sample, and a validation set drawn
   from training under-rewarded the most shift-robust model.
2. **Label noise** — conflicting labels on identical articles cap achievable accuracy.
3. **Limited tuning and seeds** — three seeds for neural models; Transformers used one learning
   rate and three epochs (XLM-R was unstable).
4. **Truncation** — 74.5% of articles exceed 512 subword tokens.
5. **No out-of-domain detection** — any input gets a topic; the app warns on low confidence.
6. **Scraping artefacts** — WordPress markup and mis-encoded apostrophes appear among features.

Full discussion and future work: report, Section 8.

## References

See `references.bib` (27 entries) and the report's reference list. Key sources:

- Niyongabo, R. A., Hong, Q., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS. *COLING 2020*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language
  models to African languages via multilingual adaptive fine-tuning. *COLING 2022*.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with
  subword information. *TACL, 5*, 135–146.
- Nzeyimana, A., & Niyongabo Rubungo, A. (2022). KinyaBERT: A morphology-aware Kinyarwanda
  language model. *ACL 2022*.
- Joshi, P., et al. (2020). The state and fate of linguistic diversity and inclusion in the NLP
  world. *ACL 2020*.
