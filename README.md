# Kinyarwanda News Topic Classification

Classifying Kinyarwanda news articles into 14 topics on the **KINNEWS** benchmark
(Niyongabo et al., 2020), comparing sparse linear models, a from-scratch BiLSTM, and
multilingual vs. African-centric Transformers — with a full error analysis and a deployed
Gradio app.

> **Build status: Phase 1 of 8 complete (scaffold).**
> Sections marked `[[TODO: ...]]` are filled by later phases. Per the project's
> no-fabricated-numbers rule, **no result is written here until the code that produces it
> has actually run.**

## Links

| | |
|---|---|
| Live app (Hugging Face Space) | `[[TODO: Phase 7]]` |
| Model on the Hub | `[[TODO: Phase 7]]` |
| Video walkthrough | `[[TODO: Phase 8]]` |
| Open in Colab | `[[TODO: needs the GitHub repo URL]]` |

## Problem and users

Kinyarwanda is the national language of Rwanda and is spoken by roughly 12 million people,
but it is under-resourced in NLP relative to that population (Joshi et al., 2020).
Kinyarwanda outlets publish news daily; automatic topic tagging helps organise, search and
route that content.

**Intended users:** newsroom and archive staff tagging articles, aggregators routing content
by topic, and researchers or developers who need a documented Kinyarwanda classifier.

## Research questions

| RQ | Question |
|---|---|
| RQ1 | How far can a simple sparse model go, and do character n-grams beat word n-grams? |
| RQ2 | Do subword embeddings trained on in-domain text help a BiLSTM over random init? |
| RQ3 | Does African-centric pretraining (AfroXLMR) beat general multilingual (XLM-R)? |
| RQ4 | What does the best model still get wrong, and why? |

## Dataset

**KINNEWS** (Niyongabo et al., 2020) — 21,268 Kinyarwanda news articles from 20 Rwandan
sources, 14 topics, **MIT licence**. The *raw* version is used, because Transformers need
natural text and this project documents its own preprocessing. Every figure below is
confirmed programmatically by the notebook, not quoted from the paper.

| | Published | As loaded |
|---|---|---|
| Train / test articles | 17,014 / 4,254 | 17,014 / 4,254 ✓ |
| Classes | 14 | 14 ✓ |
| Fields | `label, kin_label, en_label, url, title, content` | ✓ |

**Three findings from the audit that shape the whole project:**

1. **55% of the training file is duplicated.** Only 7,629 of 17,014 training rows carry
   distinct text; 9,379 rows repeat earlier text byte-for-byte and one article appears **80
   times**. Duplicates are removed *before* the validation split is drawn — otherwise copies
   of one article land on both sides of it and validation would measure memorisation.
   Training set: 17,014 → **7,626** rows. Imbalance falls from **117.5× to 53.2×**.
   The test set is never modified.

2. **Official train and test do not share a class distribution.** Environment is ~19× more
   common in test than in train, fashion ~11×, while economy is ~0.2×. Total variation
   distance is 0.383, and deduplication only narrows it to 0.320 — this is a real property
   of the published benchmark. A model that faithfully learns the training prior is
   *penalised* by this test set, which makes class weighting essential rather than optional.

3. **Articles are far longer than the context window.** Median 292 words / 716 XLM-R subword
   tokens; **94.9% exceed 256 tokens and 74.5% exceed 512**. Even at `max_length=512` the
   model never sees the tail of three-quarters of the corpus — which makes ablation A2 a
   central experiment rather than a footnote.

**Also recorded:** 3 training articles were verbatim copies of test articles (removed); 61
test articles (1.4%) have a near-duplicate in train (reported, not removed); 156 groups of
identical text carry conflicting labels (1,446 rows), an irreducible ceiling on accuracy.

**Morphology, quantified.** 231,516 distinct word types over 2.38M tokens, and **61.6% of
types appear exactly once**. Both tokenizers split Kinyarwanda at **2.47 subword tokens per
word**. This is the concrete evidence behind RQ1 and RQ2.

| Split | Articles | Notes |
|---|---|---|
| train | 6,863 | deduplicated, stratified 90% |
| validation | 763 | stratified 10%; all tuning and early stopping |
| test | 4,254 | official split, untouched, evaluated once per final model |

Indices are persisted to `data/splits.json`; the notebook asserts that no article text is
shared across the three splits.

## Method

Seven experiments and two ablations, all on identical splits:

| ID | Model | RQ |
|---|---|---|
| B0 | Majority class | floor |
| B1 | TF-IDF word 1–2 grams + Logistic Regression | baseline |
| E1 | TF-IDF char 2–5 grams (`char_wb`) + Logistic Regression | RQ1 |
| E2a | BiLSTM, random embeddings | RQ2 |
| E2b | BiLSTM + fastText skip-gram embeddings (trained on train split only) | RQ2 |
| E3 | XLM-R base, fine-tuned | RQ3 |
| E4 | AfroXLMR base, fine-tuned | RQ3 |
| A1 | Best Transformer, title only vs title + content | ablation |
| A2 | Best Transformer, `max_length` 256 vs 512 | ablation |

Primary metric is **macro-F1** (the classes are imbalanced and every topic matters equally
to users), reported with bootstrap 95% confidence intervals and McNemar significance tests.

## Results

*Generated from `results/runs/*.json` — see `results/experiments.csv` and
`report_materials/tables.md`. E2–E4 and the ablations are pending (Phases 4–5).*

| ID | Model | RQ | Val macro-F1 | Test macro-F1 | 95% CI | Test acc. | Params |
|---|---|---|---|---|---|---|---|
| E1 | TF-IDF char 2-5 grams + LR | RQ1 | 0.6995 | **0.5870** | 0.571–0.603 | 0.7261 | 1,359,736 |
| B1 | TF-IDF word 1-2 grams + LR | baseline | 0.6391 | **0.5109** | 0.493–0.527 | 0.6921 | 2,800,014 |
| B0 | Majority class | floor | 0.0274 | **0.0134** | 0.012–0.015 | 0.1039 | 1 |

Primary metric is **macro-F1**. B0's accuracy (10.4%) is nearly **8× its macro-F1** (1.3%),
which is the whole argument for the metric on a set this imbalanced.

## Key findings

**RQ1 — Do character n-grams beat word n-grams? Yes, decisively.** E1 reaches **0.5870**
test macro-F1 against B1's **0.5109**, a **+7.6 point** gain with **non-overlapping**
confidence intervals, using **less than half the parameters**. The gain is concentrated
almost entirely in the smallest training classes — environment (38 train articles) improves
+0.295 F1, tourism +0.234, fashion +0.201 — while the largest classes barely move. Character
n-grams share evidence across the inflected forms of one stem, which is what an agglutinative
language with **61.6% single-occurrence word types** needs most.

**RQ2, RQ3, RQ4** — `[[TODO: Phases 4–6]]`

## Reproducing this work

### Locally (CPU — classical models and smoke tests)

```bash
git clone <repo-url> && cd Summative
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# regenerate the notebook from its source of truth
jupytext --to ipynb notebooks/kinyarwanda_news_classification.py
jupyter lab notebooks/kinyarwanda_news_classification.ipynb
```

`CONFIG["fast_mode"] = True` subsamples the data and shortens training so the whole notebook
runs in ~10 minutes on CPU. **Fast-mode numbers are a smoke test and are never reported** —
result files produced that way are tagged `"fast_mode": true`.

### On Colab (GPU — required for E3/E4)

`[[TODO: Phase 5/8 — ordered cell list with measured runtimes]]`

## Running the app

`[[TODO: Phase 7]]`

## Repository structure

```
README.md  requirements.txt  .gitignore
notebooks/          kinyarwanda_news_classification.py  (source of truth)
                    kinyarwanda_news_classification.ipynb  (generated)
app/                app.py, requirements.txt, README.md
data/               splits.json, audit/
results/            runs/*.json, predictions/*.csv, experiments.csv
figures/            *.png
report_materials/   tables.md, figures_index.md, related_work_notes.md,
                    report_map.md, references.bib
```

## Attribution

Built on **KINNEWS** (Niyongabo et al., 2020, MIT licence) and the pretrained checkpoints
`xlm-roberta-base` and `Davlan/afro-xlmr-base`, using PyTorch, Hugging Face
`transformers`/`datasets`, scikit-learn, gensim and Gradio (versions pinned in
`requirements.txt`). Everything else — preprocessing, the leakage audit, the BiLSTM, the
training and evaluation code, and the app — was written for this project. Results are
produced by code that ran; no metric here is estimated by a language model.

## Limitations

`[[TODO: Phase 8]]`

## References

`[[TODO: Phase 8 — see report_materials/references.bib; every entry verified against the
paper or model card, anything unconfirmed marked [VERIFY]]]`
