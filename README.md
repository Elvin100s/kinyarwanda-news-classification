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

`[[TODO: Phase 2 — source, size, licence, class counts and imbalance ratio, length stats,
duplicate/leakage audit, all confirmed programmatically]]`

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

`[[TODO: Phase 3–6 — generated from results/runs/*.json, never typed by hand]]`

## Key findings

`[[TODO: Phase 8 — one per RQ, with numbers]]`

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
README.md  PRD.md  DECISIONS.md  ATTRIBUTION.md  DEFENSE_NOTES.md  VIDEO_SCRIPT.md
requirements.txt  .gitignore
notebooks/          kinyarwanda_news_classification.py  (source of truth)
                    kinyarwanda_news_classification.ipynb  (generated)
app/                app.py, requirements.txt, README.md
data/               splits.json, audit/
results/            runs/*.json, predictions/*.csv, experiments.csv
figures/            *.png
report_materials/   tables.md, figures_index.md, related_work_notes.md,
                    report_map.md, references.bib
```

- **`PRD.md`** — the full specification this repo is built against.
- **`DECISIONS.md`** — non-obvious engineering choices, each with the alternative rejected.
- **`ATTRIBUTION.md`** — what was reused vs. what was built here.

## Limitations

`[[TODO: Phase 8]]`

## References

`[[TODO: Phase 8 — see report_materials/references.bib; every entry verified against the
paper or model card, anything unconfirmed marked [VERIFY]]]`
