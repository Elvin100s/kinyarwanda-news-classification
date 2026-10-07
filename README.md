# Kinyarwanda News Topic Classification

Classifying Kinyarwanda news articles into 14 topics on the **KINNEWS** benchmark
(Niyongabo et al., 2020), comparing sparse linear models, a from-scratch BiLSTM, and
multilingual vs. African-centric Transformers — with a full error analysis and a deployed
Gradio app.

> **Build status: all 8 phases built; B0/B1/E1 have final results.**
> E2–E4 and the ablations are **code-complete and smoke-tested** but need a GPU — see
> `COLAB_RUNBOOK.md`. Per the project's no-fabricated-numbers rule, their numbers are shown
> as `[[TODO: Colab]]` rather than estimated.

## Links

| | |
|---|---|
| Live app (Hugging Face Space) | `[[TODO: needs an HF account — app is built and tested, see app/]]` |
| Model on the Hub | `[[TODO: needs HF_USERNAME + HF_TOKEN — push cell is written, notebook §8.2]]` |
| Video walkthrough | `[[TODO: record after the Colab runs]]` |
| Open in Colab | `[[TODO: needs the GitHub repo URL once pushed]]` |

## Problem and users

Kinyarwanda is an official language of Rwanda, spoken by an estimated 30 million people across
Rwanda, Uganda, DR Congo and Tanzania (figure cited by Niyongabo et al., 2020), but it is
under-resourced in NLP relative to that population (Joshi et al., 2020).
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
`results/tables.md`. E2–E4 and the ablations are pending (Phases 4–5).*

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

**RQ2 (fastText embeddings), RQ3 (African-centric pretraining), RQ4 (error analysis)** —
`[[TODO: Colab]]`. All code is written and smoke-tested end to end; these need ~3–4 GPU-hours.
See `COLAB_RUNBOOK.md`.

One RQ3 finding is already settled without training, from primary sources: **XLM-R never saw
Kinyarwanda** (absent from CC-100's 116 languages and from its 94-language model card) while
AfroXLMR's card lists it — and the two share an *identical* tokenizer (fertility 2.47 each).
So RQ3 tests whether adaptive fine-tuning can add a language the base model never saw, using
a vocabulary built without it. That is a sharper question than the brief anticipated.

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

### On Colab (GPU — required for E2 onward)

Open the notebook in Colab, set a **T4 GPU** runtime, set `CONFIG["fast_mode"] = False`, and
run top to bottom. Full ordered cell list, time estimates and crash-recovery steps are in
**`COLAB_RUNBOOK.md`**. Roughly 3–4 GPU-hours total.

Storage is split so nothing large touches your machine: the raw corpus, the Hugging Face
model cache and all checkpoints go to **Google Drive**; only small evidence files
(`results/`, `figures/`, `data/splits.json`) come back to the repo.

Every experiment skips itself if its result JSON already exists, so a dropped session
resumes rather than restarts.

## Running the app

The Gradio app loads the model **from the Hugging Face Hub**, so the repo stays small.

```bash
pip install -r app/requirements.txt
MODEL_ID=<username>/kinyarwanda-news-classifier python app/app.py
```

It shows the top-3 topics with probabilities, flags predictions below a threshold chosen on
**validation**, handles empty and very short input, and ships example buttons drawn from real
test articles — **including one known failure**, labelled as such. The "About the model" tab
carries the metrics table, per-class F1 and the limitations.

If the Space is down, run it locally with the command above.

## Repository structure

```
README.md  requirements.txt  references.bib  .gitignore
notebooks/   kinyarwanda_news_classification.py    (source of truth, diffable)
             kinyarwanda_news_classification.ipynb (generated, runs on Colab)
app/         app.py, requirements.txt, README.md   (the Gradio Space)
data/        splits.json, audit/                   (split indices + audit evidence)
figures/     *.png                                 (8 generated figures)
results/     runs/*.json, predictions/*.csv,       (every number traces to here)
             experiments.csv, tables.md, model_card.md
```

## Attribution

Built on **KINNEWS** (Niyongabo et al., 2020, MIT licence) and the pretrained checkpoints
`xlm-roberta-base` and `Davlan/afro-xlmr-base`, using PyTorch, Hugging Face
`transformers`/`datasets`, scikit-learn, gensim and Gradio (versions pinned in
`requirements.txt`). Everything else — preprocessing, the leakage audit, the BiLSTM, the
training and evaluation code, and the app — was written for this project. Results are
produced by code that ran; no metric here is estimated by a language model.

## Limitations

1. **Train and test distributions differ sharply.** Environment is ~19× more common in test
   than train, fashion ~11× (TV distance 0.383; 0.320 after dedup). Test scores describe a
   deliberately reweighted sample, not Kinyarwanda news in general.
2. **Single seed.** Bootstrap CIs capture test-set sampling and McNemar captures paired
   disagreement, but neither captures initialisation variance.
3. **Label noise.** 156 groups of byte-identical articles carry conflicting labels
   (1,446 rows) — a hard ceiling on achievable accuracy.
4. **Severe truncation.** 74.5% of articles exceed 512 subword tokens.
5. **No out-of-domain awareness.** English, French, Kirundi and random characters all get a
   confident Kinyarwanda news topic.
6. **Scraping artefacts.** `caption`/`alignnone` (WordPress markup) rank among the top
   features for `education`.

Full discussion and future work in the report (`report/`, Section 8).

## References

See `references.bib` — 27 entries, each verified against a Crossref DOI, the
ACL Anthology, or the model/dataset card cited. Key sources:

- Niyongabo, R. A., Hong, Q., Kreutzer, J., & Huang, L. (2020). KINNEWS and KIRNEWS. *COLING 2020*.
- Conneau, A., et al. (2020). Unsupervised cross-lingual representation learning at scale. *ACL 2020*.
- Alabi, J. O., Adelani, D. I., Mosbach, M., & Klakow, D. (2022). Adapting pre-trained language
  models to African languages via multilingual adaptive fine-tuning. *COLING 2022*.
- Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching word vectors with
  subword information. *TACL, 5*, 135–146.
- Joshi, P., et al. (2020). The state and fate of linguistic diversity and inclusion in the
  NLP world. *ACL 2020*.

**Verified directly against primary sources for this project:** Kinyarwanda (`rw`) is **absent**
from CC-100's 116 language files and from `xlm-roberta-base`'s 94-language card, but **present**
in `Davlan/afro-xlmr-base`'s card.
