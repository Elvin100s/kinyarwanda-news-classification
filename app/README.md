---
title: Kinyarwanda News Topic Classifier
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
python_version: "3.12"
app_file: app.py
pinned: false
license: mit
---

# Kinyarwanda News Topic Classifier

Classifies a Kinyarwanda news article or headline into one of 14 topics: politics, sport,
economy, health, entertainment, history, technology, tourism, culture, fashion, religion,
environment, education, relationship.

The model is **E1 — TF-IDF character 2–5-grams + logistic regression**, selected on validation
macro-F1 among the nine models compared in the project (test macro-F1 0.587, accuracy 72.6%).
It is small (10.8 MB) and runs on a CPU, so the Space needs no GPU. The Space uses the free
ZeroGPU tier only because the free CPU tier was not available to this account; `app.py`
declares the one `@spaces.GPU` function that tier requires but never calls it.

## Running locally

```bash
pip install -r app/requirements.txt
python app/app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | The Gradio interface and inference code |
| `model/e1_pipeline.joblib` | Fitted TF-IDF vectoriser + logistic regression (one scikit-learn `Pipeline`) |
| `model/preprocess.json` | The exact training preprocessing settings, applied to every input |
| `model_meta.json` | Test metrics and the validation-chosen confidence threshold |
| `examples.json` | Real KINNEWS test headlines, including a known failure and an English example |

All of these except `app.py` are written by Section 8 of the notebook, so the numbers shown in
the app trace back to the experiment rather than being typed here.

## How a request flows through the app

```
  1. User types Kinyarwanda text into the textbox
  2. Guard      empty / fewer than 3 words -> return a prompt, no model call
  3. Normalise  NFC, lower-case, URLs / e-mails / numbers -> placeholder tokens,
                punctuation removed except apostrophes   (same as training)
  4. Features   TF-IDF over character 2-5-grams within word boundaries
  5. Classify   logistic regression -> probability for each of the 14 topics
  6. Top-3      the three most likely topics with their probabilities
  7. Threshold  top probability < 0.46 (chosen on VALIDATION) -> low-confidence warning
```

**Why there is a confidence threshold at all.** The notebook's out-of-domain probes (Section
7.6) showed the model returns a topic for English, French, Kirundi and random characters
alike. A classifier over 14 topics cannot say "none of these"; the threshold lets the
interface tell the user when the model is near the edge of what it knows.

## Known limitations

The model cannot detect out-of-domain input, rare topics (about 30 training articles each)
are much weaker, some topics overlap, and it ignores word order. See the "About the model" tab.
