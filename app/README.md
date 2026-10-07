---
title: Kinyarwanda News Topic Classifier
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
---

# Kinyarwanda News Topic Classifier

Classifies a Kinyarwanda news article into one of 14 topics: politics, sport, economy,
health, entertainment, history, technology, tourism, culture, fashion, religion,
environment, education, relationship.

The model is loaded **from the Hugging Face Hub at startup**, not stored in this Space.
Set the `MODEL_ID` environment variable (Settings → Variables) to point at it; it defaults
to `REPLACE_ME/kinyarwanda-news-classifier`.

## Running locally

```bash
pip install -r app/requirements.txt
MODEL_ID=<username>/kinyarwanda-news-classifier python app/app.py
```

## Files

| File | Purpose |
|---|---|
| `app.py` | The Gradio interface and inference code |
| `model_meta.json` | Metrics and the validation-chosen confidence threshold (generated) |
| `examples.json` | Real KINNEWS test articles, including one known failure (generated) |

`model_meta.json` and `examples.json` are written by Section 8 of the notebook, so the
numbers shown in the app trace back to an actual experiment rather than being typed here.

## How a request flows through the app

What happens between a user typing text and seeing three topics:

```
  1. User types Kinyarwanda text into the Gradio textbox
  2. Guard      empty / < 3 words  -> return a prompt, no model call
  3. Normalise  preprocess_transformer: NFC + whitespace only
                (or preprocess_classical, if the deployed model is sparse)
  4. Tokenize   text -> input_ids + attention_mask, truncated to max_length
  5. Forward    model(...) -> logits, shape [1, 14]
  6. Softmax    logits -> probabilities summing to 1
  7. Top-3      argsort, take three highest
  8. Threshold  top probability < tau (chosen on VALIDATION) -> show warning
  9. Render     Gradio Label component + the note
```

**Why there is a confidence threshold at all.** Section 7.6 of the notebook showed the model
returns a topic for English, French, Kirundi and random characters alike. A softmax over 14
classes cannot say "none of these". The threshold does not fix that — nothing in this
architecture can — but it lets the interface tell the user when the model is near the edge
of what it knows.

## Known limitations

The model cannot detect out-of-domain input: given English, French, Kirundi or random
characters it still returns a Kinyarwanda news topic. The app shows a low-confidence note,
but that is a hint rather than a guarantee. See the "About the model" tab.
