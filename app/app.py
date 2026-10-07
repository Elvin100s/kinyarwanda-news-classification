"""Gradio demo for the Kinyarwanda news topic classifier.

Loads the model from the Hugging Face Hub (not from a file in this repo), so the Space
stays small and the model can be updated independently of the app.

Run locally:  python app/app.py
"""

from __future__ import annotations

import json
import os
import re
import time
import unicodedata
from pathlib import Path

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
MODEL_ID = os.environ.get("MODEL_ID", "REPLACE_ME/kinyarwanda-news-classifier")
MAX_LENGTH = 512
MIN_WORDS = 3

# Chosen on the VALIDATION split (see notebook Section 9.7), never on test. Falls back to
# 0.50 if the metadata file is absent.
_meta = Path(__file__).parent / "model_meta.json"
META = json.loads(_meta.read_text()) if _meta.exists() else {}
THRESHOLD = float(META.get("threshold", 0.50))

LABELS = ["politics", "sport", "economy", "health", "entertainment", "history", "technology",
          "tourism", "culture", "fashion", "religion", "environment", "education", "relationship"]

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
print(f"loading {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()
ID2LABEL = [model.config.id2label[i] for i in range(model.config.num_labels)] \
    if getattr(model.config, "id2label", None) else LABELS
print(f"loaded. {model.config.num_labels} classes, CPU inference.")


def preprocess(text: str) -> str:
    """Match the notebook's transformer preprocessing exactly: NFC + whitespace only."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


def classify(text: str):
    """Return (top-3 label->probability, status message)."""
    text = preprocess(text or "")
    if not text:
        return {}, "Enter a Kinyarwanda news article or headline to classify."
    if len(text.split()) < MIN_WORDS:
        return {}, (f"Only {len(text.split())} word(s). Give at least {MIN_WORDS} for a "
                    "meaningful prediction — a headline is usually enough.")

    started = time.time()
    with torch.no_grad():
        enc = tokenizer(text, truncation=True, max_length=MAX_LENGTH, return_tensors="pt")
        probs = torch.softmax(model(**enc).logits, dim=1)[0]
    elapsed = (time.time() - started) * 1000

    top = torch.topk(probs, k=min(3, len(probs)))
    result = {ID2LABEL[i]: float(p) for p, i in zip(top.values, top.indices)}
    best = float(top.values[0])

    n_tokens = int(enc["input_ids"].shape[1])
    note = f"Predicted in {elapsed:.0f} ms from {n_tokens} subword tokens."
    if n_tokens >= MAX_LENGTH:
        note += f" Input was truncated at {MAX_LENGTH} tokens."
    if best < THRESHOLD:
        note = (f"**Low confidence ({best:.0%}).** This text may be outside what the model "
                f"was trained on — it only knows Kinyarwanda *news* articles, and it will "
                f"still name a topic for anything it is given.\n\n{note}")
    return result, note


# --------------------------------------------------------------------------------------
# Examples — real KINNEWS test articles, including a documented failure
# --------------------------------------------------------------------------------------
_ex = Path(__file__).parent / "examples.json"
EXAMPLES = json.loads(_ex.read_text()) if _ex.exists() else []
EXAMPLE_TEXTS = [[e["text"]] for e in EXAMPLES] or [
    ["Ikipe ya Rayon Sports yatsinze umukino wayo wa nyuma mu gikombe cy'amahoro."],
    ["Minisiteri y'ubuzima yatangaje gahunda nshya yo kurwanya indwara z'ibyorezo."],
]

ABOUT = f"""
## About this model

Fine-tuned on **KINNEWS** (Niyongabo et al., 2020) — Kinyarwanda news articles from 20
Rwandan sources, 14 topics, MIT licence.

{META.get('about_table', '_Metrics are filled in from the notebook run._')}

### How it works

Your text → SentencePiece tokenizer → Transformer encoder → `<s>` representation →
linear layer → softmax over 14 topics. The three highest are shown.

### Limitations you should know about

- **It cannot tell when it is out of its depth.** Given English, French, Kirundi or random
  characters, it still returns a Kinyarwanda news topic, sometimes confidently. The
  low-confidence note (below {THRESHOLD:.0%}) is a hint, not a guarantee.
- **Rare topics are much weaker.** Some classes had fewer than 40 training articles, and the
  official test set over-represents exactly those. See the per-class table.
- **Long articles are truncated** at {MAX_LENGTH} subword tokens; most KINNEWS articles are
  longer than that, so the tail is unseen.
- **Not for consequential decisions** about people, moderation, or language detection.

Built for an academic assignment. Full code, experiments and error analysis:
see the linked repository.
"""

with gr.Blocks(title="Kinyarwanda News Topic Classifier") as demo:
    gr.Markdown("# Kinyarwanda News Topic Classifier\n"
                "Paste a Kinyarwanda news article or headline and get the three most likely "
                "topics out of 14.")
    with gr.Tab("Classify"):
        with gr.Row():
            with gr.Column(scale=3):
                inp = gr.Textbox(lines=10, label="Kinyarwanda news text",
                                 placeholder="Andika inkuru cyangwa umutwe w'inkuru hano...")
                btn = gr.Button("Classify", variant="primary")
            with gr.Column(scale=2):
                out = gr.Label(num_top_classes=3, label="Top 3 topics")
                msg = gr.Markdown()
        gr.Examples(examples=EXAMPLE_TEXTS, inputs=inp,
                    label="Examples (the last is a known failure case)")
        btn.click(classify, inputs=inp, outputs=[out, msg])
        inp.submit(classify, inputs=inp, outputs=[out, msg])
    with gr.Tab("About the model"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
