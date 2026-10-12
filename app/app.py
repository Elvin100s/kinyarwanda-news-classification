"""Gradio demo for the Kinyarwanda news topic classifier.

Serves E1 — TF-IDF character 2-5-grams + logistic regression — the model selected on
validation in the notebook. The notebook (Section 8.1) saves the fitted vectoriser and
classifier as one scikit-learn Pipeline in model/, together with the exact preprocessing
settings, so the app applies the same normalisation as training.

Run locally:  python app/app.py
"""

from __future__ import annotations

import json
import re
import time
import unicodedata
from pathlib import Path

import gradio as gr
import joblib
import numpy as np

HERE = Path(__file__).parent
MIN_WORDS = 3

# The Space runs on ZeroGPU hardware (the free tier for this account), which expects at least
# one @spaces.GPU function at startup. E1 runs on the CPU, so this placeholder is never called
# and uses no GPU quota; outside Hugging Face, `spaces` is absent and this block is skipped.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_placeholder():
        return None
except ImportError:
    pass

# ---------------------------------------------------------------------------------------
# Model and preprocessing, both written by the notebook
# ---------------------------------------------------------------------------------------
PRE = json.loads((HERE / "model" / "preprocess.json").read_text())
PIPELINE = joblib.load(HERE / "model" / PRE["model_file"])
LABELS = PRE["labels"]

_meta = HERE / "model_meta.json"
META = json.loads(_meta.read_text()) if _meta.exists() else {}
# Chosen on the VALIDATION split (notebook Section 7.7); 0.50 if the file is absent.
THRESHOLD = float(META.get("threshold", 0.50))

URL_RE = re.compile(PRE["url_re"])
EMAIL_RE = re.compile(PRE["email_re"])
NUM_RE = re.compile(PRE["num_re"])
KEEP_RE = re.compile(PRE["keep_re"])


def preprocess(text: str) -> str:
    """Identical to preprocess_classical in the notebook (Section 1.6)."""
    text = unicodedata.normalize(PRE["unicode_normalisation"], str(text)).lower()
    text = URL_RE.sub(f" {PRE['url_token']} ", text)
    text = EMAIL_RE.sub(f" {PRE['email_token']} ", text)
    text = NUM_RE.sub(f" {PRE['num_token']} ", text)
    text = KEEP_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def classify(text: str):
    """Return (top-3 label -> probability, status message)."""
    raw = (text or "").strip()
    if not raw:
        return {}, "Enter a Kinyarwanda news article or headline to classify."
    n_words = len(raw.split())
    if n_words < MIN_WORDS:
        return {}, (f"Only {n_words} word(s). Give at least {MIN_WORDS} for a meaningful "
                    "prediction — a headline is usually enough.")

    started = time.time()
    probs = PIPELINE.predict_proba([preprocess(raw)])[0]
    elapsed = (time.time() - started) * 1000

    order = np.argsort(-probs)[:3]
    classes = PIPELINE.classes_
    result = {LABELS[int(classes[i])]: float(probs[i]) for i in order}
    best = float(probs[order[0]])

    note = f"Predicted in {elapsed:.0f} ms from {n_words} words."
    if best < THRESHOLD:
        note = (f"**Low confidence ({best:.0%}).** This text may be outside what the model "
                f"was trained on — it only knows Kinyarwanda *news*, and it still names a "
                f"topic for anything it is given.\n\n{note}")
    return result, note


# ---------------------------------------------------------------------------------------
# Examples — real KINNEWS test headlines (written by the notebook), incl. a known failure
# ---------------------------------------------------------------------------------------
_ex = HERE / "examples.json"
EXAMPLES = json.loads(_ex.read_text()) if _ex.exists() else []
EXAMPLE_TEXTS = [[e["text"]] for e in EXAMPLES]
EXAMPLE_LABELS = [e["label"] for e in EXAMPLES]

ABOUT = f"""
## About this model

**E1 — TF-IDF character n-grams + logistic regression**, trained on **KINNEWS**
(Niyongabo et al., 2020): Kinyarwanda news from fifteen news websites and five newspapers,
14 topics, MIT licence. It was selected because it had the best validation macro-F1 of the
nine models compared in the project.

{META.get('about_table', '')}

### How it works

Your text is normalised (Unicode NFC, lower-case, numbers and links replaced by placeholder
tokens, punctuation removed except apostrophes) exactly as in training. It is then split into
overlapping character pieces of 2–5 letters within each word, weighted by TF-IDF, and a
logistic regression scores all 14 topics. The three most likely are shown.

Character pieces suit Kinyarwanda because one word stem appears in many forms
(*umukino*, *imikino*, *abakinnyi*); the pieces they share let the model recognise the stem in
all of them.

### Limitations you should know about

- **It cannot tell when it is out of its depth.** Given English, French, Kirundi or random
  characters, it still returns a Kinyarwanda news topic. The low-confidence note (below
  {THRESHOLD:.0%}, chosen on validation data) is a hint, not a guarantee.
- **Rare topics are much weaker.** Some topics had only about 30 training articles after
  removing duplicates; see the per-class table.
- **Topics overlap.** Politics vs history, fashion vs entertainment and education vs
  relationship are the most common confusions, and some source labels are inconsistent.
- **It ignores word order** — it relies on which word pieces appear, not on sentence structure.
- **Not for consequential decisions** about people, content moderation, or language detection.

Built for an academic project. The code, experiments and error analysis are in the project
repository.
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
        if EXAMPLE_TEXTS:
            gr.Examples(examples=EXAMPLE_TEXTS, inputs=inp, example_labels=EXAMPLE_LABELS,
                        label="Examples from the test set (includes a known failure)")
        btn.click(classify, inputs=inp, outputs=[out, msg])
        inp.submit(classify, inputs=inp, outputs=[out, msg])
    with gr.Tab("About the model"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.launch()
