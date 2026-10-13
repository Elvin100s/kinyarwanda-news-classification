"""Gradio demo for the Kinyarwanda news topic classifier.

Serves E1 — TF-IDF character 2-5-grams + logistic regression — the model selected on
validation in the notebook. The notebook (Section 8.1) saves the fitted vectoriser and
classifier as one scikit-learn Pipeline in model/, together with the exact preprocessing
settings, so the app applies the same normalisation as training.

Because E1 is linear, every prediction can be explained exactly: a topic's score is the sum
of (TF-IDF weight x coefficient) over the character n-grams in the text. The "Why this topic?"
panel shows that sum broken down by word and by n-gram.

Run locally:  python app/app.py
"""

from __future__ import annotations

import html
import json
import re
import time
import unicodedata
from collections import Counter
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

# Kinyarwanda section names, from the dataset's kin_label column (notebook Section 1.2).
KIN = {
    "politics": "politiki", "sport": "imikino", "economy": "ubukungu", "health": "ubuzima",
    "entertainment": "imyidagaduro", "history": "amateka", "technology": "ikoranabuhanga",
    "tourism": "ubukerarugendo", "culture": "umuco", "fashion": "imideri",
    "religion": "iyobokamana", "environment": "ibidukikije", "education": "uburezi",
    "relationship": "urukundo",
}

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


# ---------------------------------------------------------------------------------------
# Exact explanation of a linear model
# ---------------------------------------------------------------------------------------
VEC = PIPELINE.named_steps["tfidf"]
CLF = PIPELINE.named_steps["clf"]
ANALYZE = VEC.build_analyzer()          # the char_wb n-gram splitter used in training
VOCAB = VEC.vocabulary_
NGRAMS = VEC.get_feature_names_out()
# Softmax ignores anything added equally to every class, so each feature's weight is measured
# against its average over the 14 topics: what remains is what separates the topics.
W_CENTRED = CLF.coef_ - CLF.coef_.mean(axis=0, keepdims=True)


def explain(raw: str, row: int):
    """Split topic `row`'s score into per-n-gram and per-word contributions.

    For each n-gram j in the text, contribution = tfidf_j x centred_weight[row, j]. A word's
    score is the sum over its n-grams, each shared equally between the places it occurs, so
    the word scores add up exactly to the n-gram total.
    """
    x = VEC.transform([preprocess(raw)])
    contrib = {int(j): float(v * W_CENTRED[row, j]) for j, v in zip(x.indices, x.data)}
    counts = Counter(g for g in ANALYZE(preprocess(raw)) if g in VOCAB)

    words = []
    for piece in re.split(r"(\s+)", raw):
        if not piece or piece.isspace():
            words.append((piece, None))
            continue
        grams = [g for g in ANALYZE(preprocess(piece)) if g in VOCAB]
        words.append((piece, sum(contrib.get(VOCAB[g], 0.0) / counts.get(g, 1) for g in grams)))
    return words, contrib


# ---------------------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------------------
def _pct(p: float) -> str:
    return f"{p:.0%}" if p >= 0.01 else "<1%"


def _signed(v: float) -> str:
    return f"{v:+.2f}" if abs(v) >= 0.01 else f"{v:+.3f}"


def _gram(j: int) -> str:
    return html.escape(NGRAMS[j].replace(" ", "_"))


def empty_card(message: str) -> str:
    return (f'<div class="kn-card kn-empty"><div class="kn-kicker">Awaiting copy</div>'
            f'<p>{html.escape(message)}</p></div>')


def result_card(order, probs, elapsed_ms: float, n_words: int) -> str:
    top = LABELS[int(PIPELINE.classes_[order[0]])]
    best = float(probs[order[0]])
    runners = "".join(
        f'<div class="kn-runner"><span>{LABELS[int(PIPELINE.classes_[i])].title()}</span>'
        f'<span class="kn-runner-bar"><i style="width:{probs[i]*100:.1f}%"></i></span>'
        f'<span class="kn-num">{_pct(float(probs[i]))}</span></div>'
        for i in order[1:3])
    warn = ""
    if best < THRESHOLD:
        warn = (f'<div class="kn-warn"><b>Low confidence.</b> Below the {THRESHOLD:.0%} threshold '
                f'chosen on validation data. This text may not be Kinyarwanda news; the model '
                f'still has to name one of its 14 topics.</div>')
    return f"""
<div class="kn-card">
  <div class="kn-kicker">Filed under</div>
  <div class="kn-topic">{top.title()}</div>
  <div class="kn-kin">{KIN.get(top, '')}</div>
  <div class="kn-meter"><i style="width:{best*100:.1f}%"></i></div>
  <div class="kn-conf"><span class="kn-num">{best:.0%}</span> confidence</div>
  {warn}
  <div class="kn-sub">Next most likely</div>
  {runners}
  <div class="kn-foot">{n_words} words · {elapsed_ms:.0f} ms on CPU · model E1</div>
</div>"""


def why_panel(raw: str, row: int) -> str:
    words, contrib = explain(raw, row)
    topic = LABELS[int(PIPELINE.classes_[row])]
    # A floor on the scale keeps weak evidence faint instead of stretching it to full colour.
    scale = max(max((abs(s) for _, s in words if s is not None), default=0.0), 0.5)
    total = sum(s for _, s in words if s is not None)
    spans = []
    for piece, score in words:
        if score is None:
            spans.append(html.escape(piece))
            continue
        strength = min(abs(score) / scale, 1.0)
        cls = "kn-for" if score > 0 else "kn-against"
        title = f"{score:+.3f} towards {topic}"
        delay = 250 + (1 - strength) * 450         # strongest words are marked first
        spans.append(f'<span class="{cls}" style="--s:{strength:.2f};--d:{delay:.0f}ms" title="{html.escape(title)}">'
                     f'{html.escape(piece)}</span>' if strength >= 0.08 else html.escape(piece))
    weak = ""
    if total < 0.5:
        weak = (f'<div class="kn-warn">The words give almost no evidence for {topic} (total '
                f'{total:+.2f}, against about +3 to +5 for a confident sports headline), so the '
                f'prediction rests mostly on the model\'s built-in topic priors.</div>')
    ranked = sorted(contrib.items(), key=lambda kv: kv[1])
    pro = [kv for kv in reversed(ranked) if kv[1] > 0][:8]
    con = [kv for kv in ranked if kv[1] < 0][:5]
    chips_for = "".join(f'<span class="kn-chip kn-chip-for" style="--i:{n}"><code>{_gram(j)}</code>'
                        f'<span class="kn-num">{_signed(v)}</span></span>' for n, (j, v) in enumerate(pro))
    chips_against = "".join(f'<span class="kn-chip kn-chip-against" style="--i:{n + len(pro)}"><code>{_gram(j)}</code>'
                            f'<span class="kn-num">{_signed(v)}</span></span>' for n, (j, v) in enumerate(con))
    return f"""
<div class="kn-why">
  <div class="kn-kicker">Why “{topic}”?</div>
  <p class="kn-why-lede">Highlighted words pushed the model <mark class="kn-key-for">towards {topic}</mark>;
  underlined ones pushed <span class="kn-key-against">against it</span>. Darker means stronger.</p>
  {weak}
  <div class="kn-copy">{''.join(spans)}</div>
  <div class="kn-cols">
    <div><div class="kn-sub">Strongest character n-grams for {topic}</div>{chips_for or '<p class="kn-muted">none</p>'}</div>
    <div><div class="kn-sub">Strongest against</div>{chips_against or '<p class="kn-muted">none</p>'}</div>
  </div>
  <p class="kn-muted kn-small">The model reads overlapping 2–5-letter pieces of each word (<code>_</code> marks a
  word edge), so <i>ikipe</i>, <i>amakipe</i> and <i>iyi kipe</i> all share <code>kipe</code>.
  Each piece's contribution is its TF-IDF weight × its logistic-regression weight for this topic,
  measured against the average over all 14 topics; the word scores add up exactly to that total.</p>
</div>"""


def index_panel(probs=None) -> str:
    """All 14 sections; before any prediction, the list itself with an instruction."""
    if probs is None:
        rows = "".join(f'<div class="kn-idx-row"><span class="kn-idx-name">{label.title()}'
                       f'<em>{KIN.get(label, "")}</em></span><span class="kn-idx-bar"></span>'
                       f'<span class="kn-num">—</span></div>' for label in LABELS)
        return (f'<div class="kn-index-wrap"><div class="kn-kicker">All 14 sections</div>'
                f'<p class="kn-muted kn-small">Classify a headline or article to see how likely '
                f'each section is.</p><div class="kn-index">{rows}</div></div>')
    rows = []
    for rank, i in enumerate(np.argsort(-probs)):
        label = LABELS[int(PIPELINE.classes_[i])]
        rows.append(f'<div class="kn-idx-row" style="--i:{rank}"><span class="kn-idx-name">{label.title()}'
                    f'<em>{KIN.get(label, "")}</em></span><span class="kn-idx-bar">'
                    f'<i style="width:{probs[i]*100:.1f}%"></i></span>'
                    f'<span class="kn-num">{probs[i]*100:.1f}%</span></div>')
    return (f'<div class="kn-index-wrap"><div class="kn-kicker">All 14 sections</div>'
            f'<div class="kn-index">{"".join(rows)}</div></div>')


def classify(text: str):
    """Return (result card, explanation, all-topics index) as HTML."""
    raw = (text or "").strip()
    if not raw:
        return empty_card("Paste a Kinyarwanda news headline or article, or pick an example below."), "", index_panel()
    n_words = len(raw.split())
    if n_words < MIN_WORDS:
        return empty_card(f"Only {n_words} word(s). Give at least {MIN_WORDS} — a headline is "
                          "usually enough."), "", index_panel()

    started = time.time()
    probs = PIPELINE.predict_proba([preprocess(raw)])[0]
    elapsed = (time.time() - started) * 1000
    order = np.argsort(-probs)
    return (result_card(order, probs, elapsed, n_words), why_panel(raw, int(order[0])),
            index_panel(probs))


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
logistic regression scores all 14 topics.

Character pieces suit Kinyarwanda because one word stem appears in many forms
(*umukino*, *imikino*, *abakinnyi*); the pieces they share let the model recognise the stem in
all of them. Because the model is linear, the "Why this topic?" panel can show exactly how much
each word and each piece added to the prediction.

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

MASTHEAD = f"""
<header class="kn-mast">
  <div class="kn-flag"><i></i><i></i><i></i></div>
  <div class="kn-mast-row">
    <h1>The Kinyarwanda Newsroom Desk</h1>
    <span class="kn-edition">14 sections · KINNEWS</span>
  </div>
  <p class="kn-dek">Paste a Kinyarwanda headline or article. The model files it under one of 14 sections
  and shows you, word by word, why.</p>
  <div class="kn-stats">
    <span><b>{META.get('test_macro_f1', 0):.3f}</b> test macro-F1</span>
    <span><b>{META.get('test_accuracy', 0):.1%}</b> test accuracy</span>
    <span><b>4,254</b> held-out articles</span>
    <span>TF-IDF character 2–5-grams + logistic regression</span>
  </div>
</header>"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
         '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700'
         '&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">')

CSS = """
:root, .light { --paper:#F4EFE4; --sheet:#FBF8F1; --ink:#1D1B17; --muted:#6E675A; --rule:#DCD3C2;
  --blue:#0083B8; --green:#20603D; --yellow:#F2C200; --hl:255,206,0; --red:181,69,43; }
.dark { --paper:#15140F; --sheet:#1E1D17; --ink:#EEE7D7; --muted:#A39B8A; --rule:#38352C;
  --blue:#41B9E6; --green:#5DBB86; --yellow:#F2C200; --hl:242,194,0; --red:232,120,96; }

main.app, .gradio-container { max-width: 1120px !important; width: 100% !important; margin: 0 auto !important;
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important; }
body, .gradio-container, .main { background: var(--paper) !important; }
footer { display: none !important; }

.kn-mast { padding: 26px 4px 18px; border-bottom: 3px double var(--ink); margin-bottom: 8px; color: var(--ink); }
.kn-flag { display:flex; width:84px; height:5px; margin-bottom:14px; }
.kn-flag i { flex:1; } .kn-flag i:nth-child(1){background:var(--blue);} .kn-flag i:nth-child(2){background:var(--yellow);} .kn-flag i:nth-child(3){background:var(--green);}
.kn-mast-row { display:flex; align-items:baseline; justify-content:space-between; gap:16px; flex-wrap:wrap; }
.kn-mast h1 { font-family:'Fraunces', Georgia, serif; font-weight:700; font-size:clamp(28px,4.4vw,46px);
  letter-spacing:-0.02em; line-height:1.05; margin:0; color:var(--ink); }
.kn-edition { font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); }
.kn-dek { font-family:'Fraunces', Georgia, serif; font-size:18px; color:var(--muted); margin:10px 0 14px; max-width:720px; }
.kn-stats { display:flex; flex-wrap:wrap; gap:6px 22px; font-size:13px; color:var(--muted); }
.kn-stats b { color:var(--ink); font-variant-numeric: tabular-nums; }

.tab-nav button, button[role="tab"] { font-family:'IBM Plex Sans', sans-serif !important; letter-spacing:.08em; text-transform:uppercase; font-size:12px !important; color: var(--muted) !important; }
.tab-nav button.selected, button[role="tab"][aria-selected="true"] { color: var(--ink) !important; border-color: var(--ink) !important; }
.tab-nav button.selected::after, button[role="tab"][aria-selected="true"]::after { background: var(--ink) !important; }

#kn-input textarea { background: var(--sheet) !important; color: var(--ink) !important;
  font-family:'Fraunces', Georgia, serif !important; font-size:18px !important; line-height:1.55 !important;
  border:1px solid var(--rule) !important; border-radius:2px !important; padding:16px 18px !important; }
#kn-input label span, .kn-label { font-size:11px !important; letter-spacing:.14em; text-transform:uppercase; color:var(--muted) !important; }
#kn-input, #kn-input > *, #kn-input label, #kn-input .container, .block.kn-plain, .form:has(#kn-input) { background: transparent !important; border: none !important; box-shadow:none !important; }
#kn-input { padding: 0 !important; } #kn-input label { padding: 0 !important; }
#kn-input span, #kn-input [data-testid="block-info"] { background: transparent !important; padding-left: 0 !important; }

#kn-go { background: var(--ink) !important; color: var(--paper) !important; border-radius: 2px !important;
  font-weight:600; letter-spacing:.12em; text-transform:uppercase; font-size:13px !important; border:none !important; }
#kn-go:hover { background: var(--green) !important; color: var(--paper) !important; }

#kn-examples-block { background: transparent !important; border:none !important; }
#kn-examples-block button { background: var(--sheet) !important; border:1px solid var(--rule) !important; border-radius:2px !important;
  color: var(--ink) !important; font-size:13px !important; }
#kn-examples-block button:hover { border-color: var(--ink) !important; }

.kn-card, .kn-why, .kn-index { color: var(--ink); }
.kn-card { background: var(--sheet); border:1px solid var(--rule); border-top:4px solid var(--ink); padding:20px 22px 16px; }
.kn-kicker { font-size:11px; letter-spacing:.16em; text-transform:uppercase; color:var(--muted); margin-bottom:6px; }
.kn-topic { font-family:'Fraunces', Georgia, serif; font-weight:700; font-size:44px; line-height:1; letter-spacing:-0.02em; }
.kn-kin { font-family:'Fraunces', Georgia, serif; font-style:italic; font-size:18px; color:var(--muted); margin:4px 0 14px; }
.kn-meter { height:8px; background: var(--rule); } .kn-meter i { display:block; height:100%; background: var(--green); }
.kn-conf { margin:6px 0 14px; font-size:14px; color:var(--muted); } .kn-conf .kn-num { color:var(--ink); font-weight:600; font-size:16px; }
.kn-num { font-family:'IBM Plex Mono', monospace; font-variant-numeric: tabular-nums; font-size:13px; }
.kn-sub { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:14px 0 6px; }
.kn-runner { display:grid; grid-template-columns: 120px 1fr 44px; align-items:center; gap:10px; font-size:14px; padding:3px 0; }
.kn-runner-bar, .kn-idx-bar { height:5px; background: var(--rule); } .kn-runner-bar i, .kn-idx-bar i { display:block; height:100%; background: var(--muted); }
.kn-runner .kn-num, .kn-idx-row .kn-num { text-align:right; }
.kn-foot { margin-top:16px; padding-top:10px; border-top:1px solid var(--rule); font-size:12px; color:var(--muted); }
.kn-warn { background: rgba(var(--red), .10); border-left:3px solid rgb(var(--red)); padding:10px 12px; font-size:13.5px; margin:4px 0 6px; }
.kn-empty p { font-family:'Fraunces', Georgia, serif; font-size:18px; color: var(--muted); margin:6px 0 4px; }

.kn-why { border-top:1px solid var(--ink); padding-top:16px; margin-top:6px; }
.kn-why-lede { font-size:14px; color: var(--muted); margin:0 0 12px; }
.kn-copy { font-family:'Fraunces', Georgia, serif; font-size:20px; line-height:1.9; background: var(--sheet);
  border:1px solid var(--rule); padding:16px 20px; }
.kn-for { background-image: linear-gradient(transparent 38%, rgba(var(--hl), calc(.12 + var(--s) * .78)) 38%); padding:0 2px; border-radius:1px; }
.dark .kn-for { background-image: linear-gradient(transparent 70%, rgba(var(--hl), calc(.25 + var(--s) * .75)) 70%); }
.dark mark.kn-key-for { background: linear-gradient(transparent 70%, rgba(var(--hl), .9) 70%); }
.kn-against { text-decoration: underline wavy rgba(var(--red), calc(.35 + var(--s) * .65)); text-decoration-thickness: 1.5px; text-underline-offset: 5px; }
mark.kn-key-for { background: linear-gradient(transparent 38%, rgba(var(--hl), .75) 38%); color: inherit; padding:0 2px; }
.kn-key-against { text-decoration: underline wavy rgb(var(--red)); text-underline-offset: 4px; }
.kn-cols { display:grid; grid-template-columns: 2fr 1fr; gap: 6px 28px; }
@media (max-width: 720px) { .kn-cols { grid-template-columns: 1fr; } .kn-topic { font-size:36px; } }
.kn-chip { display:inline-flex; align-items:center; gap:8px; margin:0 6px 6px 0; padding:3px 9px; border:1px solid var(--rule); background: var(--sheet); }
.kn-chip code { font-family:'IBM Plex Mono', monospace; font-size:13.5px; background:none; padding:0; color: var(--ink); }
.kn-chip-for { border-left:3px solid rgb(var(--hl)); } .kn-chip-for .kn-num { color: var(--green); }
.kn-chip-against { border-left:3px solid rgb(var(--red)); } .kn-chip-against .kn-num { color: rgb(var(--red)); }
.kn-muted { color: var(--muted); } .kn-small { font-size:12.5px; line-height:1.55; margin-top:12px; }
.kn-small code { font-family:'IBM Plex Mono', monospace; background:none; padding:0; }

.kn-index { columns: 2 280px; column-gap: 32px; padding-top:4px; }
.kn-idx-row { display:grid; grid-template-columns: 1fr 90px 52px; gap:10px; align-items:center; padding:6px 0;
  border-bottom:1px dotted var(--rule); break-inside: avoid; font-size:14px; }
.kn-idx-name em { font-family:'Fraunces', Georgia, serif; color: var(--muted); margin-left:6px; font-size:13px; }
.kn-index-wrap { border-top:1px solid var(--ink); padding-top:16px; margin-top:6px; color: var(--ink); }
.kn-index-wrap .kn-small { margin: 0 0 8px; }

/* Reveal: the result is already computed and on screen; these only animate it into place. */
@keyframes kn-grow { from { width: 0; } }
@keyframes kn-rise { from { opacity: 0; transform: translateY(6px); } to { opacity: 1; transform: none; } }
@keyframes kn-sweep { from { background-size: 0% 100%; } to { background-size: 100% 100%; } }
@keyframes kn-ink { from { text-decoration-color: transparent; } }
.kn-card:not(.kn-empty) .kn-kicker, .kn-topic, .kn-kin { animation: kn-rise .45s ease-out both; }
.kn-kin { animation-delay: .08s; }
.kn-meter i { animation: kn-grow .8s cubic-bezier(.2,.7,.2,1) .15s both; }
.kn-conf, .kn-warn, .kn-runner, .kn-foot { animation: kn-rise .45s ease-out .3s both; }
.kn-runner-bar i { animation: kn-grow .7s cubic-bezier(.2,.7,.2,1) .4s both; }
.kn-for { background-repeat: no-repeat; background-size: 100% 100%; animation: kn-sweep .5s ease-out var(--d, .3s) both; }
.kn-against { animation: kn-ink .6s ease-out var(--d, .3s) both; }
.kn-chip { animation: kn-rise .35s ease-out both; animation-delay: calc(.55s + var(--i, 0) * .045s); }
.kn-idx-row { animation: kn-rise .35s ease-out both; animation-delay: calc(var(--i, 0) * .03s); }
.kn-idx-bar i { animation: kn-grow .7s cubic-bezier(.2,.7,.2,1) both; animation-delay: calc(.1s + var(--i, 0) * .03s); }
@media (prefers-reduced-motion: reduce) { .kn-card *, .kn-why *, .kn-index * { animation: none !important; } }

.kn-about { color: var(--ink); max-width: 760px; }
.kn-about h2, .kn-about h3 { font-family:'Fraunces', Georgia, serif !important; }
.kn-about table { border-collapse: collapse !important; border: none !important; font-size: 14px; }
.kn-about th, .kn-about td { border: none !important; border-bottom: 1px solid var(--rule) !important; padding: 6px 18px 6px 0 !important; background: transparent !important; }
.kn-about th { border-bottom: 1.5px solid var(--ink) !important; font-size: 11px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.kn-about td { font-variant-numeric: tabular-nums; }
.prose.kn-about table tr, .prose.kn-about table thead, .prose.kn-about table tbody, .prose.kn-about table { border-left: 0 !important; border-right: 0 !important; border-top: 0 !important; box-shadow: none !important; }
"""

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("IBM Plex Sans"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#F4EFE4", body_background_fill_dark="#15140F",
    block_background_fill="transparent", block_background_fill_dark="transparent",
    block_border_width="0px", block_shadow="none", block_shadow_dark="none",
    body_text_color="#1D1B17", body_text_color_dark="#EEE7D7",
    border_color_primary="#DCD3C2", border_color_primary_dark="#38352C",
)

INITIAL = empty_card("Paste a Kinyarwanda news headline or article, or pick an example below.")

with gr.Blocks(title="Kinyarwanda News Topic Classifier", theme=THEME, css=CSS, head=FONTS) as demo:
    gr.HTML(MASTHEAD)
    with gr.Tab("Classify"):
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                inp = gr.Textbox(lines=7, label="Your copy", elem_id="kn-input", show_label=True,
                                 placeholder="Andika inkuru cyangwa umutwe w'inkuru hano...")
                btn = gr.Button("Classify", elem_id="kn-go")
                if EXAMPLE_TEXTS:
                    gr.Examples(examples=EXAMPLE_TEXTS, inputs=inp, example_labels=EXAMPLE_LABELS,
                                label="Real test-set headlines (one is a known failure)",
                                elem_id="kn-examples-block", examples_per_page=10)
            with gr.Column(scale=2):
                card = gr.HTML(INITIAL, elem_classes="kn-plain")
        why = gr.HTML("", elem_classes="kn-plain")
        index = gr.HTML(index_panel(), elem_classes="kn-plain")
        btn.click(classify, inputs=inp, outputs=[card, why, index])
        inp.submit(classify, inputs=inp, outputs=[card, why, index])
    with gr.Tab("About the model"):
        gr.Markdown(ABOUT, elem_classes="kn-about")

if __name__ == "__main__":
    demo.launch()
