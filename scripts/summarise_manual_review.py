"""Summarise the manual label/error review filled in by a Kinyarwanda reader.

Usage:
    cp data/audit/manual_review_TEMPLATE.csv data/audit/manual_review.csv
    # fill in verdict / correct_label / cause / note
    python scripts/summarise_manual_review.py

verdict: label_correct | label_wrong | ambiguous
cause (errors only): topic_overlap | label_noise | short_text | truncation |
                     code_switching | rare_names | model_error | other
Writes results/manual_review_summary.json.
"""
import json
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "audit" / "manual_review.csv"
OUT = ROOT / "results" / "manual_review_summary.json"

VERDICTS = {"label_correct", "label_wrong", "ambiguous"}
CAUSES = {"topic_overlap", "label_noise", "short_text", "truncation",
          "code_switching", "rare_names", "model_error", "other"}


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found - copy the TEMPLATE there and fill it in first.")
    df = pd.read_csv(SRC, dtype=str).fillna("")
    df["verdict"] = df["verdict"].str.strip().str.lower()
    df["cause"] = df["cause"].str.strip().str.lower()

    bad_v = sorted(set(df.verdict) - VERDICTS - {""})
    bad_c = sorted(set(c for cell in df.cause for c in cell.split(";") if c.strip()) - CAUSES)
    if bad_v or bad_c:
        raise SystemExit(f"Unknown values - verdict: {bad_v}, cause: {bad_c}")

    done = df[df.verdict != ""]
    summary = {"rows_total": len(df), "rows_reviewed": len(done), "by_source": {}}
    for source, g in done.groupby("source"):
        counts = Counter(g.verdict)
        entry = {"reviewed": len(g), "verdicts": dict(counts),
                 "label_wrong_share": round(counts["label_wrong"] / len(g), 3)}
        causes = Counter(c.strip() for cell in g.cause for c in cell.split(";") if c.strip())
        if causes:
            entry["causes"] = dict(causes.most_common())
        summary["by_source"][source] = entry

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nsaved -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
