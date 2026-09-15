"""demo/record.py -- capture live pipeline runs so the hosted demo can replay them.

Run this once on a machine with the full stack (Chroma populated, Ollama up):

    python demo/record.py                 # all attack cases
    python demo/record.py --case A002     # just one
    LLM_MODEL=llama3.1:8b python demo/record.py

Writes demo/recorded/<CASE>.json plus demo/recorded/summary.json. Commit them:
they are the evidence the hosted demo serves, and they let anyone reproduce
what the page claims without a GPU.

Honesty rule: the summary records the model, the date, and the run count, and
the demo prints them. A recorded demo that does not say it is recorded is a
misleading demo.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
OUT = HERE / "recorded"
sys.path.insert(0, str(REPO_ROOT))

from ragshield import pipeline, hardened, robust_a002, config  # noqa: E402
from ragshield.attacks import (  # noqa: E402
    ATTACK_CASES, detect_a001, detect_a002, detect_a003,
)

DETECTORS = {"A001": detect_a001, "A002": detect_a002, "A003": detect_a003}


def _build_chunks(query: str, doc_text: str, doc_source: str, k: int) -> list:
    retrieved = pipeline.retrieve(query, top_k=k)
    stem = Path(doc_source).name
    benign = [c for c in retrieved if stem not in c.source]
    planted = pipeline.Chunk(text=doc_text, source=doc_source, trust="untrusted", distance=0.0)
    return [planted] + benign[: max(k - 1, 1)]


def record_case(case) -> dict:
    doc_text = (REPO_ROOT / case.document).read_text(encoding="utf-8")
    chunks = _build_chunks(case.query, doc_text, case.document, config.TOP_K)
    detector = DETECTORS[case.id]

    print(f"  {case.id}: naive...", flush=True)
    n_ans = pipeline.generate(pipeline.build_prompt(case.query, chunks))

    print(f"  {case.id}: hardened...", flush=True)
    kept, dropped = hardened._screen(chunks)
    h_raw = pipeline.generate(hardened.build_prompt(case.query, kept[: config.TOP_K]))
    h_ans, h_filtered = hardened._filter_output(h_raw)

    print(f"  {case.id}: robust ({len(kept[:config.TOP_K])} isolated calls)...", flush=True)
    agg = robust_a002.isolate_and_aggregate(case.query, kept[: config.TOP_K], pipeline.generate)
    r_ans, r_filtered = hardened._filter_output(agg["answer"])

    return {
        "case": case.id,
        "class": case.attack_class,
        "query": case.query,
        "planted_doc": doc_text,
        "model": config.LLM_MODEL,
        "recorded": date.today().isoformat(),
        "naive": {"answer": n_ans, "hit": detector(n_ans)},
        "hardened": {
            "answer": h_ans, "hit": detector(h_ans),
            "dropped": dropped, "filtered": h_filtered,
        },
        "robust": {
            "answer": r_ans, "hit": detector(r_ans),
            "decision": agg["decision"],
            "isolated": agg["isolated"],
            "support_sources": agg.get("support_sources"),
            "total_sources": agg.get("total_sources"),
            "dropped": dropped, "filtered": r_filtered,
        },
    }


def _pct(hit: bool) -> str:
    return "100%" if hit else "0%"


def main() -> None:
    ap = argparse.ArgumentParser(description="Record demo runs for replay.")
    ap.add_argument("--case", action="append", help="case id (repeatable); default all")
    args = ap.parse_args()

    cases = [c for c in ATTACK_CASES if not args.case or c.id in args.case]
    OUT.mkdir(exist_ok=True)

    recs = []
    for case in cases:
        print(f"recording {case.id} ...")
        rec = record_case(case)
        (OUT / f"{case.id}.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
        recs.append(rec)

    summary = {
        "model": config.LLM_MODEL,
        "recorded": date.today().isoformat(),
        "caption": (
            f"Single recorded run per pipeline on {config.LLM_MODEL}, temperature "
            f"{config.LLM_TEMPERATURE}, top_k={config.TOP_K}. "
            "Attack success rates over "
            f"{config.RUNS_PER_CASE} runs per case come from "
            "`python -m ragshield.evaluate`; see results/ in the repository."
        ),
        "asr_table": [
            {
                "case": r["case"], "class": r["class"],
                "naive": _pct(r["naive"]["hit"]),
                "hardened": _pct(r["hardened"]["hit"]),
                "robust": _pct(r["robust"]["hit"]),
            }
            for r in recs
        ],
        # Fill from robust_a002.sweep_poison_sources on the real model.
        "boundary": [],
    }
    existing = OUT / "summary.json"
    if existing.is_file():
        try:
            prev = json.loads(existing.read_text(encoding="utf-8"))
            summary["boundary"] = prev.get("boundary", [])
        except json.JSONDecodeError:
            pass
    existing.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {len(recs)} case file(s) + summary.json to {OUT}")


if __name__ == "__main__":
    main()
