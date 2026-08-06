"""Evaluation harness: run every case N times, log everything, report ASR.

Writes one JSON object per run to results/<pipeline>-<timestamp>.jsonl so that
individual runs stay inspectable -- when a number looks surprising you want the
prompt and the retrieved chunks that produced it, not just the verdict.

Two numbers come out of this:
  ASR      -- how often attacks succeed (want it low)
  Utility  -- how often benign answers stay correct (want it unchanged)
Reporting only the first is how projects end up with a useless perfect score.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime
from pathlib import Path

from . import config
from .attacks import ATTACK_CASES, BENIGN_CASES

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _load_pipeline(name: str):
    if name == "naive":
        from . import pipeline

        return pipeline.answer
    if name == "hardened":
        try:
            from . import hardened
        except ImportError as exc:  # not built yet
            raise SystemExit(
                "hardened pipeline not implemented yet -- run with --pipeline naive"
            ) from exc
        return hardened.answer
    raise SystemExit(f"unknown pipeline: {name}")


def _retrieved_planted(result, document: str) -> bool:
    """An attack that is never retrieved never fires. Track this separately so a
    low ASR caused by weak retrieval is not mistaken for a working defense.

    Checks both the chunks that made it into the prompt and any the hardened
    pipeline screened out (result.meta["dropped_chunks"]) -- otherwise a
    working screen would look identical to a retrieval miss.
    """
    stem = Path(document).name
    sources = [c.source for c in result.chunks]
    sources += [d.get("source", "") for d in result.meta.get("dropped_chunks", [])]
    return any(stem in s for s in sources)


def run(pipeline_name: str, runs: int | None = None) -> dict:
    answer = _load_pipeline(pipeline_name)
    runs = runs or config.RUNS_PER_CASE
    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = RESULTS_DIR / f"{pipeline_name}-{stamp}.jsonl"

    per_case: dict[str, list[bool]] = {}
    retrieval: dict[str, list[bool]] = {}
    latencies: list[int] = []

    with log_path.open("w", encoding="utf-8") as log:
        # --- attacks ---
        for case in ATTACK_CASES:
            per_case[case.id] = []
            retrieval[case.id] = []
            for i in range(runs):
                result = answer(case.query)
                success = case.detector(result.answer)
                planted = _retrieved_planted(result, case.document)
                per_case[case.id].append(success)
                retrieval[case.id].append(planted)
                latencies.append(result.latency_ms)
                log.write(
                    json.dumps(
                        {
                            "kind": "attack",
                            "case": case.id,
                            "class": case.attack_class,
                            "run": i,
                            "query": case.query,
                            "success": success,
                            "planted_retrieved": planted,
                            "answer": result.answer,
                            "sources": result.sources,
                            "latency_ms": result.latency_ms,
                            "prompt": result.prompt,
                            "meta": result.meta,
                        }
                    )
                    + "\n"
                )
                print(f"  {case.id} run {i + 1}/{runs}: {'HIT' if success else 'blocked'}")

        # --- benign ---
        benign_ok: list[bool] = []
        for case in BENIGN_CASES:
            for i in range(runs):
                result = answer(case["query"])
                low = result.answer.lower()
                ok = all(tok.lower() in low for tok in case["must_include"])
                refused = "don't know" in low or "do not know" in low
                benign_ok.append(ok and not refused)
                latencies.append(result.latency_ms)
                log.write(
                    json.dumps(
                        {
                            "kind": "benign",
                            "case": case["id"],
                            "run": i,
                            "query": case["query"],
                            "correct": ok,
                            "refused": refused,
                            "answer": result.answer,
                            "sources": result.sources,
                            "latency_ms": result.latency_ms,
                            "meta": result.meta,
                        }
                    )
                    + "\n"
                )
                print(f"  {case['id']} run {i + 1}/{runs}: {'ok' if ok else 'MISS'}")

    asr_by_case = {cid: sum(v) / len(v) for cid, v in per_case.items()}
    overall_asr = statistics.mean(asr_by_case.values()) if asr_by_case else 0.0
    utility = sum(benign_ok) / len(benign_ok) if benign_ok else 0.0

    summary = {
        "pipeline": pipeline_name,
        "model": config.LLM_MODEL,
        "runs_per_case": runs,
        "asr_by_case": asr_by_case,
        "overall_asr": overall_asr,
        "retrieval_rate": {
            cid: sum(v) / len(v) for cid, v in retrieval.items()
        },
        "utility": utility,
        "median_latency_ms": statistics.median(latencies) if latencies else 0,
        "log": str(log_path),
        "timestamp": stamp,
    }
    (RESULTS_DIR / f"{pipeline_name}-{stamp}-summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def report(summary: dict) -> None:
    print("\n" + "=" * 52)
    print(f"pipeline: {summary['pipeline']}   model: {summary['model']}")
    print("=" * 52)
    for cid, asr in summary["asr_by_case"].items():
        got = summary["retrieval_rate"][cid]
        note = "" if got else "   <- planted doc never retrieved"
        print(f"  {cid}  ASR {asr:6.0%}   retrieved {got:4.0%}{note}")
    print("-" * 52)
    print(f"  overall ASR      {summary['overall_asr']:6.0%}   (lower is better)")
    print(f"  benign utility   {summary['utility']:6.0%}   (must stay high)")
    print(f"  median latency   {summary['median_latency_ms']} ms")
    print(f"\n  runs logged to {summary['log']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure ASR and utility.")
    ap.add_argument("--pipeline", default="naive", choices=["naive", "hardened"])
    ap.add_argument("--runs", type=int, default=None)
    args = ap.parse_args()

    started = time.time()
    summary = run(args.pipeline, runs=args.runs)
    report(summary)
    print(f"  completed in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
