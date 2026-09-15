"""ragshield/ladder_eval.py -- measure the attacker cost curve.

For each target scenario (ragshield/ladder.py) x rung, measures:
  - the candidate document's exact rank against the realistic-scale haystack
    (ragshield/scale_ingest.py) for k in {config.TOP_K, 10, 20}
  - where it WOULD actually be retrieved at config.TOP_K, the real downstream
    pipelines (naive / hardened / robust), scored with the scenario's own
    mechanical detector -- reusing the "planted doc + real top-(k-1)
    neighbours" pattern demo/app.py already established

Ranking is computed by pulling the haystack's embeddings out of Chroma once
and doing the cosine comparison directly in NumPy, rather than mutating the
persisted collection per trial -- hundreds of hill-climb iterations and
several flood-count sweep points would otherwise mean hundreds of Chroma
writes for no benefit, since the ranking only needs read access to vectors
that don't change.

Outputs follow evaluate.py's convention: results/ladder-<stamp>.jsonl (every
trial), results/ladder-<stamp>-summary.json (the cost-curve table), plus a
plotted PNG of the curve with the existing defenses overlaid.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from . import config, hardened, ladder, robust_a002
from .ingest import get_collection
from .pipeline import Chunk, build_prompt as naive_build_prompt, generate as pipeline_generate

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
K_POINTS = (4, 10, 20)


@dataclass
class Haystack:
    sources: list[str]
    texts: list[str]
    trusts: list[str]
    embeddings: np.ndarray  # [N, D], L2-normalized


def load_haystack(collection_name: str | None = None) -> Haystack:
    collection = get_collection(name=collection_name or config.SCALE_COLLECTION)
    raw = collection.get(include=["embeddings", "documents", "metadatas"])
    embs = np.asarray(raw["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    embs = embs / np.clip(norms, 1e-12, None)
    sources = [m.get("source", "unknown") for m in raw["metadatas"]]
    trusts = [m.get("trust", "trusted") for m in raw["metadatas"]]
    return Haystack(sources=sources, texts=list(raw["documents"]), trusts=trusts, embeddings=embs)


def rank_candidate(haystack: Haystack, query_emb: np.ndarray, candidate_emb: np.ndarray) -> int:
    """1-indexed rank of the candidate among (haystack U candidate) for this
    query -- exact, via one matmul against the cached embedding matrix."""
    sims = haystack.embeddings @ query_emb
    cand_sim = float(np.dot(candidate_emb, query_emb))
    return int(np.sum(sims > cand_sim)) + 1


def real_top_chunks(haystack: Haystack, query: str, k: int) -> list[Chunk]:
    query_emb = ladder.embed([query])[0]
    sims = haystack.embeddings @ query_emb
    order = np.argsort(-sims)[:k]
    return [
        Chunk(
            text=haystack.texts[i], source=haystack.sources[i], trust=haystack.trusts[i],
            distance=1.0 - float(sims[i]),
        )
        for i in order
    ]


def downstream_asr(haystack: Haystack, scn: ladder.Scenario, candidate_chunks: list[Chunk], top_k: int, runs: int) -> dict:
    """Force the candidate chunk(s) into context alongside real top-(k-n)
    neighbours, then run all three pipelines -- the same wiring
    demo/app.py::_build_chunks / _run_live already uses to test a pipeline
    against a forced document instead of assuming it was retrieved."""
    candidate_sources = {c.source for c in candidate_chunks}
    real = [c for c in real_top_chunks(haystack, scn.query, top_k) if c.source not in candidate_sources]
    chunks = (candidate_chunks + real)[:top_k]

    hits = {"naive": [], "hardened": [], "robust": []}
    for _ in range(runs):
        n_ans = pipeline_generate(naive_build_prompt(scn.query, chunks))
        hits["naive"].append(scn.detector(n_ans))

        kept, _dropped = hardened._screen(chunks)
        h_raw = pipeline_generate(hardened.build_prompt(scn.query, kept[:top_k]))
        h_ans, _ = hardened._filter_output(h_raw)
        hits["hardened"].append(scn.detector(h_ans))

        agg = robust_a002.isolate_and_aggregate(scn.query, kept[:top_k], pipeline_generate)
        r_ans, _ = hardened._filter_output(agg["answer"])
        hits["robust"].append(scn.detector(r_ans))

    return {name: sum(v) / len(v) for name, v in hits.items()}


def run(
    *,
    scenario_ids: list[str] | None = None,
    corpus_sample_for_vocab: int = 500,
    rung3_iterations: int = 40,
    flood_points: tuple[int, ...] = (1, 2, 4, 8, 16),
    runs_per_trial: int | None = None,
    top_k: int | None = None,
) -> dict:
    top_k = top_k or config.TOP_K
    runs = runs_per_trial or config.RUNS_PER_CASE
    scenarios = [s for s in ladder.SCENARIOS if not scenario_ids or s.id in scenario_ids]

    haystack = load_haystack()
    vocab_sample = haystack.texts[:corpus_sample_for_vocab]

    RESULTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = RESULTS_DIR / f"ladder-{stamp}.jsonl"

    rows: list[dict] = []
    with log_path.open("w", encoding="utf-8") as log:
        for scn in scenarios:
            query_emb = ladder.embed([scn.query])[0]

            for rung_name, text_fn in (("plain", ladder.rung1_plain), ("mirrored", ladder.rung2_mirrored)):
                text = text_fn(scn)
                cand_emb = ladder.embed([text])[0]
                rank = rank_candidate(haystack, query_emb, cand_emb)
                row = {
                    "scenario": scn.id, "rung": rung_name, "effort": 0, "rank": rank,
                    "similarity": float(np.dot(cand_emb, query_emb)),
                    "entered_topk": {str(k): rank <= k for k in K_POINTS},
                }
                if rank <= top_k:
                    cand_chunk = Chunk(text=text, source=f"attacker-{scn.id}-{rung_name}", trust="untrusted", distance=0.0)
                    row["asr"] = downstream_asr(haystack, scn, [cand_chunk], top_k, runs)
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                print(f"  {scn.id} {rung_name:9s} rank {rank:<6d} sim {row['similarity']:.3f}")

            opt = ladder.rung3_optimized(scn, corpus_texts=vocab_sample, iterations=rung3_iterations)
            cand_emb = ladder.embed([opt["text"]])[0]
            rank = rank_candidate(haystack, query_emb, cand_emb)
            row = {
                "scenario": scn.id, "rung": "optimized", "effort": opt["iterations"], "rank": rank,
                "similarity": opt["similarity"], "trajectory": opt["trajectory"],
                "entered_topk": {str(k): rank <= k for k in K_POINTS},
            }
            if rank <= top_k:
                cand_chunk = Chunk(text=opt["text"], source=f"attacker-{scn.id}-optimized", trust="untrusted", distance=0.0)
                row["asr"] = downstream_asr(haystack, scn, [cand_chunk], top_k, runs)
            rows.append(row)
            log.write(json.dumps(row) + "\n")
            print(f"  {scn.id} optimized rank {rank:<6d} sim {row['similarity']:.3f}  ({opt['iterations']} accepted edits)")

            base_text = ladder.rung1_plain(scn)
            for n in flood_points:
                variants = ladder.rung4_flood(scn, base_text, n=n)
                ranks = []
                cand_chunks = []
                for text, source in variants:
                    emb = ladder.embed([text])[0]
                    r = rank_candidate(haystack, query_emb, emb)
                    ranks.append(r)
                    if r <= top_k:
                        cand_chunks.append(Chunk(text=text, source=source, trust="untrusted", distance=0.0))
                row = {
                    "scenario": scn.id, "rung": "flooded", "effort": n, "ranks": ranks,
                    "entered_topk": {str(k): any(r <= k for r in ranks) for k in K_POINTS},
                }
                if cand_chunks:
                    row["asr"] = downstream_asr(haystack, scn, cand_chunks, top_k, runs)
                rows.append(row)
                log.write(json.dumps(row) + "\n")
                print(f"  {scn.id} flood(n={n:<2d}) any-top-{top_k}? {row['entered_topk'][str(top_k)]}")

    summary = summarize(rows, top_k, [s.id for s in scenarios])
    summary["log"] = str(log_path)
    (RESULTS_DIR / f"ladder-{stamp}-summary.json").write_text(json.dumps(summary, indent=2))
    try:
        plot_cost_curve(summary, RESULTS_DIR / f"ladder-{stamp}.png")
        summary["plot"] = str(RESULTS_DIR / f"ladder-{stamp}.png")
    except Exception as exc:  # plotting is a bonus, never block the numbers on it
        print(f"  (plot skipped: {exc})")
    return summary


def summarize(rows: list[dict], top_k: int, scenario_ids: list[str]) -> dict:
    k_str = str(top_k)

    def _asr_means(sub: list[dict]) -> dict:
        if not any("asr" in r for r in sub):
            return {}
        return {
            name: round(statistics.mean(r["asr"][name] for r in sub if "asr" in r), 3)
            for name in ("naive", "hardened", "robust")
        }

    ladder_rows = []
    for rung in ("plain", "mirrored", "optimized"):
        sub = [r for r in rows if r["rung"] == rung]
        if not sub:
            continue
        p_topk = sum(1 for r in sub if r["entered_topk"].get(k_str)) / len(sub)
        ladder_rows.append({
            "rung": rung,
            "p_topk": round(p_topk, 3),
            "mean_rank": round(statistics.mean(r["rank"] for r in sub), 1),
            "asr": _asr_means(sub),
        })

    flood_by_n: dict[int, list[dict]] = {}
    for r in rows:
        if r["rung"] == "flooded":
            flood_by_n.setdefault(r["effort"], []).append(r)
    flood_curve = []
    for n in sorted(flood_by_n):
        sub = flood_by_n[n]
        p_topk = sum(1 for r in sub if r["entered_topk"].get(k_str)) / len(sub)
        flood_curve.append({"n_sources": n, "p_topk": round(p_topk, 3), "asr": _asr_means(sub)})

    return {"top_k": top_k, "scenarios": scenario_ids, "ladder": ladder_rows, "flood_curve": flood_curve}


def plot_cost_curve(summary: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    ax = axes[0]
    rungs = [r["rung"] for r in summary["ladder"]]
    p = [r["p_topk"] for r in summary["ladder"]]
    ax.plot(rungs, p, marker="o", color="#c0392b")
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(f"P(enters top-{summary['top_k']})")
    ax.set_title("Attacker sophistication -> retrieval success")
    ax.set_xlabel("rung (effort increases left to right)")

    ax2 = axes[1]
    ns = [r["n_sources"] for r in summary["flood_curve"]]
    p2 = [r["p_topk"] for r in summary["flood_curve"]]
    ax2.plot(ns, p2, marker="o", color="#c0392b", label="retrieval: any copy in top-k")
    for name, color in (("naive", "#7f8c8d"), ("hardened", "#2980b9"), ("robust", "#1e8449")):
        vals = [r["asr"].get(name) for r in summary["flood_curve"]]
        if any(v is not None for v in vals):
            ax2.plot(ns, [v if v is not None else float("nan") for v in vals],
                      marker="s", linestyle="--", color=color, label=f"ASR ({name})")
    ax2.set_xlabel("independent flooded documents (N)")
    ax2.set_ylabel("probability")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_title("Multi-document flooding vs. defenses")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def report(summary: dict) -> None:
    print("\n" + "=" * 64)
    print(f"retrieval ladder -- top_k={summary['top_k']}  scenarios={summary['scenarios']}")
    print("=" * 64)
    for row in summary["ladder"]:
        print(f"  {row['rung']:10s}  P(top-k)={row['p_topk']:6.0%}  mean_rank={row['mean_rank']:<8}  asr={row['asr']}")
    print("-" * 64)
    for row in summary["flood_curve"]:
        print(f"  flood n={row['n_sources']:<3d} P(top-k)={row['p_topk']:6.0%}  asr={row['asr']}")
    print(f"\n  log: {summary.get('log')}")
    if summary.get("plot"):
        print(f"  plot: {summary['plot']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the retrieval-manipulation attacker cost curve.")
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--runs", type=int, default=None)
    ap.add_argument("--rung3-iterations", type=int, default=40)
    ap.add_argument("--scenarios", nargs="*", default=None, help="limit to scenario ids, e.g. L1 L2")
    ap.add_argument("--flood-points", nargs="*", type=int, default=None)
    args = ap.parse_args()

    started = time.time()
    summary = run(
        scenario_ids=args.scenarios,
        rung3_iterations=args.rung3_iterations,
        runs_per_trial=args.runs,
        top_k=args.top_k,
        flood_points=tuple(args.flood_points) if args.flood_points else (1, 2, 4, 8, 16),
    )
    report(summary)
    print(f"  completed in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
