"""Build the realistic-scale "haystack" corpus for the retrieval-manipulation
cost-curve experiment (ragshield/ladder_eval.py).

Why this exists: RAGShield's 15-page FastAPI corpus makes retrieval trivial --
a planted attack document is nearly the only thing in the index relevant to
its target query, so "does it get retrieved" was never really being tested.
This module ingests a few thousand REAL security Q&A threads (Security
StackExchange, via the Stack Exchange Data Dump) into a second, separate
Chroma collection, so a planted document has to rank against genuine
competitors -- the thing the small corpus can't provide.

Source: flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl
on Hugging Face -- a pre-parsed mirror of the official Stack Exchange Data
Dump (CC BY-SA). Downloaded via huggingface_hub (plain HTTP/curl mishandles
HF's Xet-backed CDN chunking and silently truncates the file -- confirmed
while building this). The raw dump is a build artifact like data/chroma/:
gitignored, reproducible from corpus/scale_corpus.yaml plus the fixed sample
seed below, never committed.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path

from . import config, ingest

MANIFEST_PATH = config.ROOT / "corpus" / "scale_corpus.yaml"
RAW_DIR = config.ROOT / "data" / "raw"


def _manifest() -> dict:
    """corpus/scale_corpus.yaml is the single source of truth for which
    dataset/file this builds from -- same role as corpus/sources.yaml for
    the legitimate FastAPI corpus, so the dataset id never drifts out of
    sync between the committed manifest and this code."""
    import yaml

    return yaml.safe_load(MANIFEST_PATH.read_text(encoding="utf-8"))


def _download(manifest: dict) -> Path:
    from huggingface_hub import hf_hub_download

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=manifest["dataset"],
        repo_type="dataset",
        filename=manifest["file"],
        local_dir=str(RAW_DIR),
    )
    return Path(path)


def _load_pairs(gz_path: Path) -> list[tuple[str, str]]:
    """Each line is [title_and_body, best_voted_answer] -- a real security
    Q&A thread. Returned in on-disk order so row index is stable across runs."""
    pairs: list[tuple[str, str]] = []
    with gzip.open(gz_path, "rt", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if isinstance(row, list) and len(row) == 2:
                pairs.append((row[0], row[1]))
    return pairs


def sample_documents(
    size: int | None = None, seed: int | None = None
) -> list[tuple[str, str]]:
    """Returns (text, source) pairs ready for ingest.ingest_documents.

    Sampling is by row INDEX in the full downloaded pool (not a re-shuffled
    sub-list), so `source` ids stay stable across reruns and across corpus
    sizes -- rerunning with a bigger SCALE_CORPUS_SIZE is a superset, not a
    different random draw.
    """
    manifest = _manifest()
    size = size or config.SCALE_CORPUS_SIZE
    seed = seed if seed is not None else config.SCALE_CORPUS_SEED
    gz_path = RAW_DIR / manifest["file"]
    if not gz_path.is_file():
        gz_path = _download(manifest)
    pairs = _load_pairs(gz_path)

    rng = random.Random(seed)
    indices = list(range(len(pairs)))
    rng.shuffle(indices)
    chosen = sorted(indices[:size])

    site = manifest["file"].removesuffix(".jsonl.gz")
    docs: list[tuple[str, str]] = []
    for i in chosen:
        title_body, answer = pairs[i]
        text = f"{title_body.strip()}\n\n{answer.strip()}"
        docs.append((text, f"{site}#{i}"))
    return docs


def build(reset: bool = False, size: int | None = None, seed: int | None = None) -> int:
    docs = sample_documents(size=size, seed=seed)
    trust = _manifest().get("trust", "trusted")
    total = ingest.ingest_documents(
        docs, trust=trust, reset=reset, collection_name=config.SCALE_COLLECTION
    )
    print(f"  sampled {len(docs)} documents -> {total} chunks")
    return total


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build the realistic-scale security-Q&A corpus for the retrieval ladder experiment."
    )
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    ap.add_argument("--size", type=int, default=None, help="override SCALE_CORPUS_SIZE")
    ap.add_argument("--seed", type=int, default=None, help="override SCALE_CORPUS_SEED")
    args = ap.parse_args()

    count = build(reset=args.reset, size=args.size, seed=args.seed)
    print(f"indexed {count} chunks into '{config.SCALE_COLLECTION}'")


if __name__ == "__main__":
    main()
