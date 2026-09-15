"""Ingestion: web pages or local files -> chunks -> embeddings -> Chroma.

Every chunk keeps a `source` and a `trust` label in metadata. Nothing uses
`trust` yet -- the naive pipeline deliberately ignores it. It exists so the
hardened pipeline can implement source provenance later without re-ingesting.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def _client():
    return chromadb.PersistentClient(path=config.CHROMA_DIR)


def get_collection(reset: bool = False, name: str | None = None):
    client = _client()
    # Local cache first -- avoids a multi-minute retry storm (five retries
    # with backoff, per file) when HF Hub's freshness check hits flaky DNS,
    # for a model that's already downloaded and never changes underneath us.
    try:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL, local_files_only=True
        )
    except Exception:
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config.EMBED_MODEL
        )
    name = name or config.COLLECTION
    if reset:
        try:
            client.delete_collection(name)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=name,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )


def _chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha1(f"{source}:{index}:{text}".encode()).hexdigest()[:16]
    return f"{digest}"


def _add(collection, texts: list[str], source: str, trust: str) -> int:
    if not texts:
        return 0
    collection.add(
        ids=[_chunk_id(source, i, t) for i, t in enumerate(texts)],
        documents=texts,
        metadatas=[
            {"source": source, "trust": trust, "chunk": i}
            for i, _ in enumerate(texts)
        ],
    )
    return len(texts)


def ingest_urls(
    urls: list[str], trust: str = "untrusted", reset: bool = False, collection_name: str | None = None
) -> int:
    """Crawl pages and index them. Web content is untrusted by default --
    that assumption is the whole point of the threat model."""
    collection = get_collection(reset=reset, name=collection_name)
    splitter = _splitter()
    total = 0
    for url in urls:
        docs = WebBaseLoader(url).load()
        for doc in docs:
            chunks = splitter.split_text(doc.page_content)
            total += _add(collection, chunks, source=url, trust=trust)
        print(f"  indexed {url}")
    return total


def ingest_paths(
    paths: list[str], trust: str = "untrusted", reset: bool = False, collection_name: str | None = None
) -> int:
    """Index local .txt/.md files -- used for authored attack documents."""
    collection = get_collection(reset=reset, name=collection_name)
    splitter = _splitter()
    total = 0
    for p in paths:
        path = Path(p)
        files = sorted(path.rglob("*.md")) + sorted(path.rglob("*.txt")) if path.is_dir() else [path]
        for f in files:
            chunks = splitter.split_text(f.read_text(encoding="utf-8"))
            total += _add(collection, chunks, source=str(f), trust=trust)
            print(f"  indexed {f}")
    return total


def ingest_documents(
    docs: list[tuple[str, str]],
    trust: str = "trusted",
    reset: bool = False,
    collection_name: str | None = None,
) -> int:
    """Index (text, source) pairs directly -- used by scale_ingest.py, where
    documents come from a downloaded dataset rather than files or URLs."""
    collection = get_collection(reset=reset, name=collection_name)
    splitter = _splitter()
    total = 0
    for text, source in docs:
        chunks = splitter.split_text(text)
        total += _add(collection, chunks, source=source, trust=trust)
    return total


def ingest_sources(yaml_path: str, reset: bool = False) -> int:
    """Rebuild the legitimate corpus from the committed source list.

    The vector store is a build artifact and is not committed; this list is what
    makes the corpus reproducible by anyone who clones the repo.
    """
    import yaml

    spec = yaml.safe_load(Path(yaml_path).read_text(encoding="utf-8"))
    return ingest_urls(
        spec.get("urls", []), trust=spec.get("trust", "trusted"), reset=reset
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest documents into the RAGShield corpus.")
    ap.add_argument("--sources", help="YAML file listing legitimate corpus URLs")
    ap.add_argument("--url", action="append", default=[], help="URL to crawl (repeatable)")
    ap.add_argument("--path", action="append", default=[], help="File or directory to index (repeatable)")
    ap.add_argument("--trust", default="untrusted", choices=["trusted", "untrusted"])
    ap.add_argument("--reset", action="store_true", help="drop the collection first")
    args = ap.parse_args()

    count = 0
    if args.sources:
        count += ingest_sources(args.sources, reset=args.reset)
        args.reset = False
    if args.url:
        count += ingest_urls(args.url, trust=args.trust, reset=args.reset)
        args.reset = False  # only reset once
    if args.path:
        count += ingest_paths(args.path, trust=args.trust, reset=args.reset)
    print(f"indexed {count} chunks into '{config.COLLECTION}'")


if __name__ == "__main__":
    main()
