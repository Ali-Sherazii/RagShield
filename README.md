# RAGShield

Defending retrieval-augmented generation against indirect prompt injection and
corpus poisoning — by building the attack first, then the defense, and measuring
the difference.

**Status:** work in progress. Naive (undefended) pipeline is in place; attack
corpus and defense layers are next.

## Why

A RAG pipeline concatenates a trusted system prompt with untrusted retrieved
content into a single block of text. The model cannot tell them apart. Anyone
who can publish a page that gets crawled can therefore influence what the
assistant tells its users — without touching the code, the model, or the
vector store.

See [THREAT_MODEL.md](THREAT_MODEL.md) for the full attacker model and scope.

## Approach

1. Build a standard, undefended RAG pipeline (this repo, current state).
2. Build a corpus of publicly documented attacks and measure how often they
   succeed against it (Attack Success Rate).
3. Add defense layers and measure ASR again — while checking that answer
   quality on benign queries does not degrade.

A defense that blocks everything has an ASR of zero and is useless. The result
that matters is **ASR down, utility held.**

## Stack

LangChain loaders/splitters · Chroma · `bge-small-en-v1.5` embeddings ·
Llama 3.1 8B locally via Ollama (temperature 0, so results are reproducible by
anyone who clones this).

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# pull the model (once)
ollama pull llama3.1:8b

# build the legitimate corpus (crawled live from the committed URL list)
python -m ragshield.ingest --sources corpus/sources.yaml --reset

# plant the authored attack documents alongside it
python -m ragshield.ingest --path corpus/attack --trust untrusted

# ask it something ordinary
python -m ragshield.pipeline "How do I declare a path parameter?" --show-context

# measure attack success rate and benign utility
python -m ragshield.evaluate --pipeline naive
```

## Layout

```
corpus/sources.yaml   legitimate corpus URL list (committed; the corpus is not)
corpus/attack/        attack documents authored for this project
ragshield/pipeline.py the naive, undefended pipeline -- the attack target
ragshield/attacks.py  attack cases, success detectors, benign utility set
ragshield/evaluate.py runs every case N times, logs JSONL, reports ASR
results/              per-run logs and summaries (gitignored)
data/chroma/          vector store -- a build artifact, rebuildable from sources
```

## Ethics and scope

All testing is performed against this self-hosted reference pipeline. No
third-party or production system is probed. Attack techniques demonstrated are
drawn from published research and cited in `references.md`.
