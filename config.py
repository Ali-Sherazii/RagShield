"""Central configuration. Everything is env-overridable so the container and
the laptop behave identically (same lesson as AegisVault: no hardcoded values).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent

# --- storage -----------------------------------------------------------------
CHROMA_DIR = os.getenv("CHROMA_DIR", str(ROOT / "data" / "chroma"))
COLLECTION = os.getenv("COLLECTION", "ragshield")

# --- embeddings --------------------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# --- chunking ----------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

# --- retrieval ---------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# --- generation --------------------------------------------------------------
# Local model keeps ASR numbers reproducible: no silent vendor updates,
# no API key in a public repo, no per-run cost.
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0"))
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# --- evaluation --------------------------------------------------------------
RUNS_PER_CASE = int(os.getenv("RUNS_PER_CASE", "3"))