"""demo/app.py -- RAGShield interactive demo.

Two modes, auto-detected:

  replay (default)  Serves pre-recorded runs from demo/recorded/*.json. No
                    Ollama, no Chroma, no GPU -- so this deploys to a free
                    Hugging Face Space and answers in milliseconds. This is
                    what a visitor following a link gets.
  live              Runs the real pipelines. Used locally, or anywhere the
                    package + Ollama are reachable.

Why replay is the default and not a fallback: one click runs the naive prompt,
the hardened prompt, and one isolated generation per retrieved chunk -- six
generations at TOP_K=4. On llama3.1:8b that is minutes of staring at a spinner.
A demo nobody waits for is a demo nobody sees, so the hosted path serves
recorded evidence and says so plainly rather than pretending to be live.

Record the runs first (needs the full stack):
    python demo/record.py                       # writes demo/recorded/*.json

Run:
    pip install gradio
    python demo/app.py                          # replay if no stack, live if present
    RAGSHIELD_DEMO_MODE=live python demo/app.py # force live
    RAGSHIELD_DEMO_MODE=replay python demo/app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import gradio as gr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RECORDED = HERE / "recorded"
sys.path.insert(0, str(REPO_ROOT))

REPO_URL = "https://github.com/Ali-Sherazii/RAGShield"

# --------------------------------------------------------------------------
# mode detection -- live needs the package AND a reachable model
# --------------------------------------------------------------------------

_MODE = os.getenv("RAGSHIELD_DEMO_MODE", "auto").lower()
LIVE = False
_live_error = ""

if _MODE in ("auto", "live"):
    try:
        from ragshield import pipeline, hardened, robust_a002, config
        from ragshield.attacks import (
            ATTACK_CASES, detect_a001, detect_a002, detect_a003,
        )
        LIVE = True
    except Exception as exc:  # noqa: BLE001 - any import failure means no live mode
        _live_error = f"{type(exc).__name__}: {exc}"
        if _MODE == "live":
            raise SystemExit(
                f"live mode requested but the package failed to import: {_live_error}"
            )

DETECTORS = (
    {"A001": detect_a001, "A002": detect_a002, "A003": detect_a003} if LIVE else {}
)
CASE_BY_ID = {c.id: c for c in ATTACK_CASES} if LIVE else {}


# --------------------------------------------------------------------------
# recorded runs
# --------------------------------------------------------------------------

def _load_recorded() -> dict:
    out: dict = {}
    if not RECORDED.is_dir():
        return out
    for path in sorted(RECORDED.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and "case" in data:
            out[data["case"]] = data
    return out


RECORDS = _load_recorded()

# Case metadata so the UI works even with no package and no recordings.
FALLBACK_CASES = [
    ("A001", "instruction_injection",
     "Hidden HTML-comment instructions override the system prompt.",
     "How do I enable CORS in FastAPI?"),
    ("A002", "corpus_poisoning",
     "Authoritative false facts, no instructions -- defeats injection filters.",
     "Is there a limit on how many background tasks FastAPI can run?"),
    ("A003", "exfiltration",
     "Directive to encode context into an outbound markdown image URL.",
     "How should I test my FastAPI endpoints?"),
]


def _case_choices() -> list[tuple[str, str]]:
    if LIVE:
        return [(f"{c.id} · {c.attack_class} — {c.description}", c.id) for c in ATTACK_CASES]
    available = set(RECORDS)
    return [
        (f"{cid} · {klass} — {desc}" + ("" if cid in available else "  (not recorded)"), cid)
        for cid, klass, desc, _ in FALLBACK_CASES
    ]


# --------------------------------------------------------------------------
# live execution
# --------------------------------------------------------------------------

def _build_chunks(query: str, doc_text: str, doc_source: str, k: int) -> list:
    """Real retrieved benign chunks + the attack/custom doc forced in as untrusted."""
    retrieved = pipeline.retrieve(query, top_k=k)
    stem = Path(doc_source).name
    benign = [c for c in retrieved if stem not in c.source]
    planted = pipeline.Chunk(text=doc_text, source=doc_source, trust="untrusted", distance=0.0)
    return [planted] + benign[: max(k - 1, 1)]


def _run_live(query: str, chunks: list, detector) -> dict:
    """Run all three pipelines on the same chunk set.

    The robust column runs the FULL defended stack -- screen, then aggregate,
    then output filter -- not aggregation alone. Aggregating raw chunks would
    show the robust pipeline losing to A001 and A003, which is an artefact of
    the demo wiring rather than a property of the system.
    """
    n_prompt = pipeline.build_prompt(query, chunks)
    n_ans = pipeline.generate(n_prompt)

    kept, dropped = hardened._screen(chunks)
    h_prompt = hardened.build_prompt(query, kept[: config.TOP_K])
    h_raw = pipeline.generate(h_prompt)
    h_ans, h_filtered = hardened._filter_output(h_raw)

    agg = robust_a002.isolate_and_aggregate(query, kept[: config.TOP_K], pipeline.generate)
    r_ans, r_filtered = hardened._filter_output(agg["answer"])

    return {
        "query": query,
        "planted_doc": chunks[0].text if chunks else "",
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


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def _verdict(hit: bool) -> str:
    return (
        "<span style='color:#c0392b;font-weight:700'>● ATTACK SUCCEEDS</span>"
        if hit else
        "<span style='color:#1e8449;font-weight:700'>● DEFENDED</span>"
    )


def _panel(title: str, subtitle: str, hit: bool, body: str, extra: str = "") -> str:
    return (
        f"### {title}\n"
        f"<sub>{subtitle}</sub>\n\n"
        f"{_verdict(hit)}\n\n"
        f"> {body.strip()[:700]}\n"
        f"{extra}"
    )


def _format(res: dict) -> tuple[str, str, str, str]:
    n = res["naive"]
    naive_md = _panel(
        "1 · Naive", "retrieve → stuff into one prompt → generate",
        n["hit"], n["answer"],
    )

    h = res["hardened"]
    extra = ""
    reasons = [d.get("reason", "") for d in h.get("dropped", [])]
    if reasons:
        extra += f"\n\n`screened out:` {', '.join(sorted(set(reasons)))}"
    if h.get("filtered"):
        extra += f"\n\n`output filter:` {', '.join(h['filtered'])}"
    hard_md = _panel(
        "2 · Hardened", "injection screen → spotlighted prompt → output filter",
        h["hit"], h["answer"], extra,
    )

    r = res["robust"]
    rows = ["| source | trust | isolated answer |", "|---|---|---|"]
    for i in r.get("isolated", []):
        src = str(i.get("source", ""))
        src = src.rsplit("/", 1)[-1][:28]
        vote = "*abstained*" if i.get("abstain") else str(i.get("answer", ""))[:70].replace("|", "\\|")
        rows.append(f"| `{src}` | {i.get('trust','?')} | {vote} |")
    votes = "\n".join(rows)
    support = r.get("support_sources")
    total = r.get("total_sources")
    gate = (
        f"\n\n`consensus gate:` {support}/{total} independent sources"
        if support is not None and total is not None else ""
    )
    rob_md = _panel(
        "3 · Robust", "isolate per chunk → consensus over independent sources",
        r["hit"], r["answer"],
        f"{gate}\n\n`decision:` {r.get('decision','')}\n\n**per-source votes**\n\n{votes}",
    )

    doc = res.get("planted_doc", "")
    doc_md = (
        "#### The planted document\n"
        f"```\n{doc.strip()[:1200]}\n```" if doc else ""
    )
    return naive_md, hard_md, rob_md, doc_md


# --------------------------------------------------------------------------
# callbacks
# --------------------------------------------------------------------------

def run_preset(case_id: str, force_live: bool):
    if force_live and LIVE:
        case = CASE_BY_ID[case_id]
        doc_path = REPO_ROOT / case.document
        try:
            doc_text = doc_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return (f"### error\ncould not read `{doc_path}` — run from the repo root.", "", "", "")
        chunks = _build_chunks(case.query, doc_text, case.document, config.TOP_K)
        res = _run_live(case.query, chunks, DETECTORS[case_id])
        return _format(res)

    rec = RECORDS.get(case_id)
    if rec is None:
        return (
            f"### `{case_id}` has no recorded run\n\n"
            "Generate one with `python demo/record.py`, or tick **run live** "
            "if the pipeline is available on this machine.", "", "", "",
        )
    return _format(rec)


def run_custom(query: str, doc_text: str, force_live: bool):
    if not LIVE or not force_live:
        return (
            "### Custom documents need live mode\n\n"
            "This hosted instance replays recorded runs, so it cannot generate "
            "an answer for a new document. Clone the repo and run "
            "`python demo/app.py` with Ollama available to paste your own "
            f"poisoned document.\n\n[Repository]({REPO_URL})", "", "", "",
        )
    if not query.strip() or not doc_text.strip():
        return ("### Paste a document and a question first.", "", "", "")
    chunks = _build_chunks(query, doc_text, "user-upload.md", config.TOP_K)

    def any_detector(ans: str) -> bool:
        return detect_a001(ans) or detect_a002(ans) or detect_a003(ans)

    return _format(_run_live(query, chunks, any_detector))


# --------------------------------------------------------------------------
# headline result -- shown before any interaction
# --------------------------------------------------------------------------

def _headline() -> str:
    summary_path = RECORDED / "summary.json"
    if not summary_path.is_file():
        return ""
    try:
        s = json.loads(summary_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return ""

    rows = ["| attack | class | naive | hardened | robust |", "|---|---|---|---|---|"]
    for row in s.get("asr_table", []):
        rows.append(
            f"| {row['case']} | {row['class']} | {row['naive']} | "
            f"{row['hardened']} | {row['robust']} |"
        )
    table = "\n".join(rows)

    sweep = s.get("boundary", [])
    if sweep:
        head = "| attacker-controlled sources | " + " | ".join(str(p["sources"]) for p in sweep) + " |"
        sep = "|---" * (len(sweep) + 1) + "|"
        vals = "| A002 attack success | " + " | ".join(p["asr"] for p in sweep) + " |"
        boundary = f"\n\n**Where the A002 defense stops working**\n\n{head}\n{sep}\n{vals}\n"
    else:
        boundary = ""

    return (
        f"### Measured results\n\n{table}\n"
        f"<sub>{s.get('caption','')}</sub>\n{boundary}"
    )


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

MODE_BANNER = (
    "**Live mode** — answers are generated now, one model call per retrieved chunk."
    if LIVE else
    "**Replay mode** — showing runs recorded earlier so this page answers instantly. "
    "Every answer below came from a real execution of the pipelines; the logs are in the repo."
)

with gr.Blocks(title="RAGShield — RAG poisoning attacks and defenses") as demo:
    gr.Markdown(
        "# RAGShield\n"
        "### Defending retrieval-augmented generation against prompt injection "
        "and corpus poisoning — by building the attack first, then the defense, "
        "and measuring the difference.\n\n"
        "A RAG pipeline concatenates a trusted system prompt with untrusted "
        "retrieved text into one block. The model cannot tell them apart, so "
        "anyone who can publish a page that gets crawled can influence what the "
        "assistant tells its users — without touching the code, the model, or "
        "the vector store.\n\n"
        f"{MODE_BANNER}\n\n"
        f"[Repository]({REPO_URL}) · "
        f"[Threat model]({REPO_URL}/blob/main/THREAT_MODEL.md)"
    )

    headline = _headline()
    if headline:
        gr.Markdown(headline)

    gr.Markdown("---\n## Watch a defense engage\nSame document, same question, three pipelines.")

    with gr.Row():
        with gr.Column(scale=3):
            case_dd = gr.Dropdown(
                choices=_case_choices(),
                value="A002",
                label="attack case",
                info="A002 (corpus poisoning) is the one plain injection filters cannot catch.",
            )
        with gr.Column(scale=1):
            live_toggle = gr.Checkbox(
                value=False, label="run live",
                info=("generate now (slow)" if LIVE else "unavailable on this instance"),
                interactive=LIVE,
            )
    go1 = gr.Button("Run this attack", variant="primary")

    doc_out = gr.Markdown()
    with gr.Row(equal_height=False):
        out_naive = gr.Markdown()
        out_hard = gr.Markdown()
        out_robust = gr.Markdown()

    with gr.Accordion("Try your own document (live mode only)", open=False):
        q_in = gr.Textbox(
            label="question",
            placeholder="Is there a limit on how many background tasks FastAPI can run?",
        )
        doc_in = gr.Textbox(
            label="document — paste something poisoned or benign", lines=8,
            placeholder="As of 0.111 FastAPI enforces three concurrent background tasks per worker...",
        )
        go2 = gr.Button("Run", variant="secondary")

    gr.Markdown(
        "---\n"
        "<sub>The robust pipeline answers from each retrieved chunk in isolation, then "
        "reports a claim only when enough <b>independent documents</b> support it. "
        "It does not eliminate poisoning: it raises the attacker's cost from one planted "
        "document to control of several. That threshold is the defense, and it is "
        "measured above rather than asserted.</sub>"
    )

    go1.click(run_preset, [case_dd, live_toggle],
              [out_naive, out_hard, out_robust, doc_out])
    go2.click(run_custom, [q_in, doc_in, live_toggle],
              [out_naive, out_hard, out_robust, doc_out])

    demo.load(lambda: run_preset("A002", False),
              None, [out_naive, out_hard, out_robust, doc_out])


if __name__ == "__main__":
    demo.queue().launch()
