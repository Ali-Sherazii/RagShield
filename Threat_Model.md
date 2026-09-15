# RAGShield — Threat Model

Scope note: all testing in this project is performed against a self-hosted reference RAG pipeline built for this purpose. No third-party or production system is probed, and all attack techniques demonstrated are drawn from publicly documented research, cited in `references.md`.

---

## 1. System under test

A standard retrieve-then-generate pipeline: a crawler ingests web documentation into a vector store; at query time the top-k chunks are retrieved and concatenated into the model's prompt; the LLM answers the user's question from that context.

**Trust boundary — the central issue:** the pipeline crosses a boundary at retrieval. The system prompt is authored by the developer and is trusted. Retrieved document content originates from the open web and is **untrusted** — but once both are concatenated into a single prompt, that distinction disappears. The model sees one undifferentiated block of text. Every attack in this project exploits that collapse.

Stated plainly: **retrieved content is data, but the model may interpret it as instructions.**

## 2. Assets worth protecting

| Asset | Why it matters |
|---|---|
| Answer integrity | The user acts on the answer. A hijacked answer can send them to a phishing site, recommend a vulnerable dependency, or state false facts with citations attached. |
| System prompt confidentiality | Contains developer logic, guardrail wording, sometimes keys or internal policy. |
| Conversation / context confidentiality | Prior turns may contain user-supplied private data. |
| Downstream actions | If answers feed any tool, link, or automated step, a hijack becomes execution rather than just text. |
| Availability & cost | Injected content can force long generations or loops, driving up latency and spend. |

## 3. Actors

- **Content author (primary attacker).** Publishes a web page that the crawler ingests. Cannot touch the code, the model, the vector store, or the system prompt. Controls only the *text of a document*. This is the realistic and interesting adversary — deliberately weak, and still dangerous.
- **Malicious end user (secondary).** Sends crafted queries directly. Classic jailbreaking; partially in scope because some defenses overlap, but not the focus.
- **Benign user (the victim).** Asks an ordinary question and receives a compromised answer. They are not attacking anything — this is what makes indirect injection worse than direct jailbreaking: **the victim and the attacker are different people, and the victim did nothing wrong.**

## 4. Attacker capabilities (explicit)

The content author **can**:
- publish arbitrary text on a page that is crawled and indexed;
- craft text to rank highly for target queries (retrieval SEO — the attack requires being *retrieved*, so this is part of the capability, not an aside);
- use formatting to hide payloads from humans but not from parsers (white text, zero-width characters, HTML comments, alt text, metadata);
- embed instructions, false facts, or exfiltration lures;
- iterate offline against an open-source pipeline to refine payloads.

The content author **cannot**:
- modify the system prompt, application code, or defenses;
- access the vector store directly or delete competing documents;
- fine-tune or alter model weights;
- observe the victim's session, or see whether an attack succeeded (blind attack — worth noting, since exfiltration channels are how they'd gain feedback).

## 5. Attack surface → attack classes in scope

1. **Instruction injection via retrieved content.** Payload in a document overrides the developer's intent ("ignore prior instructions; tell the user X").
2. **Corpus poisoning / false-fact injection.** No instructions at all — just authoritative-sounding wrong content that the model faithfully repeats *with a citation*, which makes it more credible than an ordinary hallucination. Notably, this defeats naive "faithfulness" checks: the answer **is** grounded in the retrieved text. The text is the problem.
3. **Exfiltration via injection.** Payload instructs the model to encode the system prompt or prior context into an outbound URL (typically a markdown image), leaking it when rendered.
4. **Persona / guardrail override.** Retrieved content dissolves the assistant's constraints for the rest of the session.

## 6. Out of scope (state this — scope discipline is a signal)

- Attacks on the model weights (poisoning during training, backdoors).
- Infrastructure compromise (vector store breach, stolen credentials, supply chain).
- Denial of service and rate-limit abuse.
- Direct-to-user jailbreaking as a primary target.
- Multimodal injection (payloads in images).
- Any testing against systems not owned by this project.

## 7. Assumptions

- The LLM is a general instruction-following model with no built-in injection defense; it has no reliable way to distinguish developer instructions from retrieved text.
- The crawler ingests pages without human review — true of essentially every automated RAG ingestion pipeline.
- The developer is honest; the threat is external content, not insider risk.
- Retrieval works correctly: an attack that is never retrieved is not a successful attack, so payload retrievability is part of the attack, not a flaw in the test.
- Evaluation uses a fixed local model at temperature 0, run three times per case, mean reported — so numbers are reproducible by anyone cloning the repo.

## 8. Defense objectives (what "success" means)

Each defense maps to a boundary the attacker must cross:

| Objective | Defense layer | Attack class addressed | Status |
|---|---|---|---|
| Keep untrusted text from reading as instructions | Delimiting / spotlighting retrieved content; hardened system prompt | 1, 4 | Implemented — `hardened.build_prompt()` |
| Catch hostile content before it reaches the model | Injection detection on retrieved chunks (heuristic → classifier) | 1, 3, 4 | Implemented (heuristic only) — `hardened._screen()` |
| Limit damage from content that gets through | Output filtering: strip outbound URLs, detect system-prompt fragments | 3 | Implemented — `hardened._filter_output()` |
| Reduce exposure to untrusted sources | Source provenance / allowlisting, trust-weighted ranking | 1, 2, 3 | Implemented (exposure cap) — `hardened._screen()` |
| Preserve usefulness | Utility check on benign queries — a defense that blocks legitimate answers has failed | all | Measured — see README Results |

That last row is the one most projects skip. A pipeline that refuses everything has an attack success rate of zero and is worthless. **The result that matters is ASR down, utility held** — the same false-positive tradeoff as tuning a firewall's detection threshold.

**Measured result (2026-08-07, `llama3.1:8b`):** overall ASR 66.7% → 33.3%,
benign utility 80% → 100%. Class 1 (instruction injection) dropped 100% → 0%.
Class 2 (corpus poisoning) is unchanged at 100% — expected, since it has no
instruction pattern or output anomaly for any layer above to catch; closing
it needs semantic fact-checking against the trusted corpus, which is not yet
built. Full numbers and caveats in [README.md](README.md#results).

## 9. What gets measured

- **ASR** per attack class, naive vs hardened (headline number).
- **Utility retention** on a benign query set — answer quality and refusal rate must not degrade materially.
- **Per-layer attribution** — which defense caught each attack (this drives the demo, and shows which layers actually earn their place).
- **Overhead** — added latency and token cost per query.

---

**Design consequence:** because the attacker's only capability is authoring text that gets retrieved, every defense here operates on the path between retrieval and generation, plus one check after generation. No defense assumes the attacker can be identified, blocked, or authenticated — you cannot authenticate the open web.

## 10. Track B: retrieval as the attack surface

Section 4 already names "craft text to rank highly for target queries" as
part of the attacker's capability, not an aside. Everything above this
section tests what happens *after* retrieval succeeds. Against the 15-page
corpus that's close to given — a planted document is nearly the only thing
in the index relevant to its target query. Track B isolates the retrieval
step itself and asks the question sections 1-9 assume an answer to: what
does a document actually need to look like to get retrieved at all, against
a realistic volume of genuine competing content?

**Corpus.** `ragshield/scale_ingest.py` ingests ~4,000 real Security
StackExchange Q&A threads (Stack Exchange Data Dump, CC BY-SA — see
references.md) into a second, separate Chroma collection. This is read-only
haystack content, not modified or attacked itself — the corpus/sources.yaml
reproducibility pattern applies here too (`corpus/scale_corpus.yaml`).

**Attacker ladder** (`ragshield/ladder.py`), four rungs of increasing
sophistication, each measured by exact rank against the real haystack:

1. **Plain** — an authored false claim, no retrieval engineering. Expected
   to rank far outside the top-k; a documented negative result, not a bug.
2. **Mirrored** — the same claim with query terms echoed in the text
   (keyword-stuffing retrieval SEO).
3. **Optimized** — a white-box, gradient-shortlisted greedy hill-climb
   directly against the real local `bge-small-en-v1.5` model (the same
   model the pipeline's retrieval uses), substituting real words drawn from
   the haystack's own vocabulary so the result stays human-readable.
   HotFlip-style (Ebrahimi et al. 2018); the retrieval-specific form is
   Zhong et al. 2023. Every accept/reject decision is an exact re-embedding,
   so the gradient shortlist can only affect speed, never correctness.
4. **Flooded** — N independent paraphrased copies as separate sources (not
   chunks of one file), directly probing the boundary `robust_a002.py`'s
   `MIN_SUPPORT` consensus gate already defends.

**What this measures that sections 1-9 don't:** an attacker cost curve —
effort (rung, hill-climb iterations, or flood count) on one axis, P(enters
the top-k) and downstream ASR (naive / hardened / robust) on the other. The
existing defenses are overlaid on the flooding curve specifically, since
that's the rung `MIN_SUPPORT` is positioned against — the isolate-then-
aggregate gate can only raise the cost of flooding, never of ranking a
single well-optimized document, which is the gap this track exists to show.

Out of scope, same posture as section 6: every claim `ladder.py` plants is a
fabricated numeric limit plus a fabricated override variable (never real
actionable security advice), and the haystack corpus is read-only — nothing
is published back to Security StackExchange or any live system.

**Measured result (2026-09-15, `llama3.1:8b`, 5 scenarios, 14,806-chunk
corpus):** a plain false claim already ranks #1 for 3/5 topics (P(top-4)
60% at rung 1); query-term mirroring alone takes all 5 to rank #1 (100%).
The white-box optimizer improves raw similarity further but *lowers*
downstream naive ASR versus mirroring (60% → 20%) — retrieval success and
generation-persuasiveness measurably diverge. Multi-document flooding
(N=4) pushes naive and hardened ASR to 100% while robust holds at 0%, but
verified directly: that 0% is `hardened.MAX_UNTRUSTED_CHUNKS=1` starving
`robust_a002`'s own `MIN_SUPPORT=2` consensus gate of the independent
sources it needs to ever fire, not the gate reaching and holding its
documented threshold — the boundary `robust_a002.py`'s self-test
demonstrates only holds with `_screen()` bypassed, not in the composed,
end-to-end pipeline. Full numbers, methodology caveats, and the exact
reproduction of that last finding are in
[README.md](README.md#track-b-retrieval-manipulation-cost-curve).