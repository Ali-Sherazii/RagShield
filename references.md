# References

Attack techniques demonstrated in this project are drawn from published
research. Nothing here is novel offensive work; the contribution is the
measured comparison of defenses against known attack classes.

- Greshake, Abdelnabi, Mishra, Endres, Holz, Fritz. *Not what you've signed up
  for: Compromising Real-World LLM-Integrated Applications with Indirect Prompt
  Injection.* AISec, 2023. — basis for attack classes A001 and A003.
- Zou, Geng, Schoenherr, Wang. *PoisonedRAG: Knowledge Corruption Attacks to
  Retrieval-Augmented Generation of Large Language Models.* 2024. — basis for
  attack class A002.
- OWASP Top 10 for LLM Applications — LLM01 (Prompt Injection), LLM03
  (Training/Data Poisoning). — taxonomy and terminology.
- Hines et al. *Defending Against Indirect Prompt Injection Attacks With
  Spotlighting.* 2024. — basis for the delimiting/spotlighting defense layer.
- Xiang, Wu, Zhang, Mittal, Wang, Henderson. *Certifiably Robust RAG against
  Retrieval Corruption* ("RobustRAG"), 2024. — basis for the isolate-then-
  aggregate defense in `robust_a002.py`.
- Zhong, Huang, Chen, Boyd-Graber. *Poisoning Retrieval Corpora by Injecting
  Adversarial Passages.* EMNLP, 2023. — basis for the white-box,
  gradient-guided embedding-similarity attack (rung 3, "optimized") in
  `ragshield/ladder.py`.
- Ebrahimi, Rao, Lowd, Dou. *HotFlip: White-Box Adversarial Examples for Text
  Classification.* ACL, 2018. — the gradient-shortlist-then-verify technique
  `ladder.py`'s optimizer adapts to embedding similarity.

All attack documents in `corpus/attack/` were authored for this project. Their
payloads are inert: they reference a non-existent package and the reserved
`example.invalid` domain, and nothing is transmitted anywhere. The same is
true of `ragshield/ladder.py`'s scenarios: every claim is a fabricated
numeric limit plus a fabricated override variable, never real actionable
security advice.

## Retrieval-scale corpus

`ragshield/scale_ingest.py` builds a realistic-scale "haystack" corpus from
real, licensed community content — not authored attack material:

- Stack Exchange Data Dump (security.stackexchange.com), licensed CC BY-SA,
  via the pre-parsed mirror
  [flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl](https://huggingface.co/datasets/flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl)
  on Hugging Face. The raw dump is a gitignored build artifact
  (`data/raw/`), reproducible from `corpus/scale_corpus.yaml` plus a fixed
  sample seed — the same reproducibility pattern `corpus/sources.yaml`
  already uses for the legitimate FastAPI corpus.
