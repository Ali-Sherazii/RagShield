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

All attack documents in `corpus/attack/` were authored for this project. Their
payloads are inert: they reference a non-existent package and the reserved
`example.invalid` domain, and nothing is transmitted anywhere.
