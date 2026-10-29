# Research notes

## Trending-first check (2026-10-29, checked before any paper research)

GitHub daily trending, fetched 2026-10-29 05:08 UTC
(https://github.com/trending?since=daily), was dominated by
prompt/skill packs and agent workspaces with no measurable code
weakness to optimize: JetBrains/thinkrail, elayadesign/ai-design-skills,
obra/superpowers ("An agentic skills framework"), HKUDS/DeepTutor,
macro-inc/macro, Nutlope/hallmark (anti-slop design skill),
pbakaus/impeccable, makecindy/cindy, nightly-labs/openbot,
expo/skills. A second source (DEV Community, "Is GitHub Trending Just
AI Agents Now?", reporting GitHub Trending on 28 Oct 2026) listed the
same shape: hallmark (+2,830 stars that day), DeepTutor (+1,598),
cindy (+1,465), impeccable (+1,449), ai-design-skills (+1,395),
thinkrail (+1,204), affaan-m/ECC (+1,165), openbot (+777),
obra/superpowers (+617). These are mostly Markdown skill/prompt
collections. Optimizing one would mean editing prompts — no
benchmarkable speed, memory, or accuracy weakness exists to measure,
so none was chosen. This is the honest fallback case in the standing
rules: a cosmetic re-skin was not forced.

News/current trends checked the same morning: the effort-economics
paper (Sun et al., below) was the most-covered new result; Anthropic's
expert-deference study (28 Oct 2026, 86% adoption of AI preferences)
and Oracle/TPU cloud news were noted and rejected as non-buildable
for this project.

## Chosen target: NVIDIA/kvpress

- Repository: https://github.com/NVIDIA/kvpress
- Stars/forks at check time (2026-10-29, via GitHub search API):
  1,240 stars, 191 forks, 145 commits, last updated 2026-10-29
  02:13 UTC. License: Apache License 2.0. Language: Python.
- What it is: "LLM KV cache compression made easy" — a library of
  training-free "presses" that prune the KV cache during prefilling
  (and, experimentally, decoding) for Hugging Face transformers
  models. Each press prunes by a score: RandomPress, KnormPress
  (inverse key norm), SnapKVPress (mean attention of the last
  queries), ExpectedAttentionPress (expected future attention),
  StreamingLLMPress (sinks + recent), TOVAPress, QFilterPress,
  PyramidKVPress, AdaKVPress (head-wise reallocation), LUKVPress,
  ChunkKVPress / ChunkPress, MergingPress, and others. Verified from
  the repository README on 2026-10-29.
- Why it is optimizable, measurably: (1) most scorer presses apply
  one keep fraction uniformly and score only from the observation
  window, so they are query-blind when the question arrives after
  compression (the shared-context reuse setting); (2) chunked
  designs add a phase coordinate that average benchmarks conceal;
  (3) eviction is irreversible — no press verifies that the retained
  set can still answer probe queries and restores within budget;
  (4) no press allocates budget by measured per-head behaviour
  combined with phase fairness. Each weakness maps to one mechanism
  in this project, and each mechanism is ablated separately.

## Papers and sources verified on arXiv abstract pages

1. Zhu, Yi, Cheng, Lv, Liu, Ying, Ma, Dong, "Periodic Weak Spots:
   Phase Sensitivity from Chunked KV-Cache Compression",
   arXiv:2609.36322, submitted 28 Sep 2026.
   https://arxiv.org/abs/2609.36322 — Chunked KV-cache compression
   adds a phase coordinate (position relative to window boundaries);
   retrieval accuracy differed by up to 40 percentage points across
   phases in large open-weight models with such compression. This is
   the paper's number, not this repo's. Basis for phase-stratified
   retention and worst-phase reporting.
2. Sun, Kwon, Hasan, Zhu, Zhou, Bianchi, "What Effort Level Should I
   Choose? On the Economics of Reasoning Effort in LLM Agents",
   arXiv:2610.34491, submitted 25 Oct 2026.
   https://arxiv.org/abs/2610.34491 — Highest reasoning effort costs
   up to 42x the lowest on commonly solvable tasks; escalate only on
   verified failure. Cited for the verified-failure-only rescue
   pattern (probe, then restore within budget), not for any number
   measured here.
3. Abdali, Ko, Cameron, "AttSVD: Prompt-Adaptive Low-Rank KV Cache
   Compression via Attention-Guided SVD", arXiv (submitted 2 Oct
   2026) — feature-axis compression instead of token eviction.
   Companion context only; not implemented (needs trained SVD
   bases on real models).
4. Patel, "TokenGate", arXiv:2610.26754, submitted 19 Oct 2026 —
   learned token gating for prefill, reporting 1.4–11.3x faster
   time-to-first-token. Companion context only; needs model
   adaptation on GPU, not reproducible here.
5. Devoto et al., kvpress paper, arXiv:2510.00636 — the kvpress
   library paper. Companion context; the library README above is
   the primary source for press behaviour.
6. Query-visibility audit: "How Query Visibility Changes KV-Cache
   Compression Rankings: A Matched-Budget Audit", arXiv:2607.11942
   (found via search 2026-10-29) — holding model, budget, instances
   and decoding fixed, query-dependent compression (SnapKV in
   particular) degrades when the query is not visible at compression
   time; trivial baselines (random, StreamingLLM, key-norm) anchor
   the audit. Basis for the shared-context reuse workload and for
   including random/streaming/knorm baselines. Numbers from that
   audit are not quoted here.

## Rejected alternatives (with reasons)

- obra/superpowers, affaan-m/ECC, design-skill repos: prompt packs,
  no benchmarkable code weakness; optimizing them would be a
  cosmetic re-skin.
- deepseek-ai/DeepSpec (7,255 stars, 693 forks, updated 2026-10-29):
  full-stack speculative-decoding training/evaluation; needs GPUs
  and target/draft model training — not CPU-reproducible here.
- z-lab/dflash (6,207 stars, 441 forks): block-diffusion draft
  model; needs training and GPU serving — not reproducible here.
- sgl-project/SpecForge (1,246 stars, 380 forks): speculative
  decoding training for SGLang; GPU-bound.
- PORA, Select-TTT, Barron softmax fix, One-Pixel/ORA: rejected by
  the Day 4 worker for stated reasons (too close to Days 2–3, needs
  real transformer fast weights, held for a later night, attack
  tooling). No new decisive reason to resurrect any of them.
- AttSVD / TokenGate as the build: need real-model adaptation and
  GPU evaluation; kept as companion context only.

## Method honesty

Baseline presses in this repo are independent re-implementations of
published scoring rules, written from descriptions, run on a
simulated NumPy attention testbed. NVIDIA/kvpress code was never
downloaded, imported, or run, and no GPU or Hugging Face model was
used. Benchmark numbers in this repo are simulation numbers from
that testbed. They show the *relative* behaviour of the scoring
rules under controlled conditions; they are not measurements of
kvpress on real models, and the README says so wherever the
comparison appears.
