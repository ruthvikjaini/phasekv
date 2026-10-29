# Adapting phasekv to real kvpress

This mapping is a design sketch, not a tested integration. Nothing in
this document has been run against a real transformer; the policy was
evaluated only on the simulated testbed in this repo. Treat it as the
starting point for a GPU follow-up.

## Where each mechanism plugs in

kvpress presses are callables that wrap a model and prune the KV
cache in forward hooks. A `ScorerPress` scores every (head, token)
pair and keeps the top-k per head. The phase-adaptive policy maps
onto that interface as follows:

| phasekv mechanism | kvpress analogue |
| --- | --- |
| Score from observation-window attention | `ScorerPress` score tensor, shape (heads, tokens) |
| Per-head adaptive budgets | Replace the uniform `compression_ratio` with a per-head keep count, as `AdaKVPress`/`LUKVPress` do with static curves — here computed from measured head concentration at compression time |
| Phase-stratified retention | Post-process the top-k selection: enforce a per-phase quota (phase = position modulo chunk stride) before filling by score |
| Recall-probe rescue swaps | After the first selection, run a small set of representative probe queries (from a query log) through the retained cache, and swap in high-scoring evicted pairs for low-scoring kept pairs, keeping the total count fixed |
| Merge-downgrade | Wrap the press with a `MergingPress`-style fold: evicted values averaged into the nearest kept neighbour by key cosine similarity |

## What to verify on real models before claiming anything

1. Probe queries must come from a query log for the deployment, not
   from the evaluation questions. Using eval questions for rescue
   would leak the test set into the policy.
2. The stride for phase computation must match the chunking stride
   of the deployment's compression design, or phases are meaningless.
3. Per-head budgets change the memory layout kvpress assumes for
   uniform compression; check `PerLayerCompressionPress` and the
   pipeline's cache-shape handling first.
4. Rescue swaps add one extra attention pass over probe queries.
   Measure that cost against the recall gain; the swap cap (10% of
   budget per head in this repo) is the knob.
5. Re-run the kvpress evaluation CLI (RULER / LongBench) with the
   new press and compare against SnapKVPress and ExpectedAttentionPress
   at matched keep fractions, reporting worst-phase-style splits
   (recall by position modulo stride) alongside averages.
