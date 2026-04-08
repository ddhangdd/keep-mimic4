# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Reproduction of the **KEEP** paper (Elhussein et al., CHIL 2025; arxiv:2510.05049) on MIMIC-IV, as a CS 598 DL4H course project. The end goal is to train KEEP medical-code embeddings and plug them into a GRASP+GRU in-hospital mortality model in PyHealth 2.0, then ablate against the model's default learned embeddings.

The work is structured as **Stories 0–11** in `keep_implementation_plan.md` — that file is the spec, and the section under each story (Context / Instructions / Acceptance criteria) is authoritative. Always re-read the relevant story before implementing.

Status as of 2026-04-07: Stories 0, 1, 2 done. Story 3 is next.

## Environment & commands

There is no `pip install`, `pytest`, or build step at the repo root — everything runs out of the PyHealth venv with absolute paths.

```bash
# The Python interpreter for everything in this repo:
/home/ddhangdd2/keep/PyHealth/.venv/bin/python

# Build the OMOP knowledge graph (Story 2):
/home/ddhangdd2/keep/PyHealth/.venv/bin/python \
    /home/ddhangdd2/keep/keep_pipeline/scripts/build_omop_graph.py

# Run the Story 2 acceptance tests (plain script, not pytest):
/home/ddhangdd2/keep/PyHealth/.venv/bin/python \
    /home/ddhangdd2/keep/keep_pipeline/scripts/test_omop_graph.py
```

Tests in `keep_pipeline/scripts/test_*.py` are **standalone scripts** that print PASS/FAIL and `sys.exit(1)` on failure, not pytest modules. New story tests should follow the same pattern unless you have a reason to switch.

## Layout & what's gitignored

Only `keep_pipeline/`, `README.md`, `keep_implementation_plan.md`, and `.gitignore` are tracked. The following live under the project root but are **not** in git and must be present locally for anything to run:

| Path | What it is |
|---|---|
| `PyHealth/` | PyHealth fork on branch `dev/grasp-full-pipeline` (editable-installed in its `.venv`) |
| `PyHealth/.venv/` | The only Python interpreter used by this project — torch 2.7.1+cu126, pyhealth 2.0.0, networkx, duckdb, node2vec, gensim |
| `keep_reference/` | Read-only clone of `G2Lab/keep`, used as copy-paste source for Stories 4 + 6 |
| `data/` | Athena OMOP vocabulary CSVs (`CONCEPT.csv`, `CONCEPT_ANCESTOR.csv`, `CONCEPT_RELATIONSHIP.csv`, …; ~800 MB, license-restricted) |
| `keep_pipeline/data/`, `keep_pipeline/embeddings/` | Regenerable pipeline outputs (graph pickles, mappings, co-occurrence matrices, embeddings) |
| `KEEP paper.pdf` | Paper PDF (arxiv:2510.05049v1) — gitignored by `*.pdf` but **locally available**. Read it directly when the paper and the G2Lab reference repo disagree; this has already happened ≥3 times (see Gotchas). |

If any of these are missing, point the user at Story 0 in `keep_implementation_plan.md` rather than trying to bootstrap them yourself.

## Architecture (the big picture)

The pipeline is a linear chain of scripts in `keep_pipeline/scripts/`, each consuming Athena CSVs and/or earlier pipeline artifacts and producing pickles / npz / npy / txt files into `keep_pipeline/data/` and `keep_pipeline/embeddings/`. The two-stage KEEP design lives in the `train_node2vec.py → train_keep_glove.py` pair: node2vec generates an initial embedding from the OMOP knowledge graph alone, then a regularized GloVe refines it on the empirical co-occurrence matrix while staying close to the node2vec init via an L2 penalty (paper Eq. 4, p. 5).

The flowchart in `README.md` ("Pipeline architecture") is the canonical map of which script reads which artifact. When adding a new script, match its inputs/outputs to that diagram.

**Execution-order subtlety — Story 4 depends on Story 5b, not Story 2.** The natural reading is "build graph (S2) → train embeddings (S4) → use them downstream", but Story 5b (count filter) has to sit in the middle: node2vec must be trained on the *filtered* graph that only contains MIMIC-IV-relevant concepts, otherwise 95% of the walks are wasted on nodes with no patient data. Practical order after S1: `S2 + S3` (parallel) → `S5` (full-graph cooc to discover which concepts have data) → `S5b` (filter graph + cooc) → `S4` (node2vec on filtered) → `S6` (KEEP GloVe) → `S7`. The reference repo's `_ct_filter` filename suffix is the giveaway.

Cross-cutting invariants worth knowing before touching anything:

- **`concept_to_idx` ordering must be deterministic** (sorted by `concept_id`) because every downstream embedding matrix is row-indexed by it across stories. Don't reorder. Re-running `build_omop_graph.py` is byte-identical for the same reason — the orphan-rescue tie-break uses `ROW_NUMBER() OVER (... ORDER BY min_levels_of_separation, ancestor_concept_id)` so it's reproducible too.
- **DuckDB is used as a query engine over the raw CSVs** (no database file is materialized). Scripts open Athena CSVs directly via `duckdb.read_csv_auto('/abs/path/CONCEPT.csv')` and stream node sets in/out via `CREATE TEMP TABLE` + `executemany`. Follow the same pattern in new scripts.
- **Sparse vs dense roll-up — do not conflate.** Story 3 (ICD→OMOP mapping) does a *sparse* roll-up: each deep ICD-10-CM code maps to its **single nearest** in-graph ancestor. Story 5 (co-occurrence matrix) does a *dense* roll-up: every concept ALSO contributes to **all** of its ancestors in the graph, so a leaf "T2DM" inflates counts at "Diabetes Mellitus", "Metabolic Disease", … up to the root (paper §A.4). The two stories use the same `CONCEPT_ANCESTOR` table but different selection logic — getting them backwards fails silently.

## Gotchas — read before implementing

These are not derivable from the code and bit us during reproduction:

- **Story 2 orphan rescue.** Building the disease graph naively (standard SNOMED Condition descendants of `4274025` within depth ≤5, edges = `min_levels_of_separation = 1`) leaves **42 nodes unreachable** from the root. Cause: some Condition concepts have their direct SNOMED parent in the **Observation** domain, so the parent gets filtered out and the child loses its only incoming edge. `build_omop_graph.py` step 3a patches this with transitive ancestor edges; the paper §A.1.1 doesn't mention it. Expected output: `nodes: 68,396 / edges: 152,347 / leaves: 41,780 / multi-parent: 46,841`.

- **Story 6 — follow the paper, not the reference repo.** `keep_reference/trained_embeddings/our_embeddings/train_glove.py` deviates from the KEEP paper in three places — all should be implemented per the paper:
  - `LAMBD` should be **`1e-3`** (paper Table 6, p. 16), not `1e-5` (reference line 56: `LAMBD = 0.00001`)
  - Optimizer should be **AdamW** (paper Algorithm 1, p. 14), not `Adagrad` (reference line 188: `torch.optim.Adagrad(...)`)
  - Regularization should be **squared L2 norm** (paper Eq. 4, p. 5), not cosine distance (reference defaults `REG_NORM = None` at line 46; the `Glove.forward` branch at line 161-163 then routes through `1 - cosine_similarity` instead of `torch.norm(..., p=reg_norm, dim=1)` at line 165-166). Pass `reg_norm=2` when constructing `Glove` to take the L2 branch.

- **Reference-code duplicate function.** `keep_reference/trained_embeddings/our_embeddings/train_node2vec.py` defines `get_vector_iso()` twice — line 69 (stale 2-arg) and line 85 (working version with mean-vector fallback). The second shadows the first; copy the line-85 version when adapting Story 4.

- **`node2vec 0.5.0` numpy quirk.** Its install metadata declares `numpy<2.0.0`, but it actually works with `numpy 2.2.6` (which PyHealth requires). pip prints a resolver-conflict warning at install time — ignore it. **Do not downgrade numpy** or PyHealth breaks.

- **CUDA channel.** torch was installed from `https://download.pytorch.org/whl/cu126`. `cu121` only ships up to torch 2.5.x; for `torch~=2.7.1` use `cu126`.

## Constants pinned by the paper

When implementing new stories, these values are fixed and should not be re-derived. If the G2Lab reference repo disagrees with anything below, the paper wins (see Gotchas).

- Knowledge graph root: **`4274025`** ("Disease"), NOT `441840` ("Clinical Finding")
- Graph depth limit: **5** levels from root
- Embedding dimension: **100**
- Final downstream task: in-hospital mortality on MIMIC-IV via GRASP+GRU
- **node2vec (Story 4)** — paper Appendix: `dimensions=100`, `walk_length=30`, `num_walks=750`, `p=q=1` (unbiased / DeepWalk-equivalent), `window=10`, `min_count=1`, `batch_words=4096`. Train on the *filtered* graph from Story 5b. Isolated nodes get the mean vector as fallback (mirror line 85 of the reference's `train_node2vec.py`, NOT line 69).
- **Regularized GloVe / KEEP (Story 6)** — paper Table 6 + Algorithm 1: `embedding_dim=100`, `lr=0.05`, `epochs=300`, `batch_size=1024`, `α=0.75`, **`λ=1e-3`**, `X_max = max(50, np.quantile(cooc[cooc > 0], 0.75))`, optimizer **AdamW**, regularization is **squared L2** (`reg_norm=2`) on `(emb_u + emb_v)/2`. Final embeddings: `(emb_u + emb_v)/2`.
- **Phenotyping filter (Story 5)**: a diagnosis code only counts toward a patient's confirmed disease set if it appears **≥2 times** in their full history. Apply this *before* the dense roll-up.

## PyHealth integration anchors (Stories 8 / 9 / 11)

The PyHealth fork is editable-installed in the venv, so edits under `PyHealth/pyhealth/...` take effect immediately. Specific touchpoints — these required cross-file tracing the first time and are easy to re-derive incorrectly:

- `PyHealth/pyhealth/models/grasp.py` line 468 constructs `EmbeddingModel(dataset, embedding_dim)`. Story 8 threads `pretrained_emb_path` / `freeze_pretrained` / `normalize_pretrained` through GRASP's `__init__` and into this call site. No other GRASP changes needed — the forward pass is embedding-agnostic.
- `PyHealth/pyhealth/models/embedding.py`: `EmbeddingModel.__init__` is at line 141 and already accepts `pretrained_emb_path` at line 145; the call site that invokes pretrained loading is at line 186-199. `init_embedding_with_pretrained` is at line 66 and calls `_iter_text_vectors` (line 25) to parse the `<token> v1 v2 ... v100` text format.
- **Token-key invariant — silent failure mode.** `keep_vectors.txt` token strings are OMOP concept IDs as decimal strings (e.g. `"201826"`). They MUST match the keys in `SequenceProcessor.code_vocab`. If your task (Story 9) outputs raw ICD-10-CM codes instead of OMOP concept IDs, the pretrained lookup matches zero rows and every embedding stays at its random init. `init_embedding_with_pretrained` returns a `loaded` count, but the call site at `embedding.py:192` **discards the return value** and neither side logs anything — the failure is completely silent. Always spot-check `sample_dataset.samples[0]["conditions"]` against keys in `keep_vectors.txt` after wiring the task.

## Test harness convention

`keep_pipeline/scripts/test_omop_graph.py` is the template for story acceptance tests: a standalone script with module-level fixture loading and a `@check("test name")` decorator that catches `AssertionError`, prints `PASS`/`FAIL`/`ERROR`, accumulates results in `_results`, prints a summary, and `sys.exit(1)` on any failure. New story tests should follow the same pattern (no pytest dependency, no test discovery, just `python test_<story>.py`). The decorator's `def _():` body is intentional — function names are throwaway, the `@check("...")` string is the human-readable name.
