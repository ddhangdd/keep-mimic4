# Per-story invariants checklist

Cross-cutting invariants distilled from `/home/ddhangdd2/keep/CLAUDE.md` (the "Cross-cutting invariants", "Gotchas", and "Constants pinned by the paper" sections), keyed by story number. When generating a test file, every entry under the relevant story should become a `@check` *in addition to* the bullets in the story's `### Acceptance criteria` section.

If you implement a story not yet listed here, **add the entry after generating the test** so the next run benefits.

---

## Story 2: Build the OMOP knowledge graph

Already implemented. Test file is `test_omop_graph.py` — use it as the canonical reference for what a finished test file looks like.

Key invariants asserted there:
- Graph is `nx.DiGraph`, DAG, no self-loops
- Node count == 68,396; edge count == 152,347 (= 152,340 base + 7 rescue)
- Root `4274025` is the only zero-in-degree node; every node reachable from root
- `concept_to_idx` ordering: keys sorted by `concept_id`, values are `0..N-1`
- `idx_to_concept` is exact inverse
- DuckDB cross-check: node set == standard Conditions ≤ depth 5 from root
- Every node has `domain_id = 'Condition'` and `standard_concept = 'S'` in CONCEPT.csv
- Orphan-rescue regression: concepts 4234597 and 44807040 both reachable

---

## Story 3: Build the ICD-10-CM to OMOP concept mapping

Output: `icd10cm_to_omop: Dict[str, int]` pickled to `OUT_DIR / "icd10cm_to_omop.pkl"`.

Invariants beyond the acceptance bullets:

- **Sparse roll-up, not dense.** Each ICD-10-CM key maps to *exactly one* in-graph OMOP concept ID — the **single nearest** in-graph ancestor when the direct map is too deep. Assert `all(isinstance(v, int) for v in icd10cm_to_omop.values())` and that values are scalar, not lists/sets. (CLAUDE.md: "Sparse vs dense roll-up — do not conflate.")
- **Every value lives in the Story 2 graph.** Load `concept_to_idx.pkl` and assert `set(icd10cm_to_omop.values()).issubset(set(concept_to_idx.keys()))`. Zero exceptions allowed — this is the explicit acceptance bullet.
- **All keys are ICD-10-CM strings.** Assert `all(isinstance(k, str) for k in icd10cm_to_omop.keys())`. ICD-10-CM codes look like `"E119"`, `"I251"` — short alphanumeric, no dot-separators after the first letter+digits.
- **DuckDB ground-truth cross-check.** Re-derive the un-rolled mapping directly from `CONCEPT_RELATIONSHIP.csv` (`relationship_id = 'Maps to'`, source vocabulary `ICD10CM`, target a standard Condition) and assert the rolled-up dict's keyset is a superset of (or equal to) the keyset implied by that join. Catches drift between the spec and the implementation.
- **Roll-up depth bound.** For every value in the dict, assert it's at depth ≤ 5 from root `4274025` per `CONCEPT_ANCESTOR.csv`. The whole point of roll-up is to enforce this — verify it.

---

## Story 4: Run Node2Vec (KEEP Stage 1)

Output: `node2vec_embeddings.npy` of shape `(num_filtered_concepts, 100)`.

Invariants:

- **Trained on the FILTERED graph from Story 5b, not the full Story 2 graph.** Load `omop_graph_filtered.pkl` (Story 5b output) and assert `embeddings.shape[0] == filtered_graph.number_of_nodes()`. CLAUDE.md flags this as the most-likely-to-bite execution-order surprise.
- **Embedding dimension == 100.** `assert embeddings.shape[1] == 100` (paper Appendix).
- **Row alignment via the filtered `concept_to_idx`.** Row `i` corresponds to the concept with index `i` in the *filtered* `concept_to_idx`. Test by spot-checking: load `concept_to_idx_filtered.pkl`, pick a known concept, look it up in the dict, and assert the embedding row at that index has nonzero norm.
- **No NaN/Inf.** `assert np.isfinite(embeddings).all()`.
- **Isolated-node fallback.** If the filtered graph contains isolated nodes (in-degree + out-degree == 0), they should have received the *mean* vector, not random init or zeros. CLAUDE.md: "mirror line 85 of the reference's `train_node2vec.py`, NOT line 69." Stub as a TODO check (semantic) unless you can identify an isolated node deterministically.
- **dtype is float32 or float64.** Whichever the paper / reference uses — assert it's not int or object.

---

## Story 5: Build the co-occurrence matrix from MIMIC-IV

Output: a `(num_concepts, num_concepts)` co-occurrence matrix as `.npz` (sparse) or `.npy`.

Invariants:

- **Symmetric.** `assert (cooc != cooc.T).nnz == 0` (sparse) or `np.array_equal(cooc, cooc.T)` (dense).
- **Dense roll-up.** Each leaf concept contributes to *all* of its ancestors in the graph, not just the nearest one. This is the opposite of Story 3's sparse roll-up. Hard to verify directly without re-running the rollup, but you can spot-check: pick a leaf concept that appears in MIMIC-IV, find its ancestors in the graph, and assert the cooc count for those ancestors is ≥ the leaf's own count.
- **Phenotyping filter applied before roll-up.** A diagnosis only counts toward a patient if it appears ≥2 times in their full history. Hard to verify post-hoc; leave as a TODO with a pointer to the spec.
- **Non-negative integer counts.** `assert (cooc.data >= 0).all()` and `assert cooc.dtype.kind in 'iu'`.
- **Row count matches `concept_to_idx`.** `assert cooc.shape == (len(concept_to_idx), len(concept_to_idx))`.

---

## Story 5b: Count filter — reduce graph to MIMIC-IV-relevant concepts

Outputs: `omop_graph_filtered.pkl`, `concept_to_idx_filtered.pkl`, `idx_to_concept_filtered.pkl`, filtered cooc matrix.

Invariants:

- **Filtered graph is a subgraph of the full Story 2 graph.** Load both, assert `set(filtered.nodes) <= set(full.nodes)` and `set(filtered.edges) <= set(full.edges)`.
- **`concept_to_idx_filtered` is still sorted-by-concept-id.** Same determinism rule as Story 2 — every downstream embedding matrix is row-indexed by it. Re-running must be byte-identical.
- **Filtered cooc matrix shape matches filtered concept_to_idx.** `assert cooc_filtered.shape[0] == len(concept_to_idx_filtered)`.
- **Filtered node set still reachable from root** (the filter shouldn't accidentally orphan anything that's still in the graph) — or alternatively, the filter explicitly drops the root, in which case document that.

---

## Story 6: Train regularized GloVe (KEEP Stage 2)

Output: `keep_embeddings.npy` of shape `(num_filtered_concepts, 100)`.

Invariants — **the paper wins over the reference repo here**, in three places:

- `λ` (LAMBD) == **`1e-3`**, not `1e-5`. The reference's `train_glove.py:56` uses `1e-5`; the paper Table 6 says `1e-3`. Hard to verify from the artifact alone — leave as a TODO that points at the training script's hyperparameter line.
- Optimizer == **AdamW**, not Adagrad. Same situation — verify by reading the script source, not the artifact.
- Regularization == **squared L2** on `(emb_u + emb_v)/2`, not cosine distance. Pass `reg_norm=2` to the `Glove` constructor. Same — script-level check.
- **Final embeddings = `(emb_u + emb_v)/2`.** Not `emb_u` alone. Hard to verify post-hoc.
- **Shape and finiteness.** `(num_filtered_concepts, 100)`, `np.isfinite(...).all()`, dtype float.
- **Row alignment with filtered `concept_to_idx`.** Same check as Story 4.
- **Pinned hyperparams** for the Story 6 script (assert by reading the script if necessary, not the .npy): `lr=0.05`, `epochs=300`, `batch_size=1024`, `α=0.75`, `X_max = max(50, np.quantile(cooc[cooc>0], 0.75))`.

---

## Story 7: Export KEEP vectors to PyHealth text format

Output: `keep_vectors.txt` — one line per concept, format `<token> v1 v2 ... v100`.

Invariants:

- **Token format is decimal OMOP concept IDs as strings**, not ICD codes. CLAUDE.md flags this as a silent-failure mode for Story 9 — `init_embedding_with_pretrained` matches by string equality against `code_vocab` keys, and a mismatch silently leaves every embedding at random init.
- **One line per concept in `concept_to_idx_filtered`.** `assert num_lines == len(concept_to_idx_filtered)`.
- **Vector length == 100** for every line. Parse the file and assert.
- **Tokens parse as positive integers.** `assert all(int(token) > 0 for token in tokens)`.
- **Token set equals `set(concept_to_idx_filtered.keys())` after `int(...)` coercion.**
- **Round-trip: loading the file and stacking vectors recovers the source `.npy`** (within float-formatting tolerance, e.g. `np.allclose(reloaded, source, atol=1e-5)`).

---

## Story 8: Modify GRASP to accept pretrained embeddings

This story modifies `PyHealth/pyhealth/models/grasp.py` and threads new kwargs into `EmbeddingModel`. There's no artifact to test — the test file should instead instantiate a tiny GRASP model with `pretrained_emb_path=` pointing at a fixture and assert the embedding weights match the file's contents.

Invariants:

- **Signature change is non-breaking.** Construct `GRASP(...)` *without* the new kwargs and assert it still works (defaults preserve the pre-Story-8 behavior).
- **`pretrained_emb_path` actually loads.** Construct with a small fixture vectors.txt, then assert `model.embedding.<weight_attr>[i]` matches the corresponding row of the fixture.
- **`freeze_pretrained=True` freezes gradients.** Assert `requires_grad == False` on the embedding weight.
- **`normalize_pretrained=True` L2-normalizes rows.** Assert each row of the loaded weight has unit norm (within tolerance).
- **Token-key mismatch surfaces clearly.** This is the silent-failure mode CLAUDE.md flags. The test should construct a fixture whose tokens *don't* match `code_vocab`, then either (a) assert the load reports `loaded == 0` (if you've added logging) or (b) document that this case currently fails silently and the test suite for Story 9 catches it.

---

## Story 9: Create a MIMIC-IV mortality task with OMOP concept IDs

This is task scaffolding inside PyHealth, not an artifact-producing pipeline script. The test should validate that the task's sample dataset uses OMOP concept ID keys (not raw ICD-10-CM).

Invariants:

- **`sample_dataset.samples[0]["conditions"]` keys are OMOP concept ID strings.** Spot-check the first sample and assert each condition code is a string of digits that parses as `int > 0`. CLAUDE.md flags this as the silent-failure pivot point — if the task emits ICD codes instead, every Story 8 pretrained embedding stays at random init and you don't notice until the ablation comes back showing zero improvement.
- **Cross-check against `keep_vectors.txt`.** Load `keep_vectors.txt`, collect token strings, and assert that *most* of `sample_dataset`'s condition codes appear in that token set. "Most" because some MIMIC codes may not have rolled up to anything in the filtered graph; "100%" is too strict, but "0%" is a silent failure. A reasonable bar is ≥80% coverage — calibrate after first run.
- **The task runs end-to-end on a tiny slice** (5–10 patients) without raising. Smoke test.

---

## Story 10: Run ablation experiments

This is an experiment runner, not a unit-testable artifact producer. Skip the test scaffolding for this story unless the user explicitly asks — instead, the "test" is a notebook/results file the user inspects manually.

If the user insists, the test should validate the *output metrics file* (e.g. AUROC values for "with KEEP" vs "without KEEP" arms) is well-formed JSON with the expected fields.

---

## Story 11: Write conversion utility and unit tests (PyHealth PR deliverables)

This story is itself a test-writing story, so the "scaffold a test" loop is recursive. The skill should still help — generate a pytest-style test file (this is the one exception to the "no pytest" rule, because Story 11 is explicitly the PyHealth PR deliverable and PyHealth uses pytest).

Invariants depend on what the conversion utility does — read the Story 11 spec carefully before assuming.
