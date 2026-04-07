# KEEP Implementation Plan for Claude Code

## Project overview

We are implementing the KEEP paper (Knowledge-preserving and Empirically refined Embedding Process, CHIL 2025) for a CS 598 DL4H course project at UIUC. The goal is to generate KEEP medical code embeddings from MIMIC-IV data and plug them into a GRASP+GRU model in PyHealth 2.0 for in-hospital mortality prediction.

**What KEEP does:** A two-stage embedding process for medical diagnosis codes.
- Stage 1: Run node2vec on an OMOP knowledge graph (is-a hierarchy of diseases) to get initial embeddings that encode ontological structure.
- Stage 2: Refine those embeddings using a regularized GloVe model trained on patient co-occurrence data. The regularization penalty (controlled by lambda) prevents embeddings from drifting too far from their Stage 1 positions, which is especially important for rare diseases.

**Repos:**
- PyHealth fork: `https://github.com/lookman-olowo/PyHealth/tree/dev/grasp-full-pipeline`
- KEEP reference: `https://github.com/G2Lab/keep`
- KEEP paper PDF is in the project knowledge

**Key facts:**
- Dataset: MIMIC-IV (PhysioNet, credentialed access held by all team members)
- Knowledge graph root: "Disease" concept ID **4274025** (NOT "Clinical Finding" 441840)
- Graph depth limit: 5 levels from root
- Embedding dimension: 100
- Final output: a text file of OMOP concept ID → 100-dim vector, loaded by PyHealth's EmbeddingModel
- Downstream task: in-hospital mortality prediction using GRASP+GRU

---

## Status & environment (last updated 2026-04-07)

**Story 0: COMPLETED 2026-04-07.** Environment ready, all dependencies installed and verified, OMOP validation queries passing. Execution log lives in `~/.claude/plans/curried-stargazing-phoenix.md` Part 2.

### Canonical paths

| Path | Purpose |
|---|---|
| `/home/ddhangdd2/keep/` | Project root (working directory) |
| `/home/ddhangdd2/keep/data/` | Athena CSVs (CONCEPT.csv, CONCEPT_ANCESTOR.csv, CONCEPT_RELATIONSHIP.csv, ...) |
| `/home/ddhangdd2/keep/PyHealth/` | PyHealth fork clone, branch `dev/grasp-full-pipeline` |
| `/home/ddhangdd2/keep/PyHealth/.venv/` | Python venv (activate or call `.venv/bin/python` directly) |
| `/home/ddhangdd2/keep/keep_reference/` | KEEP reference repo (read-only, source of copy-paste material for Stories 4 + 6) |
| `/home/ddhangdd2/keep/keep_pipeline/scripts/` | All KEEP pipeline scripts (`build_omop_graph.py`, etc.) |
| `/home/ddhangdd2/keep/keep_pipeline/data/` | Pipeline data outputs (graphs, mappings, co-occurrence matrices) |
| `/home/ddhangdd2/keep/keep_pipeline/embeddings/` | Pipeline embedding outputs (node2vec, KEEP vectors) |

### Verified environment (built in Story 0)

- Python 3.12.3 (system), in venv at `/home/ddhangdd2/keep/PyHealth/.venv/`
- torch 2.7.1+cu126, CUDA available, **NVIDIA GeForce RTX 4070 SUPER**
- numpy 2.2.6, pandas 2.3.3, scipy 1.17.1, networkx 3.6.1
- duckdb 1.5.1, gensim 4.4.0, node2vec 0.5.0
- pyhealth 2.0.0 (editable install from `/home/ddhangdd2/keep/PyHealth`)

**CUDA channel note:** torch was installed from `https://download.pytorch.org/whl/cu126`. The `cu121` channel only ships up to torch 2.5.x; for `torch~=2.7.1` use `cu126` (or whichever `cu12x` channel is current at https://pytorch.org/get-started/locally/).

**numpy version conflict (cosmetic — do NOT downgrade):** `node2vec 0.5.0` declares `numpy<2.0.0` in its install metadata, but actually works with `numpy 2.2.6` (smoke-tested end-to-end in Story 0: graph build → walks → skip-gram fit → vector extraction). pip prints a resolver-conflict warning at install time; ignore it. **Do not downgrade numpy** — it would violate PyHealth's `numpy~=2.2.0` constraint and break PyHealth.

---

## Story 0: Repository setup and dependency check

### Context
Before any implementation, we need the PyHealth fork cloned and the KEEP reference code available. We also need to verify that key dependencies are installable.

### Instructions
1. Clone the PyHealth fork: `git clone --branch dev/grasp-full-pipeline https://github.com/lookman-olowo/PyHealth.git`
2. Clone the KEEP reference repo: `git clone https://github.com/G2Lab/keep.git`
3. Verify these Python packages are available (install if needed):
   - `networkx` (graph construction)
   - `node2vec` (pip install node2vec)
   - `torch` (GloVe training)
   - `numpy`, `pandas`, `scipy`
   - `duckdb` (for querying OMOP vocabulary files)
   - `gensim` (used by node2vec internally)
4. Verify the PyHealth fork's GRASP model exists at `pyhealth/models/grasp.py`
5. Verify the EmbeddingModel at `pyhealth/models/embedding.py` has a `pretrained_emb_path` parameter in its `__init__` (it should, around line 145)

### Acceptance criteria — ALL PASSED 2026-04-07
- [x] Both repos cloned successfully (PyHealth at `/home/ddhangdd2/keep/PyHealth`, KEEP reference at `/home/ddhangdd2/keep/keep_reference`)
- [x] All dependencies install without errors (see "Status & environment" block above for the full version list)
- [x] `from pyhealth.models import GRASP` succeeds; signature is `(self, dataset, static_key, embedding_dim, hidden_dim, **kwargs)` — the upstream branch does NOT yet have the pretrained params, so Story 8 is still required
- [x] `from node2vec import Node2Vec` succeeds; smoke test (graph → walks → fit → vector extraction) passes
- [x] DuckDB validation: concept 4274025 is `('Disease', 'Condition', 'SNOMED', 'Disorder', 'S')`; 68,396 standard condition concepts within depth 5 from root

---

## Story 1: Download and inspect OMOP vocabulary files

### Context
KEEP's knowledge graph is built from OMOP vocabulary flat files downloaded from Athena (https://athena.ohdsi.org). We need three specific CSV files. These are large files (~750MB total for SNOMED+ICD10CM) but are just CSV tables that can be queried with DuckDB locally.

### Note — VALIDATED 2026-04-07
Athena CSV files are present at `/home/ddhangdd2/keep/data/`. Validation was first performed in a prior session and was **re-confirmed during Story 0** (2026-04-07) using DuckDB:
- Concept 4274025: `('Disease', 'Condition', 'SNOMED', 'Disorder', 'S')` ✓ (also confirmed by paper §A.1.1, p. 15: *"the root node, 'Disease' (concept ID: 4274025)"*)
- **68,396 standard condition concepts within depth 5** ✓ (depth distribution: 0→1, 1→167, 2→4,247, 3→16,930, 4→24,900, 5→22,151)
- ICD10CM "Maps to" relationships present (not re-counted in Story 0 — will be exercised by Story 3)
- File sizes: CONCEPT.csv (189 MB), CONCEPT_ANCESTOR.csv (158 MB), CONCEPT_RELATIONSHIP.csv (396 MB) ✓

The validation queries can be re-run via the DuckDB snippet in Step 0.9 of `~/.claude/plans/curried-stargazing-phoenix.md`. A standalone `keep_pipeline/scripts/validate_athena.py` is optional and not blocking for downstream stories.

### Instructions
1. Go to https://athena.ohdsi.org and download the vocabulary bundle selecting **SNOMED and ICD10CM only**. (The proposal listed ICD10PCS, RxNorm, and NDC as well, but KEEP operates exclusively on condition/disease concepts — those other vocabularies are not used anywhere in the pipeline.)
2. Extract the ZIP. The key files are:
   - `CONCEPT.csv` — master table of all medical concepts (concept_id, concept_name, domain_id, vocabulary_id, etc.)
   - `CONCEPT_ANCESTOR.csv` — hierarchical relationships (ancestor_concept_id, descendant_concept_id, min_levels_of_separation, max_levels_of_separation)
   - `CONCEPT_RELATIONSHIP.csv` — cross-vocabulary mappings (concept_id_1, concept_id_2, relationship_id). We need "Maps to" relationships to go from ICD-10-CM → OMOP standard concepts.
3. Write a quick validation script using DuckDB to verify:
   - Concept 4274025 exists in CONCEPT.csv and its concept_name is "Disease"
   - CONCEPT_ANCESTOR.csv contains rows where ancestor_concept_id = 4274025
   - Count how many condition-domain concepts are within 5 levels of 4274025

### Acceptance criteria
- All three CSV files downloaded and accessible locally
- Concept 4274025 ("Disease") confirmed present
- A count of concepts within depth 5 — should be **~68,396 unfiltered** from the raw SNOMED hierarchy. (The paper never states a specific node count. The 5,686 number found in the KEEP repo's `configs.py` is the UK Biobank count after filtering to only concepts that appear in patient data. Your MIMIC-IV count will differ.)

### Example validation query (DuckDB)
```sql
-- Check root concept exists
SELECT concept_id, concept_name, domain_id 
FROM 'CONCEPT.csv' 
WHERE concept_id = 4274025;

-- Count condition concepts within depth 5
SELECT COUNT(DISTINCT ca.descendant_concept_id)
FROM 'CONCEPT_ANCESTOR.csv' ca
JOIN 'CONCEPT.csv' c ON ca.descendant_concept_id = c.concept_id
WHERE ca.ancestor_concept_id = 4274025
  AND ca.min_levels_of_separation <= 5
  AND c.domain_id = 'Condition'
  AND c.standard_concept = 'S';
```

---

## Story 2: Build the OMOP knowledge graph

### Context
This builds the NetworkX directed graph that node2vec will walk on. The graph contains only condition (disease) concepts connected by "is-a" hierarchical relationships, limited to 5 levels of depth from the root "Disease" (concept 4274025).

The KEEP paper (Appendix A.1.1) says: filter for condition concepts, use CONCEPT_ANCESTOR where min_levels_of_separation = 1 to get parent-child edges, and exclude concepts more than 5 levels from root.

### Instructions
1. Create a script `build_omop_graph.py`
2. Using DuckDB, query CONCEPT.csv to get all standard condition concepts
3. Using CONCEPT_ANCESTOR.csv, find all concepts where ancestor_concept_id = 4274025 and min_levels_of_separation <= 5. These are the nodes.
4. Build edges: for each pair in CONCEPT_ANCESTOR where min_levels_of_separation = 1 AND both ancestor and descendant are in our node set, add a directed edge from parent → child
5. Construct a NetworkX DiGraph
6. Save the graph as a pickle file
7. Also save a `concept_to_idx` dictionary mapping each concept_id to an integer index (0 to N-1), and an `idx_to_concept` reverse mapping. Save these as pickles too.
8. Print graph stats: number of nodes, number of edges, verify root node has no parents

### Acceptance criteria
- NetworkX DiGraph with **~68,396 nodes** (the full unfiltered SNOMED condition hierarchy at depth ≤ 5). This will be reduced later in Story 5b after intersecting with MIMIC-IV patient data.
- Graph is a DAG (directed acyclic graph) with root 4274025
- Every node is reachable from the root
- Saved as `omop_graph.pkl`, `concept_to_idx.pkl`, `idx_to_concept.pkl`

### Key detail
The concept_to_idx mapping is critical — it defines the row ordering for all embedding matrices. Node2vec will produce vectors indexed by this mapping, the co-occurrence matrix will use this mapping, and the final exported text file will use the concept_id strings from this mapping.

**Important:** This initial graph and mapping will be rebuilt after Story 5b (count filtering). The final graph will only contain concepts that actually appear in MIMIC-IV patient data — likely a few thousand nodes, not 68K. Stories 2 and 4 (node2vec) will need to be re-run on the filtered graph. See Story 5b for details.

---

## Story 3: Build the ICD-10-CM to OMOP concept mapping

### Context
MIMIC-IV stores diagnoses as ICD-10-CM codes (e.g., "E119", "I251"). KEEP operates on OMOP standard concept IDs. We need a mapping table that translates each ICD-10-CM code to its corresponding OMOP concept, then rolls it up to depth 5 if needed.

**Dependency: Story 1 (Athena files), NOT Story 2.** The core mapping query only needs CONCEPT_RELATIONSHIP.csv and CONCEPT.csv from the Athena download. The roll-up filtering at the end (remapping deep concepts to their depth-5 ancestors) needs the graph node set from Story 2, but that's a small final step — the bulk of Story 3 can start as soon as Story 1 is done.

### Instructions
1. Create a script `build_icd_to_omop_mapping.py`
2. From CONCEPT_RELATIONSHIP.csv, extract all rows where `relationship_id = 'Maps to'` and the source concept is ICD10CM vocabulary
3. Join with CONCEPT.csv to get the target standard concept's domain and check it's a Condition
4. For target concepts that are deeper than 5 levels from root 4274025, roll up to their nearest ancestor at depth ≤ 5. Use CONCEPT_ANCESTOR to find the closest ancestor that's in our graph node set (from Story 2). **This roll-up step is the only part that needs Story 2's output — the rest of Story 3 is independent.**
5. Output: a dictionary `icd10cm_to_omop: Dict[str, int]` mapping ICD-10-CM code strings → OMOP concept IDs that are in our graph
6. Save as pickle

### Acceptance criteria
- Mapping covers the majority of ICD-10-CM codes found in MIMIC-IV diagnoses_icd table
- Every target OMOP concept ID exists in the graph from Story 2
- Codes deeper than level 5 are correctly rolled up (spot-check a few)

### Roll-up logic
The KEEP paper says: "Instead of excluding codes more than five levels away from the root node, we implement a roll-up procedure that maps each code to its parent codes present in the graph." For the co-occurrence matrix specifically, they do a "dense roll-up" — mapping a deep code to ALL of its ancestors in the graph, not just the nearest one. For the knowledge graph itself, codes beyond depth 5 are simply excluded.

---

## Story 4: Run Node2Vec (KEEP Stage 1)

### Context
Node2vec generates initial embeddings by simulating biased random walks on the knowledge graph, then training a skip-gram model on those walks. This is Stage 1 of KEEP.

**Important: Run this on the FILTERED graph from Story 5b, not the full 68K-node graph from Story 2.** Running node2vec on the full graph would waste compute on tens of thousands of concepts that never appear in patient data. The KEEP repo confirms this — their files all use the `_ct_filter` suffix.

Reference code: `keep/trained_embeddings/our_embeddings/train_node2vec.py`

### Instructions
1. Create a script `train_node2vec.py`
2. Load the **filtered** NetworkX graph from Story 5b
3. Run node2vec with the paper's exact hyperparameters:
   - `dimensions=100`
   - `walk_length=30`
   - `num_walks=750`
   - `p=1, q=1` (unbiased random walk, equivalent to DeepWalk)
   - `workers=4` (adjust based on your machine)
4. Fit the model: `model = node2vec.fit(window=10, min_count=1, batch_words=4096)`
5. Extract the embedding matrix: for each concept in concept_to_idx order, get its vector from the fitted model. If a concept has no vector (isolated node), use the mean vector as fallback (this is what the reference code does at line 85-94).
6. Save the resulting numpy matrix of shape `(num_concepts, 100)` as `node2vec_embeddings.npy`
7. Print: embedding shape, a few sample concept names and their nearest neighbors by cosine similarity (sanity check that siblings are close)

### Acceptance criteria
- Output matrix shape is `(num_filtered_concepts, 100)` where num_filtered_concepts matches Story 5b's filtered graph
- Cosine similarity sanity check: T1DM and T2DM should be closer to each other than to random diseases
- Saved as `node2vec_embeddings.npy`
- Runtime should be under 30 minutes on CPU

### Reference code adaptation
The KEEP repo's `train_node2vec.py` loads a pre-built graph from a pickle at line 100-101. Replace that with loading your `omop_graph_filtered.pkl` from Story 5b. The rest of the script (lines 109-133) is directly usable — it runs node2vec, builds an index mapping, and extracts vectors in the right order.

**Reference-code gotcha — duplicate `get_vector_iso`:** `train_node2vec.py` defines `get_vector_iso()` **twice** — first at line 69 (a stale 2-arg signature without `index_mapping`/`mean_vector`) and again at line 85 (the working version with the mean-vector fallback). The second definition shadows the first; copy the line-85 version and ignore the line-69 stub.

**numpy / node2vec version quirk:** `node2vec 0.5.0`'s install metadata declares `numpy<2.0.0`, but it actually works with `numpy 2.2.6` (verified end-to-end in Story 0). pip prints a resolver-conflict warning at install time; ignore it. **Do not downgrade numpy** — that would break PyHealth's `numpy~=2.2.0` constraint.

---

## Story 5: Build the co-occurrence matrix from MIMIC-IV

### Context
The co-occurrence matrix counts how many patients have both disease i and disease j in their complete medical history. This matrix feeds into GloVe training in Stage 2.

Key rules from the paper:
- A diagnosis must appear ≥2 times in a patient's history to count (phenotyping filter)
- Co-occurrence is across complete patient history, not per-visit
- The paper uses a "dense roll-up": deep codes are mapped to ALL their ancestors in the graph

### Instructions
1. Create a script `build_cooccurrence_matrix.py`
2. Load MIMIC-IV's `diagnoses_icd` table (from the MIMIC-IV data files or via PyHealth's MIMIC4Dataset)
3. For each patient:
   a. Collect all their ICD-10-CM diagnosis codes across all admissions
   b. Map each code to OMOP concept ID(s) using the mapping from Story 3
   c. Apply the dense roll-up: each OMOP concept maps to itself AND all its ancestors in the graph (use CONCEPT_ANCESTOR). This creates multiple entries per original code.
   d. Count occurrences of each rolled-up concept. Keep only concepts with count ≥ 2.
   e. The patient's confirmed disease set = all concepts passing the ≥2 filter
4. Build the V×V co-occurrence matrix: for each patient, for every pair (i,j) in their confirmed disease set where i≠j, increment X[i][j] by 1
5. Use a sparse matrix (scipy.sparse) since most pairs will be zero
6. Save as `cooccurrence_matrix.npz` (sparse) or `cooccurrence_matrix.npy` (dense, if it fits in memory)
7. Print: matrix shape, sparsity percentage, top-10 most frequent co-occurring pairs with their names

### Acceptance criteria
- Matrix shape is `(V, V)` where V = number of nodes from Story 2 (full ~68K graph)
- Matrix is symmetric (X[i][j] == X[j][i])
- Diagonal is zero (a disease doesn't co-occur with itself)
- Top co-occurring pairs make medical sense (e.g., hypertension + diabetes should be high)
- The 75th percentile of non-zero values is computed and printed (this becomes X_max for GloVe)
- **Record which concept IDs have at least one non-zero entry** — this feeds into Story 5b's count filter

### Important detail about dense roll-up
From the paper Appendix A.4: "we implement a roll-up procedure that maps each code to its parent codes present in the graph. Adopting a dense roll-up approach, we map every code to all of its parents, creating multiple entries when a code has multiple parent nodes."

This means if a patient has "Type 2 Diabetes" (a leaf node), it also counts as having "Diabetes Mellitus" (parent) and "Metabolic Disease" (grandparent) and so on up to the root. This inflates co-occurrence counts for higher-level concepts, which is intentional.

---

## Story 5b: Count filter — reduce graph to MIMIC-IV-relevant concepts

### Context
The full SNOMED hierarchy at depth 5 has ~68,396 concepts, but most of them never appear in MIMIC-IV patient records. The KEEP repo applies a "count filter" (visible in the `_ct_filter` suffix on all their file paths) that discards concepts with zero patients. For UK Biobank, this reduced 68K → 5,686. For MIMIC-IV (ICU patients, narrower disease range), expect an even smaller number.

This step is not explicitly described in the paper but is clearly present in the KEEP repo's pipeline. Without it, you'd train node2vec and GloVe on a graph where most nodes have no patient data — wasting compute and producing meaningless embeddings for unused concepts.

### Instructions
1. Create a script `apply_count_filter.py`
2. From Story 5's co-occurrence matrix, identify which concepts have at least one non-zero co-occurrence entry (i.e., they appeared in at least one patient's confirmed disease set)
3. Build a **filtered node set**: only concepts that appear in patient data, plus their ancestors up to root (to keep the graph connected)
4. Rebuild the graph: subgraph of Story 2's graph using only filtered nodes
5. Rebuild `concept_to_idx` and `idx_to_concept` for the filtered set
6. Save as `omop_graph_filtered.pkl`, `concept_to_idx_filtered.pkl`, `idx_to_concept_filtered.pkl`
7. Rebuild the co-occurrence matrix using the new (smaller) index mapping
8. Print: filtered node count, compare to unfiltered count

### Acceptance criteria
- Filtered graph is significantly smaller than 68K (likely in the low thousands)
- Filtered graph is still connected (root can reach all nodes)
- Co-occurrence matrix is resized to match the filtered node count
- All downstream stories (4 re-run, 6, 7) use the filtered outputs

### Why this matters
After this step, **re-run Story 4 (node2vec) on the filtered graph**. Node2vec random walks on a 68K-node graph where 95% of nodes have no data is wasteful and may produce poor embeddings. The filtered graph gives node2vec a much more meaningful topology to walk on.

### Pipeline adjustment
The practical execution order becomes:
1. Stories 2, 3 (build full graph + mapping) — in parallel after S1
2. Story 5 (co-occurrence using full graph mapping) — identifies which concepts have data
3. Story 5b (count filter) — reduces graph to data-relevant concepts
4. Story 4 (node2vec on filtered graph) — now runs on the small, meaningful graph
5. Story 6 (GloVe) — uses filtered co-occurrence + filtered node2vec

---

## Story 6: Train regularized GloVe (KEEP Stage 2)

### Context
This is the core KEEP novelty. Initialize a GloVe model with the node2vec embeddings from Story 4, then train on the co-occurrence matrix from Story 5 with a regularization penalty that prevents embeddings from drifting too far from their node2vec starting positions.

Reference code: `keep/trained_embeddings/our_embeddings/train_glove.py` — the `Glove` class (lines 130-174) and `train_glove` function (lines 177-228) are directly reusable.

### Instructions
1. Create a script `train_keep_glove.py`
2. Copy the `GloveDataset` class and `Glove` model class from the KEEP reference code (`train_glove.py` lines 108-174). These implement the regularized GloVe objective.
3. Load:
   - **Filtered** co-occurrence matrix from Story 5b
   - Node2vec embeddings from Story 4 (trained on filtered graph — this becomes `embeddings_init`)
4. Compute X_max: `X_max = max(50, np.quantile(cooc_matrix[cooc_matrix > 0], 0.75))`
5. Train with the paper's hyperparameters:
   - `embedding_dim = 100`
   - `learning_rate = 0.05`
   - `num_epochs = 300`
   - `batch_size = 1024`
   - `alpha = 0.75`
   - `lambda = 0.001` (regularization strength)
   - Set `INIT_EMBEDDING = True` and `REGULARIZATION = True`
6. For the regularization norm: set `reg_norm=2` to use L2 norm matching the paper's equation. The reference code defaults to cosine distance when reg_norm=None, but the paper specifies L2.
7. The optimizer: the paper says AdamW but the reference code uses Adagrad. Try AdamW first to match the paper. If results look off, try Adagrad as a fallback.
8. After training, extract final embeddings: `combined = (embeddings_u + embeddings_v) / 2` (this is the `load_glove_embeddings` function at line 230-234 of the reference)
9. Save as `keep_embeddings.npy`

### Acceptance criteria
- Output shape: `(num_filtered_concepts, 100)`
- Training loss decreases over epochs (print every 10 epochs)
- Regularization loss is non-zero (confirms regularization is active)
- Sanity check: cosine similarity between KEEP vectors for known related diseases (e.g., T2DM and Obesity) should be higher than random pairs
- Runtime: under 2 hours on a single GPU (per the paper)

### Key discrepancies in reference code to watch for
1. Line 56: `LAMBD = 0.00001` — this is wrong, use **0.001**. Two independent sources confirm: (a) paper **Table 6, p. 16** lists `λ = 1×10⁻³`; (b) the KEEP repo's `extrinsic_evaluation/configs.py` line 29 references the production filename `..._REG_0.001.pickle`. The reference repo's hardcoded `1e-5` deviates from the paper by 100×.
2. Line 188: uses Adagrad — paper says **AdamW** (Algorithm 1, p. 14: *"Update embeddings using AdamW: wᵢ ← wᵢ − η · ∇J(W), ∀i ∈ V"*). Replace `torch.optim.Adagrad(model.parameters(), lr=lr)` with `torch.optim.AdamW(model.parameters(), lr=lr)`.
3. Line 162: uses cosine distance by default — paper specifies **squared L2 norm**. Equation 4 (p. 5) and Algorithm 1 (p. 14) both define the regularization term as `λ Σᵢ ‖wᵢ − wⁿ²ᵛᵢ‖²`. Pass `reg_norm=2` when constructing the `Glove` model so the regularization term uses L2 instead of `1 - cosine_similarity`.
4. Lines 159-160: the regularization is computed on `(embedding_i + embeddings_u(i_indices))/2`, i.e., the average of both GloVe matrices, not just one. This is correct and should be preserved.

---

## Story 7: Export KEEP vectors to PyHealth text format

### Context
PyHealth's `EmbeddingModel` loads pretrained vectors from a text file where each line is: `token_string value1 value2 ... value100`. The `init_embedding_with_pretrained` function in `pyhealth/models/embedding.py` (line 66) reads this file and matches token strings against the `code_vocab` dictionary built by `SequenceProcessor`.

### Instructions
1. Create a script `export_keep_vectors.py`
2. Load `keep_embeddings.npy` from Story 6 and `idx_to_concept_filtered.pkl` from Story 5b
3. Write a text file `keep_vectors.txt` where each line is:
   ```
   4274025 0.123 -0.456 0.789 ... (100 values)
   201826 0.234 -0.567 0.890 ...
   ```
   The first token on each line is the OMOP concept_id as a string.
4. Verify the file has exactly `num_concepts` lines
5. Verify a few spot-checked concept IDs match the expected vectors from the numpy matrix

### Acceptance criteria
- Text file with one line per concept, space-separated
- First token on each line is the OMOP concept ID (as string)
- Remaining 100 values are floats
- File is loadable by PyHealth's `_iter_text_vectors` function

### Why this format matters
The `SequenceProcessor` in PyHealth builds `code_vocab` from the raw code strings it sees in patient samples. If your task function outputs OMOP concept IDs (e.g., "201826") as the condition codes, then code_vocab will contain {"201826": 5, ...}. The `init_embedding_with_pretrained` function then looks up "201826" in the text file and loads the vector into `embedding.weight[5]`. The token strings MUST match between the text file and code_vocab.

---

## Story 8: Modify GRASP to accept pretrained embeddings

### Context
The GRASP model in the PyHealth fork creates its EmbeddingModel at line 468 of `pyhealth/models/grasp.py`:
```python
self.embedding_model = EmbeddingModel(dataset, embedding_dim)
```
This doesn't pass any pretrained path. The EmbeddingModel already supports pretrained loading via `pretrained_emb_path`, `freeze_pretrained`, and `normalize_pretrained` parameters. We just need to thread these through GRASP's constructor.

### Instructions
1. Edit `pyhealth/models/grasp.py`
2. Add parameters to GRASP's `__init__`:
   ```python
   def __init__(
       self,
       dataset: SampleDataset,
       static_key: Optional[str] = None,
       embedding_dim: int = 128,
       hidden_dim: int = 128,
       pretrained_emb_path=None,
       freeze_pretrained: bool = False,
       normalize_pretrained: bool = False,
       **kwargs
   ):
   ```
3. Update line 468 to pass these through:
   ```python
   self.embedding_model = EmbeddingModel(
       dataset, embedding_dim,
       pretrained_emb_path=pretrained_emb_path,
       freeze_pretrained=freeze_pretrained,
       normalize_pretrained=normalize_pretrained,
   )
   ```
4. No other changes needed — the rest of GRASP's forward pass works the same regardless of how embeddings are initialized.

### Acceptance criteria
- `GRASP(dataset, embedding_dim=100)` still works (backward compatible, no pretrained)
- `GRASP(dataset, embedding_dim=100, pretrained_emb_path="keep_vectors.txt")` loads KEEP vectors
- `GRASP(dataset, embedding_dim=100, pretrained_emb_path="keep_vectors.txt", freeze_pretrained=True)` loads and freezes them
- Write a unit test with synthetic vectors to verify the pretrained path works

---

## Story 9: Create a MIMIC-IV mortality task with OMOP concept IDs

### Context
The existing `MortalityPredictionMIMIC4` task at `pyhealth/tasks/in_hospital_mortality_mimic4.py` only uses lab timeseries — not diagnosis codes. The `MortalityPredictionMIMIC4` at `pyhealth/tasks/mortality_prediction.py` (line 172) outputs raw ICD codes as condition sequences.

For KEEP to work, the condition codes in patient samples need to be OMOP concept ID strings (matching the keys in `keep_vectors.txt`). There are two options:
- Option A: Create a custom task that outputs OMOP concept IDs directly
- Option B: Use the existing task with ICD codes + SequenceProcessor's code_mapping to translate

Your team already restored the code_mapping infrastructure. If code_mapping can map ICD-10-CM codes to OMOP concept IDs, use that (Option B). If not, create a custom task (Option A) that applies the icd_to_omop mapping from Story 3.

### Instructions
1. Determine which option works with your current code_mapping setup
2. If Option A: create `MortalityPredictionMIMIC4_OMOP` task class that:
   - Loads the `icd10cm_to_omop` mapping from Story 3
   - In the `__call__` method, converts each ICD code to OMOP concept ID
   - Outputs `{"conditions": [list of OMOP concept ID strings], "mortality": 0 or 1}`
3. If Option B: configure SequenceProcessor with the appropriate code_mapping tuple
4. Verify that after setting the task, `sample_dataset.samples[0]["conditions"]` contains OMOP concept ID strings that match keys in `keep_vectors.txt`

### Acceptance criteria
- Patient samples contain OMOP concept ID strings as conditions
- These strings match the concept IDs in `keep_vectors.txt`
- The task produces a reasonable number of samples with mortality labels

---

## Story 10: Run ablation experiments

### Context
The core experiment: compare three embedding strategies on the same GRASP+GRU backbone for in-hospital mortality prediction on MIMIC-IV. This tests our three hypotheses from the proposal.

### Instructions
1. Create a script `run_ablation.py`
2. Set up the three configurations:
   ```python
   # Config 1: Random initialization (baseline)
   model_random = GRASP(dataset, embedding_dim=100, cluster_num=12)
   
   # Config 2: Code mapping only (your team's PyHealth contribution)
   # Use SequenceProcessor with code_mapping enabled
   model_codemap = GRASP(dataset_with_mapping, embedding_dim=100, cluster_num=12)
   
   # Config 3: KEEP pretrained vectors
   model_keep = GRASP(dataset, embedding_dim=100, cluster_num=12,
                       pretrained_emb_path="keep_vectors.txt")
   ```
3. For each configuration:
   - Split data: 80/10/10 train/val/test by patient
   - Train with PyHealth's Trainer
   - Evaluate: AUROC, AUPRC, F1, accuracy
   - Run with 5 random seeds minimum
4. Report means and standard deviations across seeds
5. Run paired statistical tests (Wilcoxon signed-rank) between configurations

### Acceptance criteria
- All three configurations train without errors
- Results table with mean ± std for each metric across seeds
- Statistical test p-values for H1 (KEEP vs random) and H2 (KEEP vs code mapping)
- Training logs saved for reproducibility

---

## Story 11: Write conversion utility and unit tests (PyHealth PR deliverables)

### Context
Per the proposal, the PyHealth pull request should include: updated GRASP model, a KEEP vector loader utility, unit tests with synthetic vectors, and an ablation example script.

### Instructions
1. Create `pyhealth/utils/keep_loader.py` — a utility that:
   - Takes KEEP's numpy output + concept_to_idx mapping
   - Exports to PyHealth's expected text format
   - Validates the output file
2. Write unit tests in `tests/test_grasp_pretrained.py`:
   - Test GRASP with no pretrained (baseline)
   - Test GRASP with pretrained vectors from a small synthetic file
   - Test that pretrained vectors are correctly loaded into the embedding layer
   - Test freeze_pretrained flag works
3. Create an example script `examples/mortality_prediction/mortality_mimic4_grasp_keep.py` showing the full pipeline

### Acceptance criteria
- All unit tests pass
- Example script runs on synthetic data
- Code follows PyHealth contribution guidelines

---

## Execution order and dependencies

```
Story 0 (setup) 
  └→ Story 1 (download Athena files)
       ├→ Story 2 (build full graph)          ← needs S1
       ├→ Story 3 (ICD→OMOP mapping)          ← needs S1 (roll-up filter needs S2's node set)
       └→ Story 8 (modify GRASP)              ← independent, only needs PyHealth fork
            └→ Story 11 (tests + utility)
       
After S2 + S3 are done:
  Story 5 (co-occurrence matrix)              ← needs S3 mapping + S2 graph + MIMIC-IV data
    └→ Story 5b (count filter)                ← reduces graph to data-relevant concepts
         ├→ Story 4 (node2vec on FILTERED graph)
         └→ (rebuild co-occurrence with filtered indices)
              └→ Story 6 (train KEEP GloVe)   ← needs filtered node2vec + filtered co-occurrence
                   └→ Story 7 (export vectors)

Story 9 (mortality task)                      ← needs S3 mapping, can start after S3
Story 10 (ablation)                           ← needs S7 + S8 + S9 all complete
```

### Three parallel work streams after S1:

1. **KEEP pipeline** (critical path): S2 → S3 → S5 → S5b → S4 → S6 → S7
2. **PyHealth modifications**: S8 → S11 → S9 (S9 needs S3's mapping)  
3. **S2 and S3 can run simultaneously** — S3 only needs Athena CSVs, not the graph. The roll-up filter at the end of S3 needs S2's node set but that's a small final step.

Everything converges at **S10 (ablation)** which needs: KEEP vectors (S7) + modified GRASP (S8) + mortality task (S9).

### Key insight: count filter changes the execution order
The original plan had node2vec (S4) running right after the graph (S2). But since the graph needs to be filtered to only data-relevant concepts first, the actual order is: build full graph → build co-occurrence (to discover which concepts have data) → count filter → THEN run node2vec on the filtered graph. This means S4 depends on S5b, not directly on S2.

---

## File inventory (what gets created)

All KEEP pipeline scripts live in `/home/ddhangdd2/keep/keep_pipeline/scripts/` (e.g. `build_omop_graph.py`, `build_icd_to_omop_mapping.py`, `build_cooccurrence_matrix.py`, `apply_count_filter.py`, `train_node2vec.py`, `train_keep_glove.py`, `export_keep_vectors.py`). Their outputs go to `keep_pipeline/data/` and `keep_pipeline/embeddings/` per the table below. The `keep_pipeline/` tree is a sibling of `PyHealth/`, `keep_reference/`, and `data/` under the project root `/home/ddhangdd2/keep/`.

| Story | Output files | Location |
|-------|-------------|----------|
| 2 | `omop_graph.pkl`, `concept_to_idx.pkl`, `idx_to_concept.pkl` (full, unfiltered) | `keep_pipeline/data/` |
| 3 | `icd10cm_to_omop.pkl` | `keep_pipeline/data/` |
| 5 | `cooccurrence_matrix_full.npz` (on full graph, used for count filtering) | `keep_pipeline/data/` |
| 5b | `omop_graph_filtered.pkl`, `concept_to_idx_filtered.pkl`, `idx_to_concept_filtered.pkl`, `cooccurrence_matrix.npz` (rebuilt with filtered indices) | `keep_pipeline/data/` |
| 4 | `node2vec_embeddings.npy` (trained on filtered graph) | `keep_pipeline/embeddings/` |
| 6 | `keep_embeddings.npy` | `keep_pipeline/embeddings/` |
| 7 | `keep_vectors.txt` | `keep_pipeline/embeddings/` |
| 8 | Modified `grasp.py` | `pyhealth/models/grasp.py` |
| 9 | New/modified task file | `pyhealth/tasks/` |
| 10 | Results CSVs, logs | `experiments/` |
| 11 | `keep_loader.py`, tests, example | `pyhealth/utils/`, `tests/`, `examples/` |
