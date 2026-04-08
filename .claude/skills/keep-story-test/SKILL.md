---
name: keep-story-test
description: Scaffold a Story-N acceptance test file for the KEEP MIMIC-IV reproduction project under /home/ddhangdd2/keep. Generates keep_pipeline/scripts/test_<script>.py following the project's standalone-script test harness convention (no pytest), with one @check per acceptance bullet from keep_implementation_plan.md plus cross-cutting invariants from CLAUDE.md. Use this skill whenever the user asks to write, scaffold, generate, or stub out tests for a KEEP story (e.g. "write tests for Story 3", "scaffold the Story 4 acceptance test", "I need a test file for the node2vec script", "stub the test for build_icd_to_omop_mapping"), even if they don't say "skill" or "scaffold". This skill is specific to the KEEP reproduction repo and should not trigger for generic pytest scaffolding requests.
---

# keep-story-test

Scaffold an acceptance test file for one of the Stories 3–11 in the KEEP MIMIC-IV reproduction project at `/home/ddhangdd2/keep`. The output is a standalone Python script that mirrors `keep_pipeline/scripts/test_omop_graph.py` — same harness, same conventions, same run command.

## Why this skill exists

Every story in `keep_implementation_plan.md` ends in an `### Acceptance criteria` section that needs to become a runnable test file. The test files all follow one specific convention (no pytest, module-level fixture loading, `@check` decorator, `_results` accumulator, `sys.exit(1)` on failure) — see `test_omop_graph.py` for the canonical reference. Hand-translating each story's acceptance bullets into that harness is mechanical busywork, and it's easy to drop a cross-cutting invariant from CLAUDE.md (e.g. forget to assert `concept_to_idx` is sorted by concept_id). This skill does the mechanical part and pulls in the right invariants automatically.

## Inputs you need from the user

Just the story number, or a script name. Examples of what counts as a trigger:

- "scaffold the test for story 3"
- "write the acceptance test for build_icd_to_omop_mapping.py"
- "test stub for the node2vec story"

If the user gives you a story number, you have everything you need. If they give you a script name, look up which story it belongs to by grepping `keep_implementation_plan.md` for that filename.

## Workflow

### 1. Read the story spec

Read `/home/ddhangdd2/keep/keep_implementation_plan.md`. The story sections are headed `## Story <N>: <title>` and end at the next `---` separator. Extract:

- **Script name** — find the line in `### Instructions` that says `` Create a script `<name>.py` ``. The test file name is derived by replacing the leading `build_` or `train_` with `test_`. Examples:
  - `build_omop_graph.py` → `test_omop_graph.py`
  - `build_icd_to_omop_mapping.py` → `test_icd_to_omop_mapping.py`
  - `train_node2vec.py` → `test_node2vec.py`
- **Output artifacts** — pickles, npy/npz, txt files mentioned in the Instructions ("Save as pickle", "Save the resulting numpy matrix as `node2vec_embeddings.npy`", etc.). These become the fixtures the test loads at module level.
- **Acceptance criteria bullets** — every bullet under `### Acceptance criteria` becomes one `@check`.
- **Pinned constants** — if the Instructions or `## Constants pinned by the paper` section in CLAUDE.md mentions hyperparameter values for this story (dimensions, λ, optimizer, etc.), they become assertion targets where they're inspectable from the saved artifact.

### 2. Pull cross-cutting invariants from CLAUDE.md

Read `/home/ddhangdd2/keep/CLAUDE.md`, especially the "Cross-cutting invariants" bullets at the end of the Architecture section, the "Gotchas" section, and the "Constants pinned by the paper" section. Then read `references/story_invariants.md` in this skill — it's a distilled per-story checklist of invariants that should always be asserted, derived from those CLAUDE.md sections. If the invariants file has an entry for the requested story, every check in that entry should appear in the generated test file (in addition to the acceptance-criteria bullets).

If a story isn't covered in `references/story_invariants.md` yet, infer from CLAUDE.md and the story's `### Reference code adaptation` / `### Roll-up logic` / gotcha subsections what extra checks belong. Add the entry to `references/story_invariants.md` after generating the test, so future runs benefit.

### 3. Compose the test file

Read `assets/test_template.py` for the verbatim harness (header docstring, imports, `_results` + `@check` decorator, summary footer). Copy that skeleton and fill in:

1. **Module docstring** — replace the story number, the artifact list, and the run command path.
2. **Path constants** — `DATA_DIR`, `OUT_DIR`, and one Path constant per output artifact.
3. **Expected-value constants** — pull from acceptance criteria where the value is concrete (counts, shapes, dimensions). If a value isn't pinned in the spec, leave it out — don't invent.
4. **Fixture loading** — `assert <PATH>.exists()` then load each artifact (pickle, np.load, plain text). Print a one-line summary of each so failures are diagnosable.
5. **Tests** — one `@check("...")` per acceptance bullet, organized into commented sections (`# Structural tests`, `# Cross-checks against ground truth`, `# Reference-spec invariants`, etc.) following the test_omop_graph.py grouping.
6. **Summary footer** — unchanged from template.

### 4. Real check vs TODO stub

For each `@check`, decide whether it's mechanical or semantic:

- **Mechanical** (write a real assertion): file exists, dict has key, matrix has shape `(N, D)`, dtype is float32, every value in set X is in set Y, sorted ordering, dict-inverse roundtrip, max/min within bounds. Anything you can verify by inspecting the artifact alone.
- **Semantic** (write a TODO stub): "T1DM and T2DM should be closer than random pairs", "spot-check a few rolled-up codes", "runtime should be under 30 minutes". These need either a separately-trained reference model or human judgment.

For semantic checks, emit:

```python
@check("T1DM/T2DM cosine sanity check")
def _():
    # TODO: pick T1DM (concept 201254) + T2DM (201826), assert cos_sim(...)
    # > cos_sim with a random concept. See keep_implementation_plan.md story 4
    # acceptance bullet 2.
    raise AssertionError("not implemented — fill in manually")
```

The `raise AssertionError("not implemented")` makes the test file fail loudly until the human fills it in, which is what we want — the alternative (silently passing) hides incomplete coverage. The TODO comment in the body tells the human exactly what to write.

### 5. Write the file

Write to `/home/ddhangdd2/keep/keep_pipeline/scripts/test_<script>.py`. Don't run it — the underlying story script may not exist yet, in which case the fixture-loading `assert *.exists()` lines will fail before any `@check` runs. The user will run it after they finish the story implementation.

After writing, tell the user:
- The path you wrote to
- How many `@check` blocks total, split into "real assertions" vs "TODO stubs"
- The exact run command (`PyHealth/.venv/bin/python keep_pipeline/scripts/test_<script>.py`)
- Any acceptance bullet you couldn't translate (and why) so they can sanity-check coverage

## Conventions to preserve

These come from `test_omop_graph.py` and aren't negotiable — they exist because the user has made decisions about test ergonomics on this project:

- **No pytest dependency.** This is a standalone script run via the PyHealth venv interpreter, not via `pytest`. Don't `import pytest`. Don't write `def test_*` functions; use `@check("human-readable name")` with throwaway `def _():` bodies.
- **Module-level fixture loading.** Load all artifacts once at import time, not inside each test. If a fixture is missing, fail fast with a clear `assert <path>.exists(), f"missing {<path>}"` *before* any check runs.
- **Absolute paths.** Use `/home/ddhangdd2/keep/...` literals for `DATA_DIR` and `OUT_DIR`, not `Path(__file__).parent`. The user runs scripts from arbitrary working directories.
- **Numeric constants pulled out to module top.** `EXPECTED_NODES = 68396  # plan §Status` — make them greppable and self-documenting with a `# plan §...` comment pointing at the source of truth.
- **Cross-checks against ground truth.** Where possible, also re-derive the expected value from the raw Athena CSVs via DuckDB and compare. `test_omop_graph.py` does this for the node set — see the `node set == DuckDB query` check. This catches drift between what the spec *says* and what the script *does*. Not every story has a clean ground-truth source, but when it does (Story 3 against `CONCEPT_RELATIONSHIP`, Story 5 against `diagnoses_icd`, Story 7 against the npy from Story 6), include one.

## Output expectations

When done, the user should be able to:

```bash
/home/ddhangdd2/keep/PyHealth/.venv/bin/python \
    /home/ddhangdd2/keep/keep_pipeline/scripts/test_<script>.py
```

…and see PASS/FAIL/ERROR lines printed for each `@check`, a summary count, and `sys.exit(1)` if any check failed. The TODO stubs should fail with `AssertionError: not implemented` so they're impossible to ignore.
