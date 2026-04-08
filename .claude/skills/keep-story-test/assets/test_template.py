"""TEMPLATE — copy this skeleton and fill in placeholders marked {{LIKE_THIS}}.

Tests for {{STORY_TITLE}} (Story {{STORY_NUM}} output).

Validates the artifacts produced by {{SCRIPT_NAME}} against the Story
{{STORY_NUM}} acceptance criteria in keep_implementation_plan.md plus
cross-cutting invariants from CLAUDE.md.

Run:
    /home/ddhangdd2/keep/PyHealth/.venv/bin/python \\
        /home/ddhangdd2/keep/keep_pipeline/scripts/test_{{SLUG}}.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Add story-specific imports here (uncomment what you need):
# import pickle
# import duckdb
# import networkx as nx
# import numpy as np

# ----------------------------------------------------------------------------
# Paths and pinned constants
# ----------------------------------------------------------------------------

DATA_DIR = Path("/home/ddhangdd2/keep/data")
OUT_DIR = Path("/home/ddhangdd2/keep/keep_pipeline/data")

# One Path per output artifact from the story's Instructions section:
# ARTIFACT_PKL = OUT_DIR / "{{artifact}}.pkl"

# Pinned constants from acceptance criteria + CLAUDE.md "Constants pinned by
# the paper". Each should carry a `# plan §...` comment so the source of
# truth is obvious from grep.
# EXPECTED_DIM = 100  # plan §Constants pinned by the paper

# ----------------------------------------------------------------------------
# Tiny test harness — no pytest, just print PASS/FAIL and sys.exit(1) on any
# failure. Identical to test_omop_graph.py by design.
# ----------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    def deco(fn):
        try:
            fn()
            _results.append((name, True, ""))
            print(f"  PASS  {name}")
        except AssertionError as e:
            _results.append((name, False, str(e)))
            print(f"  FAIL  {name}")
            print(f"        {e}")
        except Exception as e:  # noqa: BLE001
            _results.append((name, False, f"{type(e).__name__}: {e}"))
            print(f"  ERROR {name}")
            traceback.print_exc()
        return fn
    return deco


# ----------------------------------------------------------------------------
# Load fixtures once at module import — every test reuses them. Fail fast
# with a clear message if any expected artifact is missing.
# ----------------------------------------------------------------------------

print("Loading artifacts...")
# assert ARTIFACT_PKL.exists(), f"missing {ARTIFACT_PKL}"
# with open(ARTIFACT_PKL, "rb") as f:
#     artifact = pickle.load(f)
# print(f"  artifact: {len(artifact):,} entries")
print()

print("Running tests...")

# ----------------------------------------------------------------------------
# Structural tests (types, shapes, sizes)
# ----------------------------------------------------------------------------

# @check("artifact has the right type")
# def _():
#     assert isinstance(artifact, dict)


# ----------------------------------------------------------------------------
# Acceptance criteria from keep_implementation_plan.md §Story {{STORY_NUM}}
# ----------------------------------------------------------------------------

# One @check per acceptance bullet. Real assertions where mechanical, TODO
# stubs (raise AssertionError("not implemented")) where semantic.


# ----------------------------------------------------------------------------
# Cross-cutting invariants from CLAUDE.md
# ----------------------------------------------------------------------------

# concept_to_idx ordering, paper hyperparams, sparse vs dense roll-up, etc.
# See references/story_invariants.md for the per-story checklist.


# ----------------------------------------------------------------------------
# Cross-checks against Athena ground truth (where applicable)
# ----------------------------------------------------------------------------

# When a story's output can be re-derived from raw CSVs via DuckDB, do that
# and compare. Catches drift between the spec and the implementation.


# ----------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------

print()
passed = sum(1 for _, ok, _ in _results if ok)
failed = len(_results) - passed
print("=" * 60)
print(f"  {passed} passed, {failed} failed, {len(_results)} total")
print("=" * 60)

if failed:
    print()
    print("Failures:")
    for name, ok, msg in _results:
        if not ok:
            print(f"  - {name}: {msg}")
    sys.exit(1)
