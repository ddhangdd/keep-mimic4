"""Comprehensive tests for the OMOP knowledge graph (Story 2 output).

Validates the pickles produced by build_omop_graph.py against the Story 2
acceptance criteria in keep_implementation_plan.md plus extra invariants
discovered during the orphan-rescue work (2026-04-07).

Run:
    /home/ddhangdd2/keep/PyHealth/.venv/bin/python \
        /home/ddhangdd2/keep/keep_pipeline/scripts/test_omop_graph.py
"""

from __future__ import annotations

import pickle
import sys
import traceback
from pathlib import Path

import duckdb
import networkx as nx

DATA_DIR = Path("/home/ddhangdd2/keep/data")
OUT_DIR = Path("/home/ddhangdd2/keep/keep_pipeline/data")
CONCEPT_CSV = DATA_DIR / "CONCEPT.csv"
ANCESTOR_CSV = DATA_DIR / "CONCEPT_ANCESTOR.csv"

GRAPH_PKL = OUT_DIR / "omop_graph.pkl"
CONCEPT_TO_IDX_PKL = OUT_DIR / "concept_to_idx.pkl"
IDX_TO_CONCEPT_PKL = OUT_DIR / "idx_to_concept.pkl"

ROOT_CONCEPT_ID = 4274025
MAX_DEPTH = 5
EXPECTED_NODES = 68396          # plan §Status, Story 1 + Story 2
EXPECTED_RESCUE_EDGES = 7       # plan §Story 2 gotcha
EXPECTED_BASE_EDGES = 152340    # plan §Story 2 gotcha
EXPECTED_TOTAL_EDGES = EXPECTED_BASE_EDGES + EXPECTED_RESCUE_EDGES

# Concrete orphan-rescue example documented in the plan.
ORPHAN_EXAMPLE = 4234597  # "Misuses drugs"
ORPHAN_DESCENDANT = 44807040  # "Misuses anabolic steroids"


# ----------------------------------------------------------------------------
# Tiny test harness — keeps everything in one file with no pytest dependency.
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
# Load fixtures once at module import — every test reuses them.
# ----------------------------------------------------------------------------

print("Loading pickles...")
assert GRAPH_PKL.exists(), f"missing {GRAPH_PKL}"
assert CONCEPT_TO_IDX_PKL.exists(), f"missing {CONCEPT_TO_IDX_PKL}"
assert IDX_TO_CONCEPT_PKL.exists(), f"missing {IDX_TO_CONCEPT_PKL}"

with open(GRAPH_PKL, "rb") as f:
    G: nx.DiGraph = pickle.load(f)
with open(CONCEPT_TO_IDX_PKL, "rb") as f:
    concept_to_idx: dict[int, int] = pickle.load(f)
with open(IDX_TO_CONCEPT_PKL, "rb") as f:
    idx_to_concept: dict[int, int] = pickle.load(f)

print(f"  graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
print(f"  concept_to_idx: {len(concept_to_idx):,} entries")
print(f"  idx_to_concept: {len(idx_to_concept):,} entries")
print()

con = duckdb.connect()
print("Running tests...")

# ----------------------------------------------------------------------------
# Structural tests
# ----------------------------------------------------------------------------

@check("graph is a networkx.DiGraph")
def _():
    assert isinstance(G, nx.DiGraph), f"got {type(G).__name__}"


@check(f"node count == {EXPECTED_NODES:,}")
def _():
    assert G.number_of_nodes() == EXPECTED_NODES, (
        f"expected {EXPECTED_NODES:,}, got {G.number_of_nodes():,}"
    )


@check(f"edge count == {EXPECTED_TOTAL_EDGES:,} (base {EXPECTED_BASE_EDGES:,} + rescue {EXPECTED_RESCUE_EDGES})")
def _():
    assert G.number_of_edges() == EXPECTED_TOTAL_EDGES, (
        f"expected {EXPECTED_TOTAL_EDGES:,}, got {G.number_of_edges():,}"
    )


@check("graph is a DAG")
def _():
    assert nx.is_directed_acyclic_graph(G)


@check("no self-loops")
def _():
    sl = list(nx.selfloop_edges(G))
    assert not sl, f"found {len(sl)} self-loops, e.g. {sl[:3]}"


@check(f"root concept {ROOT_CONCEPT_ID} is a node")
def _():
    assert ROOT_CONCEPT_ID in G


@check("root in-degree == 0")
def _():
    deg = G.in_degree(ROOT_CONCEPT_ID)
    assert deg == 0, f"in-degree={deg}"


@check("every node reachable from root")
def _():
    reach = nx.descendants(G, ROOT_CONCEPT_ID) | {ROOT_CONCEPT_ID}
    missing = set(G.nodes) - reach
    assert not missing, f"{len(missing):,} unreachable, e.g. {list(missing)[:5]}"


@check("only the root has in-degree 0")
def _():
    zeros = [n for n in G.nodes if G.in_degree(n) == 0]
    assert zeros == [ROOT_CONCEPT_ID], f"unexpected zero-in-degree nodes: {zeros[:5]}"


@check("all node IDs are Python int")
def _():
    bad = [n for n in G.nodes if not isinstance(n, int)]
    assert not bad, f"non-int nodes: {bad[:5]}"


# ----------------------------------------------------------------------------
# concept_to_idx / idx_to_concept invariants
# ----------------------------------------------------------------------------

@check("|concept_to_idx| == |graph nodes|")
def _():
    assert len(concept_to_idx) == G.number_of_nodes()


@check("concept_to_idx keys exactly match graph nodes")
def _():
    assert set(concept_to_idx.keys()) == set(G.nodes)


@check("indices are exactly 0..N-1")
def _():
    n = len(concept_to_idx)
    assert set(concept_to_idx.values()) == set(range(n))


@check("idx_to_concept is the inverse of concept_to_idx")
def _():
    assert len(idx_to_concept) == len(concept_to_idx)
    for cid, i in concept_to_idx.items():
        assert idx_to_concept[i] == cid, f"mismatch at concept {cid}"


@check("concept_to_idx ordered by sorted concept_id (reproducibility)")
def _():
    sorted_ids = sorted(concept_to_idx.keys())
    for i, cid in enumerate(sorted_ids):
        assert concept_to_idx[cid] == i, (
            f"index {i}: expected concept {cid}, got {idx_to_concept[i]}"
        )


# ----------------------------------------------------------------------------
# Cross-check against Athena ground truth
# ----------------------------------------------------------------------------

@check("node set == DuckDB query (depth<=5 standard Conditions under root)")
def _():
    rows = con.execute(
        f"""
        SELECT DISTINCT c.concept_id
        FROM read_csv_auto('{ANCESTOR_CSV}') ca
        JOIN read_csv_auto('{CONCEPT_CSV}') c
          ON ca.descendant_concept_id = c.concept_id
        WHERE ca.ancestor_concept_id = {ROOT_CONCEPT_ID}
          AND ca.min_levels_of_separation <= {MAX_DEPTH}
          AND c.domain_id = 'Condition'
          AND c.standard_concept = 'S'
        """
    ).fetchall()
    expected = {int(r[0]) for r in rows}
    actual = set(G.nodes)
    only_in_g = actual - expected
    only_in_q = expected - actual
    assert not only_in_g and not only_in_q, (
        f"+{len(only_in_g)} only in graph, +{len(only_in_q)} only in query"
    )


@check("every node is a standard Condition concept in CONCEPT.csv")
def _():
    con.execute("CREATE OR REPLACE TEMP TABLE check_nodes (concept_id BIGINT)")
    con.executemany(
        "INSERT INTO check_nodes VALUES (?)", [(int(n),) for n in G.nodes]
    )
    bad = con.execute(
        f"""
        SELECT n.concept_id
        FROM check_nodes n
        LEFT JOIN read_csv_auto('{CONCEPT_CSV}') c
          ON n.concept_id = c.concept_id
        WHERE c.concept_id IS NULL
           OR c.domain_id <> 'Condition'
           OR c.standard_concept <> 'S'
        LIMIT 5
        """
    ).fetchall()
    assert not bad, f"non-standard-Condition nodes: {bad}"


@check("every node has min_levels_of_separation <= 5 from root in CONCEPT_ANCESTOR")
def _():
    con.execute("CREATE OR REPLACE TEMP TABLE check_nodes2 (concept_id BIGINT)")
    con.executemany(
        "INSERT INTO check_nodes2 VALUES (?)", [(int(n),) for n in G.nodes]
    )
    bad = con.execute(
        f"""
        SELECT n.concept_id, COALESCE(MIN(ca.min_levels_of_separation), -1) AS d
        FROM check_nodes2 n
        LEFT JOIN read_csv_auto('{ANCESTOR_CSV}') ca
          ON ca.descendant_concept_id = n.concept_id
         AND ca.ancestor_concept_id = {ROOT_CONCEPT_ID}
        GROUP BY n.concept_id
        HAVING d < 0 OR d > {MAX_DEPTH}
        LIMIT 5
        """
    ).fetchall()
    assert not bad, f"nodes with bad depth-from-root: {bad}"


@check("base edges (sans rescue) == direct parent->child pairs in CONCEPT_ANCESTOR")
def _():
    con.execute("CREATE OR REPLACE TEMP TABLE node_set (concept_id BIGINT PRIMARY KEY)")
    con.executemany(
        "INSERT INTO node_set VALUES (?)", [(int(n),) for n in G.nodes]
    )
    edge_rows = con.execute(
        f"""
        SELECT ca.ancestor_concept_id, ca.descendant_concept_id
        FROM read_csv_auto('{ANCESTOR_CSV}') ca
        JOIN node_set p ON ca.ancestor_concept_id   = p.concept_id
        JOIN node_set c ON ca.descendant_concept_id = c.concept_id
        WHERE ca.min_levels_of_separation = 1
        """
    ).fetchall()
    base_edges = {(int(p), int(c)) for p, c in edge_rows}
    graph_edges = set(G.edges)
    missing = base_edges - graph_edges
    assert not missing, f"{len(missing)} CONCEPT_ANCESTOR edges missing from G"
    extra_count = len(graph_edges - base_edges)
    assert extra_count == EXPECTED_RESCUE_EDGES, (
        f"expected exactly {EXPECTED_RESCUE_EDGES} rescue edges, got {extra_count}"
    )
    assert len(base_edges) == EXPECTED_BASE_EDGES, (
        f"base edge count {len(base_edges)} != expected {EXPECTED_BASE_EDGES}"
    )


# ----------------------------------------------------------------------------
# Orphan-rescue regression — the concrete example from the plan
# ----------------------------------------------------------------------------

@check(f"orphan-rescue example: concept {ORPHAN_EXAMPLE} is in graph and reachable")
def _():
    assert ORPHAN_EXAMPLE in G, f"concept {ORPHAN_EXAMPLE} missing"
    assert nx.has_path(G, ROOT_CONCEPT_ID, ORPHAN_EXAMPLE), (
        f"no path from root to {ORPHAN_EXAMPLE}"
    )
    assert G.in_degree(ORPHAN_EXAMPLE) >= 1


@check(f"orphan-rescue example: descendant {ORPHAN_DESCENDANT} is reachable")
def _():
    assert ORPHAN_DESCENDANT in G
    assert nx.has_path(G, ROOT_CONCEPT_ID, ORPHAN_DESCENDANT)


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
