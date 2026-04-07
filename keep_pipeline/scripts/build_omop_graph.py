"""Build the OMOP knowledge graph for KEEP (Story 2).

Constructs a NetworkX directed graph of standard SNOMED Condition concepts
within 5 hierarchical levels of the root "Disease" concept (4274025), with
edges given by direct parent->child "is-a" relationships from CONCEPT_ANCESTOR.

Inputs (read-only):
    /home/ddhangdd2/keep/data/CONCEPT.csv
    /home/ddhangdd2/keep/data/CONCEPT_ANCESTOR.csv

Outputs (written to /home/ddhangdd2/keep/keep_pipeline/data/):
    omop_graph.pkl       - networkx.DiGraph, parent -> child
    concept_to_idx.pkl   - dict {concept_id (int) : row index (int)}
    idx_to_concept.pkl   - dict {row index (int) : concept_id (int)}

Per the KEEP paper (Appendix A.1.1, p. 15) and the implementation plan
(Story 2 in keep_implementation_plan.md), the node set is the descendants of
4274025 within depth <= 5 that are standard ('S') Condition concepts; the
edge set is direct parent-child pairs (min_levels_of_separation = 1) where
both endpoints are in the node set.

The concept_to_idx ordering is "sorted by concept_id" so that re-running this
script always produces identical row indices, which matters for downstream
embedding-matrix consistency across stories.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import duckdb
import networkx as nx

# --- Paths -------------------------------------------------------------------

DATA_DIR = Path("/home/ddhangdd2/keep/data")
OUT_DIR = Path("/home/ddhangdd2/keep/keep_pipeline/data")

CONCEPT_CSV = DATA_DIR / "CONCEPT.csv"
ANCESTOR_CSV = DATA_DIR / "CONCEPT_ANCESTOR.csv"

GRAPH_PKL = OUT_DIR / "omop_graph.pkl"
CONCEPT_TO_IDX_PKL = OUT_DIR / "concept_to_idx.pkl"
IDX_TO_CONCEPT_PKL = OUT_DIR / "idx_to_concept.pkl"

ROOT_CONCEPT_ID = 4274025  # "Disease" (paper §A.1.1, p. 15)
MAX_DEPTH = 5  # paper §A.1.1, p. 15: exclude concepts > 5 levels from root


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()

    # --- 1. Node set ---------------------------------------------------------
    # All standard Condition concepts within MAX_DEPTH of the Disease root.
    # This includes the root itself (min_levels_of_separation = 0 row in
    # CONCEPT_ANCESTOR for ancestor=descendant=4274025).
    print(f"[1/5] Querying node set (root={ROOT_CONCEPT_ID}, depth<={MAX_DEPTH})...")
    node_rows = con.execute(
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
    node_ids = [int(r[0]) for r in node_rows]
    node_set = set(node_ids)
    print(f"      -> {len(node_ids):,} nodes")

    if ROOT_CONCEPT_ID not in node_set:
        raise RuntimeError(
            f"root concept {ROOT_CONCEPT_ID} missing from node set; "
            "check filter conditions"
        )

    # --- 2. Edge set ---------------------------------------------------------
    # Direct parent->child pairs where both endpoints are in the node set.
    # We pass the node set into DuckDB via a temporary table for the join.
    print("[2/5] Querying direct parent->child edges (min_levels_of_separation=1)...")
    con.execute(
        "CREATE TEMP TABLE node_set (concept_id BIGINT PRIMARY KEY)"
    )
    con.executemany(
        "INSERT INTO node_set VALUES (?)", [(nid,) for nid in node_ids]
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
    edges = [(int(p), int(c)) for p, c in edge_rows]
    print(f"      -> {len(edges):,} edges")

    # --- 3. Build NetworkX DiGraph -------------------------------------------
    print("[3/5] Building NetworkX DiGraph...")
    G = nx.DiGraph()
    G.add_nodes_from(node_ids)
    G.add_edges_from(edges)  # parent -> child

    # Sanity checks
    assert G.number_of_nodes() == len(node_ids), "node count mismatch after add"
    assert nx.is_directed_acyclic_graph(G), "graph is not a DAG"
    root_in_degree = G.in_degree(ROOT_CONCEPT_ID)
    if root_in_degree != 0:
        raise RuntimeError(
            f"root {ROOT_CONCEPT_ID} has in-degree {root_in_degree}, expected 0"
        )

    # --- 3a. Rescue orphans via transitive ancestor edges --------------------
    # Some Condition concepts have their direct parent in a non-Condition
    # domain (e.g. 4234597 "Misuses drugs" -> 4042889 "Finding relating to
    # drug misuse behavior" which is Observation). Those edges are filtered
    # out by the Condition+standard node set, leaving the descendant orphaned.
    # For each primary orphan, find its nearest in-node-set ancestor (using
    # min_levels_of_separation > 0, take the minimum) and add a rescue edge.
    primary_orphans = [
        n for n in G.nodes if n != ROOT_CONCEPT_ID and G.in_degree(n) == 0
    ]
    if primary_orphans:
        print(
            f"      -> {len(primary_orphans)} primary orphans "
            "(in_degree=0, not root); patching with transitive edges..."
        )
        con.execute("CREATE TEMP TABLE orphan_ids (concept_id BIGINT PRIMARY KEY)")
        con.executemany(
            "INSERT INTO orphan_ids VALUES (?)", [(o,) for o in primary_orphans]
        )
        # For each orphan, pick the in-node-set ancestor with the smallest
        # min_levels_of_separation > 0. ROW_NUMBER breaks ties deterministically
        # by preferring the smaller ancestor_concept_id.
        rescue_rows = con.execute(
            f"""
            SELECT descendant, nearest_ancestor, depth
            FROM (
                SELECT
                    ca.descendant_concept_id AS descendant,
                    ca.ancestor_concept_id AS nearest_ancestor,
                    ca.min_levels_of_separation AS depth,
                    ROW_NUMBER() OVER (
                        PARTITION BY ca.descendant_concept_id
                        ORDER BY ca.min_levels_of_separation,
                                 ca.ancestor_concept_id
                    ) AS rn
                FROM read_csv_auto('{ANCESTOR_CSV}') ca
                JOIN orphan_ids o ON ca.descendant_concept_id = o.concept_id
                JOIN node_set n ON ca.ancestor_concept_id = n.concept_id
                WHERE ca.min_levels_of_separation > 0
            )
            WHERE rn = 1
            """
        ).fetchall()
        rescue_edges = [(int(p), int(d)) for d, p, _depth in rescue_rows]
        G.add_edges_from(rescue_edges)
        rescue_depths = sorted({int(r[2]) for r in rescue_rows})
        print(
            f"      -> added {len(rescue_edges)} rescue edges "
            f"(transitive depths used: {rescue_depths})"
        )
        if len(rescue_edges) != len(primary_orphans):
            unrescued = set(primary_orphans) - {d for _, d in rescue_edges}
            raise RuntimeError(
                f"could not find an in-set ancestor for {len(unrescued)} "
                f"orphans: {list(unrescued)[:5]}"
            )
        assert nx.is_directed_acyclic_graph(G), (
            "graph is no longer a DAG after adding rescue edges"
        )

    reachable = nx.descendants(G, ROOT_CONCEPT_ID) | {ROOT_CONCEPT_ID}
    unreachable = node_set - reachable
    if unreachable:
        raise RuntimeError(
            f"{len(unreachable):,} nodes are still unreachable from root "
            f"{ROOT_CONCEPT_ID} after rescue; first few: {list(unreachable)[:5]}"
        )
    print(
        f"      -> DAG ✓  root in-degree=0 ✓  all {len(node_ids):,} nodes "
        f"reachable from root ✓"
    )

    # --- 4. Build concept_to_idx / idx_to_concept ----------------------------
    # Sorted by concept_id for reproducibility across runs.
    print("[4/5] Building concept_to_idx / idx_to_concept (sorted by concept_id)...")
    sorted_ids = sorted(node_ids)
    concept_to_idx = {cid: i for i, cid in enumerate(sorted_ids)}
    idx_to_concept = {i: cid for cid, i in concept_to_idx.items()}
    assert len(concept_to_idx) == len(idx_to_concept) == len(node_ids)
    print(f"      -> indices 0..{len(sorted_ids) - 1}")

    # --- 5. Write pickles ----------------------------------------------------
    print(f"[5/5] Writing pickles to {OUT_DIR}/ ...")
    with open(GRAPH_PKL, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CONCEPT_TO_IDX_PKL, "wb") as f:
        pickle.dump(concept_to_idx, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(IDX_TO_CONCEPT_PKL, "wb") as f:
        pickle.dump(idx_to_concept, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"      -> {GRAPH_PKL.name}  ({GRAPH_PKL.stat().st_size / 1e6:.2f} MB)")
    print(f"      -> {CONCEPT_TO_IDX_PKL.name}  ({CONCEPT_TO_IDX_PKL.stat().st_size / 1e6:.2f} MB)")
    print(f"      -> {IDX_TO_CONCEPT_PKL.name}  ({IDX_TO_CONCEPT_PKL.stat().st_size / 1e6:.2f} MB)")

    # --- Summary -------------------------------------------------------------
    leaf_count = sum(1 for n in G.nodes if G.out_degree(n) == 0)
    parents_per_node = [G.in_degree(n) for n in G.nodes if n != ROOT_CONCEPT_ID]
    multi_parent = sum(1 for d in parents_per_node if d > 1)
    print()
    print("=" * 60)
    print("OMOP knowledge graph built")
    print(f"  nodes:        {G.number_of_nodes():,}")
    print(f"  edges:        {G.number_of_edges():,}")
    print(f"  leaves:       {leaf_count:,}")
    print(f"  multi-parent: {multi_parent:,} (nodes with > 1 parent)")
    print(f"  root:         {ROOT_CONCEPT_ID} (Disease)")
    print("=" * 60)


if __name__ == "__main__":
    main()
