"""Generate the JS data bundle consumed by keep_pipeline/viz/index.html.

Loads the rescued OMOP graph (Story 2 output), enriches it with concept names
from CONCEPT.csv, derives a small visualization-friendly slice, and writes it
to ``graph_data.js`` as ``window.graphData = {...}`` so the static HTML page
can be opened directly as ``file://`` without needing a local web server.

The slice contains:

* ``stats``               – global counts (nodes, edges, leaves, multi-parent)
* ``depth_distribution``  – BFS depth -> node count, for the layered chart
* ``sampled_subgraph``    – ~200 nodes sampled by BFS from the root, with the
                            edges among them, for the interactive force-graph
* ``orphan_cases``        – the 7 primary orphans the rescue patch fixed,
                            each with its filtered-out direct parent and the
                            rescue parent that was added in step 3a
"""

from __future__ import annotations

import json
import pickle
from collections import defaultdict, deque
from pathlib import Path

import duckdb
import networkx as nx

DATA_DIR = Path("/home/ddhangdd2/keep/data")
PIPELINE_DATA = Path("/home/ddhangdd2/keep/keep_pipeline/data")
OUT_JS = Path("/home/ddhangdd2/keep/keep_pipeline/viz/graph_data.js")

CONCEPT_CSV = DATA_DIR / "CONCEPT.csv"
ANCESTOR_CSV = DATA_DIR / "CONCEPT_ANCESTOR.csv"

ROOT = 4274025
SAMPLE_SIZE = 220  # ~enough nodes to feel "real" without choking the layout


def load_concept_names(con: duckdb.DuckDBPyConnection, ids: set[int]) -> dict[int, str]:
    """Look up concept_name for the given concept ids in CONCEPT.csv."""
    con.execute("CREATE OR REPLACE TEMP TABLE name_lookup_ids (concept_id BIGINT PRIMARY KEY)")
    con.executemany(
        "INSERT INTO name_lookup_ids VALUES (?)", [(int(i),) for i in ids]
    )
    rows = con.execute(
        f"""
        SELECT c.concept_id, c.concept_name
        FROM read_csv_auto('{CONCEPT_CSV}') c
        JOIN name_lookup_ids n ON c.concept_id = n.concept_id
        """
    ).fetchall()
    return {int(cid): name for cid, name in rows}


def main() -> None:
    OUT_JS.parent.mkdir(parents=True, exist_ok=True)

    print("[1/6] loading rescued graph...")
    G: nx.DiGraph = pickle.load(open(PIPELINE_DATA / "omop_graph.pkl", "rb"))
    print(f"      G: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

    con = duckdb.connect()

    # ------------------------------------------------------------------ stats
    print("[2/6] computing stats + depth distribution...")
    leaves = sum(1 for n in G.nodes if G.out_degree(n) == 0)
    multi_parent = sum(1 for n in G.nodes if n != ROOT and G.in_degree(n) > 1)

    # BFS from root to assign depth
    depths: dict[int, int] = {ROOT: 0}
    q: deque[int] = deque([ROOT])
    while q:
        u = q.popleft()
        for v in G.successors(u):
            if v not in depths:
                depths[v] = depths[u] + 1
                q.append(v)
    depth_counts: dict[int, int] = defaultdict(int)
    for d in depths.values():
        depth_counts[d] += 1
    depth_distribution = [
        {"depth": d, "count": depth_counts[d]} for d in sorted(depth_counts)
    ]
    print(f"      depths: {depth_distribution}")

    # --------------------------------------------------------- sampled subgraph
    print(f"[3/6] sampling subgraph ({SAMPLE_SIZE} nodes by BFS)...")
    sampled: list[int] = []
    sampled_set: set[int] = set()
    q = deque([ROOT])
    while q and len(sampled) < SAMPLE_SIZE:
        u = q.popleft()
        if u in sampled_set:
            continue
        sampled.append(u)
        sampled_set.add(u)
        # Bound out-edges per node so deep subtrees don't dominate the BFS
        children = list(G.successors(u))[:8]
        for v in children:
            if v not in sampled_set:
                q.append(v)
    sampled_edges = [
        (u, v) for u in sampled_set for v in G.successors(u) if v in sampled_set
    ]
    print(f"      sampled: {len(sampled)} nodes, {len(sampled_edges)} edges")

    # ------------------------------------------------------------ rebuild naive
    print("[4/6] rebuilding naive (un-rescued) graph to find primary orphans...")
    naive_node_rows = con.execute(
        f"""
        SELECT DISTINCT c.concept_id
        FROM read_csv_auto('{ANCESTOR_CSV}') ca
        JOIN read_csv_auto('{CONCEPT_CSV}') c
          ON ca.descendant_concept_id = c.concept_id
        WHERE ca.ancestor_concept_id = {ROOT}
          AND ca.min_levels_of_separation <= 5
          AND c.domain_id = 'Condition'
          AND c.standard_concept = 'S'
        """
    ).fetchall()
    naive_nodes = [int(r[0]) for r in naive_node_rows]
    naive_set = set(naive_nodes)

    con.execute(
        "CREATE OR REPLACE TEMP TABLE naive_set (concept_id BIGINT PRIMARY KEY)"
    )
    con.executemany(
        "INSERT INTO naive_set VALUES (?)", [(n,) for n in naive_nodes]
    )
    naive_edge_rows = con.execute(
        f"""
        SELECT ca.ancestor_concept_id, ca.descendant_concept_id
        FROM read_csv_auto('{ANCESTOR_CSV}') ca
        JOIN naive_set p ON ca.ancestor_concept_id   = p.concept_id
        JOIN naive_set c ON ca.descendant_concept_id = c.concept_id
        WHERE ca.min_levels_of_separation = 1
        """
    ).fetchall()

    G_naive = nx.DiGraph()
    G_naive.add_nodes_from(naive_nodes)
    G_naive.add_edges_from((int(p), int(c)) for p, c in naive_edge_rows)

    primary_orphans = sorted(
        n for n in G_naive.nodes if n != ROOT and G_naive.in_degree(n) == 0
    )
    print(f"      primary orphans: {len(primary_orphans)}")

    # For each orphan: who is its filtered-out direct parent in CONCEPT_ANCESTOR,
    # and who did the rescue patch wire it to?
    print("[5/6] resolving orphan context (filtered parents + rescue parents)...")
    orphan_cases = []
    for orphan in primary_orphans:
        # Direct parents (min_levels=1) in the *full* CONCEPT_ANCESTOR table
        direct_parents = con.execute(
            f"""
            SELECT c.concept_id, c.concept_name, c.domain_id, c.standard_concept,
                   c.vocabulary_id, c.concept_class_id
            FROM read_csv_auto('{ANCESTOR_CSV}') ca
            JOIN read_csv_auto('{CONCEPT_CSV}') c
              ON ca.ancestor_concept_id = c.concept_id
            WHERE ca.descendant_concept_id = {orphan}
              AND ca.min_levels_of_separation = 1
            ORDER BY c.concept_id
            """
        ).fetchall()

        # Classify each filtered parent into a "reason it was filtered" category:
        #   - "domain_mismatch"     : not Condition+standard
        #   - "outside_disease_tree": is Condition+standard but is NOT a
        #                             descendant of 4274025 within depth <= 5
        #     (these are real Conditions in SNOMED, but they live in a sibling
        #      subtree like "Clinical finding"/441840 instead of "Disease")
        filtered_with_reason = []
        for p in direct_parents:
            pid, pname, pdomain, pstd, pvocab, pclass = p
            in_naive = int(pid) in naive_set
            if in_naive:
                # This shouldn't actually happen for an orphan's direct parent,
                # because if any parent were in the naive set the orphan would
                # have an incoming edge. Mark for diagnostic completeness.
                reason = "in_node_set"
            elif pdomain != "Condition" or pstd != "S":
                reason = "domain_mismatch"
            else:
                reason = "outside_disease_tree"
            filtered_with_reason.append(
                {
                    "id": int(pid),
                    "name": pname,
                    "domain": pdomain,
                    "standard": pstd,
                    "vocabulary": pvocab,
                    "class": pclass,
                    "in_node_set": in_naive,
                    "filter_reason": reason,
                }
            )

        # Rescue parent: in the rescued graph, the orphan now has predecessors
        rescue_preds = list(G.predecessors(orphan))
        # The orphan's depth from root
        orphan_depth = depths.get(orphan, -1)
        # Downstream descendants in the *naive* graph (those that became
        # unreachable as a side-effect of this orphan). Note: some of these
        # may have other parents in the node set, so they're not all
        # ultimately unreachable -- this is an upper bound on the subtree
        # affected by this single orphan.
        downstream = list(nx.descendants(G_naive, orphan))
        # Roll up the categories so the front-end can group by primary cause
        category_set = sorted({fp["filter_reason"] for fp in filtered_with_reason})
        if category_set == ["domain_mismatch"]:
            primary_category = "domain_mismatch"
        elif category_set == ["outside_disease_tree"]:
            primary_category = "outside_disease_tree"
        else:
            primary_category = "mixed"
        orphan_cases.append(
            {
                "orphan_id": int(orphan),
                "depth_from_root": int(orphan_depth),
                "primary_category": primary_category,
                "filtered_parents": filtered_with_reason,
                "rescue_parents": [int(p) for p in rescue_preds],
                "downstream_count": len(downstream),
                "downstream_sample": [int(d) for d in downstream[:5]],
            }
        )

    # ------------------------------------------------------- name lookups
    print("[6/6] resolving names + writing JS bundle...")
    needed_ids: set[int] = set(sampled_set)
    needed_ids.add(ROOT)
    for case in orphan_cases:
        needed_ids.add(case["orphan_id"])
        for fp in case["filtered_parents"]:
            needed_ids.add(fp["id"])
        for rp in case["rescue_parents"]:
            needed_ids.add(rp)
        for d in case["downstream_sample"]:
            needed_ids.add(d)
    names = load_concept_names(con, needed_ids)

    # Hydrate
    sampled_subgraph = {
        "nodes": [
            {
                "id": str(n),
                "name": names.get(n, "?"),
                "depth": depths.get(n, -1),
                "is_root": n == ROOT,
            }
            for n in sampled
        ],
        "edges": [
            {"source": str(u), "target": str(v)} for u, v in sampled_edges
        ],
    }
    for case in orphan_cases:
        case["orphan_name"] = names.get(case["orphan_id"], "?")
        for rp in case["filtered_parents"]:
            pass  # already has name
        case["rescue_parent_details"] = [
            {"id": rid, "name": names.get(rid, "?")}
            for rid in case["rescue_parents"]
        ]
        case["downstream_sample_details"] = [
            {"id": d, "name": names.get(d, "?")}
            for d in case["downstream_sample"]
        ]

    bundle = {
        "stats": {
            "total_nodes": int(G.number_of_nodes()),
            "total_edges": int(G.number_of_edges()),
            "leaves": int(leaves),
            "multi_parent": int(multi_parent),
            "root_id": ROOT,
            "root_name": names.get(ROOT, "Disease"),
            "max_depth": max(depths.values()),
            "naive_orphans": len(primary_orphans),
            "naive_unreachable": sum(
                1 for n in naive_set if n != ROOT and not nx.has_path(G_naive, ROOT, n)
            ),
        },
        "depth_distribution": depth_distribution,
        "sampled_subgraph": sampled_subgraph,
        "orphan_cases": orphan_cases,
        "generated_at": "build_omop_graph 2026-04-07",
    }

    with open(OUT_JS, "w") as f:
        f.write("// Generated by keep_pipeline/viz/generate_viz_data.py\n")
        f.write("// DO NOT EDIT BY HAND -- regenerate from omop_graph.pkl instead.\n")
        f.write("window.graphData = ")
        json.dump(bundle, f, indent=2, ensure_ascii=False)
        f.write(";\n")

    size_kb = OUT_JS.stat().st_size / 1024
    print(f"      wrote {OUT_JS} ({size_kb:.1f} KB)")
    print(
        f"      stats: total_nodes={bundle['stats']['total_nodes']:,}, "
        f"primary_orphans={bundle['stats']['naive_orphans']}, "
        f"naive_unreachable={bundle['stats']['naive_unreachable']}"
    )


if __name__ == "__main__":
    main()
