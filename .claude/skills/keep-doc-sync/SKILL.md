---
name: keep-doc-sync
description: Keep the three load-bearing docs in the KEEP reproduction project (CLAUDE.md, README.md, keep_implementation_plan.md) consistent with each other. Use this skill any time a fact that lives in more than one of these docs is about to change — numeric constants pinned by the paper (LAMBD, embedding dim, node/edge counts), line-number references into PyHealth or reference code, story status updates ("Story 3 done"), newly discovered gotchas, or file path / script renames. Auto-trigger this skill whenever the user is editing one of the three docs, when the user reports a code change that any of the docs document, when the user says a story is finished, or when the user describes a new gotcha — even if they don't explicitly ask for "doc sync." It is much cheaper to propagate a fact across three files in one turn than to discover months later that the docs disagree.
---

# keep-doc-sync

This project's three top-level docs all describe the same KEEP reproduction work from different angles, and they overlap heavily. When a fact changes in one and not the others, future-Claude (and future-user) silently get the wrong information. This skill exists to make doc consistency a one-step operation instead of a thing the user has to remember.

## The three docs (absolute paths)

- `/home/ddhangdd2/keep/CLAUDE.md` — agent-facing project briefing. Has the **Status as of YYYY-MM-DD** line, the **Constants pinned by the paper** block, the **Gotchas** section, and the **PyHealth integration anchors** with line refs.
- `/home/ddhangdd2/keep/README.md` — public-facing reproduction guide. Has its own **## Status** section, the **Pipeline architecture** flowchart, and **Notes for re-implementers**.
- `/home/ddhangdd2/keep/keep_implementation_plan.md` — the authoritative spec. Has **## Status & environment (last updated YYYY-MM-DD)** and one section per Story 0–11, each with Context / Instructions / Acceptance criteria / sometimes a Gotcha subsection.

These paths are stable. Always reference them by absolute path so there's no ambiguity.

## When to fire

Trigger yourself proactively in any of these situations — don't wait for the user to ask:

1. **The user is editing one of the three docs.** Before you finish, check whether the edit touches a fact (constant, line ref, status, gotcha, path) that the *other* two docs also mention.
2. **The user reports a code change that the docs document.** Examples: "I bumped the embedding dim to 128", "Story 4 is done", "I refactored embedding.py and the call site moved to line 210", "I renamed `train_keep_glove.py` to `train_glove_keep.py`".
3. **The user describes a newly discovered gotcha** ("turns out the reference repo's X is wrong because Y"). Gotchas almost always belong in CLAUDE.md *and* the relevant story body in `keep_implementation_plan.md`, and sometimes in `README.md`'s Notes for re-implementers.
4. **The user explicitly asks** ("propagate this", "update all three docs", "is anything stale?").

If the change is genuinely scoped to one doc — say, a typo fix in a README sentence that has no parallel anywhere else — note that you checked and move on. The point of this skill is to make the *check* automatic, not to force three-file edits when they aren't warranted.

## The five fact categories (and where they live)

This is the mental model. Each row tells you which docs to grep when this kind of fact changes.

| Category | CLAUDE.md | README.md | keep_implementation_plan.md |
|---|---|---|---|
| **(a) Numeric constants** (LAMBD, dim, walk length, node/edge counts) | "Constants pinned by the paper" + sometimes Gotchas | sometimes "Notes for re-implementers" or "Pipeline architecture" annotations | story body (Instructions or Acceptance criteria) |
| **(b) Line-number refs** (e.g. `embedding.py:192`) | "PyHealth integration anchors" + Gotchas | rarely | story body, esp. Stories 8 / 9 |
| **(c) Story status** ("Story N done", "Story N is next") | the **Status as of YYYY-MM-DD** line | the **## Status** section | the **## Status & environment** block + each story's Acceptance criteria heading ("ALL PASSED YYYY-MM-DD") |
| **(d) Gotchas** (paper-vs-reference, silent failure modes, env quirks) | "Gotchas — read before implementing" | sometimes "Notes for re-implementers" | the relevant story's body, often as a `### Reference-code gotcha` or `### Key discrepancies` subsection |
| **(e) File paths / script names** (renames, moves) | wherever it appears (commands, anchors, examples) | "Repo structure" + "Pipeline architecture" + "Quick start" | wherever it appears in story Instructions |

When in doubt, grep all three. It's cheap.

## Workflow

Work in this order. Don't skip steps — the propose-then-apply split is the whole point.

### 1. Identify the fact and the category

State the change in one sentence: *"The user wants to bump LAMBD from 1e-3 to 1e-4."* / *"Story 3 is now done as of 2026-04-08."* / *"`embedding.py:192` is now `:195` after a refactor."*

Then classify it into one of (a)–(e) above. This determines which docs you need to check.

### 2. Find every occurrence

Use Grep across all three docs with a pattern that will match every relevant phrasing of the fact, not just the literal string. Examples:

- For `LAMBD = 1e-3`: grep for `1e-3`, `LAMBD`, `λ`, and `1e\-3` (some places use unicode lambda + scientific notation, some use the variable name, some prose).
- For a story status change: grep for the old story-status string, the date line, and `Story N` mentions.
- For a line-number ref: grep for the filename plus `:\d+` and prose forms like `line \d+`.
- For a renamed script: grep for the old name *and* the new name (the new name might already exist as a partial reference).

If a doc has zero hits, say so explicitly in step 3. Zero hits is information, not silence.

### 3. Propose a unified diff before touching anything

Show the user, in one message, every edit you intend to make across all three files. Format each as a small `path:line — old → new` block with enough surrounding context that the user can verify it's the right occurrence. Group by file. End with: *"Apply these N edits across M files? Confirm and I'll run them."*

Do not call Edit yet. The user has been burned by silent doc drift; they want to eyeball the whole picture first.

If you think a doc *should* contain the fact but doesn't, propose adding it (with the exact location) and explain why. Don't just skip the doc — call out the gap.

### 4. Apply on go-ahead

When the user confirms, run the Edits. If the user pushes back on one of the proposed edits, drop just that one and apply the rest — don't re-propose the whole batch.

### 5. Update the date stamps

Two of the three docs carry "last updated" or "as of" dates:
- `CLAUDE.md`: the line `Status as of YYYY-MM-DD: ...`
- `keep_implementation_plan.md`: the line `## Status & environment (last updated YYYY-MM-DD)`

If the change is substantive (story status, new gotcha, constant correction — anything that materially changes what either doc *means*), bump these to today's date as part of the same batch. Trivial cosmetic edits don't warrant a date bump.

## Why this matters (the why, not the what)

The user is the sole owner of this project, working solo on a tight reproduction loop. They've already had three incidents where the paper and the G2Lab reference code disagreed (LAMBD, optimizer, regularization norm), and each incident generated a gotcha that needed to land in CLAUDE.md *and* the relevant story body *and* sometimes the README. When that propagation slips, future-Claude reads the stale doc, makes a bad recommendation, and the user has to re-discover the same gotcha. The cost of a missed propagation isn't a typo — it's a re-bug.

The propose-diff-first step is non-negotiable because:
- Doc edits are easy to get *almost* right (touching the wrong occurrence of a number, picking the wrong story to add a gotcha to, missing a phrasing variant). The user can spot these in five seconds; you can't spot them at all without their eyes.
- Three-file edits applied silently are nearly impossible to review after the fact, because the user doesn't know what to look for.
- The user explicitly asked for this check to be a proposal, not a silent action.

## Things to avoid

- **Don't propagate facts that genuinely live in only one doc.** Not every constant in CLAUDE.md is in the plan; not every story note in the plan belongs in the README. Use judgment — if the other docs don't currently mention the fact and don't *need* to, leave them alone and say so.
- **Don't blindly trust the categorization table.** It's a starting point. If your grep finds a fact in a doc the table doesn't list, update there too — the docs evolve.
- **Don't batch a doc-sync proposal with unrelated edits.** Keep the diff focused so the user can review it as one logical change.
- **Don't bump the date stamp on cosmetic changes.** It dilutes the signal.
- **Don't skip the grep step even if you're "sure" you know where the fact lives.** The user has been burned by exactly this kind of overconfidence.
