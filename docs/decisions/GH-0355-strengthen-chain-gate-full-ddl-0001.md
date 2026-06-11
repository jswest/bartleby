# Strengthen the upgrade-chain equivalence gate to full DDL

`test_upgrade_chain_walks_from_v4_through_current` is the regression gate that
keeps the hand-built upgrade chain (`bartleby/db/upgrades.py`) in lockstep with
the canonical schema (`bartleby/db/schema.py`). Its `_schema()` helper compared
only table names plus *sorted column names per table* — so dropped indexes,
column type drift, `NOT NULL` / `DEFAULT` / `CHECK` constraint drift, and FK
clause drift were all invisible to it (exactly the drift the gate exists to
catch).

## Decision

`_schema()` now compares the whitespace-normalized `sqlite_master.sql` of every
`type IN ('table', 'index')` row between the chain-upgraded DB and a
freshly-created one. Normalization, in order:

1. **Strip `--` line comments.** `schema.py` annotates columns (e.g. the `#254`
   anchor-splitting note); the chain's hand-built `CREATE` strings don't. A
   comment is layout, not structure.
2. **Collapse all whitespace** to single spaces, then uppercase — so multi-line
   `CREATE`s vs single-line hand-built strings don't false-positive.
3. **For `CREATE TABLE`, split the body on top-level (paren-depth-0) commas and
   sort the resulting column/constraint defs.** This makes the comparison
   order-independent. It must be: the chain reaches v8 via `ALTER TABLE ADD
   COLUMN`, which *appends*, so a chain-built table lists the same columns in a
   different physical order than `schema.py` declares them — an equivalent
   schema we must not flag. Splitting only at depth 0 keeps a
   `CHECK (x IN ('a','b'))` or a multi-column `PRIMARY KEY (a, b)` intact.
4. **Indexes (and any non-plain-table `CREATE`) compare whole.** A dropped index
   becomes a missing dict key; an altered one becomes a changed value.

Verified directly: this catches column type drift, `NOT NULL`/constraint drift,
FK-clause drift, `CHECK`-value drift, and dropped indexes, while reconciling all
pure-layout differences (column reordering from `ALTER`, the `#254` comment,
trailing-paren whitespace) that the old check never saw and that a naive
whitespace-only normalization *would* have false-positived on.

## Why not naive whitespace-only normalization

The issue's acceptance criterion says "compare whitespace-normalized
`sqlite_master.sql`," but it also warns to "normalize aggressively … so the gate
doesn't false-positive on layout differences." Whitespace-only is insufficient:
the chain's `ALTER`-appended column order and `schema.py`'s embedded SQL comments
differ structurally-irrelevantly. Without comment-stripping and order-independent
column comparison, the gate would go *red* (not merely xfail) the moment the
v0.9.0 assembly commit removes the xfail — even on a correct schema. The
order-independent + comment-stripped form is the smallest normalization that
catches the target drift without that false-positive.

## xfail left in place (deliberate)

The test stays `@pytest.mark.xfail(reason="held at 8 …")`. `SCHEMA_VERSION` is
held at 8 while `schema.py` and a dormant `_upgrade_v8_to_v9` step already carry
the v9 (`#114` value-tags, `#254` anchor-splitting) columns. The chain walks
`v < 8` only, so the v8→v9 step never fires: the chain-upgraded `tags` /
`document_tags` lack their v9 columns while a fresh DB has them. The strengthened
gate correctly surfaces *only* that genuine, expected held-at-8 difference (no
spurious layout diffs remain), so the test still xfails — confirming the gate is
not vacuously satisfied. The v0.9.0 assembly commit (not this change) bumps
`SCHEMA_VERSION` to 9, removes the xfail, and extends the chain-strip steps; the
strengthened comparison then proves the full chain end-to-end.

## Scope

Test-only change in `tests/test_project.py` (and a `re` import). No production
code, no schema change, `SCHEMA_VERSION` untouched, xfail marker untouched.
