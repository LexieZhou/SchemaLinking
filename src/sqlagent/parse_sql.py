"""SQL → schema links: which tables and columns a query references.

Used for the gold `schema_links` in the data pipeline, the linking-F1 metric, and the R_link reward.

Approach: parse with sqlglot, run `qualify` with the schema so every column gets an explicit table,
then walk each scope (one per SELECT level) and map every column to its source:
- a base table   → record `table.column`
- a subquery/CTE → skip; its base columns are recorded inside that scope
- nothing found  → record as unresolved (counts against precision, so made-up names cost reward)

SQLite quirks handled here:
- Identifiers are case-insensitive; resolved names use the schema's original spelling.
- A double-quoted token that matches no column is a string literal (`WHERE x = "Village"`),
  not an identifier. Spider's gold SQL relies on this heavily.

Wildcards (`SELECT *`, `t.*`, `COUNT(*)`) link the table only, never every column.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify
from sqlglot.optimizer.scope import Scope, traverse_scope

from sqlagent.schema import Schema

UNKNOWN = "?"  # table part of a column whose table cannot be determined


class SQLParseError(ValueError):
    """The SQL could not be parsed or qualified, or is not a single query."""


@dataclass
class SchemaLinks:
    tables: set[str] = field(default_factory=set)
    columns: set[str] = field(default_factory=set)  # "table.column", including unresolved ones
    unresolved: set[str] = field(default_factory=set)  # tables/columns not found in the schema

    def to_dict(self) -> dict:
        return {k: sorted(getattr(self, k)) for k in ("tables", "columns", "unresolved")}


def extract_links(sql: str, schema: Schema) -> SchemaLinks:
    tree = _parse_and_qualify(sql, schema)
    links = SchemaLinks()
    for scope in traverse_scope(tree):
        for source in scope.sources.values():
            if isinstance(source, exp.Table):
                _add_table(links, source.name, schema)
        for col in scope.columns:
            _add_column(links, col, scope, schema)
    return links


def _parse_and_qualify(sql: str, schema: Schema) -> exp.Expression:
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except SqlglotError as e:
        raise SQLParseError(f"parse failed: {e}") from e
    # Rejects DDL/DML and multi-statement input ("SELECT 1; DROP TABLE t" parses as a Block).
    if not isinstance(tree, exp.Query):
        raise SQLParseError(f"not a single query: {type(tree).__name__}")

    # Tag double-quoted identifiers before qualify; meta is preserved through the rewrite.
    for ident in tree.find_all(exp.Identifier):
        start = ident.meta.get("start")
        if ident.quoted and start is not None and sql[start] == '"':
            ident.meta["double_quoted"] = True

    try:
        return qualify(
            tree,
            schema=_qualify_schema(schema),
            dialect="sqlite",
            expand_stars=False,               # `*` links the table only
            validate_qualify_columns=False,   # keep unknown columns so we can report them
            allow_partial_qualification=True, # ...including `T1.foo` on a real table
            quote_identifiers=False,
        )
    except SqlglotError as e:
        raise SQLParseError(f"qualify failed: {e}") from e


def _qualify_schema(schema: Schema) -> dict[str, dict[str, str]]:
    # qualify only needs names; a placeholder type avoids parsing dataset-specific type strings.
    return {t.name: {c.name: "TEXT" for c in t.columns} for t in schema.tables}


def _add_table(links: SchemaLinks, name: str, schema: Schema) -> None:
    table = schema.table(name)
    if table is not None:
        links.tables.add(table.name)
    else:
        links.tables.add(name.lower())
        links.unresolved.add(name.lower())


def _add_column(links: SchemaLinks, col: exp.Column, scope: Scope, schema: Schema) -> None:
    source = _find_source(scope, col.table) if col.table else None

    if isinstance(source, Scope):
        return  # output column of a subquery or CTE, not a base column
    if source is None and col.this.meta.get("double_quoted"):
        return  # SQLite reads an unresolvable "..." as a string literal

    if isinstance(source, exp.Table) and (table := schema.table(source.name)) is not None:
        if (column := table.column(col.name)) is not None:
            links.columns.add(f"{table.name}.{column.name}")
            return
        name = f"{table.name}.{col.name.lower()}"
    elif isinstance(source, exp.Table):
        name = f"{source.name.lower()}.{col.name.lower()}"  # column of an unknown table
    else:
        name = f"{UNKNOWN}.{col.name.lower()}"  # unknown or ambiguous column
    links.columns.add(name)
    links.unresolved.add(name)


def _find_source(scope: Scope, alias: str) -> exp.Table | Scope | None:
    """Look an alias up in this scope, then in enclosing scopes (correlated subqueries)."""
    s = scope
    while s is not None:
        if alias in s.sources:
            return s.sources[alias]
        s = s.parent
    return None
