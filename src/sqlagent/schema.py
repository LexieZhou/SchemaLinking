"""Database schemas: one in-memory representation, two loaders, two prompt serializations.

Loaders
- `Schema.from_spider(entry)`: one entry of a Spider-format tables.json (BIRD, Spider and SNAILS
  all ship this format). Works without the database file, e.g. for BIRD train on the Mac.
- `Schema.from_sqlite(path)`: reads the real database with PRAGMAs. The tool environment sees
  exactly this, so it is the ground truth when the two disagree.

Serializations (single-turn baseline prompt)
- S1 (compact): `table(col1, col2, ...)`, one table per line.
- S2 (detailed): CREATE TABLE-like, with column types, primary keys and foreign keys.

SQLite identifiers are case-insensitive, so every lookup goes through `_key()` while the original
spelling is kept for display.
"""
from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path


def _key(name: str) -> str:
    """Normalized lookup key for an identifier (SQLite compares identifiers case-insensitively)."""
    return name.lower()


_SIMPLE_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def quote_ident(name: str) -> str:
    """Backtick-quote identifiers that need it, e.g. BIRD's `Free Meal Count (K-12)`."""
    if _SIMPLE_IDENT.match(name):
        return name
    return "`" + name.replace("`", "``") + "`"


@dataclass
class Column:
    name: str
    type: str = ""


@dataclass
class Table:
    name: str
    columns: list[Column] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)  # composite keys have several columns

    def column(self, name: str) -> Column | None:
        k = _key(name)
        return next((c for c in self.columns if _key(c.name) == k), None)


@dataclass(frozen=True)
class ForeignKey:
    table: str
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Schema:
    db_id: str
    tables: list[Table]
    foreign_keys: list[ForeignKey] = field(default_factory=list)

    def __post_init__(self):
        self._by_key = {_key(t.name): t for t in self.tables}

    def table(self, name: str) -> Table | None:
        return self._by_key.get(_key(name))

    def tables_with_column(self, column: str) -> list[Table]:
        """All tables that have a column with this name; used to resolve unqualified columns."""
        return [t for t in self.tables if t.column(column) is not None]

    @property
    def num_columns(self) -> int:
        return sum(len(t.columns) for t in self.tables)

    # ---------------------------------------------------------------- loaders

    @classmethod
    def from_spider(cls, entry: dict) -> Schema:
        names = entry["table_names_original"]
        types = entry["column_types"]
        # SNAILS omits the type of the [-1, "*"] column, so its list is one shorter; realign it.
        if len(types) == len(entry["column_names_original"]) - 1:
            types = ["text"] + types
        tables = [Table(name=n) for n in names]
        # column_names_original is a flat list of [table_index, column_name]; index 0 is [-1, "*"].
        flat: list[tuple[int, str]] = []
        for i, (t_idx, c_name) in enumerate(entry["column_names_original"]):
            flat.append((t_idx, c_name))
            if t_idx >= 0:
                tables[t_idx].columns.append(Column(c_name, types[i]))
        # primary_keys holds column indices; BIRD uses a nested list for composite keys.
        for pk in entry["primary_keys"]:
            for c_idx in pk if isinstance(pk, list) else [pk]:
                t_idx, c_name = flat[c_idx]
                tables[t_idx].primary_key.append(c_name)
        fks = []
        for src, dst in entry["foreign_keys"]:
            (st, sc), (dt, dc) = flat[src], flat[dst]
            fks.append(ForeignKey(names[st], sc, names[dt], dc))
        return cls._without_internal(entry["db_id"], tables, fks)

    @classmethod
    def from_sqlite(cls, path: str | Path, db_id: str | None = None) -> Schema:
        path = Path(path)
        con = connect_readonly(path)
        try:
            tables, fks = [], []
            rows = con.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY rowid"
            ).fetchall()
            for (t_name,) in rows:
                q = '"' + t_name.replace('"', '""') + '"'
                # table_info rows: (cid, name, type, notnull, default, pk_position)
                info = con.execute(f"PRAGMA table_info({q})").fetchall()
                cols = [Column(r[1], r[2] or "") for r in info]
                pk = [r[1] for r in sorted((r for r in info if r[5] > 0), key=lambda r: r[5])]
                tables.append(Table(t_name, cols, pk))
                # foreign_key_list rows: (id, seq, ref_table, from, to, ...); `to` is NULL when
                # the FK implicitly references the parent's primary key.
                for r in con.execute(f"PRAGMA foreign_key_list({q})").fetchall():
                    fks.append(ForeignKey(t_name, r[3], r[2], r[4] or ""))
        finally:
            con.close()
        return cls._without_internal(db_id or path.stem, tables, fks)

    @classmethod
    def _without_internal(cls, db_id, tables, fks) -> Schema:
        # Drop SQLite bookkeeping tables such as sqlite_sequence (some Spider tables.json list it).
        keep = [t for t in tables if not _key(t.name).startswith("sqlite_")]
        names = {_key(t.name) for t in keep}
        fks = [fk for fk in fks if _key(fk.table) in names]
        return cls(db_id, keep, fks)


def connect_readonly(path: str | Path) -> sqlite3.Connection:
    """Open a SQLite file read-only. A file: URI is used so paths with spaces etc. are escaped."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    return sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)


def load_spider_tables(path: str | Path) -> dict[str, Schema]:
    """Load every schema in a tables.json, keyed by db_id."""
    with open(path) as f:
        return {e["db_id"]: Schema.from_spider(e) for e in json.load(f)}


# ---------------------------------------------------------------- serializers


def serialize_s1(schema: Schema) -> str:
    """S1: one line per table, `table(col1, col2, ...)`."""
    return "\n".join(
        f"{quote_ident(t.name)}({', '.join(quote_ident(c.name) for c in t.columns)})"
        for t in schema.tables
    )


def serialize_s2(schema: Schema) -> str:
    """S2: CREATE TABLE-like blocks with types, PRIMARY KEY and FOREIGN KEY lines."""
    fks_by_table: dict[str, list[ForeignKey]] = {}
    for fk in schema.foreign_keys:
        fks_by_table.setdefault(_key(fk.table), []).append(fk)

    blocks = []
    for t in schema.tables:
        lines = [f"  {quote_ident(c.name)} {c.type}".rstrip() for c in t.columns]
        if t.primary_key:
            lines.append(f"  PRIMARY KEY ({', '.join(quote_ident(c) for c in t.primary_key)})")
        for fk in fks_by_table.get(_key(t.name), []):
            ref = f"{quote_ident(fk.ref_table)}({quote_ident(fk.ref_column)})" if fk.ref_column \
                else quote_ident(fk.ref_table)
            lines.append(f"  FOREIGN KEY ({quote_ident(fk.column)}) REFERENCES {ref}")
        blocks.append(f"CREATE TABLE {quote_ident(t.name)} (\n" + ",\n".join(lines) + "\n);")
    return "\n\n".join(blocks)
