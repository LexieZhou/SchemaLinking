import json
import sqlite3
from pathlib import Path

import pytest

from sqlagent.schema import Schema, load_spider_tables, quote_ident, serialize_s1, serialize_s2

ROOT = Path(__file__).resolve().parents[1]
BIRD_DEV = ROOT / "data/raw/bird/dev_20240627"
SPIDER = ROOT / "data/raw/spider/spider_data"

# A Spider-format entry covering the awkward cases: the [-1, "*"] column, a composite primary key
# (BIRD's nested list), a column name with spaces, and a foreign key.
SPIDER_ENTRY = {
    "db_id": "toy",
    "table_names_original": ["Student", "Enroll", "sqlite_sequence"],
    "column_names_original": [
        [-1, "*"], [0, "id"], [0, "Full Name"], [1, "student_id"], [1, "course"],
        [2, "name"], [2, "seq"],
    ],
    "column_types": ["text", "number", "text", "number", "text", "text", "text"],
    "primary_keys": [1, [3, 4]],
    "foreign_keys": [[3, 1]],
}


@pytest.fixture
def toy_sqlite(tmp_path):
    # The directory name has a space, to exercise URI escaping in connect_readonly.
    path = tmp_path / "my dbs" / "toy.sqlite"
    path.parent.mkdir()
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE Student (id INTEGER PRIMARY KEY AUTOINCREMENT, "Full Name" TEXT);
        CREATE TABLE Enroll (
            student_id INTEGER REFERENCES Student(id),
            course TEXT,
            PRIMARY KEY (student_id, course)
        );
        INSERT INTO Student ("Full Name") VALUES ('Ada');
        """
    )
    con.close()
    return path


def check_toy(schema: Schema):
    # sqlite_sequence is filtered out by both loaders.
    assert [t.name for t in schema.tables] == ["Student", "Enroll"]
    assert schema.table("student") is schema.table("STUDENT")  # case-insensitive
    assert schema.table("Enroll").primary_key == ["student_id", "course"]
    assert schema.table("Student").column("full name").name == "Full Name"
    assert [t.name for t in schema.tables_with_column("ID")] == ["Student"]
    assert len(schema.foreign_keys) == 1
    fk = schema.foreign_keys[0]
    assert (fk.table, fk.column, fk.ref_table, fk.ref_column) == ("Enroll", "student_id", "Student", "id")
    assert schema.num_columns == 4


def test_from_spider():
    check_toy(Schema.from_spider(SPIDER_ENTRY))


def test_from_spider_without_star_type():
    # SNAILS lists no type for the [-1, "*"] column.
    entry = {**SPIDER_ENTRY, "column_types": SPIDER_ENTRY["column_types"][1:]}
    s = Schema.from_spider(entry)
    check_toy(s)
    assert s.table("Student").column("id").type == "number"


def test_from_sqlite(toy_sqlite):
    s = Schema.from_sqlite(toy_sqlite)
    assert s.db_id == "toy"
    check_toy(s)
    assert s.table("Student").column("id").type == "INTEGER"


def test_readonly(toy_sqlite):
    from sqlagent.schema import connect_readonly

    con = connect_readonly(toy_sqlite)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        con.execute("DELETE FROM Student")


def test_quote_ident():
    assert quote_ident("abc_1") == "abc_1"
    assert quote_ident("Free Meal Count (K-12)") == "`Free Meal Count (K-12)`"
    assert quote_ident("1st") == "`1st`"


def test_serializers():
    s = Schema.from_spider(SPIDER_ENTRY)
    assert serialize_s1(s) == "Student(id, `Full Name`)\nEnroll(student_id, course)"
    s2 = serialize_s2(s)
    assert "CREATE TABLE Enroll (\n  student_id number,\n  course text," in s2
    assert "PRIMARY KEY (student_id, course)" in s2
    assert "FOREIGN KEY (student_id) REFERENCES Student(id)" in s2


@pytest.mark.data
@pytest.mark.parametrize(
    "tables_json, db_root",
    [(BIRD_DEV / "dev_tables.json", BIRD_DEV / "dev_databases"), (SPIDER / "tables.json", SPIDER / "database")],
)
def test_tables_json_matches_sqlite(tables_json, db_root):
    """tables.json and the real database must expose the same tables and columns (ignoring case)."""
    if not tables_json.exists():
        pytest.skip("raw data not downloaded")

    def shape(s: Schema):
        return {t.name.lower(): sorted(c.name.lower() for c in t.columns) for t in s.tables}

    for db_id, from_json in load_spider_tables(tables_json).items():
        from_db = Schema.from_sqlite(db_root / db_id / f"{db_id}.sqlite")
        assert shape(from_json) == shape(from_db), db_id
