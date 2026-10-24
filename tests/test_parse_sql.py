import json
from pathlib import Path

import pytest

from sqlagent.parse_sql import SQLParseError, extract_links
from sqlagent.schema import Column, ForeignKey, Schema, Table, load_spider_tables

ROOT = Path(__file__).resolve().parents[1]
BIRD_DEV = ROOT / "data/raw/bird/dev_20240627"
SPIDER = ROOT / "data/raw/spider/spider_data"

# singer and concert share `name` and `singer_id`, so unqualified uses of those are ambiguous.
SCHEMA = Schema(
    "toy",
    [
        Table("Singer", [Column("singer_id"), Column("Name"), Column("Age"), Column("Full Name")]),
        Table("concert", [Column("concert_id"), Column("singer_id"), Column("name"), Column("year")]),
    ],
    [ForeignKey("concert", "singer_id", "Singer", "singer_id")],
)


def links(sql):
    return extract_links(sql, SCHEMA)


def test_alias_and_join():
    l = links("SELECT T1.name FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id "
              "WHERE T2.year > 2000")
    assert l.tables == {"Singer", "concert"}
    assert l.columns == {"Singer.Name", "Singer.singer_id", "concert.singer_id", "concert.year"}
    assert l.unresolved == set()


def test_unqualified_columns_resolved_via_schema():
    l = links("SELECT age FROM singer JOIN concert ON singer.singer_id = concert.singer_id WHERE year = 2014")
    assert l.columns == {"Singer.Age", "Singer.singer_id", "concert.singer_id", "concert.year"}


def test_case_insensitive_and_canonical_spelling():
    assert links("SELECT NAME FROM SINGER WHERE aGe > 3").columns == {"Singer.Name", "Singer.Age"}


def test_ambiguous_unqualified_column_is_unresolved():
    l = links("SELECT name FROM singer JOIN concert ON singer.singer_id = concert.singer_id")
    assert "?.name" in l.columns and l.unresolved == {"?.name"}


def test_subquery_in_where():
    l = links("SELECT name FROM singer WHERE singer_id IN (SELECT singer_id FROM concert WHERE year = 2014)")
    assert l.tables == {"Singer", "concert"}
    assert l.columns == {"Singer.Name", "Singer.singer_id", "concert.singer_id", "concert.year"}


def test_correlated_subquery_resolves_outer_alias():
    l = links("SELECT T1.name FROM singer AS T1 WHERE EXISTS "
              "(SELECT 1 FROM concert AS T2 WHERE T2.singer_id = T1.singer_id)")
    assert l.columns == {"Singer.Name", "Singer.singer_id", "concert.singer_id"}
    assert l.unresolved == set()


def test_derived_table_columns_are_not_links():
    l = links("SELECT t.n FROM (SELECT name AS n, count(*) AS c FROM singer GROUP BY name) AS t WHERE t.c > 1")
    assert l.tables == {"Singer"}
    assert l.columns == {"Singer.Name"}  # t.n and t.c are derived


def test_cte():
    l = links("WITH x AS (SELECT singer_id, count(*) AS c FROM concert GROUP BY singer_id) "
              "SELECT s.name FROM singer s JOIN x ON s.singer_id = x.singer_id ORDER BY x.c DESC")
    assert l.tables == {"Singer", "concert"}  # the CTE name is not a table
    assert l.columns == {"concert.singer_id", "Singer.Name", "Singer.singer_id"}


def test_set_operation():
    l = links("SELECT name FROM singer WHERE age > 30 UNION SELECT name FROM concert")
    assert l.columns == {"Singer.Name", "Singer.Age", "concert.name"}


def test_wildcards_link_table_only():
    assert links("SELECT * FROM singer").to_dict() == {"tables": ["Singer"], "columns": [], "unresolved": []}
    assert links("SELECT T1.* FROM singer AS T1").columns == set()
    assert links("SELECT count(*) FROM concert").columns == set()


def test_order_by_select_alias_is_not_a_column():
    assert links("SELECT name, count(*) AS cnt FROM concert GROUP BY name ORDER BY cnt").columns == {"concert.name"}


def test_double_quoted_string_literal_is_dropped():
    l = links('SELECT age FROM singer WHERE name = "Village"')
    assert l.columns == {"Singer.Age", "Singer.Name"}
    assert l.unresolved == set()


def test_double_quoted_identifier_is_kept():
    assert links('SELECT "Full Name" FROM singer').columns == {"Singer.Full Name"}


def test_backtick_identifier_with_spaces():
    assert links("SELECT `Full Name` FROM singer").columns == {"Singer.Full Name"}


def test_made_up_names_are_unresolved():
    l = links("SELECT foo FROM singer")
    assert l.columns == {"?.foo"} and l.unresolved == {"?.foo"}
    l = links("SELECT T1.foo FROM singer AS T1")
    assert l.columns == {"Singer.foo"} and l.unresolved == {"Singer.foo"}
    l = links("SELECT name FROM nosuch")
    assert "nosuch" in l.tables and "nosuch" in l.unresolved


@pytest.mark.parametrize("sql", ["", "SELEC name FRM singer", "DROP TABLE singer", "SELECT 1; DROP TABLE singer"])
def test_invalid_or_non_query_sql_raises(sql):
    with pytest.raises(SQLParseError):
        links(sql)


def test_trailing_semicolon_ok():
    assert links("SELECT name FROM concert;").columns == {"concert.name"}


@pytest.mark.data
@pytest.mark.parametrize(
    "tables_json, questions, key",
    [
        (BIRD_DEV / "dev_tables.json", BIRD_DEV / "dev.json", "SQL"),
        (SPIDER / "tables.json", SPIDER / "train_spider.json", "query"),
    ],
)
def test_gold_parse_rate(tables_json, questions, key):
    """Phase 1 gate: >98% of gold SQL parses, and almost nothing is left unresolved."""
    if not questions.exists():
        pytest.skip("raw data not downloaded")
    schemas = load_spider_tables(tables_json)
    examples = json.load(open(questions))
    failed, unresolved = [], []
    for ex in examples:
        try:
            l = extract_links(ex[key], schemas[ex["db_id"]])
        except SQLParseError as e:
            failed.append((ex[key], str(e)))
            continue
        if l.unresolved:
            unresolved.append((ex[key], sorted(l.unresolved)))
    n = len(examples)
    assert len(failed) / n < 0.02, failed[:5]
    assert len(unresolved) / n < 0.01, unresolved[:5]
