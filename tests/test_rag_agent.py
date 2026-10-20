from pathlib import Path
import importlib
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def _load_agent(monkeypatch):
    monkeypatch.setenv("VOYAGE_API_KEY", "test-voyage-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    module = importlib.import_module("rag.agent")
    # Importing reloads .env; keep tests on the local temp database
    monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
    return module


def test_execute_sql_query_truncates_large_result_sets(tmp_path, monkeypatch):
    agent = _load_agent(monkeypatch)
    db_path = tmp_path / "agent.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("CREATE TABLE sample AS SELECT range AS row_id FROM range(100)")
    conn.close()
    monkeypatch.setattr(agent, "DB_PATH", db_path)

    result = agent.execute_sql_query.invoke(
        {"sql_query": "SELECT row_id FROM sample ORDER BY row_id"}
    )

    assert "| 49 |" in result
    assert "| 50 |" not in result
    assert "Result truncated to at most 50 rows" in result


def test_execute_sql_query_bounds_long_cell_and_output(tmp_path, monkeypatch):
    agent = _load_agent(monkeypatch)
    db_path = tmp_path / "agent-long.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute(
        "CREATE TABLE sample AS "
        "SELECT repeat('x', 2000) AS value FROM range(100)"
    )
    conn.close()
    monkeypatch.setattr(agent, "DB_PATH", db_path)

    result = agent.execute_sql_query.invoke(
        {"sql_query": "SELECT value FROM sample"}
    )

    assert "cell truncated" in result
    assert len(result) < agent.MAX_SQL_OUTPUT_CHARACTERS + 300
    assert "Result truncated" in result
