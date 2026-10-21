# scripts/rag/agent.py
import os
from pathlib import Path
from typing import Optional

from scripts.rag.environment import (
    anthropic_workspace_headers,
    load_project_environment,
)

load_project_environment()

import duckdb
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_voyageai import VoyageAIEmbeddings
from typing import List, Dict, Any, Optional, Literal
from scripts.rag.router import PreProcessingRouter
from scripts.rag.database import (
    LOCAL_DB_PATH as DB_PATH,
    connect_duckdb,
    use_motherduck,
)

EMBEDDING_DIMENSIONS = 512
MAX_SQL_RESULT_ROWS = 50
MAX_SQL_CELL_CHARACTERS = 1_000
MAX_SQL_OUTPUT_CHARACTERS = 12_000

from scripts.rag.schema import DataResultPayload


def _format_query_result(columns, rows, *, rows_truncated=False) -> str:
    """Render a bounded DuckDB result as Markdown without requiring pandas."""

    def cell(value):
        if value is None:
            return ""
        value = str(value).replace("|", "\\|").replace("\r", " ").replace("\n", " ")
        if len(value) > MAX_SQL_CELL_CHARACTERS:
            value = value[:MAX_SQL_CELL_CHARACTERS] + "... [cell truncated]"
        return value

    header = "| " + " | ".join(cell(column) for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    output = [header, separator]
    truncated_by_size = False
    for row in rows:
        rendered_row = "| " + " | ".join(cell(value) for value in row) + " |"
        if sum(map(len, output)) + len(rendered_row) + 1 > MAX_SQL_OUTPUT_CHARACTERS:
            truncated_by_size = True
            break
        output.append(rendered_row)

    result = "\n".join(output)
    if rows_truncated or truncated_by_size:
        result += (
            f"\n\nResult truncated to at most {MAX_SQL_RESULT_ROWS} rows and "
            f"{MAX_SQL_OUTPUT_CHARACTERS} characters. Add filters or SQL LIMIT "
            "to narrow the result, or explicitly request a specific page."
        )
    return result


embeddings_model = VoyageAIEmbeddings(model="voyage-3-lite")


@tool
def vector_search_bills(query: str, state_code: Optional[str] = None, top_k: int = 5) -> str:
    """Perform semantic vector similarity search over legislative activity documents.
    Use this tool when searching for bill contents, summaries, specific topics, or legislative intent.

    Args:
        query: Natural language query string.
        state_code: Optional 2-letter postal state code filter (e.g., 'WA', 'CA', 'NY').
        top_k: Number of nearest neighbor documents to return (default 5).
    """
    query_vector = embeddings_model.embed_query(query)
    if len(query_vector) != EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"Voyage returned a {len(query_vector)}-dimension query vector; "
            f"expected {EMBEDDING_DIMENSIONS}."
        )

    conn = connect_duckdb(DB_PATH)
    try:
        embedding_column = next(
            (
                row[1]
                for row in conn.execute("DESCRIBE main.doc_embeddings").fetchall()
                if row[0] == "embedding"
            ),
            None,
        )
        expected_type = f"FLOAT[{EMBEDDING_DIMENSIONS}]"
        if embedding_column != expected_type:
            raise RuntimeError(
                "The stored vector index is incompatible with Voyage "
                f"({embedding_column or 'embedding column missing'}). "
                "Rebuild it by running scripts/rag/indexer.py."
            )

        if not use_motherduck():
            conn.execute("LOAD vss;")
            conn.execute("SET hnsw_enable_experimental_persistence = true")

        sql = """
              SELECT page_content, chamber, state_code, party_name, activity_type,
                     array_distance(embedding, ?::FLOAT[512]) AS distance
              FROM main.doc_embeddings
              WHERE 1=1 \
              """
        params = [query_vector]

        if state_code:
            sql += " AND state_code = ?"
            params.append(state_code.upper())

        sql += " ORDER BY distance ASC LIMIT ?"
        params.append(top_k)

        results = conn.execute(sql, params).fetchall()
    finally:
        conn.close()

    if not results:
        return "No relevant legislative documents found."

    formatted = []
    for row in results:
        content, chamber, state, party, act_type, dist = row
        formatted.append(f"[Distance: {dist:.4f} | {chamber}-{state} ({party}) - {act_type}]\n{content}")

    return "\n\n---\n\n".join(formatted)


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


@tool(response_format="content_and_artifact")
def execute_sql_query(sql_query: str):
    """Execute a structured SQL query against DuckDB.
    Table: mart_legislative_activity
    Columns & Conventions:
      - target_member_state: Full state name in UPPERCASE (e.g., 'WASHINGTON', 'NEW YORK')
      - target_member_state_code: Two-letter uppercase postal code (e.g., 'WA', 'NY')
      - relationship: Exact values are 'sponsor', 'cosponsor',
        'amendment_sponsor', 'sponsored bill received amendment'
      - bill_title, bill_id, activity_type, target_member_name, target_member_party

    When the user asks for both a count and examples, return both in one query.
    Use a CTE grouped to one row per bill_id (with a representative bill_title),
    COUNT(*) OVER () AS total_bills, and LIMIT 10 for examples. The mart
    repeats activities by county/member/activity, so deduplicate bill IDs
    before counting.
    """
    conn = connect_duckdb(DB_PATH)
    try:
        cursor = conn.execute(sql_query)
        if cursor.description is None:
            return "SQL Error: query did not return a result set.", None
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(MAX_SQL_RESULT_ROWS + 1)
        rows_truncated = len(rows) > MAX_SQL_RESULT_ROWS
        rows = rows[:MAX_SQL_RESULT_ROWS]
        content = _format_query_result(columns, rows, rows_truncated=rows_truncated)
        # The artifact carries the real rows so the server never relies on the model to copy them
        artifact = {
            "columns": columns,
            "rows": [[_json_safe(v) for v in row] for row in rows],
            "truncated": rows_truncated,
        }
        return content, artifact
    except Exception as e:
        return f"SQL Error: {str(e)}", None
    finally:
        conn.close()


# Initialize LLM, Tools, Agent, and Pre-Processor
llm = ChatAnthropic(
    model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
    default_headers=anthropic_workspace_headers(),
)
tools = [vector_search_bills, execute_sql_query]
agent_executor = create_agent(
    llm,
    tools,
    response_format = DataResultPayload,
    system_prompt = (
        "Answer legislative data questions using the available tools.\n\n"
        "QUERY EXECUTION & SQL GUIDELINES:\n"
        "- For requests combining an aggregate and examples, prefer one SQL query "
        "returning the aggregate alongside a small representative sample; do not "
        "issue separate exploratory count and listing queries.\n"
        "- Deduplicate bill counts by bill_id because mart rows repeat by member and county.\n"
        "- Keep SQL result requests narrow and respect the tool's truncation notice.\n"
        "- When performing semantic vector search for a specific state, pass the 2-letter uppercase "
        "state_code parameter to `vector_search_bills`.\n\n"
        "STATE FILTERING RULES (DuckDB mart_legislative_activity):\n"
        "- Use `target_member_state_code = 'WA'` for 2-letter postal codes ('WA', 'CA', 'NY').\n"
        "- Use `target_member_state = 'WASHINGTON'` if filtering on full uppercase state names.\n"
        "- NEVER filter `target_member_state = 'WA'` (this column holds full names, so it will return 0 rows).\n\n"
        "RESPONSE FORMATTING CONTRACT (DataResultPayload):\n"
        "The server fills in table rows from your SQL results, so never retype or invent them.\n"
        "1. METRIC RESPONSES (display_type='metric'):\n"
        "   - rows: [{'label': '<Metric Name>', 'value': '<Number>', 'subtext': '<Context>'}]. Never output empty dicts.\n\n"
        "2. TABLE RESPONSES / BILL LISTINGS (display_type='table'):\n"
        "   - Use for bills, statuses and any multi-row listing; leave `rows` empty.\n"
        "   - Set `columns` to readable headers in the same order as the SQL result columns "
        "(e.g. ['Bill ID', 'Title', 'Status', 'Latest Action Date']).\n"
        "   - Put dataset scope notes (e.g. 'Federal HR/S bills only, no state AB/SB') and thematic takeaways in 'summary'.\n\n"
        "3. LIST RESPONSES (display_type='list'):\n"
        "   - Use for topic/theme breakdowns. Fill `list_items` (leave `rows` empty): one item per topic, 4-8 items, "
        "e.g. title='Health care', description='~12% of bills, e.g. HR-123'.\n"
        "   - `summary` is 1-3 sentences: the headline number or takeaway only. Put detail and caveats "
        "(methodology, scope) in the rows, never in a long summary paragraph.\n"
        "   - Use 'text' only for a short narrative when no breakdown exists.\n\n"
        "Do not run placeholder queries such as SELECT 1."
    ),
)
router = PreProcessingRouter()


def run_pipeline(user_query: str):
    """Executes pre-processing query routing followed by agent execution."""
    print(f"\n[Raw User Query]: {user_query}")
    print("=" * 60)

    # 1. Pre-process query to normalize parameters & inject instructions
    analysis = router.process(user_query)
    print(f"[Router Analysis] Intent: {analysis.intent} | State Code: {analysis.state_code}")
    print(f"[Augmented Agent Prompt]: {analysis.rewritten_prompt}\n" + "-" * 60)

    # 2. Stream agent execution
    for chunk in agent_executor.stream({"messages": [("user", analysis.rewritten_prompt)]}):
        print(chunk)


if __name__ == "__main__":
    prompt = "How many bills were sponsored by members from Washington state , and what are the main topics of those bills?"
    run_pipeline(prompt)