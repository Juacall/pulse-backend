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

DB_PATH = Path(__file__).resolve().parents[2] / "dev.duckdb"
EMBEDDING_DIMENSIONS = 512
MAX_SQL_RESULT_ROWS = 50
MAX_SQL_CELL_CHARACTERS = 1_000
MAX_SQL_OUTPUT_CHARACTERS = 12_000


class DataResultPayload(BaseModel):
    display_type: Literal["metric", "table", "list", "text"] = Field(
        description=(
            "Type of UI element to render:\n"
            "- 'metric': Single key performance indicators or counts.\n"
            "- 'table': Multi-column tabular data.\n"
            "- 'list': Key-value or bullet-point summaries.\n"
            "- 'text': Pure narrative answers."
        )
    )
    title: Optional[str] = Field(default=None, description="Header text for the UI card.")
    summary: Optional[str] = Field(default=None, description="Detailed text explanation or takeaway.")
    columns: Optional[List[str]] = Field(
        default=None,
        description="List of column headers for 'table' display type (e.g. ['Bill ID', 'Title', 'Status'])."
    )
    rows: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "Data rows matching the display type:\n"
            "• CRITICAL FOR 'metric': Must be a list with a dict formatted exactly like:\n"
            "  [{'label': 'Total Sponsored Bills', 'value': '323', 'subtext': 'WA Delegation'}]\n"
            "• FOR 'table': List of dicts keyed by column names e.g. [{'Bill ID': 'HR-123', ...}]\n"
            "• FOR 'list': List of dicts e.g. [{'title': 'Key Focus', 'description': '...'}]"
        )
    )


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

    conn = duckdb.connect(DB_PATH, read_only=True)
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


@tool
def execute_sql_query(sql_query: str) -> str:
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
    conn = duckdb.connect(DB_PATH, read_only=True)
    try:
        cursor = conn.execute(sql_query)
        if cursor.description is None:
            return "SQL Error: query did not return a result set."
        columns = [column[0] for column in cursor.description]
        rows = cursor.fetchmany(MAX_SQL_RESULT_ROWS + 1)
        rows_truncated = len(rows) > MAX_SQL_RESULT_ROWS
        return _format_query_result(
            columns,
            rows[:MAX_SQL_RESULT_ROWS],
            rows_truncated=rows_truncated,
        )
    except Exception as e:
        return f"SQL Error: {str(e)}"
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
        "1. METRIC RESPONSES (display_type='metric'):\n"
        "   - NEVER output empty dicts [{}] in rows.\n"
        "   - MUST include rows: [{'label': '<Title/Metric Name>', 'value': '<Number>', 'subtext': '<Context>'}]\n\n"
        "2. TABLE RESPONSES (display_type='table'):\n"
        "   - NEVER output empty dicts [{}] in rows.\n"
        "   - Every dict in `rows` MUST contain key-value pairs corresponding to your `columns`.\n"
        "   - Limit `rows` to 5-10 representative items to avoid output truncation or empty objects.\n\n"
        "3. QUALITATIVE RESPONSES (display_type='text' or 'list'):\n"
        "   - Use 'text' or 'list' when no tabular rows exist."
        "2. BILL LISTINGS / TABULAR SEARCHES:\n"
        "   - Set display_type = 'table'.\n"
        "   - NEVER put raw bill lists or bulleted status breakdowns solely inside 'summary'.\n"
        "   - Populate 'columns' with key fields (e.g. ['Bill ID', 'Title', 'Status', 'Latest Action Date', 'Theme']).\n"
        "   - Populate 'rows' with representative bill dicts (e.g. S-160, HR-3922, S-1462, HR-345).\n"
        "   - Put dataset scope notes (e.g. 'Federal HR/S bills only, no state AB/SB') and overall thematic takeaways in 'summary'.\n\n"
        "3. QUALITATIVE OVERVIEWS:\n"
        "   - Set display_type = 'list' or 'text' ONLY when no structured numbers or tabular rows exist."
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