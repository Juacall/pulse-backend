# server.py
import os
import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from langchain_anthropic.chat_models import AnthropicAuthenticationError

# Import your existing agent & router modules
from scripts.rag.agent import agent_executor, DB_PATH, DataResultPayload
from scripts.rag.router import PreProcessingRouter, QueryAnalysis

app = FastAPI(title="LegisPulse API", version="1.0.0")

# Enable CORS for Vite dev server (and production domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

router = PreProcessingRouter()

CHAMBER_ALIASES = {
    "h": "house",
    "house": "house",
    "house of representatives": "house",
    "s": "senate",
    "senate": "senate",
}

class ChatRequest(BaseModel):
    query: str

class VoteRequest(BaseModel):
    politician_id: str
    vote_type: str  # 'approve', 'disapprove', or 'neutral'

class CommentRequest(BaseModel):
    politician_id: str
    text: str
    vote_type: str

# -------------------------------------------------------------------
# 1. POLITICIANS & DELEGATION DIRECTORY ENDPOINTS
# -------------------------------------------------------------------

@app.get("/api/politicians")
def get_politicians(state: Optional[str] = None, chamber: Optional[str] = None):
    """Fetch the member directory with pre-aggregated legislative metrics."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)

    where_clauses = ["1=1"]
    params = []

    if state and state.upper() != "ALL":
        where_clauses.append("members.state_code = ?")
        params.append(state.upper())

    normalized_chamber = CHAMBER_ALIASES.get((chamber or "").strip().casefold())
    if normalized_chamber:
        where_clauses.append("LOWER(TRIM(members.chamber)) = ?")
        params.append(normalized_chamber)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            members.member_id AS id,
            members.member_name AS name,
            members.state_code AS state,
            members.state_name AS stateName,
            members.chamber AS chamber,
            members.party_name AS party,
            COALESCE(metrics.sponsored_count, 0) AS sponsoredCount,
            COALESCE(metrics.cosponsored_count, 0) AS cosponsoredCount,
            COALESCE(metrics.enacted_count, 0) AS enactedCount,
            metrics.pass_rate AS passRate
        FROM main.stg_members AS members
        LEFT JOIN main.dim_members AS metrics
            ON members.member_id = metrics.bioguide_id
        WHERE {where_sql}
        ORDER BY members.member_name ASC
        LIMIT 100;
    """

    try:
        df = conn.execute(sql, params).fetchdf()
        conn.close()
        return df.astype(object).where(df.notna(), None).to_dict(orient="records")
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database Query Error: {str(e)}")


@app.get("/api/politicians/{politician_id}/donors")
def get_politician_donors(politician_id: str, cycle: Optional[int] = None):
    """Fetch cached OpenFEC top contributors for a politician."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        donor_table = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'member_top_donors'
        """).fetchone()[0]
        if not donor_table:
            raise HTTPException(
                status_code=503,
                detail="Donor data is not available; run the OpenFEC donor ingestion first.",
            )

        sql = """
            SELECT donor_name AS donorName, amount, industry, cycle
            FROM main.member_top_donors
            WHERE bioguide_id = ?
        """
        params = [politician_id]
        if cycle is not None:
            sql += " AND cycle = ?"
            params.append(str(cycle))
        sql += " ORDER BY amount DESC, donor_name ASC"
        return conn.execute(sql, params).fetchdf().to_dict(orient="records")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Donor Query Error: {str(e)}"
        )
    finally:
        conn.close()


@app.get("/api/politicians/{politician_id}/donations/summary")
def get_politician_donation_summary(politician_id: str):
    """Fetch aggregate OpenFEC campaign finance totals for a politician."""
    conn = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        summary_table = conn.execute("""
            SELECT COUNT(*)
            FROM information_schema.tables
            WHERE table_schema = 'main' AND table_name = 'member_fec_summary'
        """).fetchone()[0]
        if not summary_table:
            raise HTTPException(
                status_code=503,
                detail="Donation summary data is not available; run the OpenFEC summary ingestion first.",
            )

        row = conn.execute(
            """
            SELECT fec_candidate_id, total_receipts, pac_contributions,
                   individual_contributions, coverage_end_date
            FROM main.member_fec_summary
            WHERE bioguide_id = ?
            """,
            [politician_id],
        ).fetchone()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No donation summary found for politician {politician_id}.",
            )

        summary_available = any(value is not None for value in row[1:4])
        return {
            "fecCandidateId": row[0],
            "totalReceipts": row[1],
            "pacContributions": row[2],
            "individualContributions": row[3],
            "coverageEndDate": row[4],
            "summaryAvailable": summary_available,
            "summaryStatus": "available" if summary_available else "unavailable",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Donation Summary Query Error: {str(e)}"
        )
    finally:
        conn.close()

@app.get("/api/politicians/{bioguide_id}/bills")
def get_politician_key_bills(bioguide_id: str) -> List[Dict[str, Any]]:
    """Fetch top sponsored and cosponsored bills dynamically from DuckDB for a member."""
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"DuckDB database file not found at: {DB_PATH}")

    conn = duckdb.connect(str(DB_PATH), read_only=True)

    sql = """
          SELECT DISTINCT
              activity_id AS id,
              coalesce(bill_title, activity_title) AS title,
              relationship AS type,
              bill_status AS statusCategory,
              CAST(bill_introduced_date AS VARCHAR) AS introducedDate,
              bill_latest_action_text AS latestAction,
              bill_latest_action_date AS actionDate
          FROM mart_legislative_activity
          WHERE target_member_id = ?
            AND activity_type = 'bill'
          ORDER BY
              CASE WHEN relationship = 'sponsor' THEN 1 ELSE 2 END,
              bill_latest_action_date DESC
              LIMIT 10; \
          """

    try:
        df = conn.execute(sql, [bioguide_id]).fetchdf()
        conn.close()

        # Handle null values gracefully
        df = df.fillna({
            "title": "Untitled Bill",
            "type": "cosponsor",
            "statusCategory": "Introduced",
            "introducedDate": "",
            "latestAction": "No recent action recorded",
            "actionDate": ""
        })

        return df.to_dict(orient="records")
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DuckDB Query Error: {str(e)}")
# -------------------------------------------------------------------
# 2. RAG AGENT EXECUTION ENDPOINT
# -------------------------------------------------------------------

@app.post("/api/chat")
def chat_with_agent(request: ChatRequest):
    """Runs input through PreProcessingRouter before invoking the LangChain agent."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    # Step 1: Pre-process query with Haiku Router
    try:
        analysis: QueryAnalysis = router.process(request.query)

        # Step 2: Execute Agent Executor
        response = agent_executor.invoke({"messages": [("user", analysis.rewritten_prompt)]})
        print(f'Raw Response: {response}\n')
        # Extract the structured DataResultPayload generated by response_format
        raw_payload: DataResultPayload = response.get("structured_response")
        print(f'Raw Structured Response {raw_payload}')

        if not raw_payload:
            raw_content = response["messages"][-1].content
            raw_payload = DataResultPayload(
                display_type="text",
                title="Result",
                summary=sanitize_content_field(raw_content)
            )
        clean_payload = sanitize_payload(raw_payload,response=response)
        print(f'Clean Payload {clean_payload}')
        # Convert to Python dict for FastAPI JSON response
        final_answer = clean_payload.model_dump()
        print(f'Final Response {final_answer}')
    except AnthropicAuthenticationError as error:
        raise HTTPException(
            status_code=503,
            detail=(
                "Anthropic rejected the configured API key. Update "
                "ANTHROPIC_API_KEY in the project .env file and restart the server."
            ),
        ) from error

    return {
        "routerAnalysis": {
            "intent": analysis.intent,
            "detectedState": analysis.state_code or "NATIONAL",
            "rewrittenPrompt": analysis.rewritten_prompt
        },
        "synthesizedAnswer": final_answer
    }

def sanitize_content_field(content: Any) -> str:
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        return "\n\n".join(text_parts)
    return str(content) if content is not None else ""


def parse_markdown_table_to_rows(markdown_str: str) -> List[Dict[str, Any]]:
    """Converts a Markdown table string from ToolMessage into structured row dicts."""
    lines = [line.strip() for line in markdown_str.strip().split("\n") if line.strip()]
    if len(lines) < 3:
        return []

    # Extract headers
    headers = [h.strip() for h in lines[0].split("|") if h.strip()]
    rows = []

    # Parse data rows (skip index 1 divider line)
    for line in lines[2:]:
        if "|" in line:
            cells = [c.strip() for c in line.split("|")[1:-1]]
            if len(cells) == len(headers):
                row_dict = {headers[i]: cells[i] for i in range(len(headers))}
                rows.append(row_dict)

    return rows


def sanitize_payload(payload: DataResultPayload, response: Optional[Dict[str, Any]] = None) -> DataResultPayload:
    """Safely cleans fields and auto-reconstructs broken/empty payload rows from execution history."""
    if payload.summary:
        payload.summary = sanitize_content_field(payload.summary)
    if payload.title:
        payload.title = sanitize_content_field(payload.title)

    # -------------------------------------------------------------
    # CASE 1: Fix Empty Metric Responses (rows = [{}])
    # -------------------------------------------------------------
    if payload.display_type == "metric":
        extracted_val = None
        import re
        if payload.rows and len(payload.rows) > 0 and isinstance(payload.rows[0], dict) and payload.rows[0]:
            first_row = payload.rows[0]
            for k in ["value", "count", "total_bills", "total"]:
                if k in first_row and first_row[k] is not None:
                    extracted_val = str(first_row[k])
                    break

        if not extracted_val and response and "messages" in response:
            import re
            for msg in reversed(response["messages"]):
                if hasattr(msg, "content") and isinstance(msg.content, str) and "|" in msg.content:
                    # Look for the first row of numbers in the Markdown table output
                    lines = [line.strip() for line in msg.content.split("\n") if line.strip()]
                    for line in lines:
                        if not line.startswith("| ---") and not "total_bills" in line and "|" in line:
                            cells = [c.strip() for c in line.split("|") if c.strip()]
                            # The first numeric column in the SQL output table is the main total
                            numeric_cells = [c for c in cells if c.isdigit()]
                            if numeric_cells:
                                extracted_val = numeric_cells[0]
                                break
                if extracted_val:
                    break

        if not extracted_val and payload.summary:
            import re
            found_numbers = re.findall(r'\b[1-9]\d*\b', payload.summary)
            if found_numbers:
                extracted_val = found_numbers[0]

        payload.rows = [{
            "label": payload.title or "Total Count",
            "value": extracted_val or "0",
            "subtext": "Query Result"
        }]

    # -------------------------------------------------------------
    # CASE 2: Fix Empty Table Responses (rows = [{}, {}])
    # -------------------------------------------------------------
    elif payload.display_type == "table":
        # Check if rows are empty dicts
        has_valid_rows = payload.rows and any(isinstance(r, dict) and len(r) > 0 for r in payload.rows)

        if not has_valid_rows and response and "messages" in response:
            # Scan history backwards for the last valid SQL table output
            for msg in reversed(response["messages"]):
                if hasattr(msg, "content") and isinstance(msg.content, str) and "|" in msg.content and "---" in msg.content:
                    reconstructed_rows = parse_markdown_table_to_rows(msg.content)
                    if reconstructed_rows:
                        # Take top 10 items max
                        payload.rows = reconstructed_rows[:10]
                        if not payload.columns and reconstructed_rows:
                            payload.columns = list(reconstructed_rows[0].keys())
                        break

    return payload

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)