# server.py
import logging
import os
import threading
import duckdb
import requests
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from langchain_anthropic.chat_models import AnthropicAuthenticationError
from supabase import create_client, Client, AuthApiError, AuthError

from scripts.server.cache import cache

# Import your existing agent & router modules
from scripts.rag.agent import agent_executor, DB_PATH, DataResultPayload
from scripts.rag.router import PreProcessingRouter, QueryAnalysis

app = FastAPI(title="Pulse API", version="1.0.0")
logger = logging.getLogger("uvicorn.error")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") # Service role for backend DB updates
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY")

supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_URL and SUPABASE_KEY
    else None
)

# One shared read-only DuckDB connection per worker; each request gets its own cursor (thread-safe)
_duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None
_duckdb_path: Optional[str] = None
_duckdb_lock = threading.Lock()


def get_db_cursor() -> duckdb.DuckDBPyConnection:
    """Return a per-request cursor on the worker's shared read-only DuckDB connection."""
    global _duckdb_conn, _duckdb_path
    with _duckdb_lock:
        if _duckdb_conn is None or _duckdb_path != str(DB_PATH):
            if not DB_PATH.exists():
                raise HTTPException(
                    status_code=500, detail=f"DuckDB database file not found at: {DB_PATH}"
                )
            if _duckdb_conn is not None:
                _duckdb_conn.close()
            _duckdb_conn = duckdb.connect(str(DB_PATH), read_only=True)
            _duckdb_path = str(DB_PATH)
        return _duckdb_conn.cursor()


@app.on_event("shutdown")
def close_db_connection():
    global _duckdb_conn, _duckdb_path
    with _duckdb_lock:
        if _duckdb_conn is not None:
            _duckdb_conn.close()
            _duckdb_conn = None
            _duckdb_path = None

class VoteRequest(BaseModel):
    politician_id: str
    vote_type: str  # 'approve', 'disapprove', or 'neutral'
    turnstile_token: str

class ChatRequest(BaseModel):
    query: str

class CommentRequest(BaseModel):
    politician_id: str
    text: str
    vote_type: str

# Enable CORS for Vite dev server (and production domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://legipulse.org","https://www.legipulse.org""*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuthApiError)
async def auth_api_error_handler(request: Request, exc: AuthApiError):
    error_msg = getattr(exc, "message", None) or str(exc)
    return JSONResponse(
        status_code=401,
        content={"detail": f"Invalid or expired session token: {error_msg}"},
    )


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError):
    error_msg = getattr(exc, "message", None) or str(exc)
    return JSONResponse(
        status_code=401,
        content={"detail": f"Authentication error: {error_msg}"},
    )


router = PreProcessingRouter()

CHAMBER_ALIASES = {
    "h": "house",
    "house": "house",
    "house of representatives": "house",
    "s": "senate",
    "senate": "senate",
}



def verify_turnstile_token(token: str):
    """Validates token with Cloudflare Turnstile API."""
    if not TURNSTILE_SECRET_KEY:
        raise HTTPException(status_code=503, detail="Bot protection service is not configured.")

    try:
        res = requests.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data={
                "secret": TURNSTILE_SECRET_KEY,
                "response": token
            },
            timeout=10,
        )
        outcome = res.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Bot protection check error: {str(e)}")

    if not outcome.get("success"):
        error_codes = outcome.get("error-codes")
        msg = f"Bot protection check failed: {', '.join(error_codes)}" if error_codes else "Bot protection check failed."
        raise HTTPException(status_code=400, detail=msg)

def get_current_user(authorization: str = Header(...)):
    """Verifies Supabase JWT and checks ID verification flag."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Invalid auth header format. Expected 'Bearer <token>'."
        )

    parts = authorization.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        raise HTTPException(
            status_code=401,
            detail="Invalid auth header format. Token is missing."
        )

    token = parts[1].strip()

    if not supabase:
        raise HTTPException(
            status_code=503,
            detail="Authentication service is not configured."
        )

    # 1. Verify token with Supabase Auth
    try:
        user_res = supabase.auth.get_user(token)
    except (AuthApiError, AuthError) as e:
        error_msg = getattr(e, "message", None) or str(e)
        raise HTTPException(
            status_code=401,
            detail=f"Invalid or expired session token: {error_msg}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=401,
            detail=f"Authentication error: {str(e)}"
        ) from e

    if not user_res or not user_res.user:
        raise HTTPException(status_code=401, detail="Invalid or expired session token.")

    user_id = user_res.user.id

    # 2. Check profile verification state in PostgreSQL
    try:
        profile = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
        if not profile.data or not profile.data.get("is_identity_verified"):
            raise HTTPException(status_code=403, detail="Identity verification required to vote.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=403,
            detail=f"Identity verification check failed: {str(e)}"
        ) from e

    return user_id

# -------------------------------------------------------------------
# HEALTH CHECK ENDPOINTS
# -------------------------------------------------------------------


@app.get("/health", status_code=200)
def health_check():
    """Simple status check for container orchestrators and AWS App Runner."""
    return {"status": "healthy", "app": "LegiPulse API"}
# -------------------------------------------------------------------
# POLITICIANS & DELEGATION DIRECTORY ENDPOINTS
# -------------------------------------------------------------------

@app.get("/api/politicians")
def get_politicians(state: Optional[str] = None, chamber: Optional[str] = None):
    """Fetch the member directory with pre-aggregated legislative metrics."""
    normalized_chamber = CHAMBER_ALIASES.get((chamber or "").strip().casefold())
    cache_key = f"members:list:{(state or 'ALL').upper()}:{normalized_chamber or 'all'}"

    # 1. Try Redis Cache
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # 2. Cache Miss -> Query DuckDB
    conn = get_db_cursor()

    where_clauses = ["1=1"]
    params = []

    if state and state.upper() != "ALL":
        where_clauses.append("members.state_code = ?")
        params.append(state.upper())

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
        records = df.astype(object).where(df.notna(), None).to_dict(orient="records")

        # 3. Store in Upstash Redis (TTL: 24 Hours)
        cache.set(cache_key, records, ex_seconds=86400)

        return records
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Database Query Error: {str(e)}")

@app.get("/api/politicians/{politician_id}")
def get_politician_by_id(politician_id: str):
    """Fetch detailed metadata for a single politician by bioguide ID."""
    cache_key = f"member:profile:{politician_id}"

    # 1. Check Redis Cache
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # 2. Cache Miss -> Query DuckDB
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"DuckDB database file not found at: {DB_PATH}")

    conn = get_db_cursor()
    try:
        sql = """
              SELECT bioguide_id AS id, name, party, state, chamber,
                     district, office_address, phone, website_url
              FROM main.members
              WHERE bioguide_id = ? \
              """
        row = conn.execute(sql, [politician_id]).fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"Politician with ID {politician_id} not found."
            )

        profile = {
            "id": row[0],
            "name": row[1],
            "party": row[2],
            "state": row[3],
            "chamber": row[4],
            "district": row[5],
            "officeAddress": row[6],
            "phone": row[7],
            "websiteUrl": row[8]
        }

        # 3. Store in Upstash Redis (TTL: 24 Hours)
        cache.set(cache_key, profile, ex_seconds=86400)

        return profile
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"DuckDB Member Query Error: {str(e)}"
        )
    finally:
        conn.close()

@app.get("/api/politicians/{politician_id}/donors")
def get_politician_donors(politician_id: str, cycle: Optional[int] = None):
    """Fetch cached OpenFEC top contributors for a politician."""
    cache_key = f"fec:donors:{politician_id}"
    if cycle is not None:
        cache_key += f":cycle_{cycle}"

    # 1. Try Redis Cache
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # 2. Cache Miss -> Query DuckDB
    conn = get_db_cursor()
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
              WHERE bioguide_id = ? \
              """
        params = [politician_id]
        if cycle is not None:
            sql += " AND cycle = ?"
            params.append(str(cycle))
        sql += " ORDER BY amount DESC, donor_name ASC"

        records = conn.execute(sql, params).fetchdf().to_dict(orient="records")

        cache.set(cache_key, records, ex_seconds=86400)

        return records
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
    cache_key = f"fec:summary:{politician_id}"

    # 1. Try Redis Cache
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # 2. Cache Miss -> Query DuckDB
    conn = get_db_cursor()
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
        result = {
            "fecCandidateId": row[0],
            "totalReceipts": row[1],
            "pacContributions": row[2],
            "individualContributions": row[3],
            "coverageEndDate": row[4],
            "summaryAvailable": summary_available,
            "summaryStatus": "available" if summary_available else "unavailable",
        }

        cache.set(cache_key, result, ex_seconds=86400)

        return result
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
    cache_key = f"bills:member:{bioguide_id}"

    # 1. Try Redis Cache
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # 2. Cache Miss -> Query DuckDB
    if not DB_PATH.exists():
        raise HTTPException(status_code=500, detail=f"DuckDB database file not found at: {DB_PATH}")

    conn = get_db_cursor()

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
              LIMIT 10;
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

        records = df.to_dict(orient="records")

        cache.set(cache_key, records, ex_seconds=43200)

        return records
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=f"DuckDB Query Error: {str(e)}")
# -------------------------------------------------------------------
# RAG AGENT EXECUTION ENDPOINT
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

# -------------------------------------------------------------------
# Vote Endpoint
# -------------------------------------------------------------------
@app.post("/api/vote")
def submit_vote(
        request: VoteRequest,
        user_id: str = Depends(get_current_user)
):
    # Step 1: Validate Cloudflare Turnstile token
    verify_turnstile_token(request.turnstile_token)

    if not supabase:
        raise HTTPException(status_code=503, detail="Database service is not configured.")

    # Step 2: Execute UPSERT in Supabase PostgreSQL
    try:
        vote_data = {
            "user_id": user_id,
            "politician_id": request.politician_id,
            "vote_type": request.vote_type
        }

        # Upsert automatically updates the vote if (user_id, politician_id) already exists
        supabase.table("user_votes").upsert(
            vote_data,
            on_conflict="user_id, politician_id"
        ).execute()

        # Sentiment counts are not invalidated here; the short-TTL cache expires on its own.
        return {"status": "success", "message": "Vote recorded successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

def fetch_sentiment_from_supabase(politician_id: str) -> Dict[str, int]:
    """Fallback query to aggregate votes directly from Supabase PostgreSQL."""
    try:
        # Query vote counts grouped by vote_type
        response = (
            supabase.table("user_votes")
            .select("vote_type")
            .eq("politician_id", politician_id)
            .execute()
        )

        data = response.data or []

        # Calculate tallies
        stats = {"approve": 0, "neutral": 0, "disapprove": 0}
        for row in data:
            v_type = row.get("vote_type")
            if v_type in stats:
                stats[v_type] += 1

        return stats
    except Exception as e:
        logger.error(f"Supabase vote aggregation error for {politician_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to calculate vote stats from database."
        )


@app.get("/api/politicians/{politician_id}/votes/stats")
def get_politician_vote_stats(politician_id: str) -> Dict[str, Any]:
    """Fetch public aggregate vote sentiment (Upstash Redis -> Supabase Fallback)."""
    cache_key = f"sentiment:{politician_id}"

    # 1. Read Path: Check Upstash Redis
    cached_stats = cache.get(cache_key)
    if cached_stats is not None:
        return {
            "politicianId": politician_id,
            "sentimentStats": cached_stats,
            "source": "cache"
        }

    # 2. Cache Miss / Fallback: Fetch from Supabase PostgreSQL
    stats = fetch_sentiment_from_supabase(politician_id)

    # 3. Write-Through: Cache in Upstash Redis (TTL: 30 seconds)
    # Votes don't invalidate this key; it expires naturally and the next reader refreshes it from Supabase
    cache.set(cache_key, stats, ex_seconds=30)

    return {
        "politicianId": politician_id,
        "sentimentStats": stats,
        "source": "database"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
