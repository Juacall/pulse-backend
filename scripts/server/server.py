# server.py
import logging
import hashlib
import os
import re
import threading
import time
import duckdb
import requests
from fastapi import FastAPI, Depends, HTTPException, Header, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
from langchain_anthropic.chat_models import AnthropicAuthenticationError
from supabase import create_client, Client, AuthApiError, AuthError

from scripts.server.cache import cache

# Import your existing agent & router modules
from scripts.rag.agent import agent_executor, DB_PATH
from scripts.rag.schema import DataResultPayload
from scripts.rag.payload import sanitize_content_field, sanitize_payload
from scripts.rag.database import connect_duckdb, database_label
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
        if _duckdb_conn is None or _duckdb_path != database_label(DB_PATH):
            if _duckdb_conn is not None:
                _duckdb_conn.close()
            try:
                _duckdb_conn = connect_duckdb(DB_PATH)
            except (FileNotFoundError, RuntimeError) as e:
                raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
            _duckdb_path = database_label(DB_PATH)
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


class RegisterRequest(BaseModel):
    email: str
    password: str

class LoginRequest(BaseModel):
    email: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: str
    email: str

# Enable CORS for Vite dev server (and production domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://127.0.0.1:5173",
                   "https://legipulse.org",
                   "https://www.legipulse.org",
                   "https://lettuce-verse-kindling.ngrok-free.dev",
                   "https://legipulse-backend.internal",
                   "https://legipulse-frontend.internal"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AuthApiError)
async def auth_api_error_handler(request: Request, exc: AuthApiError):
    error_msg = getattr(exc, "message", None) or str(exc)
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": f"Invalid or expired session token: {error_msg}"},
    )


@app.exception_handler(AuthError)
async def auth_error_handler(request: Request, exc: AuthError):
    error_msg = getattr(exc, "message", None) or str(exc)
    return JSONResponse(
        status_code=status.HTTP_401_UNAUTHORIZED,
        content={"detail": f"Authentication error: {error_msg}"},
    )


router = PreProcessingRouter()

ROUTER_CACHE_VERSION = "v1"  # bump when the router prompt or QueryAnalysis schema changes
ROUTER_CACHE_TTL_SECONDS = 7 * 86400


def normalize_query(query: str) -> str:
    """Collapses case, whitespace and trailing punctuation so equivalent questions share a cache entry."""
    return re.sub(r"\s+", " ", query.strip().casefold()).rstrip(" ?!..")


def analyze_query(query: str) -> QueryAnalysis:
    """Runs the pre-processing router, reusing cached analyses to avoid repeat LLM calls."""
    digest = hashlib.sha256(normalize_query(query).encode("utf-8")).hexdigest()[:32]
    model = os.environ.get("ANTHROPIC_ROUTER_MODEL", "claude-haiku-5-5")
    cache_key = f"router:{ROUTER_CACHE_VERSION}:{model}:{digest}"

    cached = cache.get(cache_key)
    if cached is not None:
        try:
            analysis = QueryAnalysis(**cached)
            logger.info("Router cache hit for key %s", cache_key)
            return analysis
        except Exception:
            logger.warning("Ignoring invalid cached router analysis for key %s", cache_key)

    analysis = router.process(query)
    cache.set(cache_key, analysis.model_dump(), ex_seconds=ROUTER_CACHE_TTL_SECONDS)
    return analysis


ANSWER_CACHE_VERSION = "v1"  # bump when the agent prompt, tools or payload format change
ANSWER_CACHE_TTL_SECONDS = 2 * 86400
DATA_VERSION_REFRESH_SECONDS = 300
_data_version_state = {"value": None, "checked_at": 0.0}


def get_data_version() -> Optional[str]:
    """Latest ingestion timestamp, so new ingests automatically invalidate cached answers."""
    now = time.monotonic()
    if (
        _data_version_state["value"] is not None
        and now - _data_version_state["checked_at"] < DATA_VERSION_REFRESH_SECONDS
    ):
        return _data_version_state["value"]

    try:
        conn = get_db_cursor()
        try:
            row = conn.execute(
                "SELECT CAST(MAX(ingested_at) AS VARCHAR) FROM main.raw_bills"
            ).fetchone()
        finally:
            conn.close()
    except Exception as error:
        logger.warning("Could not determine data version; answer caching disabled: %s", error)
        return None

    if not row or not row[0]:
        return None
    _data_version_state.update(value=row[0], checked_at=now)
    return row[0]


def answer_cache_key(query: str, analysis: QueryAnalysis) -> Optional[str]:
    """Key on the normalized question, router-detected scope and data version; None disables caching."""
    data_version = get_data_version()
    if data_version is None:
        return None
    scope = f"{normalize_query(query)}|{analysis.state_code or ''}|{analysis.chamber or ''}|{data_version}"
    digest = hashlib.sha256(scope.encode("utf-8")).hexdigest()[:32]
    return f"chat:answer:{ANSWER_CACHE_VERSION}:{digest}"


def is_cacheable_answer(answer: Dict[str, Any]) -> bool:
    """Only cache real structured results, never fallbacks, empty payloads or text-only downgrades."""
    if answer.get("display_type") not in ("table", "list", "metric"):
        return False
    rows = answer.get("rows")
    return bool(rows) and all(isinstance(r, dict) and r for r in rows)

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
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Bot protection service is not configured.")

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
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Bot protection check error: {str(e)}")

    if not outcome.get("success"):
        error_codes = outcome.get("error-codes")
        msg = f"Bot protection check failed: {', '.join(error_codes)}" if error_codes else "Bot protection check failed."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

def get_current_user(authorization: str = Header(...)):
    """Verifies Supabase JWT and checks ID verification flag."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth header format. Expected 'Bearer <token>'."
        )

    parts = authorization.split(" ", 1)
    if len(parts) < 2 or not parts[1].strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth header format. Token is missing."
        )

    token = parts[1].strip()

    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication service is not configured."
        )

    # 1. Verify token with Supabase Auth
    try:
        user_res = supabase.auth.get_user(token)
    except (AuthApiError, AuthError) as e:
        error_msg = getattr(e, "message", None) or str(e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired session token: {error_msg}"
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        ) from e

    if not user_res or not user_res.user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired session token.")

    user_id = user_res.user.id

    # 2. Check profile verification state in PostgreSQL
    try:
        profile = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
        if not profile.data or not profile.data.get("is_identity_verified"):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Identity verification required to vote.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Identity verification check failed: {str(e)}"
        ) from e

    return user_id

# -------------------------------------------------------------------
# HEALTH CHECK ENDPOINTS
# -------------------------------------------------------------------


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check():
    """Simple status check for container orchestrators and AWS App Runner."""
    return {"status": "healthy", "app": "LegiPulse API"}


# -------------------------------------------------------------------
# AUTH ENDPOINTS
# -------------------------------------------------------------------
@app.post("/api/auth/register", status_code=status.HTTP_201_CREATED)
def register_user(payload: RegisterRequest):
    try:
        # Registers user with Supabase Auth
        res = supabase.auth.sign_up({
            "email": payload.email,
            "password": payload.password
        })

        if not res.user:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Registration failed.")

        return {
            "message": "User registered successfully",
            "userId": res.user.id,
            "email": res.user.email
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

@app.post("/api/auth/login", response_model=LoginResponse)
def login(payload: LoginRequest):
    try:
        # Authenticate credentials against Supabase Auth
        res = supabase.auth.sign_in_with_password({
            "email": payload.email,
            "password": payload.password
        })

        if not res.session or not res.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        return LoginResponse(
            access_token=res.session.access_token,
            refresh_token=res.session.refresh_token,
            user_id=res.user.id,
            email=res.user.email
        )

    except Exception as e:
        # Pass through specific authentication errors from Supabase
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid login credentials"
        )
# -------------------------------------------------------------------
# POLITICIANS & DELEGATION DIRECTORY ENDPOINTS
# -------------------------------------------------------------------

from typing import Optional
from fastapi import APIRouter, Query, HTTPException, status
import duckdb

router = APIRouter()

@app.get("/api/politicians")
def get_politicians(
        search: Optional[str] = Query(None, min_length=2),
        state: Optional[str] = None,
        chamber: Optional[str] = None,
        limit: int = 50
):
    """Fetch the member directory with pre-aggregated legislative metrics and optional search."""
    normalized_chamber = CHAMBER_ALIASES.get((chamber or "").strip().casefold())
    clean_search = search.strip() if search else None

    # 1. Redis Cache Strategy:
    # Only cache state/chamber filter results. Bypass cache if a dynamic search query is active.
    cache_key = f"members:list:v2:{(state or 'ALL').upper()}:{normalized_chamber or 'all'}:{limit}" if not clean_search else None

    if cache_key:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

    # 2. Query Construction
    conn = get_db_cursor()

    where_clauses = ["1=1"]
    params = []

    if state and state.upper() != "ALL":
        where_clauses.append("members.state_code = ?")
        params.append(state.upper())

    if normalized_chamber:
        where_clauses.append("LOWER(TRIM(members.chamber)) = ?")
        params.append(normalized_chamber)

    # Inject case-insensitive name search across member_name
    if clean_search:
        where_clauses.append("members.member_name ILIKE ?")
        params.append(f"%{clean_search}%")

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
            metrics.pass_rate AS passRate,
            att.total_eligible_votes AS rc_total,
            att.missed_votes AS rc_missed,
            att.present_abstain_votes AS rc_present
        FROM main.stg_members AS members
        LEFT JOIN main.dim_members AS metrics
            ON members.member_id = metrics.bioguide_id
        LEFT JOIN main.mart_member_attendance AS att
            ON members.member_id = att.member_id
        WHERE {where_sql}
        ORDER BY members.member_name ASC
        LIMIT ?;
    """
    params.append(limit)

    try:
        df = conn.execute(sql, params).fetchdf()
        conn.close()

        records = df.astype(object).where(df.notna(), None).to_dict(orient="records")

        for record in records:
            total = record.pop("rc_total")
            missed = record.pop("rc_missed")
            present = record.pop("rc_present")
            record["rollCallVotes"] = None if not total else {
                "totalVotes": total,
                "votesCast": total - missed,
                "missedVotes": missed,
                "presentVotes": present,
                "attendancePercentage": round((total - missed) / total * 100.0, 2),
            }

        # 3. Store non-search results in Upstash Redis (TTL: 24 Hours)
        if cache_key:
            cache.set(cache_key, records, ex_seconds=86400)

        return records

    except Exception as e:
        conn.close()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database Query Error: {str(e)}"
        )

@app.get("/api/politicians/{politician_id}")
def get_politician_by_id(politician_id: str):
    """Fetch detailed metadata for a single politician by bioguide ID."""
    cache_key = f"member:profile:{politician_id}"

    # 1. Check Redis Cache
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    # 2. Cache Miss -> Query DuckDB
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
                status_code=status.HTTP_404_NOT_FOUND,
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"DuckDB Member Query Error: {str(e)}"
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
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Donor Query Error: {str(e)}"
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
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
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
                status_code=status.HTTP_404_NOT_FOUND,
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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Donation Summary Query Error: {str(e)}"
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
    conn = get_db_cursor()

    sql = """
          SELECT DISTINCT
              activity_id AS id,
              coalesce(bill_title, activity_title) AS title,
              relationship AS type,
              bill_status AS statusCategory,
              CAST(bill_introduced_date AS VARCHAR) AS introducedDate,
              bill_latest_action_text AS latestAction,
              CAST(bill_latest_action_date AS VARCHAR) AS actionDate
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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"DuckDB Query Error: {str(e)}")
# -------------------------------------------------------------------
# RAG AGENT EXECUTION ENDPOINT
# -------------------------------------------------------------------

@app.post("/api/chat")
def chat_with_agent(request: ChatRequest):
    """Runs input through PreProcessingRouter before invoking the LangChain agent."""
    if not request.query.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Query string cannot be empty.")

    # Step 1: Pre-process query with Haiku Router
    try:
        analysis: QueryAnalysis = analyze_query(request.query)
        answer_key = answer_cache_key(request.query, analysis)
        router_analysis = {
            "intent": analysis.intent,
            "detectedState": analysis.state_code or "NATIONAL",
            "rewrittenPrompt": analysis.rewritten_prompt
        }

        # Step 2: Serve a previously computed answer for this data version if we have one
        if answer_key:
            cached_answer = cache.get(answer_key)
            if cached_answer is not None:
                logger.info("Chat answer cache hit for key %s", answer_key)
                return {
                    "routerAnalysis": router_analysis,
                    "synthesizedAnswer": cached_answer,
                    "cached": True,
                }

        # Step 3: Execute Agent Executor
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
        final_answer = clean_payload.model_dump(exclude={"list_items"})
        print(f'Final Response {final_answer}')

        if answer_key and is_cacheable_answer(final_answer):
            cache.set(answer_key, final_answer, ex_seconds=ANSWER_CACHE_TTL_SECONDS)
    except AnthropicAuthenticationError as error:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Anthropic rejected the configured API key. Update "
                "ANTHROPIC_API_KEY in the project .env file and restart the server."
            ),
        ) from error

    return {
        "routerAnalysis": router_analysis,
        "synthesizedAnswer": final_answer,
        "cached": False,
    }


# -------------------------------------------------------------------
# Vote Endpoints
# -------------------------------------------------------------------

@app.get("/api/politicians/{politician_id}/roll-call-votes")
def get_official_roll_call_votes(politician_id: str):
    """Returns official floor attendance and roll-call voting stats from MotherDuck."""
    cache_key = f"votes:rollcall:{politician_id}"
    cached_data = cache.get(cache_key)
    if cached_data is not None:
        return cached_data

    conn = get_db_cursor()
    query = """
            SELECT total_eligible_votes, missed_votes, present_abstain_votes
            FROM main.mart_member_attendance
            WHERE member_id = ?
            """
    try:
        row = conn.execute(query, [politician_id]).fetchone()
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"DuckDB Query Error: {str(e)}")
    finally:
        conn.close()

    if not row or not row[0]:
        # No roll-call records: report unknown rather than a misleading 100% attendance
        result = {
            "totalVotes": 0,
            "votesCast": 0,
            "missedVotes": 0,
            "presentVotes": 0,
            "attendancePercentage": None,
        }
    else:
        total, missed, present = row
        result = {
            "totalVotes": total,
            "votesCast": total - missed,
            "missedVotes": missed,
            "presentVotes": present,
            "attendancePercentage": round((total - missed) / total * 100.0, 2),
        }

    cache.set(cache_key, result, ex_seconds=86400)
    return result

@app.post("/api/vote")
def submit_vote(
        request: VoteRequest,
        user_id: str = Depends(get_current_user)
):
    # Step 1: Validate Cloudflare Turnstile token
    verify_turnstile_token(request.turnstile_token)

    if not supabase:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service is not configured.")

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
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")

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
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to calculate vote stats from database."
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
