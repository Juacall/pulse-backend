# server.py
import logging
import hashlib
import os
import re
import threading
import time
import duckdb
import requests
import stripe
import httpx
from datetime import datetime, timezone
from fastapi import FastAPI,APIRouter,Query, Depends, HTTPException, Header, Request, status,BackgroundTasks
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
from scripts.server.rate_limiter import RateLimiter
router = APIRouter()

app = FastAPI(title="Pulse API", version="1.0.0")
logger = logging.getLogger("uvicorn.error")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") # Service role for backend DB updates
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")

supabase: Optional[Client] = (
    create_client(SUPABASE_URL, SUPABASE_KEY)
    if SUPABASE_URL and SUPABASE_KEY
    else None
)

# One shared read-only DuckDB connection per worker; each request gets its own cursor (thread-safe)
_duckdb_conn: Optional[duckdb.DuckDBPyConnection] = None
_duckdb_path: Optional[str] = None
_duckdb_lock = threading.Lock()

standard_limiter = RateLimiter(requests_per_minute=120, burst_limit=30)
search_limiter = RateLimiter(requests_per_minute=30, burst_limit=10)
ai_console_limiter = RateLimiter(requests_per_minute=5, burst_limit=2)

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

httpx_client = httpx.AsyncClient(timeout=3.0)

class VoteRequest(BaseModel):
    politician_id: str
    vote_type: str  # 'approve', 'disapprove', or 'neutral'

class ChatRequest(BaseModel):
    query: str

class TurnstileVerifyRequest(BaseModel):
    turnstile_token: str

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

# Pydantic schema matching the payload sent by PoliticianActionButtons.jsx
class ClickEvent(BaseModel):
    bioguide_id: str
    button_type: str
    destination_url: str
    timestamp: Optional[str] = None

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

async def verify_authenticated_user(authorization: str = Header(None)) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.split(" ")[1]

    try:
        # Validate JWT session directly via Supabase Auth
        user_response = supabase.auth.get_user(token)
        if not user_response or not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")

        user = user_response.user
        return {
            "sub": user.id,
            "email": user.email
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")




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
# TELEMETRY ENDPOINTS
# -------------------------------------------------------------------
def log_click_to_db(event_data: dict, client_ip: Optional[str], user_agent: Optional[str]):
    """Background task to store telemetry events without blocking the HTTP response."""

    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not configured."
        )
    try:
        created_at = event_data.get("timestamp") or datetime.now(timezone.utc).isoformat()

        payload = {
            "bioguide_id": event_data["bioguide_id"],
            "button_type": event_data["button_type"],
            "destination_url": event_data["destination_url"],
            "client_ip": client_ip,
            "user_agent": user_agent,
            "clicked_at": created_at,
        }

        # Insert directly into Supabase table
        response = supabase.table("raw_telemetry_clicks").insert(payload).execute()

    except Exception as e:
        logger.error(f"Failed to log click telemetry to Supabase: {str(e)}")


@app.post("/api/telemetry/click", status_code=status.HTTP_204_NO_CONTENT)
def track_click(
        event: ClickEvent,
        background_tasks: BackgroundTasks,
        request: Request
):

    client_ip = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")


    # Pass uncalled function handle followed by its positional/keyword arguments
    background_tasks.add_task(
        log_click_to_db,
        event_data=event.model_dump(),
        client_ip=client_ip,
        user_agent=user_agent,
    )

    return None

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

@app.post("/api/identity/create-session")
async def create_verification_session(user: dict = Depends(verify_authenticated_user)):
    user_id = user["sub"]  # Supabase User UUID
    # Explicit fallback check for API key
    if not stripe.api_key:
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe API key is unconfigured on server.")

    frontend_url = os.getenv("FRONTEND_URL", "https://lettuce-verse-kindling.ngrok-free.dev")
    return_url = f"{frontend_url}/account?verification=success"
    session = stripe.identity.VerificationSession.create(
        type="document",
        client_reference_id=user_id,  # 👈 Binds Stripe session directly to Supabase user ID
        provided_details={
            "email": user["email"],
        },
        options={
            "document": {
                "require_matching_selfie": True,
                "require_address": True,
            }
        },
        return_url=return_url
    )
    return {"client_secret": session.client_secret, "url": session.url}



@app.post("/api/webhooks/stripe")
async def stripe_webhook(request: Request):
    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(status_code=400, detail="Missing stripe-signature header")

    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, os.getenv("STRIPE_WEBHOOK_SECRET")
        )
    except stripe.error.SignatureVerificationError as e:
        print(f"Webhook signature verification failed: {e}")
        raise HTTPException(status_code=400, detail="Invalid signature")
    except ValueError as e:
        print(f"Invalid webhook payload: {e}")
        raise HTTPException(status_code=400, detail="Invalid payload")
    except Exception as e:
        print(f"Webhook handling error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    if event["type"] == "identity.verification_session.verified":
        session = event["data"]["object"]
        args = session.to_dict()
        supabase_user_id = args.get("client_reference_id") # 👈 Retreive original Supabase UUID

        # Fetch verified address details from Stripe Identity redacts
        verified_outputs = args.get("verified_outputs", {})
        address = verified_outputs.get("address", {})
        print(address)
        city = address.get("city")
        state = address.get("state")
        country = address.get("country")
        current_timestamp = datetime.now(timezone.utc).isoformat()

        # Update public.profiles in Supabase
        supabase.table("profiles").update({
            "state_code": state,
            "country_code": country,
            "verified_at": current_timestamp,
            "is_identity_verified": True,
            "stripe_verification_id": args["id"]
        }).eq("id", supabase_user_id).execute()

        #Update Supabase Auth user_metadata (syncs session in frontend)
        supabase.auth.admin.update_user_by_id(
            supabase_user_id,
            {
                "user_metadata": {
                    "is_identity_verified": True,
                    "state_code": state,
                    "city": city
                }
            }
        )

    return {"status": "success"}

@app.post("/api/auth/verify-turnstile")
async def verify_login_turnstile(payload: TurnstileVerifyRequest, req: Request):
    # Extract original client IP (handles proxy/ngrok setups)

    x_forwarded_for = req.headers.get("x-forwarded-for")
    if x_forwarded_for:
        client_ip = x_forwarded_for.split(",")[0].strip()
    else:
        client_ip = req.client.host if req.client else None

    await verify_turnstile_token(payload.turnstile_token, client_ip=client_ip)
    return {"status": "success", "message": "Bot check passed."}

async def verify_turnstile_token(token: str, client_ip:str):
    """Validates token with Cloudflare Turnstile API."""
    if not TURNSTILE_SECRET_KEY:
        print("❌ ERROR: CLOUDFLARE_TURNSTILE_SECRET_KEY is missing from environment variables!")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Bot protection service is not configured.")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Turnstile verification token."
        )

    payload = {
        "secret": TURNSTILE_SECRET_KEY,
        "response": token,
    }


    payload["remoteip"] = client_ip

    try:

        res = await httpx_client.post(
            "https://challenges.cloudflare.com/turnstile/v0/siteverify",
            data=payload,
        )
        outcome = res.json()
    except httpx.HTTPError as e:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Bot protection service unavailable: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error during bot validation: {str(e)}"
        )

    if not outcome.get("success"):
        error_codes = outcome.get("error-codes", [])
        print(f"❌ Turnstile Verification Failed! Cloudflare error-codes: {errors}")
        msg = f"Bot protection check failed: {', '.join(error_codes)}" if error_codes else "Bot protection check failed."
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=msg)

    print("✅ Turnstile token verified successfully!")
    return True
# -------------------------------------------------------------------
# POLITICIANS & DELEGATION DIRECTORY ENDPOINTS
# -------------------------------------------------------------------
import json

def fetch_vote_aggregates_from_supabase() -> Dict[str, Dict[str, int]]:
    """
    Query Supabase for aggregated vote counts per politician.
    Returns: { "bioguide_id_123": {"approve": 12, "disapprove": 2, "neutral": 1}, ... }
    """
    # RPC call or SQL query grouping by politician_id
    response = supabase.rpc("get_politician_vote_counts",{}).execute()
    vote_map = {}
    for row in response.data:
        p_id = row["politician_id"]
        vote_map[p_id] = {
            "approve": row.get("approve_count", 0),
            "disapprove": row.get("disapprove_count", 0),
            "neutral": row.get("neutral_count", 0),
        }

    return vote_map

@app.get("/api/politicians/list", dependencies=[Depends(standard_limiter)])
def get_politicians_list(
        state: Optional[str] = None,
        limit: Optional[int] = Query(None, ge=1, le=500),
):
    """
    Returns politicians for state-based client caching:
    - If `state` is ALL/None: returns top 50 incumbents for initial dashboard render.
    - If `state` is provided (e.g., 'CA'): returns the full state dataset for local UI filtering.
    """
    clean_state = state.strip().upper() if state and state.upper() != "ALL" else None

    # Apply limit of 50 only when requesting 'ALL' states
    effective_limit = limit if limit is not None else (None if clean_state else 50)

    # Bump cache version key to invalidate stale payloads lacking votesWithParty
    cache_key = f"politicians:list:v5:{clean_state or 'ALL'}:{effective_limit or 'unlimited'}"

    try:
        cached_data = cache.get(cache_key)
        if cached_data:
            return json.loads(cached_data) if isinstance(cached_data, (str, bytes)) else cached_data
    except Exception:
        pass  # Fallback to DB query on cache error

    conn = get_db_cursor()
    where_clauses = ["1=1"]
    params = []

    if clean_state:
        where_clauses.append("p.state = ?")
        params.append(clean_state)

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            p.id                        AS id,
            p.name                      AS name,
            p.state                     AS state,
            p.chamber                   AS chamber,
            p.district                  AS district,
            p.party                     AS party,
            p.website                   AS website,
            p.campaign                  AS campaign,
            COALESCE(p.imageUrl)        AS imageUrl,
            p.is_incumbent              AS isIncumbent,
            p.status                    AS status,    
            COALESCE(p.sponsored_count, 0)   AS sponsoredCount,
            COALESCE(p.cosponsored_count, 0) AS cosponsoredCount,
            COALESCE(p.enacted_count, 0)     AS enactedCount,
            p.pass_rate                 AS passRate,
            COALESCE(p.total_receipts, 0.0)  AS totalReceipts,
            p.first_file_date           AS firstFileDate,
            att.total_eligible_votes    AS rc_total,
            att.missed_votes            AS rc_missed,
            att.present_abstain_votes   AS rc_present,
            pl.party_loyalty_pct        AS votesWithParty
        FROM main.dim_politicians AS p
        LEFT JOIN main.mart_member_attendance AS att
            ON p.bioguide_id = att.member_id AND p.is_incumbent = TRUE
        LEFT JOIN main.mart_party_loyalty AS pl
            ON p.bioguide_id = pl.member_id AND p.is_incumbent = TRUE
        WHERE {where_sql}
        ORDER BY p.is_incumbent DESC, p.name ASC
    """

    if effective_limit:
        sql += " LIMIT ?"
        params.append(effective_limit)

    try:
        df = conn.execute(sql, params).fetchdf()
        records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
        sentiment_vote_map = fetch_vote_aggregates_from_supabase()

        for record in records:
            p_id = record["id"]
            total = record.pop("rc_total", None)
            missed = record.pop("rc_missed", None)
            present = record.pop("rc_present", None)

            if total and total > 0:
                votes_cast = total - (missed or 0)
                record["rollCallVotes"] = {
                    "totalVotes": total,
                    "votesCast": votes_cast,
                    "missedVotes": missed or 0,
                    "presentVotes": present or 0,
                    "attendancePercentage": round((votes_cast / total) * 100.0, 2),
                }
            else:
                record["rollCallVotes"] = None

            if record.get("firstFileDate"):
                record["firstFileDate"] = str(record["firstFileDate"])

            stats = sentiment_vote_map.get(p_id, {"approve": 0, "disapprove": 0, "neutral": 0})
            stats["totalVotes"] = stats["approve"] + stats["disapprove"] + stats["neutral"]
            record["sentimentStats"] = stats

            # Format votesWithParty to 1 decimal place or integer
            raw_loyalty = record.get("votesWithParty")
            if raw_loyalty is not None:
                record["votesWithParty"] = round(float(raw_loyalty), 1)

        try:
            cache.set(cache_key, json.dumps(records), ex=86400)
        except Exception:
            pass

        return records

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database Query Error: {str(e)}"
        )
    finally:
        conn.close()


@app.get("/api/politicians/search", dependencies=[Depends(search_limiter)])
def search_politicians(
        q: str = Query(..., min_length=2, description="Search query string"),
        state: Optional[str] = None,
        chamber: Optional[str] = None,
        party: Optional[str] = None,
        candidate_status: Optional[str] = Query(None, alias="status"),
        limit: int = Query(50, ge=1, le=200),
):
    """Executes a targeted search query across all candidates with active filter parameters."""
    normalized_chamber = CHAMBER_ALIASES.get((chamber or "").strip().casefold())
    clean_search = q.strip()
    clean_party = party.strip() if party and party.upper() != "ALL" else None
    clean_state = state.strip().upper() if state and state.upper() != "ALL" else None
    clean_status = candidate_status.strip() if candidate_status and candidate_status.upper() != "ALL" else None

    conn = get_db_cursor()

    where_clauses = ["LOWER(p.name) LIKE LOWER(?)"]
    params = [f"%{clean_search}%"]

    if clean_state:
        where_clauses.append("p.state = ?")
        params.append(clean_state)

    if normalized_chamber:
        where_clauses.append("LOWER(TRIM(p.chamber)) = ?")
        params.append(normalized_chamber)

    if clean_party:
        where_clauses.append("LOWER(TRIM(p.party)) = LOWER(?)")
        params.append(clean_party)

    if clean_status:
        target_status = clean_status.lower()
        if target_status == "incumbent":
            where_clauses.append("(p.is_incumbent = TRUE OR LOWER(p.status) = 'incumbent')")
        elif target_status == "challenger":
            where_clauses.append("(LOWER(p.status) = 'challenger' OR (p.is_incumbent = FALSE OR LOWER(p.status) = 'candidate'))")
        elif target_status == "open":
            where_clauses.append("LOWER(p.status) IN ('open', 'open_seat')")

    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT 
            p.id                        AS id,
            p.name                      AS name,
            p.state                     AS state,
            p.chamber                   AS chamber,
            p.district                  AS district,
            p.party                     AS party,
            p.website                   AS website,
            p.campaign                  AS campaign,
            COALESCE(p.imageUrl)        AS imageUrl,
            p.is_incumbent              AS isIncumbent,
            p.status                    AS status,    
            COALESCE(p.sponsored_count, 0)   AS sponsoredCount,
            COALESCE(p.cosponsored_count, 0) AS cosponsoredCount,
            COALESCE(p.enacted_count, 0)     AS enactedCount,
            p.pass_rate                 AS passRate,
            COALESCE(p.total_receipts, 0.0)  AS totalReceipts,
            p.first_file_date           AS firstFileDate,
            att.total_eligible_votes    AS rc_total,
            att.missed_votes            AS rc_missed,
            att.present_abstain_votes   AS rc_present,
            pl.party_loyalty_pct        AS votesWithParty
        FROM main.dim_politicians AS p
        LEFT JOIN main.mart_member_attendance AS att
            ON p.bioguide_id = att.member_id AND p.is_incumbent = TRUE
        LEFT JOIN main.mart_party_loyalty AS pl
            ON p.bioguide_id = pl.member_id AND p.is_incumbent = TRUE
        WHERE {where_sql}
        ORDER BY p.is_incumbent DESC, p.name ASC
        LIMIT ?;
    """
    params.append(limit)

    try:
        df = conn.execute(sql, params).fetchdf()
        records = df.astype(object).where(df.notna(), None).to_dict(orient="records")
        sentiment_vote_map = fetch_vote_aggregates_from_supabase()

        for record in records:
            p_id = record["id"]
            total = record.pop("rc_total", None)
            missed = record.pop("rc_missed", None)
            present = record.pop("rc_present", None)

            if total and total > 0:
                votes_cast = total - (missed or 0)
                record["rollCallVotes"] = {
                    "totalVotes": total,
                    "votesCast": votes_cast,
                    "missedVotes": missed or 0,
                    "presentVotes": present or 0,
                    "attendancePercentage": round((votes_cast / total) * 100.0, 2),
                }
            else:
                record["rollCallVotes"] = None

            if record.get("firstFileDate"):
                record["firstFileDate"] = str(record["firstFileDate"])

            stats = sentiment_vote_map.get(p_id, {"approve": 0, "disapprove": 0, "neutral": 0})
            stats["totalVotes"] = stats["approve"] + stats["disapprove"] + stats["neutral"]
            record["sentimentStats"] = stats

            # Format votesWithParty to 1 decimal place or integer
            raw_loyalty = record.get("votesWithParty")
            if raw_loyalty is not None:
                record["votesWithParty"] = round(float(raw_loyalty), 1)
        return records

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database Query Error: {str(e)}"
        )
    finally:
        conn.close()

@app.get("/api/politicians/{politician_id}", dependencies=[Depends(standard_limiter)])
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
              FROM main.dim_politicians
              WHERE politician_id = ? 
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
                                   WHERE table_schema = 'main' AND table_name = 'politician_top_donors'
                                   """).fetchone()[0]
        if not donor_table:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Donor data is not available; run the OpenFEC donor ingestion first.",
            )

        sql = """
              SELECT donor_name AS donorName, amount, industry, cycle
              FROM main.stg_fec_donors
              WHERE politician_id = ? 
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
                                     WHERE table_schema = 'main' AND table_name = 'politician_fec_summary'
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
            FROM main.politician_fec_summary
            WHERE politician_id = ?
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

@app.post("/api/chat", dependencies=[Depends(ai_console_limiter)] )
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
        # Extract the structured DataResultPayload generated by response_format
        raw_payload: DataResultPayload = response.get("structured_response")

        if not raw_payload:
            raw_content = response["messages"][-1].content
            raw_payload = DataResultPayload(
                display_type="text",
                title="Result",
                summary=sanitize_content_field(raw_content)
            )
        clean_payload = sanitize_payload(raw_payload,response=response)
        # Convert to Python dict for FastAPI JSON response
        final_answer = clean_payload.model_dump(exclude={"list_items"})

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
        user_id: str = Depends(get_current_user),
):
    if not supabase:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service is not configured."
        )

    try:
        # 1. Upsert vote in Supabase PostgreSQL
        vote_data = {
            "user_id": user_id,
            "politician_id": request.politician_id,
            "vote_type": request.vote_type
        }

        supabase.table("user_votes").upsert(
            vote_data,
            on_conflict="user_id, politician_id"
        ).execute()

        # 2. Invalidate old Redis cache key
        cache_key = f"sentiment:{request.politician_id}"
        try:
            cache.delete(cache_key)
        except Exception as cache_err:
            logger.warning(f"Failed to delete Redis cache key {cache_key}: {cache_err}")

        # 3. Calculate fresh database tallies
        fresh_stats = fetch_sentiment_from_supabase(request.politician_id)

        # 4. Write fresh counts back to Redis cache (30s TTL)
        cache.set(cache_key, fresh_stats, ex_seconds=30)

        # 5. Return updated tallies directly to the client
        return {
            "status": "success",
            "message": "Vote recorded successfully.",
            "user_vote": request.vote_type,
            "sentimentStats": fresh_stats
        }

    except Exception as e:
        logger.error(f"Failed to process vote for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {str(e)}"
        )

def fetch_sentiment_from_supabase(politician_id: str) -> Dict[str, int]:
    """Fallback query aggregating vote tallies efficiently."""
    try:
        # Efficient aggregation query filtering directly by politician_id
        response = (
            supabase.table("user_votes")
            .select("vote_type")
            .eq("politician_id", politician_id)
            .execute()
        )

        data = response.data or []
        stats = {"approve": 0, "neutral": 0, "disapprove": 0}
        for row in data:
            v_type = row.get("vote_type")
            if v_type in stats:
                stats[v_type] += 1

        return stats
    except Exception as e:
        logger.error(f"Supabase vote aggregation error for {politician_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to calculate vote stats from database."
        )


@app.get("/api/votes/stats/{politician_id}")
@app.get("/api/politicians/{politician_id}/votes/stats")
def get_politician_vote_stats(politician_id: str) -> Dict[str, Any]:
    """Fetch aggregate vote sentiment with Upstash Redis caching layer."""
    cache_key = f"sentiment:{politician_id}"

    # 1. Read Path: Check Upstash Redis Cache
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
    cache.set(cache_key, stats, ex_seconds=30)

    return {
        "politicianId": politician_id,
        "sentimentStats": stats,
        "source": "database"
    }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
