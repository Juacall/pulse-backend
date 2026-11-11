# server.py
import logging
import hashlib
import os
import re
import threading
import time
import duckdb
import json
import stripe
import httpx
from datetime import datetime, timezone
from fastapi import FastAPI, Query, Depends, HTTPException, Header, Request, status, BackgroundTasks, Response
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from langchain_anthropic.chat_models import AnthropicAuthenticationError
from supabase import create_client, Client, AuthApiError, AuthError

from scripts.server.cache import cache

# Import your existing agent & router modules
from scripts.rag.agent import agent_executor, DB_PATH
from scripts.rag.schema import DataResultPayload
from scripts.rag.payload import sanitize_content_field, sanitize_payload
from scripts.database import connect_duckdb, database_label
from scripts.rag.router import PreProcessingRouter, QueryAnalysis
from scripts.server.rate_limiter import RateLimiter

app = FastAPI(title="Pulse API", version="1.0.0")
logger = logging.getLogger("uvicorn.error")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_KEY") # Service role for backend DB updates
TURNSTILE_SECRET_KEY = os.getenv("TURNSTILE_SECRET_KEY")
STRIPE_WEBHOOK_SECRET = os.getenv("STRIPE_WEBHOOK_SECRET")
CACHE_TTL_USER_VOTES = 86400
CACHE_TTL_ELECTIONS = 3600  # Cache duration: 1 hour (3600 seconds)
CACHE_TTL_24 = 3600 * 24
bills_cache = 3600

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

class ElectionCandidateResponse(BaseModel):
    politician_id: str
    name: str
    party: Optional[str] = None
    campaign_url: Optional[str] = None
    is_incumbent: bool                          # Incumbent during race
    is_currently_in_seat: bool = False          # Active seat holder today
    is_winner: bool = False                     # Race winner flag
    total_receipts: float = 0.0

class ElectionSlateResponse(BaseModel):
    election_id: str = Field(..., example="2026_WA_H_01", description="Composite election slate key")
    office: str = Field(..., example="House", description="Office sought: 'House' or 'Senate'")
    district: str = Field(..., example="01", description="Two-character district code ('00' for Senate)")
    candidates: List[ElectionCandidateResponse] = Field(default_factory=list, description="Contenders in this race")

    class Config:
        from_attributes = True

class StateElectionsPayload(BaseModel):
    state_code: str = Field(..., example="WA", description="Two-letter state postal code")
    election_cycle: int = Field(..., example=2026, description="Election cycle year")
    total_races: int = Field(..., description="Total active contest slates in the state")
    races: List[ElectionSlateResponse] = Field(default_factory=list, description="List of grouped race slates")

    class Config:
        from_attributes = True

# Pydantic schema matching the payload sent by PoliticianActionButtons.jsx
class ClickEvent(BaseModel):
    bioguide_id: str
    button_type: str
    destination_url: str
    timestamp: Optional[str] = None

class UserSummaryResponse(BaseModel):
    user_id: str
    is_identity_verified: bool  # 👈 Fixed: bool instead of str
    stripe_verification_id: Optional[str] = None  # 👈 Fixed: Optional[str] allows None
    votes_map: Dict[str, str]  # 👈 Fixed: Dict[str, str] for politician_id -> vote_type map

    class Config:
        from_attributes = True

class NationalDonationSummary(BaseModel):
    total_grassroots_dollars: float
    total_large_donor_dollars: float
    total_pac_dollars: float
    total_receipts_dollars: float
    national_grassroots_pct: float

class StateDonationSummary(BaseModel):
    state: str
    candidate_count: int
    total_grassroots_dollars: float
    total_large_donor_dollars: float
    total_pac_dollars: float
    total_receipts_dollars: float
    grassroots_pct: float

class IncumbentVsChallengerMetrics(BaseModel):
    category: str
    candidate_count: int
    total_grassroots_dollars: float
    total_large_donor_dollars: float
    total_pac_dollars: float
    total_receipts_dollars: float
    avg_grassroots_pct: float

class TopDonorItem(BaseModel):
    donor_rank: int
    donor_name: str
    industry: str
    cycle: str
    total_amount: float

class CandidateDistrictFinancials(BaseModel):
    politician_id: str
    politician_name: str
    party: Optional[str]
    is_incumbent: bool
    total_receipts: float
    grassroots_contributions: float
    large_donor_contributions: float
    pac_contributions: float
    grassroots_percentage: float
    coverage_end_date: Optional[str]

class DistrictRaceSummaryResponse(BaseModel):
    state: str
    district: str
    total_race_receipts: float
    candidate_count: int
    candidates: List[CandidateDistrictFinancials]

class StateIncumbentSplitMetrics(BaseModel):
    state: str
    category: str
    candidate_count: int
    total_grassroots_dollars: float
    total_large_donor_dollars: float
    total_pac_dollars: float
    total_receipts_dollars: float
    grassroots_pct: float

class TopDonorResponse(BaseModel):
    donor_name: str
    donor_type: str
    cycle: str
    total_amount_contributed: float
    politicians_funded: int
    top_recipient_name: Optional[str] = None
    top_recipient_amount: Optional[float] = None

class LeaderboardResponse(BaseModel):
    cycle: str
    total_count: int
    donors: List[TopDonorResponse]

class SuperPacImpactItem(BaseModel):
    super_pac_name: str
    cycle: str
    total_support_spending: float
    total_oppose_spending: float
    total_independent_expenditures: float
    candidates_targeted: int
    top_target_candidate: Optional[str] = None
    top_target_amount: Optional[float] = None

class SuperPacImpactResponse(BaseModel):
    cycle: str
    total_count: int
    super_pacs: List[SuperPacImpactItem]

# Enable CORS for Vite dev server (and production domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173",
                   "http://127.0.0.1:5173",
                   "https://legipulse.org",
                   "https://www.legipulse.org",
                   "https://lettuce-verse-kindling.ngrok-free.dev",
                   "https://legipulse-backend.internal",
                   "https://pulse-frontend.juacallb.workers.dev"],
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
        if not profile.data:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Identity verification required to vote.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Identity verification check failed: {str(e)}"
        ) from e

    return user_id

def get_authenticated_user_id(authorization: str = Header(...)) -> str:
    """
    Lightweight auth dependency: Validates token with Supabase
    WITHOUT throwing HTTP 403 if the user is unverified.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid auth header format. Expected 'Bearer <token>'."
        )

    token = authorization.split(" ", 1)[1].strip()

    try:
        user_res = supabase.auth.get_user(token)
        if not user_res or not user_res.user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")
        return user_res.user.id
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication error: {str(e)}"
        )

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
@app.get("/api/usersummary",response_model=UserSummaryResponse)
def get_user_summary(current_user: str = Depends(get_current_user)):
    """
    Returns user profile, Stripe verification status, and a mapping of
    politician_id -> vote_type for the authenticated user.
    Leverages Redis caching for fast state hydration on app load.
    """
    user_id = current_user
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity could not be verified."
        )
    cache_key = f"user:{user_id}:summary"

    # 1. Check Redis Cache First
    try:
        cached_data = cache.get(cache_key)
        if cached_data:
            if isinstance(cached_data, (bytes, bytearray)):
                cached_data = cached_data.decode("utf-8")
            return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
    except Exception as err:
        logger.warning(f"Redis read error for user summary (user_id={user_id}): {err}")

    # 2. Query Supabase (Profile Verification & User Votes)
    try:
        # Fetch profile verification data
        profile_res = (
            supabase.table("profiles")
            .select("is_identity_verified, stripe_verification_id")
            .eq("id", user_id)
            .single()
            .execute()
        )

        # Fetch voting history
        votes_res = (
            supabase.table("user_votes")
            .select("politician_id, vote_type")
            .eq("user_id", user_id)
            .execute()
        )
        profile_data = profile_res.data or {}
        raw_votes = votes_res.data or []

        is_verified = profile_data.get("is_identity_verified", False)
        stripe_verification_id = profile_data.get("stripe_verification_id")

        # 3. Stripe Self-Healing Check
        # If DB status is False but a Stripe session exists, verify status directly with Stripe
        if stripe_verification_id and not is_verified:
            try:
                session = stripe.identity.VerificationSession.retrieve(stripe_verification_id)
                if session.get("status") == "verified":
                    is_verified = True
                    # Sync updated state back to Supabase
                    supabase.table("profiles").update(
                        {"is_identity_verified": True}
                    ).eq("id", user_id).execute()
            except stripe.error.StripeError as stripe_err:
                logger.error(f"Failed to check Stripe verification status for session {stripe_verification_id}: {stripe_err}")

        # Construct vote map: { "politician_id_123": "YEA", "politician_id_456": "NAY" }
        vote_map = {row["politician_id"]: row["vote_type"] for row in raw_votes}

        # Build payload
        summary_payload = {
            "user_id": user_id,
            "is_identity_verified": is_verified,
            "stripe_verification_id": stripe_verification_id,
            "votes_map": vote_map
        }

        # 4. Write back to Redis Cache
        try:
            cache.set(
                cache_key,
                json.dumps(summary_payload),
                ex_seconds=CACHE_TTL_USER_VOTES
            )
        except Exception as err:
            logger.warning(f"Redis write error for user summary (user_id={user_id}): {err}")

        return summary_payload

    except Exception as e:
        logger.error(f"Failed to fetch user summary for user_id={user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user voting and verification records: {str(e)}"
        )



@app.post("/api/identity/create-session")
async def create_verification_session(user: dict = Depends(verify_authenticated_user)):
    user_id = user["sub"]  # Supabase User UUID
    # Explicit fallback check for API key
    if not stripe.api_key:
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")

    if not stripe.api_key:
        raise HTTPException(status_code=500, detail="Stripe API key is unconfigured on server.")

    frontend_url = os.getenv("FRONTEND_URL")
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
        session_id = session['id']
        user_id = session['client_reference_id']
        if user_id:
            report_id = session['last_verification_report']
            report = stripe.identity.VerificationReport.retrieve(report_id)
            doc_data = report.to_dict()
        # Fetch verified address details from Stripe Identity redacts
            verified_outputs = doc_data.get("document", {})
            address = verified_outputs.get("address", {})
            city = address.get("city")
            state = address.get("state")
            country = address.get("country")
            postal_code = address.get("postal_code")
            current_timestamp = datetime.now(timezone.utc).isoformat()

            # Update public.profiles in Supabase
            supabase.table("profiles").update({
                "state_code": state,
                "country_code": country,
                "postal_code" : postal_code,
                "city" : city,
                "verified_at": current_timestamp,
                "is_identity_verified": True,
                "stripe_verification_id": session_id
            }).eq("id", user_id).execute()

            #Update Supabase Auth user_metadata (syncs session in frontend)
            supabase.auth.admin.update_user_by_id(
                user_id,
                {
                    "user_metadata": {
                        "is_identity_verified": True,
                        "state_code": state,
                        "city": city
                    }
                }
            )
            cache.invalidate_user_cache(user_id)

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


@app.get("/api/seats/{seat_id}")
def get_seat_detail(seat_id: str):
    """
    Returns seat metadata, active incumbent, and historical tenures.
    Normalizes seat_id to handle unpadded or zero-padded district numbers.
    """
    conn = get_db_cursor()
    try:
        # Normalize WA-H-1 -> WA-H-01
        parts = seat_id.upper().strip().split("-")
        if len(parts) == 3 and parts[1] == "H":
            clean_seat_id = f"{parts[0]}-H-{parts[2].zfill(2)}"
        else:
            clean_seat_id = seat_id.upper().strip()

        cache_key = f"seat:detail:{clean_seat_id}"
        cached_seat = cache.get(cache_key)
        if cached_seat is not None:
            return cached_seat

        # 1. Fetch seat metadata and current active incumbent
        seat_query = """
                     SELECT
                         s.seat_id,
                         s.state_code,
                         s.chamber,
                         s.district,
                         s.seat_class,
                         p.bioguide_id AS incumbent_id,
                         p.name AS incumbent_name,
                         p.party AS incumbent_party,
                         p.imageUrl AS incumbent_image,
                         t.start_year AS tenure_start_year,
                         t.congress AS tenure_congress
                     FROM main.dim_seats s
                              LEFT JOIN main.fct_seat_tenures t
                                        ON s.seat_id = t.seat_id AND t.is_current_incumbent = TRUE
                              LEFT JOIN main.dim_politicians p
                                        ON t.politician_id = p.bioguide_id
                     WHERE s.seat_id = ?; \
                     """
        seat_row = conn.execute(seat_query, [clean_seat_id]).fetchone()

        if not seat_row:
            raise HTTPException(status_code=404, detail=f"Seat {seat_id} not found")

        # 2. Fetch complete tenure history
        tenure_query = """
                       SELECT
                           t.tenure_id,
                           t.congress,
                           t.start_year,
                           t.end_year,
                           t.is_current_incumbent,
                           p.bioguide_id,
                           p.name,
                           p.party
                       FROM main.fct_seat_tenures t
                                JOIN main.dim_politicians p ON t.politician_id = p.bioguide_id
                       WHERE t.seat_id = ?
                       ORDER BY t.start_year DESC, t.congress DESC; \
                       """
        history_rows = conn.execute(tenure_query, [clean_seat_id]).fetchall()

        history = [
            {
                "tenure_id": r[0],
                "congress": r[1],
                "start_year": r[2],
                "end_year": r[3],
                "is_current_incumbent": bool(r[4]),
                "politician_id": r[5],
                "name": r[6],
                "party": r[7]
            }
            for r in history_rows
        ]

        payload = {
            "seat_id": seat_row[0],
            "state_code": seat_row[1],
            "chamber": seat_row[2],
            "district": seat_row[3],
            "seat_class": seat_row[4],
            "incumbent": {
                "bioguide_id": seat_row[5],
                "name": seat_row[6],
                "party": seat_row[7],
                "image_url": seat_row[8],
                "start_year": seat_row[9],
                "congress": seat_row[10]
            } if seat_row[5] else None,
            "tenure_history": history
        }
        cache.set(cache_key, payload, ex_seconds=CACHE_TTL_24)
        return payload
    finally:
        conn.close()

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

@app.get(
    "/api/politicians/{politician_id}/donors",
    response_model=List[Dict[str, Any]],
    summary="Get Top Donors for a Specific Politician"
)
def get_politician_donors(politician_id: str, cycle: Optional[int] = Query(None, description="FEC Election Cycle")):
    """Fetch top itemized contributors for a politician."""
    cache_key = f"fec:donors:{politician_id}"
    if cycle is not None:
        cache_key += f":cycle_{cycle}"

    # 1. Try Redis Cache
    try:
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            if isinstance(cached_data, (bytes, bytearray)):
                cached_data = cached_data.decode("utf-8")
            return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
    except Exception as err:
        logger.warning(f"Redis read failure for key {cache_key}: {err}")

    # 2. Database Fetch Path (DuckDB)
    conn = get_db_cursor()
    try:
        # Verify table existence against the actual table used in the query
        table_check = conn.execute("""
                                   SELECT COUNT(*)
                                   FROM information_schema.tables
                                   WHERE table_schema = 'main' AND table_name = 'politician_top_donors'
                                   """).fetchone()[0]

        if not table_check:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Donor data is not available; run the donor ingestion pipeline first.",
            )

        # Build SQL query on the correct table
        sql = """
              SELECT
                  donor_name AS donorName,
                  amount,
                  industry,
                  cycle
              FROM main.politician_top_donors
              WHERE politician_id = ? \
              """
        params = [politician_id]

        if cycle is not None:
            sql += " AND cycle = ?"
            params.append(cycle)  # Pass integer directly to match schema

        sql += " ORDER BY amount DESC, donor_name ASC LIMIT 5"

        records = conn.execute(sql, params).fetchdf().to_dict(orient="records")

        # 3. Write to Cache
        try:
            cache.set(
                cache_key,
                json.dumps(jsonable_encoder(records)),
                ex_seconds=86400
            )
        except Exception as err:
            logger.warning(f"Redis write failure for key {cache_key}: {err}")

        return records

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Donor Query Error: {str(e)}"
        )
    finally:
        if conn:
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
@app.get("/api/votes/usersummary",response_model=UserSummaryResponse)
def get_user_summary(current_user: str = Depends(get_current_user)):
    """
    Returns user profile, Stripe verification status, and a mapping of
    politician_id -> vote_type for the authenticated user.
    Leverages Redis caching for fast state hydration on app load.
    """
    user_id = current_user
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User identity could not be verified."
        )
    cache_key = f"user:{user_id}:summary"

    # 1. Check Redis Cache First
    try:
        cached_data = cache.get(cache_key)
        if cached_data:
            if isinstance(cached_data, (bytes, bytearray)):
                cached_data = cached_data.decode("utf-8")
            return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
    except Exception as err:
        logger.warning(f"Redis read error for user summary (user_id={user_id}): {err}")

    # 2. Query Supabase (Profile Verification & User Votes)
    try:
        # Fetch profile verification data
        profile_res = (
            supabase.table("profiles")
            .select("is_identity_verified, stripe_verification_id")
            .eq("id", user_id)
            .single()
            .execute()
        )

        # Fetch voting history
        votes_res = (
            supabase.table("user_votes")
            .select("politician_id, vote_type")
            .eq("user_id", user_id)
            .execute()
        )
        profile_data = profile_res.data or {}
        raw_votes = votes_res.data or []

        is_verified = profile_data.get("is_identity_verified", False)
        stripe_verification_id = profile_data.get("stripe_verification_id")

        # 3. Stripe Self-Healing Check
        # If DB status is False but a Stripe session exists, verify status directly with Stripe
        if stripe_verification_id and not is_verified:
            try:
                session = stripe.identity.VerificationSession.retrieve(stripe_verification_id)
                if session.get("status") == "verified":
                    is_verified = True
                    # Sync updated state back to Supabase
                    supabase.table("profiles").update(
                        {"is_identity_verified": True}
                    ).eq("id", user_id).execute()
            except stripe.error.StripeError as stripe_err:
                logger.error(f"Failed to check Stripe verification status for session {stripe_verification_id}: {stripe_err}")

        # Construct vote map: { "politician_id_123": "YEA", "politician_id_456": "NAY" }
        vote_map = {row["politician_id"]: row["vote_type"] for row in raw_votes}

        # Build payload
        summary_payload = {
            "user_id": user_id,
            "is_identity_verified": is_verified,
            "stripe_verification_id": stripe_verification_id,
            "votes_map": vote_map
        }

        # 4. Write back to Redis Cache
        try:
            cache.set(
                cache_key,
                json.dumps(summary_payload),
                ex_seconds=CACHE_TTL_USER_VOTES
            )
        except Exception as err:
            logger.warning(f"Redis write error for user summary (user_id={user_id}): {err}")

        return summary_payload

    except Exception as e:
        logger.error(f"Failed to fetch user summary for user_id={user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve user voting and verification records: {str(e)}"
        )

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

        # 2. Invalidate public politician sentiment cache
        politician_cache_key = f"sentiment:{request.politician_id}"
        try:
            cache.delete(politician_cache_key)
        except Exception as cache_err:
            logger.warning(f"Failed to delete public sentiment cache key {politician_cache_key}: {cache_err}")

        # 3. Update active user's personal vote summary in Redis
        user_summary_cache_key = f"user:{user_id}:votes_summary"
        try:
            cached_user_votes = cache.get(user_summary_cache_key)
            if cached_user_votes:
                vote_map = json.loads(cached_user_votes) if isinstance(cached_user_votes, (str, bytes)) else cached_user_votes
                vote_map[request.politician_id] = request.vote_type
                cache.set(user_summary_cache_key, json.dumps(vote_map), ex=86400)
            else:
                cache.delete(user_summary_cache_key)
        except Exception as user_cache_err:
            logger.warning(f"Failed to update user votes summary cache key {user_summary_cache_key}: {user_cache_err}")

        # 4. Calculate fresh database tallies
        fresh_stats = fetch_sentiment_from_supabase(request.politician_id)

        # 5. Write fresh politician counts back to Redis cache
        try:
            cache.set(politician_cache_key, fresh_stats, ex_seconds=30)
        except Exception as cache_err:
            logger.warning(f"Failed to write fresh stats to Redis: {cache_err}")

        # 6. Return updated tallies and user stance directly to client
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
# -------------------------------------------------------------------
# Election Endpoints
# -------------------------------------------------------------------
@app.get("/api/elections/state", response_model=StateElectionsPayload, summary="Get state election slates and candidates")
def get_state_elections(
        state_code: str = Query(..., max_length=2, description="Two-letter state code, e.g. WA"),
        cycle: Optional[int] = Query(2026, description="Election cycle year")
):
    state_clean = state_code.upper().strip()
    cache_key = f"elections:{cycle}:{state_clean}"

    # 1. Read Path: Check Redis Cache First
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return StateElectionsPayload(**data)
    except Exception as err:
        logger.warning(f"Redis read error for election cache key {cache_key}: {err}")

    # 2. Database Query (Querying mart_elections directly)
    try:
        conn = get_db_cursor()
        query = """
                SELECT
                    election_id,
                    office_sought,
                    district,
                    politician_id,
                    candidate_name,
                    party,
                    campaign_url,
                    is_incumbent_during_race,
                    is_currently_in_seat,
                    is_winner,
                    COALESCE(total_receipts, 0.0) AS total_receipts
                FROM main.mart_elections
                WHERE state_code = ? AND election_cycle = ?
                ORDER BY office_sought DESC, district ASC, total_receipts DESC;
                """
        rows = conn.execute(query, [state_clean, cycle]).fetchall()
        conn.close()

        races_map = {}
        for r in rows:
            election_id = r[0]
            if election_id not in races_map:
                races_map[election_id] = {
                    "election_id": election_id,
                    "office": r[1],
                    "district": r[2],
                    "candidates": []
                }

            races_map[election_id]["candidates"].append(
                ElectionCandidateResponse(
                    politician_id=r[3],
                    name=r[4],
                    party=r[5],
                    campaign_url=r[6],
                    is_incumbent=bool(r[7]),
                    is_currently_in_seat=bool(r[8]),
                    is_winner=bool(r[9]),
                    total_receipts=float(r[10])
                )
            )

        races_list = [ElectionSlateResponse(**data) for data in races_map.values()]

        payload = StateElectionsPayload(
            state_code=state_clean,
            election_cycle=cycle,
            total_races=len(races_list),
            races=races_list
        )

        # 3. Write Path: Cache result in Redis (TTL: 1 Hour)
        try:
            cache.set(
                cache_key,
                json.dumps(payload.model_dump()),
                ex_seconds=CACHE_TTL_ELECTIONS
            )
        except Exception as err:
            logger.warning(f"Redis write error for election cache key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve elections for {state_clean}: {str(e)}"
        )

# -------------------------------------------------------------------
# Analytics Endpoints
# -------------------------------------------------------------------

@app.get(
    "/api/analytics/super-pacs/impact",
    response_model=SuperPacImpactResponse,
    summary="Get Super PAC Independent Expenditure Impact Analysis"
)
def get_super_pac_impact(
        cycle: str = Query("2026", description="FEC Election Cycle"),
        limit: int = Query(50, ge=1, le=500, description="Number of Super PACs to return")
):
    conn = None
    cache_key = f"analytics:super_pacs:impact:{cycle}:limit_{limit}"

    # 1. Redis Cache Read
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return SuperPacImpactResponse(**data)
    except Exception as err:
        logger.warning(f"Redis read error for key {cache_key}: {err}")

    # 2. Database Read (DuckDB)
    try:
        conn = get_db_cursor()

        # Aggregates by normalized super_pac_name, filters out non-committee filers, & excludes >$50M outliers
        query = """
                SELECT
                    REGEXP_REPLACE(UPPER(TRIM(super_pac_name)), '\\s+', ' ', 'g') AS norm_pac_name,
                    cycle,
                    ROUND(SUM(CASE WHEN stance = 'SUPPORT' THEN amount ELSE 0 END), 2) AS total_support_spending,
                    ROUND(SUM(CASE WHEN stance = 'OPPOSE' THEN amount ELSE 0 END), 2) AS total_oppose_spending,
                    ROUND(SUM(amount), 2) AS total_independent_expenditures,
                    COUNT(DISTINCT fec_candidate_id) AS candidates_targeted,
                    ARG_MAX(candidate_name, amount) AS top_target_candidate,
                    ROUND(MAX(amount), 2) AS top_target_amount
                FROM main.stg_fec_independent_expenditures
                WHERE cycle = ?
                  AND super_pac_id LIKE 'C%'
                  AND amount BETWEEN 1.0 AND 50000000.0
                GROUP BY norm_pac_name, cycle
                ORDER BY total_independent_expenditures DESC
                    LIMIT ?; \
                """
        results = conn.execute(query, [cycle, limit]).fetchall()

        items = [
            SuperPacImpactItem(
                super_pac_name=row[0],
                cycle=row[1],
                total_support_spending=float(row[2]),
                total_oppose_spending=float(row[3]),
                total_independent_expenditures=float(row[4]),
                candidates_targeted=int(row[5]),
                top_target_candidate=row[6],
                top_target_amount=float(row[7]) if row[7] is not None else None
            )
            for row in results
        ]

        payload = SuperPacImpactResponse(
            cycle=cycle,
            total_count=len(items),
            super_pacs=items
        )

        # 3. Redis Cache Write (24-Hour TTL)
        try:
            cache.set(
                cache_key,
                json.dumps(jsonable_encoder(payload)),
                ex_seconds=CACHE_TTL_24
            )
        except Exception as err:
            logger.warning(f"Redis write error for key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Super PAC impact analytics: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


# Target Candidate details for a specific Super PAC
@app.get("/api/analytics/super-pacs/{super_pac_name}/targets")
def get_super_pac_targets(super_pac_name: str, cycle: str = "2026"):
    # Normalize input parameter for cache key
    norm_input_name = " ".join(super_pac_name.strip().upper().split())
    cache_key = f"analytics:super_pacs:targets:{cycle}:{norm_input_name.lower()}"

    try:
        cached_targets = cache.get(cache_key)
        if cached_targets is not None:
            if isinstance(cached_targets, (bytes, bytearray)):
                cached_targets = cached_targets.decode("utf-8")
            return json.loads(cached_targets) if isinstance(cached_targets, str) else cached_targets
    except Exception as err:
        logger.warning(f"Redis read error for key {cache_key}: {err}")

    conn = get_db_cursor()
    try:
        # Match using normalized super_pac_name and candidate_name
        query = """
                SELECT
                    REGEXP_REPLACE(UPPER(TRIM(candidate_name)), '\\s+', ' ', 'g') AS norm_candidate_name,
                    stance,
                    ROUND(SUM(amount), 2) AS total_amount
                FROM main.stg_fec_independent_expenditures
                WHERE REGEXP_REPLACE(UPPER(TRIM(super_pac_name)), '\\s+', ' ', 'g') = ?
                  AND cycle = ?
                  AND candidate_name IS NOT NULL
                  AND super_pac_id LIKE 'C%'
                  AND amount BETWEEN 1.0 AND 50000000.0
                GROUP BY norm_candidate_name, stance
                ORDER BY total_amount DESC; \
                """
        results = conn.execute(query, [norm_input_name, cycle]).fetchall()

        targets = [
            {
                "candidate_name": row[0],
                "stance": row[1],
                "amount": float(row[2])
            }
            for row in results
        ]

        payload = {"super_pac_name": norm_input_name, "targets": targets}

        try:
            cache.set(cache_key, json.dumps(jsonable_encoder(payload)), ex_seconds=CACHE_TTL_24)
        except Exception as err:
            logger.warning(f"Redis write error for key {cache_key}: {err}")

        return payload
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve Super PAC targets: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

@app.get(
    "/api/donors/leaderboard",
    response_model=LeaderboardResponse,
    summary="Get Top Direct Organizational & PAC Donors Leaderboard"
)
def get_top_donors_leaderboard(
        cycle: str = Query("2026", description="FEC Election Cycle"),
        limit: int = Query(50, ge=1, le=500, description="Number of top donors to return")
):
    conn = None
    cache_key = f"analytics:donors:leaderboard:{cycle}:limit_{limit}"

    # 1. Cache Read Path (Upstash Redis)
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return LeaderboardResponse(**data)
    except Exception as err:
        logger.warning(f"Redis read error for key {cache_key}: {err}")

    # 2. Database Fetch Path (DuckDB)
    try:
        conn = get_db_cursor()
        query = """
                SELECT
                    donor_name,
                    donor_type,
                    cycle,
                    total_amount_contributed,
                    politicians_funded,
                    top_recipient_name,
                    top_recipient_amount
                FROM main.v_global_top_donors_leaderboard
                WHERE cycle = ?
                ORDER BY total_amount_contributed DESC
                    LIMIT ?; \
                """
        results = conn.execute(query, [cycle, limit]).fetchall()

        donor_items = [
            TopDonorResponse(
                donor_name=row[0],
                donor_type=row[1] if row[1] else "PAC",
                cycle=row[2],
                total_amount_contributed=float(row[3]),
                politicians_funded=int(row[4]),
                top_recipient_name=row[5],
                top_recipient_amount=float(row[6]) if row[6] is not None else None
            )
            for row in results
        ]

        payload = LeaderboardResponse(
            cycle=cycle,
            total_count=len(donor_items),
            donors=donor_items
        )

        # 3. Cache Write Path (24-Hour TTL)
        try:
            cache.set(
                cache_key,
                json.dumps(jsonable_encoder(payload)),
                ex_seconds=CACHE_TTL_24
            )
        except Exception as err:
            logger.warning(f"Redis write error for key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve donor leaderboard: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

@app.get(
    "/api/analytics/donations/national-summary",
    response_model=NationalDonationSummary,
    summary="Get Macro National Campaign Finance Summary"
)
def get_national_donation_summary():
    conn = None
    cache_key = "analytics:donations:national"

    # 1. Cache Read Path
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return NationalDonationSummary(**data)
    except Exception as err:
        logger.warning(f"Redis read error for national donation cache key {cache_key}: {err}")

    # 2. Database Read Path
    try:
        conn = get_db_cursor()
        query = """
                SELECT
                    ROUND(COALESCE(SUM(s.individual_unitemized), 0.0), 2) AS total_grassroots_dollars,
                    ROUND(COALESCE(SUM(s.individual_itemized), 0.0), 2) AS total_large_donor_dollars,
                    ROUND(COALESCE(SUM(s.pac_contributions), 0.0), 2) AS total_pac_dollars,
                    ROUND(COALESCE(SUM(s.total_receipts), 0.0), 2) AS total_receipts_dollars,
                    ROUND(
                            COALESCE(
                                    (SUM(s.individual_unitemized) / NULLIF(SUM(s.total_receipts), 0.0)) * 100,
                                    0.0
                            ), 2
                    ) AS national_grassroots_pct
                FROM main.politician_fec_summary s; \
                """
        row = conn.execute(query).fetchone()

        if not row or row[0] is None:
            payload = NationalDonationSummary(
                total_grassroots_dollars=0.0,
                total_large_donor_dollars=0.0,
                total_pac_dollars=0.0,
                total_receipts_dollars=0.0,
                national_grassroots_pct=0.0
            )
        else:
            payload = NationalDonationSummary(
                total_grassroots_dollars=float(row[0]),
                total_large_donor_dollars=float(row[1]),
                total_pac_dollars=float(row[2]),
                total_receipts_dollars=float(row[3]),
                national_grassroots_pct=float(row[4])
            )

        # 3. Cache Write Path
        try:
            cache.set(
                cache_key,
                json.dumps(payload.model_dump()),
                ex_seconds=CACHE_TTL_24
            )
        except Exception as err:
            logger.warning(f"Redis write error for national donation cache key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve donation analytics: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

@app.get(
    "/api/analytics/donations/by-state",
    response_model=List[StateDonationSummary],
    summary="Get Campaign Finance Summaries Aggregated by State"
)
def get_donations_by_state(
        min_receipts: Optional[float] = Query(0.0, description="Filter out states with total receipts below this threshold"),
        limit: Optional[int] = Query(50, ge=1, le=50, description="Limit the number of states returned")
):
    conn = None
    # 1. Parameterized Cache Key
    cache_key = f'analytics:donations:by_state:min_{min_receipts}:limit_{limit}'

    # 2. Cache Read Path
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return data
    except Exception as err:
        logger.warning(f"Redis read error for donations by state cache key {cache_key}: {err}")

    # 3. Cache Miss Path
    try:
        conn = get_db_cursor()
        query = """
                SELECT
                    p.state,
                    COUNT(DISTINCT s.politician_id) AS candidate_count,
                    ROUND(COALESCE(SUM(s.individual_unitemized), 0.0), 2) AS total_grassroots_dollars,
                    ROUND(COALESCE(SUM(s.individual_itemized), 0.0), 2) AS total_large_donor_dollars,
                    ROUND(COALESCE(SUM(s.pac_contributions), 0.0), 2) AS total_pac_dollars,
                    ROUND(COALESCE(SUM(s.total_receipts), 0.0), 2) AS total_receipts_dollars,
                    ROUND(
                            COALESCE(
                                    (SUM(s.individual_unitemized) / NULLIF(SUM(s.total_receipts), 0.0)) * 100,
                                    0.0
                            ), 2
                    ) AS grassroots_pct
                FROM main.politician_fec_summary s
                         JOIN main.dim_politicians p ON s.politician_id = p.id
                WHERE p.state IS NOT NULL AND p.state != ''
                GROUP BY p.state
                HAVING total_receipts_dollars >= ?
                ORDER BY total_receipts_dollars DESC
                    LIMIT ?; \
                """
        rows = conn.execute(query, [min_receipts, limit]).fetchall()

        payload = [
            StateDonationSummary(
                state=row[0],
                candidate_count=row[1],
                total_grassroots_dollars=float(row[2]),
                total_large_donor_dollars=float(row[3]),
                total_pac_dollars=float(row[4]),
                total_receipts_dollars=float(row[5]),
                grassroots_pct=float(row[6])
            )
            for row in rows
        ]

        # 4. Cache Write Path
        try:
            cache.set(
                cache_key,
                json.dumps(jsonable_encoder(payload)),
                ex_seconds=CACHE_TTL_24
            )
        except Exception as err:
            logger.warning(f"Redis write error for donation by state cache key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve by-state donation analytics: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@app.get(
    "/api/analytics/donations/incumbent-vs-challenger",
    response_model=List[IncumbentVsChallengerMetrics],
    summary="Compare Financial Totals Between Incumbents and Challengers"
)
def get_incumbent_vs_challenger_donations():
    conn = None
    cache_key = "analytics:donations:incumbent-vs-challenger"

    # 1. Cache Read Path (Upstash Redis)
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return [IncumbentVsChallengerMetrics(**item) for item in data]
    except Exception as err:
        logger.warning(f"Redis read error for key {cache_key}: {err}")

    # 2. Database Fetch Path (DuckDB)
    try:
        conn = get_db_cursor()
        query = """
                SELECT
                    CASE WHEN p.is_incumbent THEN 'Incumbents' ELSE 'Challengers' END AS category,
                    COUNT(DISTINCT s.politician_id) AS candidate_count,
                    ROUND(COALESCE(SUM(s.individual_unitemized), 0.0), 2) AS total_grassroots_dollars,
                    ROUND(COALESCE(SUM(s.individual_itemized), 0.0), 2) AS total_large_donor_dollars,
                    ROUND(COALESCE(SUM(s.pac_contributions), 0.0), 2) AS total_pac_dollars,
                    ROUND(COALESCE(SUM(s.total_receipts), 0.0), 2) AS total_receipts_dollars,
                    ROUND(
                            COALESCE(
                                    (SUM(s.individual_unitemized) / NULLIF(SUM(s.total_receipts), 0.0)) * 100,
                                    0.0
                            ), 2
                    ) AS avg_grassroots_pct
                FROM main.politician_fec_summary s
                         JOIN main.dim_politicians p ON s.politician_id = p.id
                GROUP BY p.is_incumbent
                ORDER BY p.is_incumbent DESC; \
                """
        rows = conn.execute(query).fetchall()

        payload = [
            IncumbentVsChallengerMetrics(
                category=row[0],
                candidate_count=row[1],
                total_grassroots_dollars=float(row[2]),
                total_large_donor_dollars=float(row[3]),
                total_pac_dollars=float(row[4]),
                total_receipts_dollars=float(row[5]),
                avg_grassroots_pct=float(row[6])
            )
            for row in rows
        ]

        # 3. Cache Write Path (TTL: 24 Hours)
        try:
            cache.set(
                cache_key,
                json.dumps(jsonable_encoder(payload)),
                ex_seconds=CACHE_TTL_24
            )
        except Exception as err:
            logger.warning(f"Redis write error for key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve incumbent vs challenger analytics: {str(e)}"
        )
    finally:
        if conn:
            conn.close()



@app.get(
    "/api/analytics/donations/district",
    response_model=DistrictRaceSummaryResponse,
    summary="Get Detailed Financial Summary for Candidates in a Specific District"
)
def get_district_donation_summary(
        state: str = Query(..., description="Two-letter state postal abbreviation"),
        district: str = Query(..., description="Two-digit district code (e.g., '01')")
):
    conn = None
    clean_state = state.upper()
    cache_key = f"analytics:donations:district:{clean_state}:{district}"

    # 1. Upstash Redis Cache Read
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return DistrictRaceSummaryResponse(**data)
    except Exception as err:
        logger.warning(f"Redis read error for key {cache_key}: {err}")

    # 2. Database Fetch Path
    try:
        conn = get_db_cursor()
        query = """
                SELECT
                    p.id AS politician_id,
                    p.name AS politician_name,
                    p.party,
                    p.is_incumbent,
                    ROUND(COALESCE(s.total_receipts, 0.0), 2) AS total_receipts,
                    ROUND(COALESCE(s.individual_unitemized, 0.0), 2) AS grassroots_contributions,
                    ROUND(COALESCE(s.individual_itemized, 0.0), 2) AS large_donor_contributions,
                    ROUND(COALESCE(s.pac_contributions, 0.0), 2) AS pac_contributions,
                    ROUND(COALESCE(s.grassroots_percentage, 0.0), 2) AS grassroots_percentage,
                    s.coverage_end_date
                FROM main.dim_politicians p
                         LEFT JOIN main.politician_fec_summary s ON p.id = s.politician_id
                WHERE UPPER(p.state) = UPPER(?) AND p.district = ?
                ORDER BY total_receipts DESC; \
                """
        rows = conn.execute(query, [clean_state, district]).fetchall()

        candidates = [
            CandidateDistrictFinancials(
                politician_id=row[0],
                politician_name=row[1],
                party=row[2],
                is_incumbent=bool(row[3]),
                total_receipts=float(row[4]),
                grassroots_contributions=float(row[5]),
                large_donor_contributions=float(row[6]),
                pac_contributions=float(row[7]),
                grassroots_percentage=float(row[8]),
                coverage_end_date=row[9]
            )
            for row in rows
        ]

        total_race_receipts = sum(c.total_receipts for c in candidates)

        response_payload = DistrictRaceSummaryResponse(
            state=clean_state,
            district=district,
            total_race_receipts=round(total_race_receipts, 2),
            candidate_count=len(candidates),
            candidates=candidates
        )

        # 3. Upstash Redis Cache Write (24-Hour TTL)
        try:
            cache.set(
                cache_key,
                json.dumps(jsonable_encoder(response_payload)),
                ex_seconds=CACHE_TTL_24
            )
        except Exception as err:
            logger.warning(f"Redis write error for key {cache_key}: {err}")

        return response_payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve district financials for {state}-{district}: {str(e)}"
        )
    finally:
        if conn:
            conn.close()


@app.get(
    "/api/analytics/donations/by-state/{state}/incumbent-vs-challenger",
    response_model=List[StateIncumbentSplitMetrics],
    summary="Get State Campaign Finance Metrics Broken Down by Incumbents vs. Challengers"
)
def get_state_incumbent_vs_challenger_donations(state: str):
    conn = None
    clean_state = state.upper()
    cache_key = f"analytics:donations:state-incumbent:{clean_state}"

    # 1. Cache Read Path
    try:
        cached_payload = cache.get(cache_key)
        if cached_payload:
            if isinstance(cached_payload, (bytes, bytearray)):
                cached_payload = cached_payload.decode("utf-8")
            data = json.loads(cached_payload) if isinstance(cached_payload, str) else cached_payload
            return [StateIncumbentSplitMetrics(**item) for item in data]
    except Exception as err:
        logger.warning(f"Redis read error for key {cache_key}: {err}")

    # 2. Database Fetch Path
    try:
        conn = get_db_cursor()
        query = """
                SELECT
                    UPPER(p.state) AS state,
                    CASE WHEN p.is_incumbent THEN 'Incumbents' ELSE 'Challengers' END AS category,
                    COUNT(DISTINCT s.politician_id) AS candidate_count,
                    ROUND(COALESCE(SUM(s.individual_unitemized), 0.0), 2) AS total_grassroots_dollars,
                    ROUND(COALESCE(SUM(s.individual_itemized), 0.0), 2) AS total_large_donor_dollars,
                    ROUND(COALESCE(SUM(s.pac_contributions), 0.0), 2) AS total_pac_dollars,
                    ROUND(COALESCE(SUM(s.total_receipts), 0.0), 2) AS total_receipts_dollars,
                    ROUND(
                            COALESCE(
                                    (SUM(s.individual_unitemized) / NULLIF(SUM(s.total_receipts), 0.0)) * 100,
                                    0.0
                            ), 2
                    ) AS grassroots_pct
                FROM main.politician_fec_summary s
                         JOIN main.dim_politicians p ON s.politician_id = p.id
                WHERE UPPER(p.state) = UPPER(?)
                GROUP BY p.state, p.is_incumbent
                ORDER BY p.is_incumbent DESC; \
                """
        rows = conn.execute(query, [clean_state]).fetchall()

        payload = [
            StateIncumbentSplitMetrics(
                state=row[0],
                category=row[1],
                candidate_count=row[2],
                total_grassroots_dollars=float(row[3]),
                total_large_donor_dollars=float(row[4]),
                total_pac_dollars=float(row[5]),
                total_receipts_dollars=float(row[6]),
                grassroots_pct=float(row[7])
            )
            for row in rows
        ]

        # 3. Cache Write Path (TTL: 24 Hours)
        try:
            cache.set(
                cache_key,
                json.dumps(jsonable_encoder(payload)),
                ex_seconds=CACHE_TTL_24
            )
        except Exception as err:
            logger.warning(f"Redis write error for key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve state incumbent comparison for {state}: {str(e)}"
        )
    finally:
        if conn:
            conn.close()



@app.get("/api/analytics/donations/by-state/{state_code}/candidates")
def get_candidates_by_state(state_code: str):
    cache_key = f"analytics:donations:by-state-candidates:{state_code.strip().upper()}"
    cached_candidates = cache.get(cache_key)
    if cached_candidates is not None:
        return cached_candidates

    conn = get_db_cursor()
    try:
        query = """
                SELECT
                    p.name,
                    p.chamber AS office,
                    p.status,
                    p.is_incumbent,
                    ROUND(COALESCE(SUM(s.individual_unitemized), 0.0), 2) AS total_grassroots_dollars,
                    ROUND(COALESCE(SUM(s.individual_itemized), 0.0), 2) AS total_large_donor_dollars,
                    ROUND(COALESCE(SUM(s.pac_contributions), 0.0), 2) AS total_pac_dollars,
                    ROUND(COALESCE(SUM(s.total_receipts), 0.0), 2) AS total_receipts
                FROM main.dim_politicians p
                         LEFT JOIN main.politician_fec_summary s
                                   ON p.fec_candidate_id = s.fec_candidate_id
                WHERE UPPER(p.state) = UPPER(?)
                GROUP BY
                    p.name,
                    p.chamber,
                    p.status,
                    p.is_incumbent
                ORDER BY total_receipts DESC;
                """
        results = conn.execute(query, [state_code]).fetchall()
        candidates = [
            {
                "name": row[0],
                "office": row[1] or "Federal Office",
                "incumbent_challenge_full": row[2] or ("Incumbent" if row[3] else "Challenger"),
                "total_grassroots_dollars": float(row[4]),
                "total_large_donor_dollars": float(row[5]),
                "total_pac_dollars": float(row[6]),
                "total_receipts": float(row[7]),
            }
            for row in results
        ]
        payload = {"state": state_code.upper(), "candidates": candidates}
        cache.set(cache_key, payload, ex_seconds=CACHE_TTL_24)
        return payload
    finally:
        conn.close()
# ------------------------------------------------------------------
# 1. Legislative Tracking Endpoints
# ------------------------------------------------------------------

def strip_html(text: Optional[str]) -> Optional[str]:
    """Strips HTML tags (<p>, <b>, etc.) from CRS summary strings."""
    if not text:
        return None
    clean = re.sub(r'<[^>]*>', '', text)
    return clean.strip() or None

def get_bills_watermark(
        conn: duckdb.DuckDBPyConnection,
        table_name: str = "main.mart_recent_bills",
        chamber: Optional[str] = None
) -> str:
    """Queries DuckDB metadata for a 1ms watermark fingerprint for standard feed caching."""
    date_col = "bill_latest_action_date" if "passed" in table_name else "bill_introduced_date"

    conditions = [f"{date_col} >= CURRENT_DATE - INTERVAL '365 days'"]
    params = []

    if chamber and chamber.lower() in ("house", "senate"):
        conditions.append("LOWER(chamber) = LOWER(?)")
        params.append(chamber)

    where_clause = f"WHERE {' AND '.join(conditions)}"

    query = f"""
        SELECT 
            COALESCE(MAX({date_col})::TEXT, '0') || '_' || COUNT(*)::TEXT
        FROM {table_name}
        {where_clause};
    """
    row = conn.execute(query, params).fetchone()
    raw_wm = row[0] if row else "0_0"
    return hashlib.md5(raw_wm.encode("utf-8")).hexdigest()[:12]



@app.get("/api/bills/recent")
def get_recent_bills(
        response: Response,
        q: Optional[str] = Query(default=None, description="Search by title, bill number, or sponsor within 1 year"),
        page: int = Query(default=1, ge=1, description="Page number for pagination"),
        per_chamber_limit: int = Query(default=10, ge=1, le=50, description="Number of recent bills per chamber per page"),
        chamber: Optional[str] = Query(default=None, description="Filter by chamber ('House' or 'Senate')")
):
    """
    Retrieves bills introduced within the past year with pagination support.
    """
    conn = get_db_cursor()

    try:
        search_term = q.strip() if q and q.strip() else None

        # 1. READ PATH (Cache default first-page requests only; skip cache for search or page > 1)
        cache_key = None
        if not search_term and page == 1:
            watermark = get_bills_watermark(conn, table_name="main.mart_recent_bills", chamber=chamber)
            chamber_tag = chamber.lower() if chamber else "all"
            cache_key = f"legipulse:recent_bills:{chamber_tag}:limit_{per_chamber_limit}:wm_{watermark}"

            try:
                cached_data = cache.get(cache_key)
                if cached_data is not None:
                    response.headers["X-Cache"] = "HIT"
                    if isinstance(cached_data, (bytes, bytearray)):
                        cached_data = cached_data.decode("utf-8")
                    return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
            except Exception as redis_err:
                logger.warning(f"[Upstash Cache Miss/Error]: {redis_err}")

        # 2. QUERY DUCKDB WITH PAGINATION WINDOW
        conditions = ["bill_introduced_date >= CURRENT_DATE - INTERVAL '365 days'"]
        params = []

        if chamber and chamber.lower() in ("house", "senate"):
            conditions.append("LOWER(chamber) = LOWER(?)")
            params.append(chamber)

        if search_term:
            conditions.append("""
                (LOWER(bill_title) LIKE LOWER(?) 
                 OR LOWER(bill_number) LIKE LOWER(?) 
                 OR LOWER(sponsor_name) LIKE LOWER(?))
            """)
            like_pattern = f"%{search_term}%"
            params.extend([like_pattern, like_pattern, like_pattern])

        where_clause = f"WHERE {' AND '.join(conditions)}"

        min_row = (page - 1) * per_chamber_limit + 1
        max_row = page * per_chamber_limit

        query = f"""
            WITH ranked_bills AS (
                SELECT 
                    bill_id, bill_number, congress, chamber, bill_title,
                    bill_policy_area, bill_summary, bill_introduced_date,
                    bill_latest_action_date, bill_latest_action_text,
                    sponsor_bioguide_id, sponsor_name, sponsor_party, sponsor_state,
                    ROW_NUMBER() OVER (
                        PARTITION BY LOWER(chamber) 
                        ORDER BY bill_introduced_date DESC
                    ) AS rn
                FROM main.mart_recent_bills
                {where_clause}
            )
            SELECT 
                bill_id, bill_number, congress, chamber, bill_title,
                bill_policy_area, bill_summary, bill_introduced_date,
                bill_latest_action_date, bill_latest_action_text,
                sponsor_bioguide_id, sponsor_name, sponsor_party, sponsor_state
            FROM ranked_bills
            WHERE rn BETWEEN ? AND ?
            ORDER BY bill_introduced_date DESC;
        """
        params.extend([min_row, max_row])

        rows = conn.execute(query, params).fetchall()

        bills = []
        for r in rows:
            (
                bill_id, bill_number, congress, chamber_name, title, policy_area,
                raw_summary, introduced_date, latest_action_date, latest_action_text,
                sponsor_id, sponsor_name, sponsor_party, sponsor_state
            ) = r

            bill_type_clean = bill_number.split()[0].lower() if bill_number else ""
            bill_num_clean = bill_number.split()[-1] if bill_number else ""

            url_type_map = {
                "hr": "house-bill", "hres": "house-resolution", "hjres": "house-joint-resolution", "hconres": "house-concurrent-resolution",
                "s": "senate-bill", "sres": "senate-resolution", "sjres": "senate-joint-resolution", "sconres": "senate-concurrent-resolution"
            }
            url_segment = url_type_map.get(bill_type_clean, "house-bill" if chamber_name == "House" else "senate-bill")
            congress_url = f"https://www.congress.gov/bill/{congress}th-congress/{url_segment}/{bill_num_clean}"

            clean_summary = strip_html(raw_summary)
            if clean_summary and not clean_summary.strip():
                clean_summary = None

            bills.append({
                "bill_id": bill_id,
                "bill_number": bill_number,
                "congress": congress,
                "chamber": chamber_name,
                "title": title,
                "policy_area": policy_area or "Uncategorized",
                "summary": clean_summary,
                "raw_summary": raw_summary,
                "introduced_date": str(introduced_date) if introduced_date else None,
                "latest_action_date": str(latest_action_date) if latest_action_date else None,
                "latest_action_text": latest_action_text,
                "sponsor_id": sponsor_id,
                "sponsor_name": sponsor_name,
                "sponsor_party": sponsor_party,
                "sponsor_state": sponsor_state,
                "congress_url": congress_url
            })

        # Check if more rows exist for subsequent pages
        has_more = len(bills) >= (per_chamber_limit if chamber else per_chamber_limit * 2)

        response_payload = {"bills": bills, "count": len(bills), "page": page, "has_more": has_more}

        # 3. WRITE PATH (Only write page 1 requests to Redis)
        if cache_key:
            try:
                cache.set(cache_key, json.dumps(jsonable_encoder(response_payload)), ex_seconds=86400)
            except Exception as set_err:
                logger.warning(f"[Upstash Cache Write Error]: {set_err}")
            response.headers["X-Cache"] = "MISS"
        else:
            response.headers["X-Cache"] = "BYPASS"

        return response_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch recent bills: {str(e)}")
    finally:
        conn.close()


@app.get("/api/analytics/legislative/track-record/{politician_id}")
def get_lawmaker_track_record(politician_id: str):
    conn = None
    clean_id = politician_id.upper()
    cache_key = f"analytics:legislative:track-record:{clean_id}"

    # Cache Read
    try:
        cached = cache.get(cache_key)
        if cached:
            return json.loads(cached if isinstance(cached, str) else cached.decode("utf-8"))
    except Exception as err:
        logger.warning(f"Redis read error for key {cache_key}: {err}")

    try:
        conn = get_db_cursor()

        # A. Funnel Query
        funnel_query = """
                       SELECT
                           member_relationship,
                           total_bills,
                           introduced_count,
                           in_committee_count,
                           passed_chamber_count,
                           enacted_count
                       FROM main.mart_bill_funnel
                       WHERE member_id = ?; \
                       """
        funnel_rows = conn.execute(funnel_query, [clean_id]).fetchall()
        funnel_data = [
            {
                "relationship": row[0],
                "total_bills": row[1],
                "introduced": row[2],
                "in_committee": row[3],
                "passed_chamber": row[4],
                "enacted": row[5]
            }
            for row in funnel_rows
        ]

        # B. Bipartisanship Query
        bipartisan_query = """
                           SELECT
                               total_sponsored_bills,
                               bipartisan_sponsored_bills,
                               ROUND(bipartisanship_score, 1) AS bipartisanship_score
                           FROM main.mart_bipartisanship_score
                           WHERE member_id = ?;
                           """
        b_row = conn.execute(bipartisan_query, [clean_id]).fetchone()
        bipartisanship_data = {
            "total_sponsored_bills": b_row[0] if b_row else 0,
            "bipartisan_sponsored_bills": b_row[1] if b_row else 0,
            "bipartisanship_score": float(b_row[2]) if b_row and b_row[2] is not None else 0.0
        }

        # C. Policy Focus Query
        policy_query = """
                       SELECT
                           policy_area,
                           bill_count,
                           percentage_of_total
                       FROM main.mart_policy_focus
                       WHERE member_id = ?
                       ORDER BY bill_count DESC
                           LIMIT 10; \
                       """
        policy_rows = conn.execute(policy_query, [clean_id]).fetchall()
        policy_data = [
            {
                "policy_area": row[0],
                "bill_count": row[1],
                "percentage": float(row[2])
            }
            for row in policy_rows
        ]

        payload = {
            "politician_id": clean_id,
            "funnel": funnel_data,
            "bipartisanship": bipartisanship_data,
            "policy_focus": policy_data
        }

        # Cache Write (24-Hour TTL)
        try:
            cache.set(cache_key, json.dumps(jsonable_encoder(payload)), ex_seconds=CACHE_TTL_24)
        except Exception as err:
            logger.warning(f"Redis write error for key {cache_key}: {err}")

        return payload

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch legislative track record for {politician_id}: {str(e)}"
        )
    finally:
        if conn:
            conn.close()

# Add or replace get_recently_passed_bills in server.py

@app.get("/api/bills/passed")
def get_recently_passed_bills(
        response: Response,
        q: Optional[str] = Query(default=None, description="Search passed bills by title, number, or sponsor within 1 year"),
        page: int = Query(default=1, ge=1, description="Page number for pagination"),
        per_chamber_limit: int = Query(default=10, ge=1, le=50, description="Number of passed bills per chamber per page"),
        chamber: Optional[str] = Query(default=None, description="Filter by chamber ('House' or 'Senate')")
):
    """
    Retrieves legislation passed within the past year (Passed House, Passed Senate, Became Law)
    with window-based equal chamber pagination.
    Standard feeds are cached via Upstash Redis; active search queries bypass Redis.
    """
    conn = get_db_cursor()

    try:
        search_term = q.strip() if q and q.strip() else None

        # 1. READ PATH (Cache default first-page requests only; skip cache for search or page > 1)
        cache_key = None
        if not search_term and page == 1:
            watermark = get_bills_watermark(conn, table_name="main.mart_recently_passed_bills", chamber=chamber)
            chamber_tag = chamber.lower() if chamber else "all"
            cache_key = f"legipulse:passed_bills:{chamber_tag}:limit_{per_chamber_limit}:wm_{watermark}"

            try:
                cached_data = cache.get(cache_key)
                if cached_data is not None:
                    response.headers["X-Cache"] = "HIT"
                    if isinstance(cached_data, (bytes, bytearray)):
                        cached_data = cached_data.decode("utf-8")
                    return json.loads(cached_data) if isinstance(cached_data, str) else cached_data
            except Exception as redis_err:
                logger.warning(f"[Upstash Cache Miss/Error]: {redis_err}")

        # 2. QUERY DUCKDB WITH PAGINATION WINDOW (365-Day Filter on bill_latest_action_date)
        conditions = ["bill_latest_action_date >= CURRENT_DATE - INTERVAL '365 days'"]
        params = []

        if chamber and chamber.lower() in ("house", "senate"):
            conditions.append("LOWER(chamber) = LOWER(?)")
            params.append(chamber)

        if search_term:
            conditions.append("""
                (LOWER(bill_title) LIKE LOWER(?) 
                 OR LOWER(bill_number) LIKE LOWER(?) 
                 OR LOWER(sponsor_name) LIKE LOWER(?))
            """)
            like_pattern = f"%{search_term}%"
            params.extend([like_pattern, like_pattern, like_pattern])

        where_clause = f"WHERE {' AND '.join(conditions)}"

        min_row = (page - 1) * per_chamber_limit + 1
        max_row = page * per_chamber_limit

        query = f"""
            WITH ranked_bills AS (
                SELECT 
                    bill_id, bill_number, congress, chamber, bill_title,
                    bill_policy_area, bill_summary, bill_introduced_date,
                    bill_latest_action_date, bill_latest_action_text, bill_status,
                    sponsor_bioguide_id, sponsor_name, sponsor_party, sponsor_state,
                    ROW_NUMBER() OVER (
                        PARTITION BY LOWER(chamber) 
                        ORDER BY bill_latest_action_date DESC
                    ) AS rn
                FROM main.mart_recently_passed_bills
                {where_clause}
            )
            SELECT 
                bill_id, bill_number, congress, chamber, bill_title,
                bill_policy_area, bill_summary, bill_introduced_date,
                bill_latest_action_date, bill_latest_action_text, bill_status,
                sponsor_bioguide_id, sponsor_name, sponsor_party, sponsor_state
            FROM ranked_bills
            WHERE rn BETWEEN ? AND ?
            ORDER BY bill_latest_action_date DESC;
        """
        params.extend([min_row, max_row])

        rows = conn.execute(query, params).fetchall()

        bills = []
        for r in rows:
            (
                bill_id, bill_number, congress, chamber_name, title, policy_area,
                raw_summary, introduced_date, latest_action_date, latest_action_text,
                bill_status, sponsor_id, sponsor_name, sponsor_party, sponsor_state
            ) = r

            bill_type_clean = bill_number.split()[0].lower() if bill_number else ""
            bill_num_clean = bill_number.split()[-1] if bill_number else ""

            url_type_map = {
                "hr": "house-bill", "hres": "house-resolution", "hjres": "house-joint-resolution", "hconres": "house-concurrent-resolution",
                "s": "senate-bill", "sres": "senate-resolution", "sjres": "senate-joint-resolution", "sconres": "senate-concurrent-resolution"
            }
            url_segment = url_type_map.get(bill_type_clean, "house-bill" if chamber_name == "House" else "senate-bill")
            congress_url = f"https://www.congress.gov/bill/{congress}th-congress/{url_segment}/{bill_num_clean}"

            clean_summary = strip_html(raw_summary)
            if clean_summary and not clean_summary.strip():
                clean_summary = None

            bills.append({
                "bill_id": bill_id,
                "bill_number": bill_number,
                "congress": congress,
                "chamber": chamber_name,
                "title": title,
                "policy_area": policy_area or "Uncategorized",
                "summary": clean_summary,
                "raw_summary": raw_summary,
                "introduced_date": str(introduced_date) if introduced_date else None,
                "latest_action_date": str(latest_action_date) if latest_action_date else None,
                "latest_action_text": latest_action_text,
                "bill_status": bill_status,
                "sponsor_id": sponsor_id,
                "sponsor_name": sponsor_name,
                "sponsor_party": sponsor_party,
                "sponsor_state": sponsor_state,
                "congress_url": congress_url
            })

        # Calculate if more rows exist for infinite scroll
        expected_count = per_chamber_limit if chamber else per_chamber_limit * 2
        has_more = len(bills) >= expected_count

        response_payload = {
            "bills": bills,
            "count": len(bills),
            "page": page,
            "has_more": has_more
        }

        # 3. WRITE PATH (Only write page 1 requests without search queries to Redis)
        if cache_key:
            try:
                cache.set(cache_key, json.dumps(jsonable_encoder(response_payload)), ex_seconds=86400)
            except Exception as set_err:
                logger.warning(f"[Upstash Cache Write Error]: {set_err}")
            response.headers["X-Cache"] = "MISS"
        else:
            response.headers["X-Cache"] = "BYPASS"

        return response_payload

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch passed bills: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
