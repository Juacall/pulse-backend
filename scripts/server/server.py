# server.py
import os
import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Import your existing agent & router modules
from scripts.rag.agent import agent_executor, DB_PATH
from scripts.rag.router import PreProcessingRouter

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
    """Fetch the member directory and attach legislative activity counts."""
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
            COALESCE(activity.sponsored_count, 0) AS sponsoredCount,
            COALESCE(activity.cosponsored_count, 0) AS cosponsoredCount
        FROM main.stg_members AS members
        LEFT JOIN (
            SELECT
                target_member_id,
                COUNT(DISTINCT CASE WHEN relationship = 'sponsor' THEN bill_id END) AS sponsored_count,
                COUNT(DISTINCT CASE WHEN relationship = 'cosponsor' THEN bill_id END) AS cosponsored_count
            FROM main.mart_legislative_activity
            GROUP BY target_member_id
        ) AS activity
            ON members.member_id = activity.target_member_id
        WHERE {where_sql}
        ORDER BY members.member_name ASC
        LIMIT 100;
    """

    try:
        df = conn.execute(sql, params).fetchdf()
        conn.close()
        return df.to_dict(orient="records")
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
    analysis = router.process(request.query)

    # Step 2: Execute Agent Executor
    response = agent_executor.invoke({"messages": [("user", analysis.rewritten_prompt)]})
    final_answer = response["messages"][-1].content

    return {
        "routerAnalysis": {
            "intent": analysis.intent,
            "detectedState": analysis.state_code or "NATIONAL",
            "rewrittenPrompt": analysis.rewritten_prompt
        },
        "synthesizedAnswer": final_answer
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)