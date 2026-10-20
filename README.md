# Pulse API

FastAPI server for Pulse. It serves member, bill, campaign-finance and vote-sentiment data from DuckDB (MotherDuck in production) and answers natural-language questions through a RAG agent. Data ingestion lives in the separate `pulse-ingestion` project.

## Run

```powershell
uv sync
uv run uvicorn scripts.server.server:app --port 8000
```

Copy `.env.example` to `.env` and fill it in. Set `MOTHERDUCK_TOKEN` to read from MotherDuck; without it the server refuses to query unless `ALLOW_LOCAL_DB_FALLBACK=true`, which uses the local `dev.duckdb`. Interactive docs are at `/docs`.

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check. Returns `{"status": "healthy"}`. |
| GET | `/api/politicians` | Lists members with sponsored, cosponsored and enacted bill counts and `passRate` (null when a member has no sponsored bills). Optional `state` (postal code) and `chamber` filters. Cached 24h. |
| GET | `/api/politicians/{politician_id}` | Profile for one member by bioguide ID: name, party, state, chamber, district, office address, phone and website. Cached 24h. |
| GET | `/api/politicians/{politician_id}/donors` | Top donor employer aggregates for a member. Optional `cycle` filters by election cycle. Reports when donor data hasn't been ingested. Cached 24h. |
| GET | `/api/politicians/{politician_id}/donations/summary` | Campaign finance totals (receipts, PAC and individual contributions). `summaryStatus` is `unavailable` when all totals are null, rather than implying zero. Returns 404 if the member has no summary. Cached 24h. |
| GET | `/api/politicians/{bioguide_id}/bills` | Key bills a member sponsored or cosponsored, including `introducedDate`. Cached 12h. |
| GET | `/api/politicians/{politician_id}/votes/stats` | Public approve / neutral / disapprove tallies for a member. Read from Redis, falling back to Supabase (cached 30s). `source` shows which was used. |
| POST | `/api/vote` | Records or updates the signed-in user's vote on a member. Body: `politician_id`, `vote_type` (`approve`, `disapprove` or `neutral`) and `turnstile_token`. Requires a Supabase JWT in the `Authorization` header and a valid Cloudflare Turnstile token. |
| POST | `/api/chat` | Answers a natural-language question. Body: `{"query": "..."}`. A router classifies the query and detects a state, then the agent runs SQL and vector search and returns `synthesizedAnswer` (a structured result: text, table, list or metric) plus `routerAnalysis`. |

## Configuration

See `.env.example`. Key variables: `ANTHROPIC_API_KEY`, `VOYAGE_API_KEY`, `SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `TURNSTILE_SECRET_KEY`, `UPSTASH_REDIS_REST_URL` / `UPSTASH_REDIS_REST_TOKEN` (cache is disabled if missing), and the MotherDuck settings above.

## Tests

```powershell
uv run python -m pytest
```
