

### Getting Started

1. Create a **private** GitHub repository from this archive and push your initial commit.
2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
3. Run `uv sync` to install dependencies.
4. Get a free [Congress API key](https://api.congress.gov/sign-up/). The inherited script reads it from the `CONGRESS_API_KEY` environment variable — `export CONGRESS_API_KEY=<your key>` in your shell, or copy `.env.example` to `.env` (gitignored) and source it before running. How the key gets into the environment is up to you.
5. `cd dbt` — run all dbt commands from the `dbt/` directory.
6. Verify your setup: `uv run dbt debug`
7. Load seed data into a local DuckDB database: `uv run dbt seed`
8. Back in the project root, run the inherited ingestion: `cd .. && uv run scripts/main.py`
9. Build the inherited dbt models: `cd dbt && uv run dbt build`

Step 8 takes roughly 10 minutes with the inherited defaults (the script makes ~800 API calls). Step 9 is sub-second.

### Run ingestion stages separately

The interactive ingestion menu offers separate House/Senate bills-only and
amendments-only runs. Amendments-only uses the bills already stored in
`raw_bills`. The same modes are available from the CLI:

```powershell
uv run scripts/main.py --house --bills-only --skip-members
uv run scripts/main.py --house --amendments-only
```

Use `--reset` with either command to fully refresh only the selected destination
table. Amendments-only runs require bills to have already been ingested.
The pipeline stages are also callable independently from `scripts.pipeline`:
`ingest_members(...)`, `ingest_member_bills(db, members, ...)`, and
`ingest_bill_amendments(db, bills, ...)`.
Bill API requests use a bounded pool of five worker threads by default; their
database writes are serialized through the calling thread.
Member ingestion now fetches current Congress members directly from the API,
then uses the census crosswalk only to attach House district GEOIDs. The
target-district and target-county member-selection options have been removed.
Amendment refresh decisions track the latest action date separately from
`raw_bills`, so loading bills first does not cause their amendments to be
mistakenly skipped. Once a bill's amendments have been fetched, unchanged bills
are skipped on later runs; use a full amendment refresh to process them again.
Bill ingestion stores the Congress API's `introducedDate` from the existing
sponsored/cosponsored legislation response; the politician bills endpoint
returns it as `introducedDate`. Existing databases gain the nullable raw column
automatically when ingestion initializes its schema.

### OpenFEC donor data

Set `FEC_API_KEY` in the project `.env` file, then run
`uv run python scripts/ingest_donars.py` from the repository root after the
dbt models have been built. This creates and refreshes the
`member_top_donors` and `member_fec_summary` tables. The summary table holds
each member's FEC candidate ID, cycle receipts, PAC and individual
contributions, and OpenFEC coverage end date. Candidate IDs are reused on
subsequent runs. `member_fec_candidate_mapping` records the bioguide-to-FEC ID
mapping, validation status, matched candidate details, and candidate options
for manual review. Mapping uses `dbt/seeds/legislators-current.csv` as its
primary Bioguide-to-FEC crosswalk. It checks the listed FEC candidate IDs
against OpenFEC for the current cycle, then validates the candidate's office
and state; matching House districts are preferred, and district mismatches are
noted in the mapping reason. Members without FEC IDs in the crosswalk fall back
to the existing name search, which accepts only a unique normalized
first/last name, office, and state match. Existing matched IDs found in the
crosswalk are reused; conflicting or unresolved mappings are refreshed from
the crosswalk. Run
`uv run python scripts/ingest_donars.py --mappings` to build or refresh mappings
without fetching financial data. Mappings are rechecked every 90 days by
default when unresolved; a matched mapping is reused and not queried again.
Set `FEC_MAPPING_MAX_AGE_DAYS` to change the retry interval for unresolved
mappings. Candidate
summaries are fetched in batches. Run
`uv run python scripts/ingest_donars.py --summaries` to refresh summaries only,
or `uv run python scripts/ingest_donars.py --donors` for detailed employer
aggregates only (not individual contributors). With no mode specified, both run.
Candidates absent from a bulk-summary response remain eligible for retry. Use
`uv run python scripts/ingest_donars.py --summary-cross-check` to fill missing
totals from the latest available reports for the candidate's non-joint-fundraising
committees; the source is recorded in `summary_source`.
Use `--force` with `--donors` to replace cached employer aggregates regardless
of the freshness window. Successfully synced
member/cycle records are skipped for seven days by default. Set
`DONOR_SYNC_MAX_AGE_DAYS` to change that freshness window (use `0` to force a
refresh every run). HTTP 429 rate limits are logged to
`logs/openfec_ingestion.log`; the run pauses cleanly, keeps completed records,
and can be rerun later. Every run also writes a completion report to the console
and that log, including summary candidates saved or missing, summary source,
donor members refreshed, employer rows replaced/saved, freshness skips, and
unmapped members. Partial runs are marked paused and report completed work.
Requests are paced four seconds apart by default; set
`FEC_REQUEST_INTERVAL_SECONDS` to adjust the delay. The API serves a politician's cached donor rows at
`GET /api/politicians/{politician_id}/donors`; an optional `cycle` query
parameter filters the results, for example
`/api/politicians/H1/donors?cycle=2026`. Aggregate campaign finance totals
are available at
`GET /api/politicians/{politician_id}/donations/summary`.
The summary response includes `summaryAvailable` and `summaryStatus`; when all
contribution totals are null, the status is `unavailable` rather than implying
zero contributions.
The politician directory returns sponsored, cosponsored, and enacted bill
counts plus `passRate` from `dim_members`. The rate is null when no sponsored
bills are present in the data, rather than reporting a misleading zero percent.
Chat responses return only `synthesizedAnswer` by default and avoid commentary
about internal SQL choices. Send `"debug": true` in the `/api/chat` request body
to include `routerAnalysis` and request a brief explanation of the data/query
approach.

The pipeline should run end-to-end. We'd recommend that you verify that before you begin making changes.

> **Note:** The DuckDB database file (`dev.duckdb`) is gitignored and should not be committed.

### Build the RAG vector index

The indexer and search agent both use Voyage AI's `voyage-3-lite` embeddings
(512 dimensions). After building the dbt models, set `VOYAGE_API_KEY` in the
environment and run `uv run python scripts/rag/indexer.py` from the repository
root. The indexer hashes chunk content and metadata, embeds only new or changed
chunks, and removes vectors absent from the current dbt view. Its first run
rebuilds legacy/incompatible embedding tables; later runs are incremental.
The RAG agent loads `ANTHROPIC_API_KEY` from the project-root `.env` before
initializing Anthropic clients; that file's key overrides a process-level
`ANTHROPIC_API_KEY`. Both Anthropic clients send the configured
`anthropic-workspace-id` header, defaulting to the workspace ID used by this
project; set `ANTHROPIC_WORKSPACE_ID` to override it. Override the agent's
Claude model with `ANTHROPIC_MODEL` or the router's classification model with
`ANTHROPIC_ROUTER_MODEL`.
The SQL tool limits responses to 50 rows and 12,000 characters, truncating long
cells; for count-and-example questions, the agent is instructed to combine the
aggregate and a small sample in one query.