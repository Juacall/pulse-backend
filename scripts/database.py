"""
scripts/database.py

Database schema initialization, connection helpers, validations, and metadata tracking for DuckDB.
"""

from pathlib import Path
import sys
import duckdb

# Add script folder to path if executed standalone or as module
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from config import DB_PATH, STATE_FIPS_TO_ABBR
except ImportError:
    from scripts.config import DB_PATH, STATE_FIPS_TO_ABBR


def get_db_connection(db_path: Path = DB_PATH):
    """Open and return a connection to the DuckDB database."""
    return duckdb.connect(str(db_path))


def check_tables_exist(db, tables: list[str] = None) -> bool:
    """
    Validate whether the specified tables exist in the database.
    Defaults to checking the 3 required raw tables.
    """
    if tables is None:
        tables = ["raw_members", "raw_bills", "raw_amendments"]

    existing = {
        row[0] for row in db.execute("SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'").fetchall()
    }
    missing = [t for t in tables if t not in existing]
    if missing:
        return False
    return True


def create_tables(db, replace: bool = False, replace_tables: list[str] | None = None):
    """
    Initialize raw tables in DuckDB with primary keys and checksum/timestamp tracking.

    Three raw tables feed the dbt models downstream:
    - raw_members:    one row per current House or Senate member
    - raw_bills:      one row per member-bill relationship (sponsor or cosponsor)
    - raw_amendments: all amendments to target bills, regardless of sponsor

    When ``replace`` is true, ``replace_tables`` optionally limits which raw
    tables are recreated. If omitted, all raw tables are replaced.
    """
    raw_tables = {"raw_members", "raw_bills", "raw_amendments"}
    tables_to_replace = raw_tables if replace_tables is None else set(replace_tables)
    unknown_tables = tables_to_replace - raw_tables
    if unknown_tables:
        raise ValueError(f"Unknown raw table(s) requested for replacement: {', '.join(sorted(unknown_tables))}")

    def create_stmt(table_name: str) -> str:
        should_replace = replace and table_name in tables_to_replace
        return "CREATE OR REPLACE TABLE" if should_replace else "CREATE TABLE IF NOT EXISTS"

    db.execute(f"""
        {create_stmt("raw_members")} main.raw_members (
            bioguide_id   VARCHAR,
            name          VARCHAR,
            state         VARCHAR,
            chamber       VARCHAR,
            district      INTEGER,
            party         VARCHAR,
            geoid_cd      VARCHAR,
            update_date   TIMESTAMP,
            row_hash      VARCHAR,
            ingested_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (bioguide_id)
        )
    """)
    db.execute(f"""
        {create_stmt("raw_bills")} main.raw_bills (
            congress            INTEGER,
            bill_type           VARCHAR,
            bill_number         VARCHAR,
            title               VARCHAR,
            latest_action_date  VARCHAR,
            latest_action_text  VARCHAR,
            member_id           VARCHAR,
            relationship        VARCHAR,
            update_date         TIMESTAMP,
            row_hash            VARCHAR,
            ingested_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (congress, bill_type, bill_number, member_id, relationship)
        )
    """)
    db.execute(f"""
        {create_stmt("raw_amendments")} main.raw_amendments (
            congress          INTEGER,
            bill_type         VARCHAR,
            bill_number       VARCHAR,
            amendment_number  VARCHAR,
            amendment_type    VARCHAR,
            description       VARCHAR,
            purpose           VARCHAR,
            sponsor_id        VARCHAR,
            update_date       TIMESTAMP,
            row_hash          VARCHAR,
            ingested_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (congress, bill_type, bill_number, amendment_number, amendment_type)
        )
    """)


    amendment_state_stmt = (
        "CREATE OR REPLACE TABLE"
        if replace and "raw_amendments" in tables_to_replace
        else "CREATE TABLE IF NOT EXISTS"
    )
    db.execute(f"""
        {amendment_state_stmt} main.bill_amendment_ingestion_state (
            congress INTEGER,
            bill_type VARCHAR,
            bill_number VARCHAR,
            latest_action_date VARCHAR,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (congress, bill_type, bill_number)
        )
    """)

    db.execute(f"""
            -- Track execution lifecycle
         CREATE TABLE IF NOT EXISTS main.ingestion_runs (
            run_id VARCHAR PRIMARY KEY,
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            status VARCHAR -- 'RUNNING', 'COMPLETED', 'FAILED'
        )
""")

    db.execute(f"""
            -- Track completed operational steps to allow seamless resumes
             CREATE TABLE IF NOT EXISTS main.ingestion_step_history (
                run_id VARCHAR,
                step_type VARCHAR,  -- 'MEMBER_BILLS', 'BILL_AMENDMENTS'
                entity_key VARCHAR, -- e.g., 'bioguide_id' or '119-HR-1234'
                completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (run_id, step_type, entity_key)
            )
""")


def get_existing_bill_timestamps(db) -> dict[tuple[int, str, str], str]:
    """
    Return the last action date processed for amendments per bill.

    This is deliberately separate from raw_bills: bill ingestion may run before
    the first amendment ingestion, and should not mark amendments as processed.
    """
    create_tables(db, replace=False)
    rows = db.execute("""
        SELECT congress, bill_type, bill_number, latest_action_date
        FROM main.bill_amendment_ingestion_state
        WHERE latest_action_date IS NOT NULL
    """).fetchall()
    return {(row[0], row[1].upper(), str(row[2])): row[3] for row in rows}


def mark_bill_amendments_processed(db, congress, bill_type, bill_number, latest_action_date):
    """Persist the bill action date after its amendments were fetched and saved."""
    db.execute("""
        INSERT INTO main.bill_amendment_ingestion_state (
            congress, bill_type, bill_number, latest_action_date, processed_at
        ) VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (congress, bill_type, bill_number) DO UPDATE SET
            latest_action_date = EXCLUDED.latest_action_date,
            processed_at = EXCLUDED.processed_at
    """, [congress, bill_type.upper(), str(bill_number), latest_action_date])


def get_existing_bills(db, chamber: str | None = None) -> list[tuple[int, str, str, str | None]]:
    """Return distinct stored bills, optionally limited to a chamber."""
    create_tables(db, replace=False)
    query = """
        SELECT bills.congress, bills.bill_type, bills.bill_number,
               MAX(bills.latest_action_date) AS latest_action_date
        FROM main.raw_bills AS bills
    """
    params = []
    if chamber and chamber.lower() in ("house", "senate"):
        query += """
            JOIN main.raw_members AS members
              ON bills.member_id = members.bioguide_id
            WHERE LOWER(members.chamber) = ?
        """
        params.append(chamber.lower())
    query += """
        GROUP BY bills.congress, bills.bill_type, bills.bill_number
        ORDER BY bills.congress, bills.bill_type, bills.bill_number
    """
    rows = db.execute(query, params).fetchall()
    return [(row[0], row[1].upper(), str(row[2]), row[3]) for row in rows]


def get_existing_members(
    db,
    member_limit: int | None = None,
    shuffle: bool = False,
    chamber: str | None = None,
) -> list[dict]:
    """
    Query existing members from raw_members table.
    Returns a list of dicts with 'bioguideId', 'name', 'chamber', etc.
    """
    create_tables(db, replace=False)
    order_clause = "random()" if shuffle else "bioguide_id"
    query = "SELECT bioguide_id, name, state, chamber, district, party, geoid_cd FROM main.raw_members"
    params = []
    if chamber and chamber.lower() in ("house", "senate"):
        query += " WHERE LOWER(chamber) = ?"
        params.append(chamber.lower())
    query += f" ORDER BY {order_clause}"
    if member_limit is not None:
        query += f" LIMIT {int(member_limit)}"
    rows = db.execute(query, params).fetchall() if params else db.execute(query).fetchall()
    return [
        {
            "bioguideId": row[0],
            "name": row[1],
            "state": row[2],
            "chamber": row[3],
            "district": row[4],
            "partyName": row[5],
            "_geoid_cd": row[6],
        }
        for row in rows
    ]


def get_district_geoids(db) -> dict[tuple[str, int], str]:
    """Map (state abbreviation, district number) to census congressional GEOID."""
    if not check_tables_exist(db, ["raw_census__cd11920_county20"]):
        raise RuntimeError(
            "Prerequisite seed table 'raw_census__cd11920_county20' is missing. "
            "Please run 'uv run dbt seed' from the dbt/ directory first."
        )

    rows = db.execute("""
        SELECT DISTINCT
            GEOID_CD119_20,
            LEFT(GEOID_CD119_20, 2) AS state_fips,
            CAST(RIGHT(GEOID_CD119_20, 2) AS INTEGER) AS district_num
        FROM main.raw_census__cd11920_county20
        WHERE GEOID_CD119_20 NOT LIKE '%ZZ'
    """).fetchall()
    return {
        (STATE_FIPS_TO_ABBR[state_fips], district_num): geoid_cd
        for geoid_cd, state_fips, district_num in rows
        if state_fips in STATE_FIPS_TO_ABBR
    }
