import argparse
import csv
import json
import logging
import os
import re
import time
from pathlib import Path

import duckdb
import httpx

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "dev.duckdb"
LOG_PATH = BASE_DIR / "logs" / "openfec_ingestion.log"
CROSSWALK_PATH = BASE_DIR / "dbt" / "seeds" / "legislators-current.csv"
FEC_BASE_URL = "https://api.open.fec.gov/v1"
DONOR_CYCLE = 2026
DONOR_SYNC_MAX_AGE_DAYS = int(os.getenv("DONOR_SYNC_MAX_AGE_DAYS", "7"))
FEC_MAPPING_MAX_AGE_DAYS = int(os.getenv("FEC_MAPPING_MAX_AGE_DAYS", "90"))
FEC_REQUEST_INTERVAL_SECONDS = max(
    0.0, float(os.getenv("FEC_REQUEST_INTERVAL_SECONDS", "4"))
)
SUMMARY_BATCH_SIZE = 50
FEC_MAPPING_BATCH_SIZE = 50
NON_COMPANY_EMPLOYERS = {
    "ENTREPRENEUR",
    "HOMEMAKER",
    "INFORMATIONREQUESTED",
    "INFORMATIONREQUESTEDPERBESTEFFORTS",
    "NA",
    "NONE",
    "NOTEMPLOYED",
    "NULL",
    "RETIRED",
    "SELF",
    "SELFEMPLOYED",
}
_last_fec_request_at = None


class OpenFECRateLimitError(RuntimeError):
    def __init__(self, endpoint: str, retry_after: str | None = None):
        super().__init__(f"OpenFEC rate limit reached at {endpoint}.")
        self.endpoint = endpoint
        self.retry_after = retry_after
        self.context = "context unavailable"


class OpenFECRequestTimeoutError(RuntimeError):
    def __init__(self, endpoint: str):
        super().__init__(f"OpenFEC request timed out at {endpoint}.")
        self.endpoint = endpoint
        self.context = "context unavailable"


def _get_ingestion_logger():
    logger = logging.getLogger("openfec_ingestion")
    log_path = LOG_PATH.resolve()
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler) and Path(
            handler.baseFilename
        ).resolve() != log_path:
            logger.removeHandler(handler)
            handler.close()
    if not any(
        isinstance(handler, logging.FileHandler)
        and Path(handler.baseFilename).resolve() == log_path
        for handler in logger.handlers
    ):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


def _with_rate_limit_context(operation, context):
    try:
        return operation()
    except (OpenFECRateLimitError, OpenFECRequestTimeoutError) as error:
        error.context = context
        raise


def _pace_fec_request():
    """Keep successive OpenFEC request starts at least the configured interval apart."""
    global _last_fec_request_at
    now = time.monotonic()
    if _last_fec_request_at is not None:
        remaining = FEC_REQUEST_INTERVAL_SECONDS - (now - _last_fec_request_at)
        if remaining > 0:
            time.sleep(remaining)
            now = time.monotonic()
    _last_fec_request_at = now


def get_fec_api_key() -> str:
    """Get the OpenFEC API key from the environment or the project .env file."""
    api_key = os.getenv("FEC_API_KEY", "").strip()
    if not api_key:
        env_file = BASE_DIR / ".env"
        if env_file.exists():
            for line in env_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    if key.strip() == "FEC_API_KEY":
                        api_key = value.strip().strip("\"'")
                        break

    if not api_key:
        raise RuntimeError("FEC_API_KEY is not set in the environment or project .env file.")
    return api_key


def init_donor_tables(conn):
    """Create donor tables in dev.duckdb."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS member_fec_summary (
            bioguide_id VARCHAR PRIMARY KEY,
            fec_candidate_id VARCHAR,
            total_receipts DOUBLE,
            pac_contributions DOUBLE,
            individual_contributions DOUBLE,
            coverage_end_date VARCHAR,
            summary_source VARCHAR
        )
    """)
    conn.execute(
        "ALTER TABLE main.member_fec_summary ADD COLUMN IF NOT EXISTS summary_source VARCHAR"
    )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS member_top_donors (
            bioguide_id VARCHAR,
            donor_name VARCHAR,
            amount DOUBLE,
            industry VARCHAR,
            cycle VARCHAR,
            PRIMARY KEY (bioguide_id, donor_name, cycle)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS member_fec_sync_state (
            bioguide_id VARCHAR,
            cycle VARCHAR,
            synced_at TIMESTAMP,
            PRIMARY KEY (bioguide_id, cycle)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS member_fec_summary_sync_state (
            bioguide_id VARCHAR,
            cycle VARCHAR,
            synced_at TIMESTAMP,
            PRIMARY KEY (bioguide_id, cycle)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS member_fec_candidate_mapping (
            bioguide_id VARCHAR PRIMARY KEY,
            member_name VARCHAR NOT NULL,
            state_code VARCHAR NOT NULL,
            chamber VARCHAR NOT NULL,
            district_number INTEGER,
            fec_candidate_id VARCHAR,
            fec_candidate_name VARCHAR,
            fec_office VARCHAR,
            fec_state VARCHAR,
            fec_district VARCHAR,
            mapping_status VARCHAR NOT NULL,
            mapping_reason VARCHAR,
            candidate_options JSON,
            checked_at TIMESTAMP NOT NULL
        )
    """)


def _get_fec_response(endpoint: str, params: dict):
    """Request an OpenFEC endpoint without exposing the API key in HTTP errors."""
    _pace_fec_request()
    try:
        response = httpx.get(
            f"{FEC_BASE_URL}/{endpoint.lstrip('/')}",
            params=params,
            timeout=60,
        )
    except httpx.TimeoutException as error:
        raise OpenFECRequestTimeoutError(endpoint) from error
    if response.is_error:
        if response.status_code == 429:
            raise OpenFECRateLimitError(
                endpoint, response.headers.get("Retry-After")
            )
        raise RuntimeError(
            f"OpenFEC request failed with HTTP {response.status_code} at {endpoint}."
        )
    return response


def _normalize_name(name: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


def _candidate_name_identity(name: str | None) -> tuple[str, str]:
    """Extract normalized surname and first name from FEC/member name formats."""
    parts = (name or "").upper().split(",", maxsplit=1)
    if len(parts) == 2:
        surname = _normalize_name(parts[0])
        given_names = re.findall(r"[A-Z]+", parts[1])
        first_name = given_names[0] if given_names else ""
        return surname, first_name

    parts = re.findall(r"[A-Z]+", (name or "").upper())
    if len(parts) < 2:
        return "", ""
    return _normalize_name(parts[-1]), parts[0]


def _normalize_district(district) -> int | None:
    if district is None or str(district).strip() == "":
        return None
    try:
        return int(district)
    except (TypeError, ValueError):
        return None


def fetch_fec_candidate_mapping(
    bioguide_id: str,
    name: str,
    state: str,
    chamber: str,
    district: int | None,
    api_key: str,
) -> dict:
    """Broaden candidate search and accept only a unique name/office/state match."""
    office = {"house": "H", "senate": "S"}.get(chamber.strip().casefold())
    if office is None:
        raise ValueError(f"Unsupported chamber for FEC mapping: {chamber}")

    name_variants = [name]
    if "," in name:
        surname, given_names = name.split(",", maxsplit=1)
        given_tokens = re.findall(r"[A-Za-z]+", given_names)
        if given_tokens:
            name_variants.append(f"{given_tokens[0]} {surname.strip()}")
        name_variants.append(surname.strip())

    candidates_by_id = {}
    expected_identity = _candidate_name_identity(name)
    for query in dict.fromkeys(variant.strip() for variant in name_variants if variant.strip()):
        response = _get_fec_response(
            "candidates/search/",
            [
                ("api_key", api_key),
                ("q", query),
                ("state", state.upper()),
                ("office", office),
                ("cycle", DONOR_CYCLE),
                ("per_page", 100),
                ("sort", "-receipts"),
            ],
        )
        for candidate in response.json().get("results", []):
            candidate_id = candidate.get("candidate_id")
            if candidate_id:
                candidates_by_id[candidate_id] = candidate

        identity_matches = [
            candidate
            for candidate in candidates_by_id.values()
            if _candidate_name_identity(candidate.get("name")) == expected_identity
            and str(candidate.get("office") or "").upper() == office
            and str(candidate.get("state") or "").upper() == state.upper()
        ]
        if identity_matches:
            break

    candidates = list(candidates_by_id.values())
    options = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "name": candidate.get("name"),
            "office": candidate.get("office"),
            "state": candidate.get("state"),
            "district": candidate.get("district"),
        }
        for candidate in candidates
    ]
    identity_matches = [
        candidate
        for candidate in candidates
        if _candidate_name_identity(candidate.get("name")) == expected_identity
        and str(candidate.get("office") or "").upper() == office
        and str(candidate.get("state") or "").upper() == state.upper()
    ]
    district_matches = [
        candidate
        for candidate in identity_matches
        if office != "H"
        or _normalize_district(candidate.get("district")) == district
    ]
    matches = district_matches or identity_matches

    if len(matches) == 1:
        candidate = matches[0]
        status = "MATCHED"
        candidate_district = _normalize_district(candidate.get("district"))
        if office == "H" and candidate_district != district:
            reason = (
                "Unique normalized first/last name, office, and state match; "
                "candidate district differs or was changed by redistricting."
            )
        else:
            reason = "Unique normalized first/last name, office, state, and district match."
    elif len(matches) > 1:
        candidate = None
        status = "AMBIGUOUS"
        reason = "Multiple FEC candidates match the member's name, office, and state."
    else:
        candidate = None
        status = "NOT_FOUND"
        reason = (
            "No unique normalized first/last name, office, and state match."
            if candidates
            else "OpenFEC returned no candidate search results."
        )

    return {
        "bioguide_id": bioguide_id,
        "member_name": name,
        "state_code": state.upper(),
        "chamber": chamber,
        "district_number": district,
        "fec_candidate_id": candidate.get("candidate_id") if candidate else None,
        "fec_candidate_name": candidate.get("name") if candidate else None,
        "fec_office": candidate.get("office") if candidate else None,
        "fec_state": candidate.get("state") if candidate else None,
        "fec_district": candidate.get("district") if candidate else None,
        "mapping_status": status,
        "mapping_reason": reason,
        "candidate_options": json.dumps(options),
    }


def load_legislator_fec_crosswalk(
    path: Path | None = None,
) -> dict[str, list[str]]:
    """Load Bioguide IDs and their FEC candidate IDs from the legislators seed."""
    path = path or CROSSWALK_PATH
    with path.open(encoding="utf-8-sig", newline="") as crosswalk_file:
        reader = csv.DictReader(crosswalk_file)
        required_columns = {"bioguide_id", "fec_ids"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"{path} must contain the columns bioguide_id and fec_ids."
            )

        crosswalk = {}
        for row in reader:
            bioguide_id = (row.get("bioguide_id") or "").strip()
            if not bioguide_id:
                continue
            if bioguide_id in crosswalk:
                raise ValueError(
                    f"{path} contains duplicate Bioguide ID {bioguide_id}."
                )
            crosswalk[bioguide_id] = list(
                dict.fromkeys(
                    candidate_id.strip()
                    for candidate_id in (row.get("fec_ids") or "").split(",")
                    if candidate_id.strip()
                )
            )
    return crosswalk


def fetch_fec_candidates_by_ids(
    fec_candidate_ids: list[str], api_key: str, cycle: int = DONOR_CYCLE
) -> list[dict]:
    """Fetch current-cycle candidate records for IDs provided by the crosswalk."""
    if not fec_candidate_ids:
        return []

    response = _get_fec_response(
        "candidates/",
        [
            ("api_key", api_key),
            ("cycle", cycle),
            ("per_page", min(len(fec_candidate_ids), 100)),
            *(("candidate_id", candidate_id) for candidate_id in fec_candidate_ids),
        ],
    )
    return response.json().get("results", [])


def _mapping_from_crosswalk_candidates(member, fec_candidate_ids, candidates):
    bioguide_id, name, state, chamber, district = member
    office = {"house": "H", "senate": "S"}.get(chamber.strip().casefold())
    if office is None:
        raise ValueError(f"Unsupported chamber for FEC mapping: {chamber}")

    candidates_by_id = {
        candidate["candidate_id"]: candidate
        for candidate in candidates
        if candidate.get("candidate_id") in fec_candidate_ids
    }
    options = [
        {
            "candidate_id": candidate.get("candidate_id"),
            "name": candidate.get("name"),
            "office": candidate.get("office"),
            "state": candidate.get("state"),
            "district": candidate.get("district"),
        }
        for candidate in candidates_by_id.values()
    ]
    eligible = [
        candidate
        for candidate in candidates_by_id.values()
        if str(candidate.get("office") or "").upper() == office
        and str(candidate.get("state") or "").upper() == state.upper()
    ]
    district_matches = [
        candidate
        for candidate in eligible
        if office != "H"
        or _normalize_district(candidate.get("district")) == district
    ]
    matches = district_matches or eligible

    candidate = matches[0] if len(matches) == 1 else None
    if candidate:
        status = "MATCHED"
        if office == "H" and _normalize_district(candidate.get("district")) != district:
            reason = (
                "Crosswalk candidate ID validated for the current cycle; "
                "FEC district differs or was changed by redistricting."
            )
        else:
            reason = "Crosswalk candidate ID validated for the current cycle."
    elif len(matches) > 1:
        status = "AMBIGUOUS"
        reason = (
            "Multiple crosswalk candidate IDs match the current cycle, office, "
            "and state."
        )
    elif not candidates_by_id:
        status = "NOT_FOUND"
        reason = f"No crosswalk candidate ID returned by OpenFEC for cycle {DONOR_CYCLE}."
    else:
        status = "NOT_FOUND"
        reason = "Crosswalk candidates do not match the member's current office and state."

    return {
        "bioguide_id": bioguide_id,
        "member_name": name,
        "state_code": state.upper(),
        "chamber": chamber,
        "district_number": district,
        "fec_candidate_id": candidate.get("candidate_id") if candidate else None,
        "fec_candidate_name": candidate.get("name") if candidate else None,
        "fec_office": candidate.get("office") if candidate else None,
        "fec_state": candidate.get("state") if candidate else None,
        "fec_district": candidate.get("district") if candidate else None,
        "mapping_status": status,
        "mapping_reason": reason,
        "candidate_options": json.dumps(options),
    }


def fetch_candidate_committees(
    fec_candidate_id: str, api_key: str, cycle: int = DONOR_CYCLE
) -> list[dict]:
    """Fetch non-joint-fundraising committees associated with a candidate."""
    response = _get_fec_response(
        f"candidate/{fec_candidate_id}/committees/",
        {"api_key": api_key, "cycle": cycle, "per_page": 100},
    )
    committees = {}
    for row in response.json().get("results", []):
        committee_id = row.get("committee_id")
        if committee_id and row.get("designation") != "J":
            committees[committee_id] = row
    return list(committees.values())


def fetch_candidate_committee_ids(
    fec_candidate_id: str, api_key: str, cycle: int = DONOR_CYCLE
) -> list[str]:
    """Fetch candidate committee IDs, excluding joint fundraising committees."""
    return [
        committee["committee_id"]
        for committee in fetch_candidate_committees(
            fec_candidate_id, api_key, cycle
        )
    ]


def fetch_candidate_committee_report_summary(
    fec_candidate_id: str, api_key: str, cycle: int = DONOR_CYCLE
) -> dict | None:
    """Build a fallback summary from each committee's latest cycle filing."""
    committees = fetch_candidate_committees(fec_candidate_id, api_key, cycle)
    committee_summaries = []
    for committee in committees:
        committee_id = committee["committee_id"]
        response = _get_fec_response(
            f"committee/{committee_id}/reports/",
            [
                ("api_key", api_key),
                ("cycle", cycle),
                ("per_page", 100),
                ("sort", "-receipt_date"),
            ],
        )
        reports = [
            report
            for report in response.json().get("results", [])
            if report.get("most_recent") is True
        ]
        if not reports:
            continue

        latest_report = max(
            reports,
            key=lambda report: (
                str(report.get("coverage_end_date") or ""),
                str(report.get("receipt_date") or ""),
            ),
        )
        if latest_report.get("total_receipts_ytd") is not None:
            committee_summaries.append(latest_report)

    if not committee_summaries:
        return None

    coverage_dates = [
        str(report["coverage_end_date"])[:10]
        for report in committee_summaries
        if report.get("coverage_end_date")
    ]
    return {
        "total_receipts": sum(
            float(report.get("total_receipts_ytd") or 0)
            for report in committee_summaries
        ),
        "pac_contributions": sum(
            float(report.get("other_political_committee_contributions_ytd") or 0)
            for report in committee_summaries
        ),
        "individual_contributions": sum(
            float(report.get("individual_itemized_contributions_ytd") or 0)
            for report in committee_summaries
        ),
        "coverage_end_date": min(coverage_dates) if coverage_dates else None,
        "summary_source": "committee_reports",
    }


def fetch_top_pac_contributors(
    fec_candidate_id: str, api_key: str, cycle: int = DONOR_CYCLE
) -> list[dict]:
    """Fetch top employer aggregates from the candidate's committees."""
    committee_ids = fetch_candidate_committee_ids(
        fec_candidate_id, api_key, cycle
    )
    if not committee_ids:
        return []

    response = _get_fec_response(
        "schedules/schedule_a/by_employer/",
        [
            ("api_key", api_key),
            ("cycle", cycle),
            ("sort", "-total"),
            ("per_page", 100),
            *(("committee_id", committee_id) for committee_id in committee_ids),
        ],
    )
    totals_by_employer = {}
    for row in response.json().get("results", []):
        employer = row.get("employer")
        donor_name = employer.strip() if isinstance(employer, str) else ""
        employer_category = re.sub(r"[^A-Z0-9]", "", donor_name.upper())
        if not donor_name or employer_category in NON_COMPANY_EMPLOYERS:
            continue
        donor_name = donor_name.title()
        amount = float(row.get("total") or 0.0)
        totals_by_employer[donor_name] = totals_by_employer.get(donor_name, 0.0) + amount

    top_employers = sorted(
        totals_by_employer.items(), key=lambda item: item[1], reverse=True
    )[:5]
    return [
        {
            "donor_name": donor_name,
            "amount": amount,
            "industry": "Employer aggregate",
        }
        for donor_name, amount in top_employers
    ]


def fetch_fec_candidate_summaries(
    fec_candidate_ids: list[str], api_key: str, cycle: int = DONOR_CYCLE
) -> dict[str, dict]:
    """Fetch summary totals for multiple candidates in a single OpenFEC request."""
    if not fec_candidate_ids:
        return {}

    response = _get_fec_response(
        "candidates/totals/",
        [
            ("api_key", api_key),
            ("cycle", cycle),
            ("per_page", max(len(fec_candidate_ids), 20)),
            *(("candidate_id", candidate_id) for candidate_id in fec_candidate_ids),
        ],
    )
    summaries = {}
    for row in response.json().get("results", []):
        candidate_id = row.get("candidate_id")
        if candidate_id:
            summaries[candidate_id] = {
                "total_receipts": row.get("receipts", row.get("total_receipts")),
                "pac_contributions": row.get("other_political_committee_contributions"),
                "individual_contributions": row.get("individual_itemized_contributions"),
                "coverage_end_date": row.get("coverage_end_date"),
                "summary_source": "candidate_totals",
            }
    return summaries


def _save_candidate_mapping(conn, mapping):
    conn.execute(
        """
        INSERT INTO main.member_fec_candidate_mapping (
            bioguide_id, member_name, state_code, chamber, district_number,
            fec_candidate_id, fec_candidate_name, fec_office, fec_state,
            fec_district, mapping_status, mapping_reason, candidate_options,
            checked_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        ON CONFLICT (bioguide_id) DO UPDATE SET
            member_name = EXCLUDED.member_name,
            state_code = EXCLUDED.state_code,
            chamber = EXCLUDED.chamber,
            district_number = EXCLUDED.district_number,
            fec_candidate_id = EXCLUDED.fec_candidate_id,
            fec_candidate_name = EXCLUDED.fec_candidate_name,
            fec_office = EXCLUDED.fec_office,
            fec_state = EXCLUDED.fec_state,
            fec_district = EXCLUDED.fec_district,
            mapping_status = EXCLUDED.mapping_status,
            mapping_reason = EXCLUDED.mapping_reason,
            candidate_options = EXCLUDED.candidate_options,
            checked_at = EXCLUDED.checked_at
        """,
        [
            mapping["bioguide_id"],
            mapping["member_name"],
            mapping["state_code"],
            mapping["chamber"],
            mapping["district_number"],
            mapping["fec_candidate_id"],
            mapping["fec_candidate_name"],
            mapping["fec_office"],
            mapping["fec_state"],
            mapping["fec_district"],
            mapping["mapping_status"],
            mapping["mapping_reason"],
            mapping["candidate_options"],
        ],
    )


def _ensure_candidate_mappings(conn, members, api_key):
    crosswalk = load_legislator_fec_crosswalk()
    mapping_rows = conn.execute("""
        SELECT bioguide_id, fec_candidate_id, mapping_status, mapping_reason, checked_at
        FROM main.member_fec_candidate_mapping
    """).fetchall()
    mappings = {}
    checked_at = {}
    mapping_reasons = {}
    for bioguide_id, fec_id, status, reason, last_checked in mapping_rows:
        checked_at[bioguide_id] = last_checked
        mapping_reasons[bioguide_id] = reason
        if status == "NOT_FOUND" and reason in {
            "No exact normalized name, office, state, and district match.",
            "OpenFEC returned no candidate search results.",
        }:
            checked_at[bioguide_id] = None
        if status == "MATCHED" and fec_id:
            mappings[bioguide_id] = fec_id

    crosswalk_lookups = []
    name_lookups = []
    for member in members:
        bioguide_id, name, state, chamber, district = member
        fec_candidate_ids = crosswalk.get(bioguide_id, [])
        existing_fec_id = mappings.get(bioguide_id)
        if existing_fec_id and (
            not fec_candidate_ids or existing_fec_id in fec_candidate_ids
        ):
            continue

        last_checked = checked_at.get(bioguide_id)
        recently_checked = last_checked is not None and conn.execute(
            "SELECT ? >= CURRENT_TIMESTAMP - (? * INTERVAL '1 day')",
            [last_checked, FEC_MAPPING_MAX_AGE_DAYS],
        ).fetchone()[0]
        mapping_reason = mapping_reasons.get(bioguide_id) or ""
        previously_checked_crosswalk = mapping_reason.startswith(
            (
                "Crosswalk candidate ID",
                "Multiple crosswalk candidate IDs",
                "No crosswalk candidate ID",
                "Crosswalk candidates do not match",
            )
        )
        if recently_checked and (
            not fec_candidate_ids or previously_checked_crosswalk
        ):
            continue

        if fec_candidate_ids:
            crosswalk_lookups.append((member, fec_candidate_ids))
        else:
            name_lookups.append(member)

    all_candidate_ids = list(
        dict.fromkeys(
            candidate_id
            for _member, candidate_ids in crosswalk_lookups
            for candidate_id in candidate_ids
        )
    )
    candidates_by_id = {}
    for offset in range(0, len(all_candidate_ids), FEC_MAPPING_BATCH_SIZE):
        batch_ids = all_candidate_ids[offset:offset + FEC_MAPPING_BATCH_SIZE]
        batch = _with_rate_limit_context(
            lambda: fetch_fec_candidates_by_ids(batch_ids, api_key),
            f"crosswalk candidate batch {offset // FEC_MAPPING_BATCH_SIZE + 1} "
            f"({len(batch_ids)} candidate IDs)",
        )
        for candidate in batch:
            candidate_id = candidate.get("candidate_id")
            if candidate_id:
                candidates_by_id[candidate_id] = candidate

    for member, fec_candidate_ids in crosswalk_lookups:
        mapping = _mapping_from_crosswalk_candidates(
            member, fec_candidate_ids, list(candidates_by_id.values())
        )
        _save_candidate_mapping(conn, mapping)
        if mapping["mapping_status"] == "MATCHED":
            mappings[mapping["bioguide_id"]] = mapping["fec_candidate_id"]
            _save_candidate_id(
                conn, mapping["bioguide_id"], mapping["fec_candidate_id"]
            )
        else:
            mappings.pop(mapping["bioguide_id"], None)
        print(
            f"FEC mapping {mapping['mapping_status']}: "
            f"{mapping['member_name']} ({mapping['bioguide_id']})"
        )

    for member in name_lookups:
        bioguide_id, name, state, chamber, district = member
        mapping = _with_rate_limit_context(
            lambda: fetch_fec_candidate_mapping(
                bioguide_id, name, state, chamber, district, api_key
            ),
            f"mapping lookup for {name} ({bioguide_id})",
        )
        _save_candidate_mapping(conn, mapping)
        if mapping["mapping_status"] == "MATCHED":
            mappings[bioguide_id] = mapping["fec_candidate_id"]
            _save_candidate_id(conn, bioguide_id, mapping["fec_candidate_id"])
        else:
            mappings.pop(bioguide_id, None)
        print(
            f"FEC mapping {mapping['mapping_status']}: {name} ({bioguide_id})"
        )

    return mappings


def _save_candidate_id(conn, bioguide_id, fec_id):
    """Persist a resolved FEC ID without replacing existing summary values."""
    conn.execute(
        """
        INSERT INTO main.member_fec_summary (bioguide_id, fec_candidate_id)
        VALUES (?, ?)
        ON CONFLICT (bioguide_id) DO UPDATE SET fec_candidate_id = EXCLUDED.fec_candidate_id
        """,
        [bioguide_id, fec_id],
    )


def _mark_synced(conn, table, bioguide_ids):
    for bioguide_id in bioguide_ids:
        conn.execute(
            f"""
            INSERT INTO main.{table} (bioguide_id, cycle, synced_at)
            VALUES (?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT (bioguide_id, cycle) DO UPDATE SET
                synced_at = EXCLUDED.synced_at
            """,
            [bioguide_id, str(DONOR_CYCLE)],
        )


def _upsert_summary(conn, bioguide_id, fec_id, summary):
    conn.execute(
        """
        INSERT INTO main.member_fec_summary (
            bioguide_id, fec_candidate_id, total_receipts, pac_contributions,
            individual_contributions, coverage_end_date, summary_source
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT (bioguide_id) DO UPDATE SET
            fec_candidate_id = EXCLUDED.fec_candidate_id,
            total_receipts = COALESCE(
                EXCLUDED.total_receipts, member_fec_summary.total_receipts
            ),
            pac_contributions = COALESCE(
                EXCLUDED.pac_contributions, member_fec_summary.pac_contributions
            ),
            individual_contributions = COALESCE(
                EXCLUDED.individual_contributions,
                member_fec_summary.individual_contributions
            ),
            coverage_end_date = COALESCE(
                EXCLUDED.coverage_end_date, member_fec_summary.coverage_end_date
            ),
            summary_source = COALESCE(
                EXCLUDED.summary_source, member_fec_summary.summary_source
            )
        """,
        [
            bioguide_id,
            fec_id,
            summary.get("total_receipts") if summary else None,
            summary.get("pac_contributions") if summary else None,
            summary.get("individual_contributions") if summary else None,
            summary.get("coverage_end_date") if summary else None,
            summary.get("summary_source") if summary else None,
        ],
    )


def _emit_ingestion_run_summary(mode, stats):
    lines = [
        f"OpenFEC ingestion {stats['status']}: mode={mode}, cycle={DONOR_CYCLE}, "
        f"members={stats['members']}, mapped={stats['mapped']}, "
        f"unmapped={stats['unmapped']}"
    ]
    if mode in ("both", "summaries"):
        lines.append(
            "Summary results: "
            f"eligible={stats['summary_eligible']}, "
            f"skipped_fresh={stats['summary_fresh']}, "
            f"attempted={stats['summary_attempted']}, "
            f"saved_with_receipts={stats['summary_saved']}, "
            f"returned_without_receipts={stats['summary_missing_total']}, "
            f"absent_from_response={stats['summary_absent']}, "
            "not_processed_or_paused="
            f"{stats['summary_eligible'] - stats['summary_saved'] - stats['summary_missing_total'] - stats['summary_absent']}, "
            f"unmapped={stats['summary_unmapped']}"
        )
        if stats["summary_sources"]:
            sources = ", ".join(
                f"{source}={count}"
                for source, count in sorted(stats["summary_sources"].items())
            )
            lines.append(f"Summary sources saved: {sources}")
    if mode == "summary-cross-check":
        lines.append(
            "Summary cross-check results: "
            f"eligible={stats['fallback_eligible']}, "
            f"attempted={stats['fallback_attempted']}, "
            f"saved={stats['fallback_saved']}, "
            f"no_usable_reports={stats['fallback_no_report']}, "
            "not_processed_or_paused="
            f"{stats['fallback_eligible'] - stats['fallback_saved'] - stats['fallback_no_report']}, "
            f"unmapped={stats['summary_unmapped']}"
        )
    if mode in ("both", "donors"):
        lines.append(
            "Donor results: "
            f"eligible={stats['donor_eligible']}, "
            f"skipped_fresh={stats['donor_fresh']}, "
            f"refreshed_members={stats['donor_refreshed']}, "
            f"employer_rows_saved={stats['donor_rows']}, "
            f"old_rows_replaced={stats['donor_rows_replaced']}, "
            f"members_with_no_qualifying_employers={stats['donor_empty']}, "
            "not_processed_or_paused="
            f"{stats['donor_eligible'] - stats['donor_refreshed']}, "
            f"unmapped={stats['donor_unmapped']}"
        )

    logger = _get_ingestion_logger()
    for line in lines:
        logger.info(line)
        print(line)


def run_fec_ingestion(mode="both", force=False):
    api_key = get_fec_api_key()
    conn = duckdb.connect(str(DB_PATH))
    stats = {
        "status": "failed",
        "members": 0,
        "mapped": 0,
        "unmapped": 0,
        "summary_eligible": 0,
        "summary_fresh": 0,
        "summary_attempted": 0,
        "summary_saved": 0,
        "summary_missing_total": 0,
        "summary_absent": 0,
        "summary_unmapped": 0,
        "summary_sources": {},
        "fallback_eligible": 0,
        "fallback_attempted": 0,
        "fallback_saved": 0,
        "fallback_no_report": 0,
        "donor_eligible": 0,
        "donor_fresh": 0,
        "donor_refreshed": 0,
        "donor_rows": 0,
        "donor_rows_replaced": 0,
        "donor_empty": 0,
        "donor_unmapped": 0,
    }
    try:
        init_donor_tables(conn)
        members = conn.execute("""
            SELECT DISTINCT member_id, member_name, state_code, chamber, district_number
            FROM main.stg_members
            WHERE member_id IS NOT NULL
              AND member_name IS NOT NULL
              AND state_code IS NOT NULL
              AND chamber IS NOT NULL
            ORDER BY member_id
        """).fetchall()

        stats["members"] = len(members)
        print(f"Found {len(members)} unique members to sync FEC data for...")
        try:
            candidate_ids = _ensure_candidate_mappings(conn, members, api_key)
        except (OpenFECRateLimitError, OpenFECRequestTimeoutError) as error:
            stats["status"] = "paused"
            _log_openfec_failure(error, mode)
            return
        stats["mapped"] = len(candidate_ids)
        stats["unmapped"] = len(members) - len(candidate_ids)
        if mode == "mappings":
            stats["status"] = "completed"
            print("FEC candidate mapping refresh complete!")
            return

        recently_synced_summaries = {
            row[0]
            for row in conn.execute(
                """
                SELECT bioguide_id
                FROM main.member_fec_summary_sync_state
                WHERE cycle = ?
                  AND synced_at >= CURRENT_TIMESTAMP - (? * INTERVAL '1 day')
                """,
                [str(DONOR_CYCLE), DONOR_SYNC_MAX_AGE_DAYS],
            ).fetchall()
        }
        summaries_with_totals = {
            row[0]
            for row in conn.execute("""
                SELECT bioguide_id
                FROM main.member_fec_summary
                WHERE total_receipts IS NOT NULL
            """).fetchall()
        }
        recently_synced_summaries.intersection_update(summaries_with_totals)
        recently_synced_donors = {
            row[0]
            for row in conn.execute(
                """
                SELECT bioguide_id
                FROM main.member_fec_sync_state
                WHERE cycle = ?
                  AND synced_at >= CURRENT_TIMESTAMP - (? * INTERVAL '1 day')
                """,
                [str(DONOR_CYCLE), DONOR_SYNC_MAX_AGE_DAYS],
            ).fetchall()
        }
        if force:
            recently_synced_summaries.clear()
            recently_synced_donors.clear()

        if mode in ("both", "summaries"):
            summary_members = [
                member for member in members
                if member[0] not in recently_synced_summaries
                and member[0] in candidate_ids
            ]
            stats["summary_eligible"] = len(summary_members)
            stats["summary_fresh"] = len(candidate_ids) - len(summary_members)
            stats["summary_unmapped"] = len(members) - len(candidate_ids)
            candidates = [
                (member[0], member[1], candidate_ids[member[0]])
                for member in summary_members
            ]
            for offset in range(0, len(candidates), SUMMARY_BATCH_SIZE):
                batch = candidates[offset:offset + SUMMARY_BATCH_SIZE]
                stats["summary_attempted"] += len(batch)
                try:
                    summaries = _with_rate_limit_context(
                        lambda: fetch_fec_candidate_summaries(
                            [candidate[2] for candidate in batch],
                            api_key,
                            cycle=DONOR_CYCLE,
                        ),
                        f"summary batch {offset // SUMMARY_BATCH_SIZE + 1} "
                        f"({len(batch)} candidates)",
                    )
                except (OpenFECRateLimitError, OpenFECRequestTimeoutError) as error:
                    _log_openfec_failure(error, mode)
                    stats["status"] = "paused"
                    return

                conn.execute("BEGIN TRANSACTION")
                try:
                    synced_bioguide_ids = []
                    for bioguide_id, _name, fec_id in batch:
                        summary = summaries.get(fec_id)
                        if summary is None:
                            stats["summary_absent"] += 1
                            continue
                        _upsert_summary(conn, bioguide_id, fec_id, summary)
                        if summary["total_receipts"] is not None:
                            synced_bioguide_ids.append(bioguide_id)
                            stats["summary_saved"] += 1
                            source = summary.get("summary_source") or "unspecified"
                            stats["summary_sources"][source] = (
                                stats["summary_sources"].get(source, 0) + 1
                            )
                        else:
                            stats["summary_missing_total"] += 1
                    _mark_synced(conn, "member_fec_summary_sync_state", synced_bioguide_ids)
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                missing_count = len(batch) - len(synced_bioguide_ids)
                print(
                    f"Synced summaries for {len(synced_bioguide_ids)} members; "
                    f"{missing_count} remain eligible for retry or report cross-check"
                )

        if mode == "summary-cross-check":
            members_needing_fallback = [
                member
                for member in members
                if member[0] in candidate_ids
                and member[0] not in summaries_with_totals
            ]
            stats["fallback_eligible"] = len(members_needing_fallback)
            stats["summary_unmapped"] = len(members) - len(candidate_ids)
            for bioguide_id, name, _state, _chamber, _district in members_needing_fallback:
                fec_id = candidate_ids[bioguide_id]
                stats["fallback_attempted"] += 1
                try:
                    fallback_summary = _with_rate_limit_context(
                        lambda: fetch_candidate_committee_report_summary(
                            fec_id, api_key, cycle=DONOR_CYCLE
                        ),
                        f"committee-report summary for {name} ({bioguide_id}, {fec_id})",
                    )
                except (OpenFECRateLimitError, OpenFECRequestTimeoutError) as error:
                    _log_openfec_failure(error, mode)
                    stats["status"] = "paused"
                    return

                if fallback_summary is None:
                    stats["fallback_no_report"] += 1
                    print(f"No usable committee reports found for {name} ({fec_id})")
                    continue

                conn.execute("BEGIN TRANSACTION")
                try:
                    _upsert_summary(
                        conn, bioguide_id, fec_id, fallback_summary
                    )
                    _mark_synced(
                        conn, "member_fec_summary_sync_state", [bioguide_id]
                    )
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                stats["fallback_saved"] += 1
                print(f"Cross-checked summary from committee reports for {name}")
            stats["status"] = "completed"
            print("Committee-report summary cross-check complete!")
            return

        if mode in ("both", "donors"):
            donor_members = [
                member for member in members
                if member[0] not in recently_synced_donors
                and member[0] in candidate_ids
            ]
            stats["donor_eligible"] = len(donor_members)
            stats["donor_fresh"] = len(candidate_ids) - len(donor_members)
            stats["donor_unmapped"] = len(members) - len(candidate_ids)
            candidates = [
                (member[0], member[1], candidate_ids[member[0]])
                for member in donor_members
            ]
            for bioguide_id, name, fec_id in candidates:
                try:
                    donors = _with_rate_limit_context(
                        lambda: fetch_top_pac_contributors(
                            fec_id, api_key, cycle=DONOR_CYCLE
                        ),
                        f"donor lookup for {name} ({bioguide_id}, {fec_id})",
                    )
                except (OpenFECRateLimitError, OpenFECRequestTimeoutError) as error:
                    _log_openfec_failure(error, mode)
                    stats["status"] = "paused"
                    return
                conn.execute("BEGIN TRANSACTION")
                try:
                    _save_candidate_id(conn, bioguide_id, fec_id)
                    existing_rows = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM main.member_top_donors
                        WHERE bioguide_id = ? AND cycle = ?
                        """,
                        [bioguide_id, str(DONOR_CYCLE)],
                    ).fetchone()[0]
                    conn.execute(
                        "DELETE FROM main.member_top_donors WHERE bioguide_id = ? AND cycle = ?",
                        [bioguide_id, str(DONOR_CYCLE)],
                    )
                    for donor in donors:
                        conn.execute(
                            """
                            INSERT INTO main.member_top_donors
                                (bioguide_id, donor_name, amount, industry, cycle)
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            [
                                bioguide_id,
                                donor["donor_name"],
                                donor["amount"],
                                donor["industry"],
                                str(DONOR_CYCLE),
                            ],
                        )
                    _mark_synced(conn, "member_fec_sync_state", [bioguide_id])
                    conn.execute("COMMIT")
                except Exception:
                    conn.execute("ROLLBACK")
                    raise
                stats["donor_refreshed"] += 1
                stats["donor_rows"] += len(donors)
                stats["donor_rows_replaced"] += existing_rows
                if not donors:
                    stats["donor_empty"] += 1
                print(f"Synced top donors for {name} ({fec_id})")
        stats["status"] = "completed"
    finally:
        conn.close()
        _emit_ingestion_run_summary(mode, stats)

    print("FEC Campaign Finance Ingestion Complete!")


def _log_openfec_failure(
    error: OpenFECRateLimitError | OpenFECRequestTimeoutError, mode: str
):
    if isinstance(error, OpenFECRateLimitError):
        failure = f"HTTP 429 at {error.endpoint}"
        retry_note = (
            f" Retry-After: {error.retry_after}."
            if error.retry_after
            else " No Retry-After header was provided."
        )
    else:
        failure = f"request timed out at {error.endpoint}"
        retry_note = " The request can be retried on the next run."
    message = (
        f"Ingestion paused in {mode} mode: {failure}; "
        f"{error.context}.{retry_note} Successful prior records were saved; "
        "the failed item was not marked synced. Rerun later to resume. "
        f"Details logged to {LOG_PATH}."
    )
    _get_ingestion_logger().warning(message)
    print(message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest OpenFEC donor and summary data.")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--summaries",
        action="store_const",
        const="summaries",
        dest="mode",
        help="Ingest only bulk candidate summaries.",
    )
    mode_group.add_argument(
        "--donors",
        action="store_const",
        const="donors",
        dest="mode",
        help="Ingest only per-candidate top donor details.",
    )
    mode_group.add_argument(
        "--mappings",
        action="store_const",
        const="mappings",
        dest="mode",
        help="Create or refresh validated bioguide-to-FEC candidate mappings only.",
    )
    mode_group.add_argument(
        "--summary-cross-check",
        action="store_const",
        const="summary-cross-check",
        dest="mode",
        help="Fill missing summary totals from candidate committee filings.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Refresh financial data even when it was synced within the freshness window.",
    )
    parser.set_defaults(mode="both")
    args = parser.parse_args()
    run_fec_ingestion(mode=args.mode, force=args.force)
