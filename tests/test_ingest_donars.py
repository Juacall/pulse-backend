import json

import duckdb
import pytest

from scripts import ingest_donars


@pytest.fixture(autouse=True)
def disable_request_pacing(monkeypatch):
    monkeypatch.setattr(ingest_donars, "FEC_REQUEST_INTERVAL_SECONDS", 0)
    monkeypatch.setattr(ingest_donars, "_last_fec_request_at", None)


def _matched_mapping(
    bioguide_id, name, state, chamber, district, fec_id="C00123456"
):
    return {
        "bioguide_id": bioguide_id,
        "member_name": name,
        "state_code": state,
        "chamber": chamber,
        "district_number": district,
        "fec_candidate_id": fec_id,
        "fec_candidate_name": name,
        "fec_office": "H" if chamber == "House" else "S",
        "fec_state": state,
        "fec_district": f"{district:02d}" if district is not None else None,
        "mapping_status": "MATCHED",
        "mapping_reason": "test match",
        "candidate_options": "[]",
    }


def test_fetch_fec_candidate_summaries_maps_bulk_openfec_totals(monkeypatch):
    class Response:
        is_error = False

        def json(self):
            return {
                "results": [
                    {
                        "candidate_id": "C00123456",
                        "receipts": 2500.0,
                        "other_political_committee_contributions": 900.0,
                        "individual_itemized_contributions": 1600.0,
                        "coverage_end_date": "2026-09-30",
                    }
                ]
            }

    def fake_get(url, *, params, timeout):
        assert url.endswith("/candidates/totals/")
        assert ("cycle", 2026) in params
        assert ("candidate_id", "C00123456") in params
        assert ("candidate_id", "C00123457") in params
        return Response()

    monkeypatch.setattr(ingest_donars.httpx, "get", fake_get)

    summaries = ingest_donars.fetch_fec_candidate_summaries(
        ["C00123456", "C00123457"], "test-key", cycle=2026
    )

    assert summaries == {
        "C00123456": {
            "total_receipts": 2500.0,
            "pac_contributions": 900.0,
            "individual_contributions": 1600.0,
            "coverage_end_date": "2026-09-30",
            "summary_source": "candidate_totals",
        }
    }


def test_fetch_fec_candidate_mapping_requires_exact_member_match(monkeypatch):
    class Response:
        is_error = False

        def json(self):
            return {
                "results": [
                    {
                        "candidate_id": "H4AZ04032",
                        "name": "GOSAR, PAUL A.",
                        "office": "H",
                        "state": "AZ",
                        "district": "04",
                    },
                    {
                        "candidate_id": "H4AZ09000",
                        "name": "OTHER, PAUL A.",
                        "office": "H",
                        "state": "AZ",
                        "district": "04",
                    },
                ]
            }

    def fake_get(url, *, params, timeout):
        assert url.endswith("/candidates/search/")
        assert ("office", "H") in params
        assert ("state", "AZ") in params
        assert ("district", "04") not in params
        assert ("cycle", 2026) in params
        return Response()

    monkeypatch.setattr(ingest_donars.httpx, "get", fake_get)

    mapping = ingest_donars.fetch_fec_candidate_mapping(
        "G000001",
        "Gosar, Paul A.",
        "AZ",
        "House",
        4,
        "test-key",
    )

    assert mapping["mapping_status"] == "MATCHED"
    assert mapping["fec_candidate_id"] == "H4AZ04032"
    assert mapping["fec_candidate_name"] == "GOSAR, PAUL A."
    assert len(json.loads(mapping["candidate_options"])) == 2


def test_fetch_fec_candidate_mapping_matches_current_name_when_fec_district_differs(
    monkeypatch,
):
    class Response:
        is_error = False

        def json(self):
            return {
                "results": [
                    {
                        "candidate_id": "H4AZ04032",
                        "name": "GOSAR, PAUL A.",
                        "office": "H",
                        "state": "AZ",
                        "district": "04",
                    }
                ]
            }

    monkeypatch.setattr(
        ingest_donars.httpx, "get", lambda *args, **kwargs: Response()
    )

    mapping = ingest_donars.fetch_fec_candidate_mapping(
        "G000001", "Gosar, Paul A.", "AZ", "House", 9, "test-key"
    )

    assert mapping["mapping_status"] == "MATCHED"
    assert mapping["fec_candidate_id"] == "H4AZ04032"
    assert "redistricting" in mapping["mapping_reason"]


def test_fetch_fec_candidate_mapping_retries_with_name_variant(monkeypatch):
    class Response:
        is_error = False

        def __init__(self, results):
            self.results = results

        def json(self):
            return {"results": self.results}

    requested_queries = []

    def fake_get(url, *, params, timeout):
        query = dict(params)["q"]
        requested_queries.append(query)
        if query == "Member, Alex B.":
            return Response([])
        return Response(
            [
                {
                    "candidate_id": "H4WA01001",
                    "name": "MEMBER, ALEX",
                    "office": "H",
                    "state": "WA",
                    "district": "01",
                }
            ]
        )

    monkeypatch.setattr(ingest_donars.httpx, "get", fake_get)

    mapping = ingest_donars.fetch_fec_candidate_mapping(
        "M000001", "Member, Alex B.", "WA", "House", 1, "test-key"
    )

    assert requested_queries == ["Member, Alex B.", "Alex Member"]
    assert mapping["mapping_status"] == "MATCHED"
    assert mapping["fec_candidate_id"] == "H4WA01001"


def test_fetch_fec_candidate_mapping_flags_ambiguous_candidates(monkeypatch):
    class Response:
        is_error = False

        def json(self):
            return {
                "results": [
                    {
                        "candidate_id": candidate_id,
                        "name": "MEMBER, ALEX",
                        "office": "S",
                        "state": "WA",
                    }
                    for candidate_id in ("S4WA00001", "S6WA00002")
                ]
            }

    monkeypatch.setattr(
        ingest_donars.httpx, "get", lambda *args, **kwargs: Response()
    )

    mapping = ingest_donars.fetch_fec_candidate_mapping(
        "S000001", "Member, Alex", "WA", "Senate", None, "test-key"
    )

    assert mapping["mapping_status"] == "AMBIGUOUS"
    assert mapping["fec_candidate_id"] is None


def test_load_legislator_fec_crosswalk_splits_candidate_ids(tmp_path):
    crosswalk_path = tmp_path / "legislators-current.csv"
    crosswalk_path.write_text(
        'bioguide_id,fec_ids\nG000001,"H6WA01001,S4WA00011"\n',
        encoding="utf-8",
    )

    assert ingest_donars.load_legislator_fec_crosswalk(crosswalk_path) == {
        "G000001": ["H6WA01001", "S4WA00011"]
    }


def test_fetch_fec_candidates_by_ids_uses_cycle_filtered_candidates_endpoint(
    monkeypatch,
):
    class Response:
        is_error = False

        def json(self):
            return {"results": [{"candidate_id": "H6WA01001"}]}

    def fake_get(url, *, params, timeout):
        assert url.endswith("/candidates/")
        assert ("candidate_id", "H6WA01001") in params
        assert ("candidate_id", "H4WA01001") in params
        assert ("cycle", 2026) in params
        return Response()

    monkeypatch.setattr(ingest_donars.httpx, "get", fake_get)

    candidates = ingest_donars.fetch_fec_candidates_by_ids(
        ["H6WA01001", "H4WA01001"], "test-key", cycle=2026
    )

    assert candidates == [{"candidate_id": "H6WA01001"}]


def test_ensure_candidate_mappings_uses_crosswalk_before_name_search(
    tmp_path, monkeypatch
):
    conn = duckdb.connect(":memory:")
    ingest_donars.init_donor_tables(conn)
    crosswalk_path = tmp_path / "legislators-current.csv"
    crosswalk_path.write_text(
        'bioguide_id,fec_ids\nG000001,"H6WA01001,H4WA01001"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(ingest_donars, "CROSSWALK_PATH", crosswalk_path)

    def fetch_candidates(candidate_ids, api_key, cycle=2026):
        assert candidate_ids == ["H6WA01001", "H4WA01001"]
        assert cycle == 2026
        return [
            {
                "candidate_id": "H6WA01001",
                "name": "MEMBER, ALEX",
                "office": "H",
                "state": "WA",
                "district": "01",
            },
            {
                "candidate_id": "H4WA01001",
                "name": "MEMBER, ALEX",
                "office": "H",
                "state": "WA",
                "district": "03",
            },
        ]

    monkeypatch.setattr(
        ingest_donars, "fetch_fec_candidates_by_ids", fetch_candidates
    )

    def unexpected_name_search(*args, **kwargs):
        raise AssertionError("Crosswalk IDs should be used before name search")

    monkeypatch.setattr(
        ingest_donars, "fetch_fec_candidate_mapping", unexpected_name_search
    )

    mappings = ingest_donars._ensure_candidate_mappings(
        conn, [("G000001", "Member, Alex", "WA", "House", 1)], "test-key"
    )

    assert mappings == {"G000001": "H6WA01001"}
    saved_mapping = conn.execute("""
        SELECT fec_candidate_id, mapping_status, mapping_reason
        FROM main.member_fec_candidate_mapping
        WHERE bioguide_id = 'G000001'
    """).fetchone()
    assert saved_mapping == (
        "H6WA01001",
        "MATCHED",
        "Crosswalk candidate ID validated for the current cycle.",
    )
    conn.close()


def test_fec_requests_wait_to_respect_configured_interval(monkeypatch):
    clock_values = iter([10.0, 11.5, 14.0])
    sleep_calls = []
    monkeypatch.setattr(ingest_donars, "FEC_REQUEST_INTERVAL_SECONDS", 4.0)
    monkeypatch.setattr(
        ingest_donars.time, "monotonic", lambda: next(clock_values)
    )
    monkeypatch.setattr(ingest_donars.time, "sleep", sleep_calls.append)

    ingest_donars._pace_fec_request()
    ingest_donars._pace_fec_request()

    assert sleep_calls == [2.5]
    assert ingest_donars._last_fec_request_at == 14.0


def test_fec_http_errors_do_not_include_api_key(monkeypatch):
    class Response:
        is_error = True
        status_code = 404

    monkeypatch.setattr(
        ingest_donars.httpx, "get", lambda *args, **kwargs: Response()
    )

    try:
        ingest_donars.fetch_fec_candidate_summaries(
            ["C00123456"], "secret-test-key", cycle=2026
        )
    except RuntimeError as error:
        assert "404" in str(error)
        assert "secret-test-key" not in str(error)
    else:
        raise AssertionError("Expected OpenFEC HTTP errors to be raised")


def test_fec_429_raises_rate_limit_error_with_retry_after(monkeypatch):
    class Response:
        is_error = True
        status_code = 429
        headers = {"Retry-After": "60"}

    monkeypatch.setattr(
        ingest_donars.httpx, "get", lambda *args, **kwargs: Response()
    )

    try:
        ingest_donars.fetch_fec_candidate_summaries(
            ["C00123456"], "secret-test-key", cycle=2026
        )
    except ingest_donars.OpenFECRateLimitError as error:
        assert error.retry_after == "60"
        assert "secret-test-key" not in str(error)
    else:
        raise AssertionError("Expected OpenFEC 429 to raise rate-limit error")


def test_fec_read_timeout_is_converted_without_exposing_url(monkeypatch):
    def timeout_get(*args, **kwargs):
        raise ingest_donars.httpx.ReadTimeout("timed out")

    monkeypatch.setattr(ingest_donars.httpx, "get", timeout_get)

    try:
        ingest_donars.fetch_fec_candidate_summaries(
            ["C00123456"], "secret-test-key", cycle=2026
        )
    except ingest_donars.OpenFECRequestTimeoutError as error:
        assert "candidates/totals/" in str(error)
        assert "secret-test-key" not in str(error)
    else:
        raise AssertionError("Expected read timeout to become a retryable error")


def test_fetch_top_pac_contributors_groups_missing_and_duplicate_employers(monkeypatch):
    class Response:
        is_error = False

        def __init__(self, results):
            self.results = results

        def json(self):
            return {"results": self.results}

    responses = iter(
        [
            Response([{"committee_id": "C00000001"}, {"committee_id": "C00000002"}]),
            Response(
                [
                    {"employer": None, "total": 100.0},
                    {"employer": "Not Employed", "total": 863139893.52},
                    {"employer": "Null", "total": 428057577.5},
                    {"employer": "Retired", "total": 368400499.12},
                    {"employer": "Self Employed", "total": 100000.0},
                    {"employer": "Self", "total": 90000.0},
                    {"employer": "None", "total": 80000.0},
                    {"employer": "N/A", "total": 70000.0},
                    {"employer": "Homemaker", "total": 60000.0},
                    {"employer": "Information Requested", "total": 50000.0},
                    {
                        "employer": "Information Requested Per Best Efforts",
                        "total": 40000.0,
                    },
                    {"employer": "Entrepreneur", "total": 30000.0},
                    {"employer": "self-employed", "total": 20000.0},
                    {"employer": "Example Corp", "total": 200.0},
                    {"employer": "EXAMPLE CORP", "total": 300.0},
                    {"employer": "Another Company", "total": 400.0},
                ]
            ),
        ]
    )

    def fake_get(url, *, params, timeout):
        if url.endswith("/candidate/C00123456/committees/"):
            assert params["cycle"] == 2026
            return next(responses)
        assert url.endswith("/schedules/schedule_a/by_employer/")
        assert ("candidate_id", "C00123456") not in params
        assert ("committee_id", "C00000001") in params
        assert ("committee_id", "C00000002") in params
        return next(responses)

    monkeypatch.setattr(ingest_donars.httpx, "get", fake_get)

    donors = ingest_donars.fetch_top_pac_contributors(
        "C00123456", "test-key", cycle=2026
    )

    assert donors == [
        {
            "donor_name": "Example Corp",
            "amount": 500.0,
            "industry": "Employer aggregate",
        },
        {
            "donor_name": "Another Company",
            "amount": 400.0,
            "industry": "Employer aggregate",
        },
    ]


def test_fetch_top_pac_contributors_returns_empty_without_committees(monkeypatch):
    class Response:
        is_error = False

        def json(self):
            return {"results": []}

    called_urls = []

    def fake_get(url, *, params, timeout):
        called_urls.append(url)
        return Response()

    monkeypatch.setattr(ingest_donars.httpx, "get", fake_get)

    donors = ingest_donars.fetch_top_pac_contributors(
        "C00123456", "test-key", cycle=2026
    )

    assert donors == []
    assert called_urls == [
        f"{ingest_donars.FEC_BASE_URL}/candidate/C00123456/committees/"
    ]


def test_fetch_candidate_committee_report_summary_uses_latest_non_jfc_filings(
    monkeypatch,
):
    class Response:
        is_error = False

        def __init__(self, results):
            self.results = results

        def json(self):
            return {"results": self.results}

    responses = iter(
        [
            Response(
                [
                    {
                        "committee_id": "C00000001",
                        "designation": "P",
                    },
                    {
                        "committee_id": "C00000002",
                        "designation": "J",
                    },
                ]
            ),
            Response(
                [
                    {
                        "most_recent": True,
                        "coverage_end_date": "2026-06-30",
                        "receipt_date": "2026-07-15",
                        "total_receipts_ytd": "1000",
                        "other_political_committee_contributions_ytd": "200",
                        "individual_itemized_contributions_ytd": "700",
                    },
                    {
                        "most_recent": False,
                        "coverage_end_date": "2026-06-30",
                        "receipt_date": "2026-07-10",
                        "total_receipts_ytd": "900",
                        "other_political_committee_contributions_ytd": "100",
                        "individual_itemized_contributions_ytd": "600",
                    },
                ]
            ),
        ]
    )

    def fake_get(url, *, params, timeout):
        if url.endswith("/candidate/C00123456/committees/"):
            return next(responses)
        assert url.endswith("/committee/C00000001/reports/")
        assert ("cycle", 2026) in params
        return next(responses)

    monkeypatch.setattr(ingest_donars.httpx, "get", fake_get)

    summary = ingest_donars.fetch_candidate_committee_report_summary(
        "C00123456", "test-key", cycle=2026
    )

    assert summary == {
        "total_receipts": 1000.0,
        "pac_contributions": 200.0,
        "individual_contributions": 700.0,
        "coverage_end_date": "2026-06-30",
        "summary_source": "committee_reports",
    }


def test_run_fec_ingestion_populates_member_top_donors(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "donors.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            chamber VARCHAR,
            district_number INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO stg_members VALUES ('H1', 'House Member', 'WA', 'House', 1)"
    )
    conn.close()

    monkeypatch.setattr(ingest_donars, "DB_PATH", db_path)
    monkeypatch.setenv("FEC_API_KEY", "test-key")
    monkeypatch.setattr(
        ingest_donars,
        "fetch_fec_candidate_mapping",
        lambda bioguide_id, name, state, chamber, district, api_key: _matched_mapping(
            bioguide_id, name, state, chamber, district
        ),
    )
    summary_calls = []

    def fetch_summary(candidate_id, api_key, cycle):
        summary_calls.append(candidate_id)
        return {
            "total_receipts": 2500.0,
            "pac_contributions": 900.0,
            "individual_contributions": 1600.0,
            "coverage_end_date": "2026-09-30",
            "summary_source": "candidate_totals",
        }

    monkeypatch.setattr(
        ingest_donars,
        "fetch_fec_candidate_summaries",
        lambda candidate_ids, api_key, cycle: {
            candidate_id: fetch_summary(candidate_id, api_key, cycle)
            for candidate_id in candidate_ids
        },
    )
    donor_calls = []

    def fetch_donors(candidate_id, api_key, cycle):
        donor_calls.append(candidate_id)
        return [
            {
                "donor_name": "Example Employer",
                "amount": 1250.0,
                "industry": "Employer aggregate",
            }
        ]

    monkeypatch.setattr(
        ingest_donars,
        "fetch_top_pac_contributors",
        fetch_donors,
    )

    ingest_donars.run_fec_ingestion()
    ingest_donars.run_fec_ingestion()
    output = capsys.readouterr().out

    conn = duckdb.connect(str(db_path), read_only=True)
    donors = conn.execute("""
        SELECT bioguide_id, donor_name, amount, industry, cycle
        FROM member_top_donors
    """).fetchall()
    summaries = conn.execute("""
        SELECT bioguide_id, fec_candidate_id, total_receipts,
               pac_contributions, individual_contributions, coverage_end_date
        FROM member_fec_summary
    """).fetchall()
    sync_states = conn.execute("""
        SELECT bioguide_id, cycle
        FROM member_fec_sync_state
    """).fetchall()
    conn.close()

    assert donors == [
        (
            "H1",
            "Example Employer",
            1250.0,
            "Employer aggregate",
            "2026",
        )
    ]
    assert summaries == [
        (
            "H1",
            "C00123456",
            2500.0,
            900.0,
            1600.0,
            "2026-09-30",
        )
    ]
    assert summary_calls == ["C00123456"]
    assert donor_calls == ["C00123456"]
    assert sync_states == [("H1", "2026")]
    assert "Summary results: eligible=1, skipped_fresh=0, attempted=1" in output
    assert "saved_with_receipts=1" in output
    assert "Summary sources saved: candidate_totals=1" in output
    assert "Donor results: eligible=1, skipped_fresh=0, refreshed_members=1" in output
    assert "employer_rows_saved=1" in output
    assert "Summary results: eligible=0, skipped_fresh=1, attempted=0" in output
    assert "Donor results: eligible=0, skipped_fresh=1, refreshed_members=0" in output


def test_run_fec_ingestion_reuses_saved_candidate_id(tmp_path, monkeypatch):
    db_path = tmp_path / "cached-candidate.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            chamber VARCHAR,
            district_number INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO stg_members VALUES ('H1', 'House Member', 'WA', 'House', 1)"
    )
    conn.execute("""
        CREATE TABLE member_fec_summary (
            bioguide_id VARCHAR PRIMARY KEY,
            fec_candidate_id VARCHAR,
            total_receipts DOUBLE,
            pac_contributions DOUBLE,
            individual_contributions DOUBLE,
            coverage_end_date VARCHAR
        )
    """)
    conn.execute("""
        INSERT INTO member_fec_summary (bioguide_id, fec_candidate_id)
        VALUES ('H1', 'C00987654')
    """)
    ingest_donars.init_donor_tables(conn)
    conn.execute("""
        INSERT INTO member_fec_candidate_mapping (
            bioguide_id, member_name, state_code, chamber, district_number,
            fec_candidate_id, fec_candidate_name, fec_office, fec_state,
            fec_district, mapping_status, mapping_reason, candidate_options,
            checked_at
        )
        VALUES (
            'H1', 'House Member', 'WA', 'House', 1, 'C00987654',
            'House Member', 'H', 'WA', '01', 'MATCHED', 'test match',
            '[]', TIMESTAMP '2020-01-01 00:00:00'
        )
    """)
    conn.close()

    monkeypatch.setattr(ingest_donars, "DB_PATH", db_path)
    monkeypatch.setenv("FEC_API_KEY", "test-key")
    monkeypatch.setattr(ingest_donars, "FEC_MAPPING_MAX_AGE_DAYS", 0)

    def unexpected_mapping_lookup(*args):
        raise AssertionError("Should reuse the saved FEC candidate mapping")

    monkeypatch.setattr(
        ingest_donars, "fetch_fec_candidate_mapping", unexpected_mapping_lookup
    )
    def unexpected_summary_fetch(*args, **kwargs):
        raise AssertionError("Donor-only mode should not fetch candidate summaries")

    monkeypatch.setattr(
        ingest_donars,
        "fetch_fec_candidate_summaries",
        unexpected_summary_fetch,
    )
    monkeypatch.setattr(
        ingest_donars,
        "fetch_top_pac_contributors",
        lambda candidate_id, api_key, cycle: [],
    )

    ingest_donars.run_fec_ingestion(mode="donors")


def test_run_fec_ingestion_summaries_mode_skips_donor_requests(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "summaries-only.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            chamber VARCHAR,
            district_number INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO stg_members VALUES ('H1', 'House Member', 'WA', 'House', 1)"
    )
    conn.close()

    monkeypatch.setattr(ingest_donars, "DB_PATH", db_path)
    monkeypatch.setenv("FEC_API_KEY", "test-key")
    monkeypatch.setattr(
        ingest_donars,
        "fetch_fec_candidate_mapping",
        lambda bioguide_id, name, state, chamber, district, api_key: _matched_mapping(
            bioguide_id, name, state, chamber, district
        ),
    )
    monkeypatch.setattr(
        ingest_donars,
        "fetch_fec_candidate_summaries",
        lambda candidate_ids, api_key, cycle: {
            "C00123456": {
                "total_receipts": 2500.0,
                "pac_contributions": 900.0,
                "individual_contributions": 1600.0,
                "coverage_end_date": "2026-09-30",
            }
        },
    )

    def unexpected_donor_fetch(*args, **kwargs):
        raise AssertionError("Summary-only mode should not fetch donor details")

    monkeypatch.setattr(
        ingest_donars, "fetch_top_pac_contributors", unexpected_donor_fetch
    )

    ingest_donars.run_fec_ingestion(mode="summaries")


def test_missing_bulk_summary_remains_eligible_for_retry(tmp_path, monkeypatch):
    db_path = tmp_path / "missing-bulk-summary.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            chamber VARCHAR,
            district_number INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO stg_members VALUES ('H1', 'House Member', 'WA', 'House', 1)"
    )
    ingest_donars.init_donor_tables(conn)
    _insert_test_mapping(conn, "H1", "House Member", "WA", "House", 1)
    conn.close()

    monkeypatch.setattr(ingest_donars, "DB_PATH", db_path)
    monkeypatch.setenv("FEC_API_KEY", "test-key")
    calls = []
    monkeypatch.setattr(
        ingest_donars,
        "fetch_fec_candidate_summaries",
        lambda candidate_ids, api_key, cycle: calls.append(candidate_ids) or {},
    )

    ingest_donars.run_fec_ingestion(mode="summaries")
    ingest_donars.run_fec_ingestion(mode="summaries")

    conn = duckdb.connect(str(db_path), read_only=True)
    sync_count = conn.execute(
        "SELECT COUNT(*) FROM member_fec_summary_sync_state"
    ).fetchone()[0]
    summary_count = conn.execute(
        "SELECT COUNT(*) FROM member_fec_summary WHERE bioguide_id = 'H1'"
    ).fetchone()[0]
    conn.close()

    assert calls == [["C00123456"], ["C00123456"]]
    assert sync_count == 0
    assert summary_count == 0


def test_summary_cross_check_populates_missing_summary_values(tmp_path, monkeypatch):
    db_path = tmp_path / "summary-fallback.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            chamber VARCHAR,
            district_number INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO stg_members VALUES ('H1', 'House Member', 'WA', 'House', 1)"
    )
    ingest_donars.init_donor_tables(conn)
    _insert_test_mapping(conn, "H1", "House Member", "WA", "House", 1)
    conn.close()

    monkeypatch.setattr(ingest_donars, "DB_PATH", db_path)
    monkeypatch.setenv("FEC_API_KEY", "test-key")
    requested = []

    def fetch_report_summary(candidate_id, api_key, cycle):
        requested.append(candidate_id)
        return {
            "total_receipts": 1200.0,
            "pac_contributions": 300.0,
            "individual_contributions": 700.0,
            "coverage_end_date": "2026-06-30",
            "summary_source": "committee_reports",
        }

    monkeypatch.setattr(
        ingest_donars,
        "fetch_candidate_committee_report_summary",
        fetch_report_summary,
    )

    ingest_donars.run_fec_ingestion(mode="summary-cross-check")

    conn = duckdb.connect(str(db_path), read_only=True)
    summary = conn.execute("""
        SELECT total_receipts, pac_contributions, individual_contributions,
               coverage_end_date, summary_source
        FROM member_fec_summary WHERE bioguide_id = 'H1'
    """).fetchone()
    conn.close()

    assert requested == ["C00123456"]
    assert summary == (1200.0, 300.0, 700.0, "2026-06-30", "committee_reports")


def test_summary_cross_check_pauses_cleanly_on_read_timeout(tmp_path, monkeypatch):
    db_path = tmp_path / "summary-timeout.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            chamber VARCHAR,
            district_number INTEGER
        )
    """)
    conn.execute(
        "INSERT INTO stg_members VALUES ('H1', 'House Member', 'WA', 'House', 1)"
    )
    ingest_donars.init_donor_tables(conn)
    _insert_test_mapping(conn, "H1", "House Member", "WA", "House", 1)
    conn.close()

    monkeypatch.setattr(ingest_donars, "DB_PATH", db_path)
    monkeypatch.setenv("FEC_API_KEY", "test-key")
    logged = []

    def timeout_summary(*args, **kwargs):
        raise ingest_donars.OpenFECRequestTimeoutError(
            "committee/C00000001/reports/"
        )

    monkeypatch.setattr(
        ingest_donars,
        "fetch_candidate_committee_report_summary",
        timeout_summary,
    )
    monkeypatch.setattr(
        ingest_donars,
        "_log_openfec_failure",
        lambda error, mode: logged.append((error, mode)),
    )

    ingest_donars.run_fec_ingestion(mode="summary-cross-check")

    conn = duckdb.connect(str(db_path), read_only=True)
    summary_count = conn.execute(
        "SELECT COUNT(*) FROM member_fec_summary WHERE total_receipts IS NOT NULL"
    ).fetchone()[0]
    sync_count = conn.execute(
        "SELECT COUNT(*) FROM member_fec_summary_sync_state"
    ).fetchone()[0]
    conn.close()

    assert summary_count == 0
    assert sync_count == 0
    assert len(logged) == 1
    assert logged[0][1] == "summary-cross-check"
    assert "House Member" in logged[0][0].context


def _insert_test_mapping(conn, bioguide_id, name, state, chamber, district):
    fec_office = "H" if chamber == "House" else "S"
    fec_district = f"{district:02d}" if district is not None else None
    conn.execute(
        """
        INSERT INTO member_fec_candidate_mapping (
            bioguide_id, member_name, state_code, chamber, district_number,
            fec_candidate_id, fec_candidate_name, fec_office, fec_state,
            fec_district, mapping_status, mapping_reason, candidate_options,
            checked_at
        ) VALUES (?, ?, ?, ?, ?, 'C00123456', ?, ?, ?, ?, 'MATCHED',
                  'test match', '[]', CURRENT_TIMESTAMP)
        """,
        [
            bioguide_id,
            name,
            state,
            chamber,
            district,
            name,
            fec_office,
            state,
            fec_district,
        ],
    )


def test_run_fec_ingestion_logs_429_and_resumes_without_marking_failed_member(
    tmp_path, monkeypatch, capsys
):
    db_path = tmp_path / "rate-limit.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            chamber VARCHAR,
            district_number INTEGER
        )
    """)
    conn.executemany(
        "INSERT INTO stg_members VALUES (?, ?, ?, ?, ?)",
        [
            ("H1", "First Member", "WA", "House", 1),
            ("H2", "Second Member", "WA", "House", 2),
        ],
    )
    ingest_donars.init_donor_tables(conn)
    conn.executemany(
        """
        INSERT INTO member_fec_candidate_mapping (
            bioguide_id, member_name, state_code, chamber, district_number,
            fec_candidate_id, mapping_status, mapping_reason, candidate_options,
            checked_at
        ) VALUES (?, ?, 'WA', 'House', ?, ?, 'MATCHED', 'test', '[]', CURRENT_TIMESTAMP)
        """,
        [
            ("H1", "First Member", 1, "H1FEC"),
            ("H2", "Second Member", 2, "H2FEC"),
        ],
    )
    conn.close()

    log_path = tmp_path / "logs" / "fec.log"
    monkeypatch.setattr(ingest_donars, "DB_PATH", db_path)
    monkeypatch.setattr(ingest_donars, "LOG_PATH", log_path)
    monkeypatch.setenv("FEC_API_KEY", "test-key")

    def fetch_donors(fec_id, api_key, cycle):
        if fec_id == "H2FEC":
            raise ingest_donars.OpenFECRateLimitError(
                "schedules/schedule_a/by_employer/", "120"
            )
        return [
            {
                "donor_name": "Example Employer",
                "amount": 100.0,
                "industry": "Corporate / Organization",
            }
        ]

    monkeypatch.setattr(
        ingest_donars, "fetch_top_pac_contributors", fetch_donors
    )

    ingest_donars.run_fec_ingestion(mode="donors")

    output = capsys.readouterr().out
    assert "Ingestion paused" in output
    assert "Retry-After: 120" in output
    assert log_path.exists()
    assert "Second Member (H2, H2FEC)" in log_path.read_text(encoding="utf-8")

    conn = duckdb.connect(str(db_path), read_only=True)
    sync_states = conn.execute("""
        SELECT bioguide_id FROM member_fec_sync_state ORDER BY bioguide_id
    """).fetchall()
    donors = conn.execute("""
        SELECT bioguide_id FROM member_top_donors
    """).fetchall()
    conn.close()
    assert sync_states == [("H1",)]
    assert donors == [("H1",)]
