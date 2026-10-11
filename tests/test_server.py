from pathlib import Path
import sys

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.server import server


def _create_politicians_db(path):
    conn = duckdb.connect(str(path))
    conn.execute(
        """
        CREATE TABLE stg_members (
            member_id VARCHAR,
            member_name VARCHAR,
            state_code VARCHAR,
            state_name VARCHAR,
            chamber VARCHAR,
            party_name VARCHAR
        )
        """
    )
    conn.executemany(
        """
        INSERT INTO stg_members VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            ("H1", "House Member", "WA", "WASHINGTON", "House", "Party A"),
            ("S1", "Senate Member", "WA", "WASHINGTON", "Senate", "Party B"),
        ],
    )
    conn.execute(
        """
        CREATE TABLE mart_legislative_activity (
            target_member_id VARCHAR,
            relationship VARCHAR,
            bill_id VARCHAR
        )
        """
    )
    conn.execute(
        "INSERT INTO mart_legislative_activity VALUES ('H1', 'sponsor', 'HR1')"
    )
    conn.close()


def test_politicians_chamber_filter_normalizes_request_and_stored_value(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "politicians.duckdb"
    _create_politicians_db(db_path)
    monkeypatch.setattr(server, "DB_PATH", db_path)

    for chamber in ("Senate", "senate", "S"):
        politicians = server.get_politicians(chamber=chamber)
        assert [politician["id"] for politician in politicians] == ["S1"]
        assert politicians[0]["sponsoredCount"] == 0


def test_politicians_without_chamber_filter_returns_all_members(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "politicians.duckdb"
    _create_politicians_db(db_path)
    monkeypatch.setattr(server, "DB_PATH", db_path)

    politicians = server.get_politicians()

    assert {politician["id"] for politician in politicians} == {"H1", "S1"}


def test_politicians_all_chambers_filter_includes_senate_members(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "politicians.duckdb"
    _create_politicians_db(db_path)
    monkeypatch.setattr(server, "DB_PATH", db_path)

    politicians = server.get_politicians(state="WA", chamber="ALL")

    assert {politician["id"] for politician in politicians} == {"H1", "S1"}


def test_politician_donors_returns_donors_and_filters_by_cycle(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "donors.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE member_top_donors (
            bioguide_id VARCHAR,
            donor_name VARCHAR,
            amount DOUBLE,
            industry VARCHAR,
            cycle VARCHAR
        )
    """)
    conn.executemany(
        "INSERT INTO member_top_donors VALUES (?, ?, ?, ?, ?)",
        [
            ("H1", "Small Donor", 100.0, "Industry A", "2024"),
            ("H1", "Large Donor", 500.0, "Industry B", "2024"),
            ("H1", "New Cycle Donor", 300.0, "Industry C", "2026"),
            ("S1", "Other Member Donor", 900.0, "Industry D", "2024"),
        ],
    )
    conn.close()
    monkeypatch.setattr(server, "DB_PATH", db_path)

    donors = server.get_politician_donors("H1", cycle=2024)

    assert donors == [
        {
            "donorName": "Large Donor",
            "amount": 500.0,
            "industry": "Industry B",
            "cycle": "2024",
        },
        {
            "donorName": "Small Donor",
            "amount": 100.0,
            "industry": "Industry A",
            "cycle": "2024",
        },
    ]


def test_politician_donors_reports_when_ingestion_has_not_run(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_path)).close()
    monkeypatch.setattr(server, "DB_PATH", db_path)

    try:
        server.get_politician_donors("H1")
    except server.HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("Expected donor endpoint to report unavailable data")


def test_politician_donation_summary_returns_summary_fields(tmp_path, monkeypatch):
    db_path = tmp_path / "summary.duckdb"
    conn = duckdb.connect(str(db_path))
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
    conn.execute(
        """
        INSERT INTO member_fec_summary VALUES (
            'H1', 'H0WA01033', 125000.0, 45000.0, 80000.0, '2026-09-30'
        )
        """
    )
    conn.close()
    monkeypatch.setattr(server, "DB_PATH", db_path)

    summary = server.get_politician_donation_summary("H1")

    assert summary == {
        "fecCandidateId": "H0WA01033",
        "totalReceipts": 125000.0,
        "pacContributions": 45000.0,
        "individualContributions": 80000.0,
        "coverageEndDate": "2026-09-30",
        "summaryAvailable": True,
        "summaryStatus": "available",
    }


def test_politician_donation_summary_marks_null_totals_unavailable(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "summary.duckdb"
    conn = duckdb.connect(str(db_path))
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
    conn.execute(
        "INSERT INTO member_fec_summary (bioguide_id, fec_candidate_id) VALUES ('H1', 'H0WA01033')"
    )
    conn.close()
    monkeypatch.setattr(server, "DB_PATH", db_path)

    summary = server.get_politician_donation_summary("H1")

    assert summary["summaryAvailable"] is False
    assert summary["summaryStatus"] == "unavailable"
    assert summary["totalReceipts"] is None


def test_politician_donation_summary_returns_404_when_member_has_no_summary(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "summary.duckdb"
    conn = duckdb.connect(str(db_path))
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
    conn.close()
    monkeypatch.setattr(server, "DB_PATH", db_path)

    try:
        server.get_politician_donation_summary("H1")
    except server.HTTPException as error:
        assert error.status_code == 404
    else:
        raise AssertionError("Expected missing politician summary to return 404")


def test_politician_donation_summary_reports_when_ingestion_has_not_run(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "empty.duckdb"
    duckdb.connect(str(db_path)).close()
    monkeypatch.setattr(server, "DB_PATH", db_path)

    try:
        server.get_politician_donation_summary("H1")
    except server.HTTPException as error:
        assert error.status_code == 503
    else:
        raise AssertionError("Expected summary endpoint to report unavailable data")
