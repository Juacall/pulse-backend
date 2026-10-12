"""
tests/test_ingestion_validation.py

Unit and integration tests for data ingestion, schema validation, checksumming,
idempotency, member sampling, and chamber filtering.
"""

from pathlib import Path
import sys
from threading import Barrier
from unittest.mock import patch

import pytest
import duckdb

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from database import (
    check_tables_exist,
    create_tables,
    get_district_geoids,
    get_existing_bills,
    get_existing_bill_timestamps,
    get_existing_members,
    mark_bill_amendments_processed,
)
from ingestion import load_amendments, load_bills, load_members
from main import parse_args
from mock_data import (
    MOCK_AMENDMENTS,
    MOCK_COSPONSORED_BILLS,
    MOCK_MEMBERS,
    MOCK_SENATE_AMENDMENTS,
    MOCK_SENATE_COSPONSORED_BILLS,
    MOCK_SENATE_MEMBERS,
    MOCK_SENATE_SPONSORED_BILLS,
    MOCK_SPONSORED_BILLS,
    create_in_memory_db_with_seeds,
)
from pipeline import (
    ingest_bill_amendments,
    ingest_member_bills,
    ingest_members,
    run_ingestion,
    setup_database,
)


def test_table_existence_validation():
    """Verify check_tables_exist accurately detects present and missing tables."""
    db = create_in_memory_db_with_seeds()

    # Raw tables and census crosswalk seed must exist in the mock database
    assert check_tables_exist(db, ["raw_members", "raw_bills", "raw_amendments"]) is True
    assert check_tables_exist(db, ["raw_census__cd11920_county20"]) is True

    # Non-existent table should return False
    assert check_tables_exist(db, ["non_existent_table"]) is False
    db.close()


def test_create_tables_adds_introduced_date_to_existing_raw_bills():
    db = duckdb.connect(":memory:")
    db.execute("""
        CREATE TABLE main.raw_bills (
            congress INTEGER,
            bill_type VARCHAR,
            bill_number VARCHAR,
            title VARCHAR,
            latest_action_date VARCHAR,
            latest_action_text VARCHAR,
            member_id VARCHAR,
            relationship VARCHAR,
            update_date TIMESTAMP,
            row_hash VARCHAR,
            ingested_at TIMESTAMP,
            PRIMARY KEY (congress, bill_type, bill_number, member_id, relationship)
        )
    """)

    create_tables(db, replace=False)

    columns = {
        row[0] for row in db.execute("DESCRIBE main.raw_bills").fetchall()
    }
    assert "introduced_date" in columns
    db.close()


def test_get_district_geoids_maps_crosswalk_rows():
    db = create_in_memory_db_with_seeds()

    geoids = get_district_geoids(db)

    assert geoids[("GA", 14)] == "1314"
    assert geoids[("TX", 21)] == "4821"
    db.close()


def test_full_refresh_replaces_only_selected_raw_tables():
    """A targeted full refresh preserves other raw tables and run history."""
    db = create_in_memory_db_with_seeds()
    db.execute("INSERT INTO main.raw_members (bioguide_id) VALUES ('M000001')")
    db.execute("""
        INSERT INTO main.raw_bills
        (congress, bill_type, bill_number, member_id, relationship)
        VALUES (119, 'HR', '1', 'M000001', 'sponsor')
    """)
    db.execute("""
        INSERT INTO main.raw_amendments
        (congress, bill_type, bill_number, amendment_number, amendment_type)
        VALUES (119, 'HR', '1', '1', 'HAM')
    """)
    mark_bill_amendments_processed(db, 119, "HR", "1", "2025-01-01")
    db.execute("""
        INSERT INTO main.ingestion_runs (run_id, status)
        VALUES ('existing-run', 'COMPLETED')
    """)

    setup_database(db, reset=True, reset_tables=["raw_bills"])

    assert db.execute("SELECT COUNT(*) FROM main.raw_members").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM main.raw_bills").fetchone()[0] == 0
    assert db.execute("SELECT COUNT(*) FROM main.raw_amendments").fetchone()[0] == 1
    assert get_existing_bill_timestamps(db) == {(119, "HR", "1"): "2025-01-01"}
    assert db.execute(
        "SELECT COUNT(*) FROM main.ingestion_runs WHERE run_id = 'existing-run'"
    ).fetchone()[0] == 1
    db.close()


def test_full_amendment_refresh_clears_processed_bill_state():
    db = create_in_memory_db_with_seeds()
    mark_bill_amendments_processed(db, 119, "HR", "1", "2025-01-01")

    setup_database(db, reset=True, reset_tables=["raw_amendments"])

    assert get_existing_bill_timestamps(db) == {}
    db.close()


def test_member_ingestion_and_data_validation():
    """Verify member records are inserted and column values match expectations."""
    db = create_in_memory_db_with_seeds()

    load_members(db, MOCK_MEMBERS)
    rows = db.execute("SELECT bioguide_id, name, state, chamber, district, party, geoid_cd, row_hash FROM main.raw_members ORDER BY bioguide_id").fetchall()

    assert len(rows) == 2

    # Validate Marjorie Taylor Greene
    mtg = rows[0]
    assert mtg[0] == "G000596"
    assert mtg[1] == "Greene, Marjorie Taylor"
    assert mtg[2] == "GA"
    assert mtg[3] == "House"
    assert mtg[4] == 14
    assert mtg[5] == "Republican"
    assert mtg[6] == "1314"
    assert mtg[7] is not None  # row_hash computed

    # Validate Chip Roy
    roy = rows[1]
    assert roy[0] == "R000614"
    assert roy[1] == "Roy, Chip"
    assert roy[2] == "TX"
    assert roy[3] == "House"
    assert roy[4] == 21
    assert roy[5] == "Republican"
    assert roy[6] == "4821"
    assert roy[7] is not None
    db.close()


def test_senate_member_ingestion():
    """Verify Senate member records are correctly inserted with chamber='Senate'."""
    db = create_in_memory_db_with_seeds()

    load_members(db, MOCK_SENATE_MEMBERS)
    rows = db.execute("SELECT bioguide_id, name, state, chamber, district, party FROM main.raw_members ORDER BY bioguide_id").fetchall()

    assert len(rows) == 2
    cruz = rows[0]
    assert cruz[0] == "C001098"
    assert cruz[1] == "Cruz, Ted"
    assert cruz[2] == "TX"
    assert cruz[3] == "Senate"
    assert cruz[4] is None

    db.close()


def test_bills_ingestion_and_relationship():
    """Verify bills are loaded with proper relationship and composite keys."""
    db = create_in_memory_db_with_seeds()

    bid = "G000596"
    sponsor_stats = load_bills(db, MOCK_SPONSORED_BILLS, bid, "sponsor")
    cosponsor_stats = load_bills(db, MOCK_COSPONSORED_BILLS, bid, "cosponsor")
    unchanged_stats = load_bills(db, MOCK_SPONSORED_BILLS, bid, "sponsor")

    assert sponsor_stats == {"total": 1, "inserted": 1, "updated": 0, "unchanged": 0}
    assert cosponsor_stats == {"total": 1, "inserted": 1, "updated": 0, "unchanged": 0}
    assert unchanged_stats == {"total": 1, "inserted": 0, "updated": 0, "unchanged": 1}

    rows = db.execute("SELECT congress, bill_type, bill_number, title, member_id, relationship FROM main.raw_bills ORDER BY bill_number").fetchall()
    assert len(rows) == 2

    # Check sponsored bill
    b1 = rows[0]
    assert b1 == (119, "HR", "101", "Protect American Energy Act", bid, "sponsor")

    # Check cosponsored bill
    b2 = rows[1]
    assert b2 == (119, "HR", "102", "Border Security Acceleration Act", bid, "cosponsor")
    introduced_dates = db.execute(
        "SELECT bill_number, introduced_date FROM main.raw_bills ORDER BY bill_number"
    ).fetchall()
    assert introduced_dates == [("101", "2025-01-03"), ("102", "2025-01-04")]
    db.close()


@patch("pipeline.fetch_legislation_for_member", return_value=(MOCK_SPONSORED_BILLS, MOCK_COSPONSORED_BILLS))
def test_house_bill_ingestion_logs_ingested_and_skipped_counts(mock_legislation, capsys):
    """House bill ingestion logs new rows and skips unchanged rows on rerun."""
    db = create_in_memory_db_with_seeds()
    house_member = [MOCK_MEMBERS[0]]

    ingest_member_bills(db, house_member, chamber="house")
    first_output = capsys.readouterr().out
    ingest_member_bills(db, house_member, chamber="house")
    second_output = capsys.readouterr().out

    assert "Bills ingested (new or changed): 2 (new: 2, updated: 0)" in first_output
    assert "Bills skipped (unchanged): 0" in first_output
    assert "Bills ingested (new or changed): 0 (new: 0, updated: 0)" in second_output
    assert "Bills skipped (unchanged): 2" in second_output
    assert mock_legislation.call_count == 2
    db.close()


def test_amendments_ingestion():
    """Verify amendments are correctly linked to parent bills."""
    db = create_in_memory_db_with_seeds()

    load_amendments(db, MOCK_AMENDMENTS, 119, "HR", "101")
    rows = db.execute("SELECT congress, bill_type, bill_number, amendment_number, amendment_type, description, purpose, sponsor_id FROM main.raw_amendments").fetchall()

    assert len(rows) == 1
    amdt = rows[0]
    assert amdt[0] == 119
    assert amdt[1] == "HR"
    assert amdt[2] == "101"
    assert amdt[3] == "1"
    assert amdt[4] == "HAMDT"
    assert "funding limits" in amdt[5]
    assert "Section 3" in amdt[6]
    assert amdt[7] == "G000596"
    db.close()


def test_idempotent_duplicate_run():
    """Verify running ingestion twice produces no duplicate rows or corrupted state."""
    db = create_in_memory_db_with_seeds()

    # First run
    load_members(db, MOCK_MEMBERS)
    load_bills(db, MOCK_SPONSORED_BILLS, "G000596", "sponsor")
    load_amendments(db, MOCK_AMENDMENTS, 119, "HR", "101")

    # Second run with exact same data
    load_members(db, MOCK_MEMBERS)
    load_bills(db, MOCK_SPONSORED_BILLS, "G000596", "sponsor")
    load_amendments(db, MOCK_AMENDMENTS, 119, "HR", "101")

    # Verify counts did NOT double
    member_count = db.execute("SELECT COUNT(*) FROM main.raw_members").fetchone()[0]
    bill_count = db.execute("SELECT COUNT(*) FROM main.raw_bills").fetchone()[0]
    amendment_count = db.execute("SELECT COUNT(*) FROM main.raw_amendments").fetchone()[0]

    assert member_count == 2
    assert bill_count == 1
    assert amendment_count == 1
    db.close()


def test_row_update_on_content_change():
    """Verify updating a record's attribute updates the database row and checksum."""
    db = create_in_memory_db_with_seeds()

    load_members(db, MOCK_MEMBERS)
    orig_hash = db.execute("SELECT row_hash FROM main.raw_members WHERE bioguide_id = 'G000596'").fetchone()[0]

    # Modify member name and re-load
    updated_members = [
        {
            "bioguideId": "G000596",
            "name": "Greene, Marjorie Taylor (Updated)",
            "state": "GA",
            "chamber": "House",
            "district": 14,
            "partyName": "Republican",
            "_geoid_cd": "1314",
            "updateDate": "2025-02-15T12:00:00Z",
        }
    ]
    load_members(db, updated_members)

    row = db.execute("SELECT name, row_hash FROM main.raw_members WHERE bioguide_id = 'G000596'").fetchone()
    assert row[0] == "Greene, Marjorie Taylor (Updated)"
    assert row[1] != orig_hash  # Hash must change
    db.close()


def test_get_existing_members():
    """Verify get_existing_members retrieves member records from raw_members table."""
    db = create_in_memory_db_with_seeds()
    load_members(db, MOCK_MEMBERS + MOCK_SENATE_MEMBERS)

    members = get_existing_members(db)
    assert len(members) == 4

    # Test chamber filter
    house_members = get_existing_members(db, chamber="house")
    assert len(house_members) == 2
    assert all(m["chamber"] == "House" for m in house_members)

    senate_members = get_existing_members(db, chamber="senate")
    assert len(senate_members) == 2
    assert all(m["chamber"] == "Senate" for m in senate_members)

    # Test limit
    members_limited = get_existing_members(db, member_limit=1)
    assert len(members_limited) == 1
    db.close()


def test_cli_flags_parsing():
    """Verify CLI task, chamber, member-scope, and sampling flags are parsed."""
    # Test without flags
    with sys_argv(["scripts/main.py", "--limit", "10"]):
        args = parse_args()
        assert args.random is False
        assert args.limit == 10
        assert args.members_only is False
        assert args.bills_only is False
        assert args.amendments_only is False
        assert args.skip_members is False
        assert args.house is False
        assert args.senate is False

    # Test with --members-only
    with sys_argv(["scripts/main.py", "--members-only"]):
        args = parse_args()
        assert args.members_only is True

    with sys_argv(["scripts/main.py", "--bills-only"]):
        args = parse_args()
        assert args.bills_only is True

    with sys_argv(["scripts/main.py", "--amendments-only"]):
        args = parse_args()
        assert args.amendments_only is True

    # Test with --skip-members
    with sys_argv(["scripts/main.py", "--skip-members"]):
        args = parse_args()
        assert args.skip_members is True

    # Test with --senate and -senate
    with sys_argv(["scripts/main.py", "--senate"]):
        args = parse_args()
        assert args.senate is True

    with sys_argv(["scripts/main.py", "-senate"]):
        args = parse_args()
        assert args.senate is True

    # Test with --house and -house
    with sys_argv(["scripts/main.py", "--house"]):
        args = parse_args()
        assert args.house is True

    with sys_argv(["scripts/main.py", "-house"]):
        args = parse_args()
        assert args.house is True


@patch("pipeline.fetch_members_for_congress", return_value=MOCK_MEMBERS)
@patch("pipeline.fetch_legislation_for_member", return_value=(MOCK_SPONSORED_BILLS, MOCK_COSPONSORED_BILLS))
@patch("pipeline.fetch_amendments_for_bill", return_value=MOCK_AMENDMENTS)
def test_ingest_members_is_callable(mock_amendments, mock_leg, mock_members):
    """The member stage can be run independently and returns ingestion stats."""
    db = create_in_memory_db_with_seeds()

    members, stats = ingest_members(db, member_limit=2, chamber="house")

    assert members == MOCK_MEMBERS
    assert stats == {"total": 2, "inserted": 2, "updated": 0, "unchanged": 0}
    assert db.execute("SELECT COUNT(*) FROM main.raw_members").fetchone()[0] == 2
    assert mock_members.called
    assert not mock_leg.called
    assert not mock_amendments.called
    db.close()


@patch("pipeline.fetch_legislation_for_member", return_value=(MOCK_SPONSORED_BILLS, MOCK_COSPONSORED_BILLS))
@patch("pipeline.fetch_amendments_for_bill")
def test_ingest_member_bills_is_callable(mock_amendments, mock_legislation):
    """The bill stage can run independently and returns distinct bills."""
    db = create_in_memory_db_with_seeds()

    bills = ingest_member_bills(db, MOCK_MEMBERS, chamber="house")

    assert list(bills) == [(119, "HR", "101"), (119, "HR", "102")]
    assert db.execute("SELECT COUNT(*) FROM main.raw_bills").fetchone()[0] == 4
    assert mock_legislation.call_count == len(MOCK_MEMBERS)
    assert not mock_amendments.called
    db.close()


def test_ingest_member_bills_fetches_members_concurrently():
    """Bill API calls run concurrently while their database writes complete safely."""
    db = create_in_memory_db_with_seeds()
    members = [
        {**MOCK_MEMBERS[0], "bioguideId": "TEST01"},
        {**MOCK_MEMBERS[1], "bioguideId": "TEST02"},
    ]
    both_workers_started = Barrier(2)

    def fetch_member_bills(*args, **kwargs):
        both_workers_started.wait(timeout=5)
        return MOCK_SPONSORED_BILLS, MOCK_COSPONSORED_BILLS

    with patch("pipeline.fetch_legislation_for_member", side_effect=fetch_member_bills) as mock_fetch:
        bills = ingest_member_bills(db, members, chamber="house")

    assert len(bills) == 2
    assert mock_fetch.call_count == 2
    assert db.execute("SELECT COUNT(*) FROM main.raw_bills").fetchone()[0] == 4
    db.close()


@patch("pipeline.fetch_amendments_for_bill", return_value=MOCK_AMENDMENTS)
def test_ingest_bill_amendments_is_callable(mock_amendments):
    """Amendments can be ingested directly from stored bill rows."""
    db = create_in_memory_db_with_seeds()
    bills = [
        (119, "HR", "101", "2025-02-01"),
        (119, "HR", "102", "2025-02-02"),
    ]

    result = ingest_bill_amendments(
        db, bills, full_refresh=True, use_async=True
    )

    assert result == (2, 2, 0)
    assert mock_amendments.call_count == 2
    assert db.execute("SELECT COUNT(*) FROM main.raw_amendments").fetchone()[0] > 0
    db.close()


@pytest.mark.parametrize("use_async", [False, True])
@patch("pipeline.fetch_amendments_for_bill", return_value=[])
def test_amendment_ingestion_logs_processed_and_unchanged_bills(
        mock_amendments, use_async, capsys
):
    """Both amendment modes report processed and unchanged bill counts."""
    db = create_in_memory_db_with_seeds()
    bills = {
        (119, "HR", "101"): {"latestAction": {"actionDate": "2025-02-01"}},
        (119, "HR", "102"): {"latestAction": {"actionDate": "2025-02-02"}},
    }
    timestamps = {(119, "HR", "102"): "2025-02-02"}

    result = ingest_bill_amendments(
        db, bills, existing_timestamps=timestamps, use_async=use_async
    )

    output = capsys.readouterr().out
    assert result == (2, 1, 1)
    assert mock_amendments.call_count == 1
    assert "Processed bill 119-HR-101: no amendments found" in output
    assert "Skipping bill 119-HR-102: no changes" in output
    assert "Bills processed: 1" in output
    assert "Bills skipped (no changes): 1" in output
    assert "Bills with no amendments: 1" in output
    db.close()


@pytest.mark.parametrize("use_async", [False, True])
@patch("pipeline.fetch_amendments_for_bill", return_value=[])
def test_amendments_process_once_after_bills_are_already_loaded(
        mock_amendments, use_async
):
    """Loading current bill rows does not mark their amendments as processed."""
    db = create_in_memory_db_with_seeds()
    load_members(db, MOCK_MEMBERS)
    load_bills(db, MOCK_SPONSORED_BILLS, "G000596", "sponsor")
    stored_bills = get_existing_bills(db, chamber="house")
    assert len(stored_bills) == 1

    # Bills already have the latest action date, but amendment state is still empty.
    initial_timestamps = get_existing_bill_timestamps(db)
    assert initial_timestamps == {}
    assert ingest_bill_amendments(
        db, stored_bills, existing_timestamps=initial_timestamps, use_async=use_async
    ) == (1, 1, 0)
    assert mock_amendments.call_count == 1

    # Once an amendment request succeeds (even when it finds none), later runs skip it.
    processed_timestamps = get_existing_bill_timestamps(db)
    assert processed_timestamps == {(119, "HR", "101"): "2025-02-01"}
    assert ingest_bill_amendments(
        db, stored_bills, existing_timestamps=processed_timestamps, use_async=use_async
    ) == (1, 0, 1)
    assert mock_amendments.call_count == 1
    db.close()


@patch("pipeline.fetch_members_for_congress", return_value=MOCK_MEMBERS)
@patch("pipeline.fetch_legislation_for_member", return_value=(MOCK_SPONSORED_BILLS, MOCK_COSPONSORED_BILLS))
@patch("pipeline.fetch_amendments_for_bill", return_value=MOCK_AMENDMENTS)
def test_run_ingestion_members_only(mock_amendments, mock_leg, mock_members):
    """Verify run_ingestion with members_only=True skips bills and amendments."""
    db = create_in_memory_db_with_seeds()

    run_ingestion(db, member_limit=2, members_only=True, chamber="house")

    member_count = db.execute("SELECT COUNT(*) FROM main.raw_members").fetchone()[0]
    bill_count = db.execute("SELECT COUNT(*) FROM main.raw_bills").fetchone()[0]
    amendment_count = db.execute("SELECT COUNT(*) FROM main.raw_amendments").fetchone()[0]

    assert member_count == 2
    assert bill_count == 0
    assert amendment_count == 0

    assert mock_members.called
    assert not mock_leg.called
    assert not mock_amendments.called
    db.close()


@patch("pipeline.fetch_members_for_congress")
@patch("pipeline.fetch_legislation_for_member", return_value=(MOCK_SPONSORED_BILLS, MOCK_COSPONSORED_BILLS))
@patch("pipeline.fetch_amendments_for_bill", return_value=MOCK_AMENDMENTS)
def test_run_ingestion_skip_members(mock_amendments, mock_leg, mock_members):
    """Verify run_ingestion with skip_members=True pulls members from table and ingests bills/amendments."""
    db = create_in_memory_db_with_seeds()

    # Pre-populate raw_members
    load_members(db, MOCK_MEMBERS)

    run_ingestion(db, skip_members=True, chamber="house")

    assert not mock_members.called
    assert mock_leg.called
    assert mock_amendments.called

    bill_count = db.execute("SELECT COUNT(*) FROM main.raw_bills").fetchone()[0]
    amendment_count = db.execute("SELECT COUNT(*) FROM main.raw_amendments").fetchone()[0]

    assert bill_count > 0
    assert amendment_count > 0
    db.close()


@patch("pipeline.fetch_members_for_congress")
@patch("pipeline.fetch_legislation_for_member", return_value=(MOCK_SPONSORED_BILLS, MOCK_COSPONSORED_BILLS))
@patch("pipeline.fetch_amendments_for_bill")
def test_run_ingestion_bills_only(mock_amendments, mock_leg, mock_members):
    """Bills-only mode writes bill records without requesting amendments."""
    db = create_in_memory_db_with_seeds()
    load_members(db, MOCK_MEMBERS)

    run_ingestion(
        db, skip_members=True, ingest_bills=True, ingest_amendments=False,
        chamber="house",
    )

    assert mock_leg.called
    assert not mock_amendments.called
    assert not mock_members.called
    assert db.execute("SELECT COUNT(*) FROM main.raw_bills").fetchone()[0] > 0
    assert db.execute("SELECT COUNT(*) FROM main.raw_amendments").fetchone()[0] == 0
    db.close()


@patch("pipeline.fetch_members_for_congress")
@patch("pipeline.fetch_legislation_for_member")
@patch("pipeline.fetch_amendments_for_bill", return_value=MOCK_AMENDMENTS)
def test_run_ingestion_amendments_only(mock_amendments, mock_leg, mock_members):
    """Amendments-only mode reads stored bills without fetching members or bills."""
    db = create_in_memory_db_with_seeds()
    db.execute("DROP TABLE main.raw_census__cd11920_county20")
    load_members(db, MOCK_MEMBERS)
    load_bills(db, MOCK_SPONSORED_BILLS, "G000596", "sponsor")
    load_bills(db, MOCK_COSPONSORED_BILLS, "G000596", "cosponsor")
    stored_bills = get_existing_bills(db, chamber="house")
    assert len(stored_bills) == 2

    run_ingestion(
        db, ingest_bills=False, ingest_amendments=True, full_refresh=True,
        chamber="house", use_async=True,
    )

    assert not mock_members.called
    assert not mock_leg.called
    assert mock_amendments.call_count == 2
    assert db.execute("SELECT COUNT(*) FROM main.raw_amendments").fetchone()[0] > 0
    db.close()


@patch("pipeline.fetch_members_for_congress", return_value=MOCK_SENATE_MEMBERS)
@patch("pipeline.fetch_legislation_for_member", return_value=(MOCK_SENATE_SPONSORED_BILLS, MOCK_SENATE_COSPONSORED_BILLS))
@patch("pipeline.fetch_amendments_for_bill", return_value=MOCK_SENATE_AMENDMENTS)
def test_run_ingestion_senate(mock_amendments, mock_leg, mock_senate_members):
    """Verify run_ingestion with chamber='senate' fetches senate members and legislation."""
    db = create_in_memory_db_with_seeds()

    run_ingestion(db, member_limit=2, chamber="senate")

    member_rows = db.execute("SELECT bioguide_id, chamber FROM main.raw_members ORDER BY bioguide_id").fetchall()
    assert len(member_rows) == 2
    assert all(r[1] == "Senate" for r in member_rows)

    bill_rows = db.execute("SELECT congress, bill_type, bill_number FROM main.raw_bills ORDER BY bill_number").fetchall()
    assert len(bill_rows) == 4
    assert all(r[1] == "S" for r in bill_rows)

    assert mock_senate_members.called
    assert mock_leg.called
    assert mock_amendments.called
    db.close()


class sys_argv:
    """Context manager to safely mock sys.argv for CLI testing."""
    def __init__(self, argv):
        self.argv = argv
        self._orig = None

    def __enter__(self):
        self._orig = sys.argv
        sys.argv = self.argv
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.argv = self._orig


if __name__ == "__main__":
    test_table_existence_validation()
    test_member_ingestion_and_data_validation()
    test_senate_member_ingestion()
    test_bills_ingestion_and_relationship()
    test_amendments_ingestion()
    test_idempotent_duplicate_run()
    test_row_update_on_content_change()
    test_get_existing_members()
    test_cli_flags_parsing()
    test_run_ingestion_members_only()
    test_run_ingestion_skip_members()
    test_run_ingestion_senate()
    print("All ingestion and data validation tests passed successfully.")
