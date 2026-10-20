from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.server import server

#
# def test_chat_hides_router_details_and_requests_concise_answer_by_default(
#     monkeypatch,
# ):
#     analysis = SimpleNamespace(
#         intent="SQL_AGGREGATION",
#         state_code="WA",
#         rewritten_prompt="Count sponsored bills for Washington.",
#     )
#     monkeypatch.setattr(server.router, "process", lambda query: analysis)
#     captured = {}
#
#     def invoke(payload):
#         captured.update(payload)
#         return {"messages": [SimpleNamespace(content="Members sponsored 323 bills.")]}
#
#     monkeypatch.setattr(server.agent_executor, "invoke", invoke)
#
#     result = server.chat_with_agent(server.ChatRequest(query="How many?"))
#
#     assert result == {"synthesizedAnswer": "Members sponsored 323 bills."}
#     prompt = captured["messages"][0][1]
#     assert "Return the direct answer only" in prompt
#     assert "Do not describe internal SQL" in prompt
#
#
# def test_chat_includes_router_details_when_debug_is_enabled(monkeypatch):
#     analysis = SimpleNamespace(
#         intent="SQL_AGGREGATION",
#         state_code="WA",
#         rewritten_prompt="Count sponsored bills for Washington.",
#     )
#     monkeypatch.setattr(server.router, "process", lambda query: analysis)
#     captured = {}
#
#     def invoke(payload):
#         captured.update(payload)
#         return {"messages": [SimpleNamespace(content="Members sponsored 323 bills.")]}
#
#     monkeypatch.setattr(server.agent_executor, "invoke", invoke)
#
#     result = server.chat_with_agent(
#         server.ChatRequest(query="How many?", debug=True)
#     )
#
#     assert result == {
#         "synthesizedAnswer": "Members sponsored 323 bills.",
#         "routerAnalysis": {
#             "intent": "SQL_AGGREGATION",
#             "detectedState": "WA",
#             "rewrittenPrompt": "Count sponsored bills for Washington.",
#         },
#     }
#     assert "Debug mode is enabled" in captured["messages"][0][1]


def test_chat_reports_invalid_anthropic_key_as_service_unavailable(monkeypatch):
    monkeypatch.setattr(
        server.router,
        "process",
        Mock(side_effect=server.AnthropicAuthenticationError(
            message="invalid x-api-key",
            response=Mock(),
            body={"error": "invalid x-api-key"},
        )),
    )

    try:
        server.chat_with_agent(server.ChatRequest(query="How many bills?"))
    except server.HTTPException as error:
        assert error.status_code == 503
        assert "ANTHROPIC_API_KEY" in error.detail
    else:
        raise AssertionError("Expected an invalid Anthropic key to return 503")


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
    conn.execute("""
        CREATE TABLE dim_members (
            bioguide_id VARCHAR PRIMARY KEY,
            sponsored_count INTEGER,
            cosponsored_count INTEGER,
            enacted_count INTEGER,
            pass_rate DOUBLE
        )
    """)
    conn.execute(
        "INSERT INTO dim_members VALUES ('H1', 10, 20, 2, 20.0)"
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
        assert politicians[0]["cosponsoredCount"] == 0
        assert politicians[0]["enactedCount"] == 0
        assert politicians[0]["passRate"] is None


def test_politicians_without_chamber_filter_returns_all_members(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "politicians.duckdb"
    _create_politicians_db(db_path)
    monkeypatch.setattr(server, "DB_PATH", db_path)

    politicians = server.get_politicians()

    assert {politician["id"] for politician in politicians} == {"H1", "S1"}
    house_member = next(member for member in politicians if member["id"] == "H1")
    assert house_member["sponsoredCount"] == 10
    assert house_member["cosponsoredCount"] == 20
    assert house_member["enactedCount"] == 2
    assert house_member["passRate"] == 20.0


def test_politicians_all_chambers_filter_includes_senate_members(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "politicians.duckdb"
    _create_politicians_db(db_path)
    monkeypatch.setattr(server, "DB_PATH", db_path)

    politicians = server.get_politicians(state="WA", chamber="ALL")

    assert {politician["id"] for politician in politicians} == {"H1", "S1"}


def test_politician_bills_includes_introduced_date(tmp_path, monkeypatch):
    db_path = tmp_path / "bills.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE mart_legislative_activity (
            target_member_id VARCHAR,
            activity_type VARCHAR,
            activity_id VARCHAR,
            bill_title VARCHAR,
            activity_title VARCHAR,
            relationship VARCHAR,
            bill_status VARCHAR,
            bill_introduced_date DATE,
            bill_latest_action_text VARCHAR,
            bill_latest_action_date DATE
        )
    """)
    conn.execute("""
        INSERT INTO mart_legislative_activity VALUES (
            'H1', 'bill', 'HR-101', 'Example bill', 'Example bill',
            'sponsor', 'In Committee', DATE '2025-01-03',
            'Referred to committee', DATE '2025-02-01'
        )
    """)
    conn.close()
    monkeypatch.setattr(server, "DB_PATH", db_path)

    bills = server.get_politician_key_bills("H1")

    assert bills[0]["introducedDate"] == "2025-01-03"
    assert str(bills[0]["actionDate"])[:10] == "2025-02-01"


def test_politician_donors_returns_donors_and_filters_by_cycle(
    tmp_path, monkeypatch
):
    db_path = tmp_path / "donors.duckdb"
    conn = duckdb.connect(str(db_path))
    conn.execute("""
        CREATE TABLE member_top_donors (\
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


def test_get_current_user_raises_401_on_malformed_auth_header():
    try:
        server.get_current_user(authorization="Basic 12345")
    except server.HTTPException as error:
        assert error.status_code == 401
        assert "Invalid auth header format" in error.detail
    else:
        raise AssertionError("Expected malformed auth header to return 401")


def test_get_current_user_raises_401_on_missing_token():
    try:
        server.get_current_user(authorization="Bearer ")
    except server.HTTPException as error:
        assert error.status_code == 401
        assert "Token is missing" in error.detail
    else:
        raise AssertionError("Expected missing token to return 401")


def test_get_current_user_returns_clear_error_on_auth_api_error(monkeypatch):
    mock_supabase = MagicMock()
    jwt_error_message = (
        "invalid JWT: unable to parse or verify signature, "
        "token is malformed: token contains an invalid number of segments"
    )
    mock_supabase.auth.get_user.side_effect = server.AuthApiError(
        message=jwt_error_message,
        status=400,
        code=None,
    )
    monkeypatch.setattr(server, "supabase", mock_supabase)

    try:
        server.get_current_user(authorization="Bearer malformed.token")
    except server.HTTPException as error:
        assert error.status_code == 401
        assert "Invalid or expired session token" in error.detail
        assert jwt_error_message in error.detail
    else:
        raise AssertionError("Expected AuthApiError to be converted to HTTPException 401")


def test_get_current_user_returns_403_when_identity_not_verified(monkeypatch):
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user.return_value = SimpleNamespace(
        user=SimpleNamespace(id="user-123")
    )
    mock_profile_result = MagicMock()
    mock_profile_result.data = {"id": "user-123", "is_identity_verified": False}
    mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = (
        mock_profile_result
    )
    monkeypatch.setattr(server, "supabase", mock_supabase)

    try:
        server.get_current_user(authorization="Bearer valid.token")
    except server.HTTPException as error:
        assert error.status_code == 403
        assert "Identity verification required" in error.detail
    else:
        raise AssertionError("Expected unverified profile to return 403")


def test_get_current_user_succeeds_for_verified_user(monkeypatch):
    mock_supabase = MagicMock()
    mock_supabase.auth.get_user.return_value = SimpleNamespace(
        user=SimpleNamespace(id="user-123")
    )
    mock_profile_result = MagicMock()
    mock_profile_result.data = {"id": "user-123", "is_identity_verified": True}
    mock_supabase.table.return_value.select.return_value.eq.return_value.single.return_value.execute.return_value = (
        mock_profile_result
    )
    monkeypatch.setattr(server, "supabase", mock_supabase)

    user_id = server.get_current_user(authorization="Bearer valid.token")
    assert user_id == "user-123"


def test_vote_endpoint_returns_json_error_instead_of_stack_trace_on_malformed_jwt(monkeypatch):
    from fastapi.testclient import TestClient

    mock_supabase = MagicMock()
    jwt_error_message = (
        "invalid JWT: unable to parse or verify signature, "
        "token is malformed: token contains an invalid number of segments"
    )
    mock_supabase.auth.get_user.side_effect = server.AuthApiError(
        message=jwt_error_message,
        status=400,
        code=None,
    )
    monkeypatch.setattr(server, "supabase", mock_supabase)

    client = TestClient(server.app, raise_server_exceptions=False)
    response = client.post(
        "/api/vote",
        headers={"Authorization": "Bearer malformed.token"},
        json={
            "politician_id": "H1",
            "vote_type": "approve",
            "turnstile_token": "dummy-token",
        },
    )

    assert response.status_code == 401
    assert jwt_error_message in response.json()["detail"]
