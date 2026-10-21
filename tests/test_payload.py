from types import SimpleNamespace

from scripts.rag.payload import sanitize_payload
from scripts.rag.schema import DataResultPayload


def tool_msg(columns, rows):
    return SimpleNamespace(
        type="tool", content="| md |", artifact={"columns": columns, "rows": rows}
    )


def response(*messages):
    return {"messages": list(messages)}


BILLS = tool_msg(
    ["bill_id", "title"], [["HR-1", "First"], ["S-2", "Second"]]
)


def test_table_with_empty_dict_rows_is_rebuilt_from_sql_result():
    payload = DataResultPayload(
        display_type="table", columns=["Bill ID", "Title"], rows=[{}, {}]
    )
    result = sanitize_payload(payload, response(BILLS))
    assert result.display_type == "table"
    assert result.columns == ["Bill ID", "Title"]
    assert result.rows == [
        {"Bill ID": "HR-1", "Title": "First"},
        {"Bill ID": "S-2", "Title": "Second"},
    ]


def test_table_with_placeholder_rows_is_rebuilt():
    payload = DataResultPayload(display_type="table", rows=[{"1": "1"}])
    result = sanitize_payload(payload, response(BILLS))
    assert result.columns == ["bill_id", "title"]
    assert result.rows[0] == {"bill_id": "HR-1", "title": "First"}


def test_probe_query_is_ignored_in_favor_of_real_result():
    probe = tool_msg(["1"], [[1]])
    payload = DataResultPayload(display_type="table", rows=[{}])
    result = sanitize_payload(payload, response(BILLS, probe))
    assert result.rows[0]["bill_id"] == "HR-1"


def test_valid_model_rows_are_kept():
    rows = [{"Bill ID": "HR-9", "Theme": "Energy"}]
    payload = DataResultPayload(
        display_type="table", columns=["Bill ID", "Theme"], rows=rows
    )
    result = sanitize_payload(payload, response(BILLS))
    assert result.rows == rows


def test_table_without_sql_result_downgrades_to_text():
    payload = DataResultPayload(display_type="table", summary="Nothing", rows=[{}])
    result = sanitize_payload(payload, response())
    assert result.display_type == "text"
    assert result.rows is None
    assert result.summary == "Nothing"


def test_empty_metric_uses_first_numeric_sql_value():
    count = tool_msg(["label", "total_bills"], [["WA", 323]])
    payload = DataResultPayload(display_type="metric", title="Bills", rows=[{}])
    result = sanitize_payload(payload, response(count))
    assert result.rows == [
        {"label": "Bills", "value": "323", "subtext": "Query Result"}
    ]


def test_metric_falls_back_to_summary_number_then_text():
    payload = DataResultPayload(
        display_type="metric", summary="Members sponsored 1,204 bills", rows=[{}]
    )
    assert sanitize_payload(payload, response()).rows[0]["value"] == "1,204"

    empty = DataResultPayload(display_type="metric", summary="No data", rows=[{}])
    assert sanitize_payload(empty, response()).display_type == "text"


def test_list_items_are_mapped_into_rows():
    payload = DataResultPayload(
        display_type="list",
        summary="1,709 bills",
        list_items=[{"title": "Health", "description": "~9%"}],
        rows=[{}],
    )
    result = sanitize_payload(payload, response())
    assert result.display_type == "list"
    assert result.rows == [{"title": "Health", "description": "~9%"}]
    assert result.list_items is None


def test_list_with_only_empty_rows_downgrades_to_text():
    payload = DataResultPayload(display_type="list", summary="Short", rows=[{}])
    result = sanitize_payload(payload, response())
    assert result.display_type == "text"
    assert result.rows is None


def test_cell_with_pipe_survives_structured_path():
    msg = tool_msg(["title"], [["A | B"], ["C"]])
    payload = DataResultPayload(display_type="table", rows=[{}])
    assert sanitize_payload(payload, response(msg)).rows[0] == {"title": "A | B"}
