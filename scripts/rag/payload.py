"""Repairs agent output so table/metric rows come from real SQL results."""
import re
from typing import Any, Dict, List, Optional

from scripts.rag.schema import DataResultPayload

MAX_TABLE_ROWS = 25


def sanitize_content_field(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
        return "\n\n".join(text_parts)
    return str(content) if content is not None else ""


def extract_sql_results(messages: Optional[List[Any]]) -> List[Dict[str, Any]]:
    """Collect structured results attached to SQL tool messages, oldest first."""
    results = []
    for msg in messages or []:
        artifact = getattr(msg, "artifact", None)
        if (
            getattr(msg, "type", None) == "tool"
            and isinstance(artifact, dict)
            and artifact.get("columns")
        ):
            results.append(artifact)
    return results


def _is_probe(result: Dict[str, Any]) -> bool:
    """Detects connectivity/sanity queries such as SELECT 1."""
    columns, rows = result["columns"], result["rows"]
    if len(columns) != 1 or len(rows) != 1:
        return False
    return bool(re.fullmatch(r"\d+|\?column\?", str(columns[0]).strip()))


def best_sql_result(messages: Optional[List[Any]]) -> Optional[Dict[str, Any]]:
    """The most recent non-empty, non-probe SQL result."""
    for result in reversed(extract_sql_results(messages)):
        if result.get("rows") and not _is_probe(result):
            return result
    return None


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _table_rows_valid(payload: DataResultPayload) -> bool:
    rows = payload.rows
    if not rows or not all(isinstance(r, dict) and r for r in rows):
        return False
    keys = {str(k).strip() for r in rows for k in r}
    if all(re.fullmatch(r"\d+", k) for k in keys):
        return False
    if payload.columns:
        wanted = {c.strip().casefold() for c in payload.columns}
        if not any(k.casefold() in wanted for k in keys):
            return False
    return True


def _metric_rows_valid(payload: DataResultPayload) -> bool:
    rows = payload.rows
    if not rows or not isinstance(rows[0], dict):
        return False
    value = rows[0].get("value")
    return value is not None and str(value).strip() != ""


def _rebuild_table(payload: DataResultPayload, result: Dict[str, Any]) -> None:
    result_columns = [str(c) for c in result["columns"]]
    # Model-chosen headers are applied by position only when they line up with the data
    if payload.columns and len(payload.columns) == len(result_columns):
        headers = list(payload.columns)
    else:
        headers = result_columns
    payload.columns = headers
    payload.rows = [
        {header: ("" if value is None else value) for header, value in zip(headers, row)}
        for row in result["rows"][:MAX_TABLE_ROWS]
    ]


def _metric_value(payload: DataResultPayload, result: Optional[Dict[str, Any]]) -> Optional[str]:
    if result:
        for value in result["rows"][0]:
            if _is_number(value):
                return str(value)
    if payload.summary:
        match = re.search(r"\b\d[\d,]*(?:\.\d+)?\b", payload.summary)
        if match:
            return match.group(0)
    return None


def _downgrade_to_text(payload: DataResultPayload) -> None:
    payload.display_type = "text"
    payload.columns = None
    payload.rows = None
    if not payload.summary:
        payload.summary = "No structured data was returned for this question."


def _clean_list_rows(payload: DataResultPayload) -> None:
    rows = [
        {"title": item.title, "description": item.description}
        for item in (payload.list_items or [])
    ]
    rows = rows or [
        r for r in (payload.rows or [])
        if isinstance(r, dict) and any(str(v).strip() for v in r.values() if v is not None)
    ]
    payload.list_items = None
    if rows:
        payload.rows = rows
    else:
        _downgrade_to_text(payload)


def sanitize_payload(
    payload: DataResultPayload, response: Optional[Dict[str, Any]] = None
) -> DataResultPayload:
    """Cleans text fields and rebuilds broken rows from the agent's SQL results."""
    if payload.summary:
        payload.summary = sanitize_content_field(payload.summary)
    if payload.title:
        payload.title = sanitize_content_field(payload.title)

    result = best_sql_result((response or {}).get("messages"))

    if payload.display_type == "table" and not _table_rows_valid(payload):
        if result:
            _rebuild_table(payload, result)
        else:
            _downgrade_to_text(payload)

    elif payload.display_type == "metric" and not _metric_rows_valid(payload):
        value = _metric_value(payload, result)
        if value is None:
            _downgrade_to_text(payload)
        else:
            payload.rows = [{
                "label": payload.title or "Total Count",
                "value": value,
                "subtext": "Query Result",
            }]

    elif payload.display_type == "list":
        _clean_list_rows(payload)

    return payload
