from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class ListItem(BaseModel):
    title: str = Field(description="Short heading, e.g. a topic name.")
    description: str = Field(description="One-sentence detail, e.g. share of bills and an example.")


class DataResultPayload(BaseModel):
    display_type: Literal["metric", "table", "list", "text"] = Field(
        description=(
            "Type of UI element to render:\n"
            "- 'metric': Single key performance indicators or counts.\n"
            "- 'table': Multi-column tabular data.\n"
            "- 'list': Key-value or bullet-point summaries.\n"
            "- 'text': Pure narrative answers."
        )
    )
    title: Optional[str] = Field(default=None, description="Header text for the UI card.")
    summary: Optional[str] = Field(default=None, description="Detailed text explanation or takeaway.")
    columns: Optional[List[str]] = Field(
        default=None,
        description=(
            "Column headers for the 'table' display type (e.g. ['Bill ID', 'Title', 'Status']). "
            "List them in the same order as the SQL result columns."
        ),
    )
    rows: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description=(
            "For 'metric': [{'label': 'Total Sponsored Bills', 'value': '323', 'subtext': 'WA Delegation'}].\n"
            "For 'list': leave empty and use `list_items` instead.\n"
            "For 'table': leave empty; the server fills rows from the SQL result."
        ),
    )
    list_items: Optional[List[ListItem]] = Field(
        default=None,
        description="Entries for the 'list' display type, e.g. one per topic or theme (4-8 items).",
    )
