import os
from pathlib import Path

import duckdb

from scripts.rag.environment import load_project_environment

load_project_environment()

LOCAL_DB_PATH = Path(__file__).resolve().parents[2] / "dev.duckdb"
MOTHERDUCK_DB_NAME = os.getenv("MOTHERDUCK_DB_NAME") or "pulse_prod"


def use_motherduck() -> bool:
    return bool(os.getenv("MOTHERDUCK_TOKEN"))


def database_label(local_path: Path = LOCAL_DB_PATH) -> str:
    return f"md:{MOTHERDUCK_DB_NAME}" if use_motherduck() else str(local_path)


def connect_duckdb(local_path: Path = LOCAL_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Connect to MotherDuck when MOTHERDUCK_TOKEN is set, else the local read-only file."""
    if use_motherduck():
        return duckdb.connect(f"md:{MOTHERDUCK_DB_NAME}")
    if not Path(local_path).exists():
        raise FileNotFoundError(f"DuckDB database file not found at: {local_path}")
    return duckdb.connect(str(local_path), read_only=True)
