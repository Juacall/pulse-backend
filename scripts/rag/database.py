import logging
import os
from pathlib import Path

import duckdb

from scripts.rag.environment import load_project_environment

load_project_environment()

logger = logging.getLogger("uvicorn.error")

LOCAL_DB_PATH = Path(__file__).resolve().parents[2] / "dev.duckdb"
MOTHERDUCK_DB_NAME = os.getenv("MOTHERDUCK_DB_NAME") or "pulse_prod"
_logged_targets: set[str] = set()


def use_motherduck() -> bool:
    return bool(os.getenv("MOTHERDUCK_TOKEN"))


def local_fallback_allowed() -> bool:
    return os.getenv("ALLOW_LOCAL_DB_FALLBACK", "").strip().lower() in {"1", "true", "yes"}


def database_label(local_path: Path = LOCAL_DB_PATH) -> str:
    return f"md:{MOTHERDUCK_DB_NAME}" if use_motherduck() else str(local_path)


def _log_once(level: int, key: str, message: str, *args) -> None:
    if key not in _logged_targets:
        _logged_targets.add(key)
        logger.log(level, message, *args)


def connect_duckdb(local_path: Path = LOCAL_DB_PATH) -> duckdb.DuckDBPyConnection:
    """Connect to MotherDuck; the local file is used only when ALLOW_LOCAL_DB_FALLBACK is true."""
    if use_motherduck():
        _log_once(logging.INFO, "md", "Using MotherDuck database '%s'.", MOTHERDUCK_DB_NAME)
        return duckdb.connect(f"md:{MOTHERDUCK_DB_NAME}")

    if not local_fallback_allowed():
        message = (
            "MOTHERDUCK_TOKEN is not set and ALLOW_LOCAL_DB_FALLBACK is not enabled. "
            "Set MOTHERDUCK_TOKEN, or set ALLOW_LOCAL_DB_FALLBACK=true to use the local "
            f"database at {local_path}."
        )
        _log_once(logging.ERROR, "missing-config", message)
        raise RuntimeError(message)

    _log_once(
        logging.WARNING,
        f"local:{local_path}",
        "MOTHERDUCK_TOKEN is not set; ALLOW_LOCAL_DB_FALLBACK is enabled, using local database %s.",
        local_path,
    )
    if not Path(local_path).exists():
        raise FileNotFoundError(f"DuckDB database file not found at: {local_path}")
    return duckdb.connect(str(local_path), read_only=True)
