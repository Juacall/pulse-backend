import pytest


@pytest.fixture(autouse=True)
def isolate_external_services(monkeypatch):
    # Tests patch DB_PATH with temp files: never use MotherDuck or the shared Redis cache
    monkeypatch.delenv("MOTHERDUCK_TOKEN", raising=False)
    try:
        from scripts.server.cache import cache
    except ImportError:
        return
    monkeypatch.setattr(cache, "client", None)
