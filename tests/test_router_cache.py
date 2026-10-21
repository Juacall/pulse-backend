from scripts.rag.router import QueryAnalysis
from scripts.server import server


class FakeCache:
    def __init__(self):
        self.store = {}

    def get(self, key):
        return self.store.get(key)

    def set(self, key, value, ex_seconds=0):
        self.store[key] = value
        return True


class CountingRouter:
    def __init__(self):
        self.calls = 0

    def process(self, query):
        self.calls += 1
        return QueryAnalysis(
            intent="SEMANTIC_SEARCH", state_code="CA", rewritten_prompt=f"rewrite: {query}"
        )


def _install(monkeypatch):
    fake_cache, fake_router = FakeCache(), CountingRouter()
    monkeypatch.setattr(server, "cache", fake_cache)
    monkeypatch.setattr(server, "router", fake_router)
    return fake_cache, fake_router


def test_repeated_query_uses_cached_analysis(monkeypatch):
    fake_cache, fake_router = _install(monkeypatch)

    first = server.analyze_query("California wildfire legislation")
    second = server.analyze_query("California wildfire legislation")

    assert fake_router.calls == 1
    assert second == first
    assert len(fake_cache.store) == 1


def test_equivalent_phrasing_shares_cache_entry(monkeypatch):
    _, fake_router = _install(monkeypatch)

    server.analyze_query("California wildfire legislation")
    server.analyze_query("  california   Wildfire legislation?? ")

    assert fake_router.calls == 1


def test_different_queries_are_not_shared(monkeypatch):
    _, fake_router = _install(monkeypatch)

    server.analyze_query("California wildfire legislation")
    server.analyze_query("Washington wildfire legislation")

    assert fake_router.calls == 2


def test_invalid_cached_entry_falls_back_to_router(monkeypatch):
    fake_cache, fake_router = _install(monkeypatch)
    server.analyze_query("q")
    key = next(iter(fake_cache.store))
    fake_cache.store[key] = {"intent": "NOPE"}

    server.analyze_query("q")

    assert fake_router.calls == 2
