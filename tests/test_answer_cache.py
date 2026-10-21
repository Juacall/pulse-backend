import pytest

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


class FakeRouter:
    def process(self, query):
        return QueryAnalysis(
            intent="SEMANTIC_SEARCH", state_code="CA", rewritten_prompt="rewrite"
        )


class FakeAgent:
    def __init__(self, payload):
        self.payload = payload
        self.calls = 0

    def invoke(self, _input):
        self.calls += 1
        return {"messages": [], "structured_response": self.payload}


def _payload(**overrides):
    base = dict(
        display_type="table",
        columns=["Bill ID"],
        rows=[{"Bill ID": "HR-1"}],
    )
    base.update(overrides)
    return server.DataResultPayload(**base)


@pytest.fixture
def env(monkeypatch):
    fake_cache = FakeCache()
    monkeypatch.setattr(server, "cache", fake_cache)
    monkeypatch.setattr(server, "router", FakeRouter())
    monkeypatch.setattr(server, "get_data_version", lambda: "2026-10-20 01:00:00")
    return fake_cache, monkeypatch


def _chat(query="California wildfire legislation"):
    return server.chat_with_agent(server.ChatRequest(query=query))


def test_second_identical_question_skips_agent(env):
    fake_cache, monkeypatch = env
    agent = FakeAgent(_payload())
    monkeypatch.setattr(server, "agent_executor", agent)

    first = _chat()
    second = _chat("  california wildfire legislation? ")

    assert agent.calls == 1
    assert first["cached"] is False
    assert second["cached"] is True
    assert second["synthesizedAnswer"] == first["synthesizedAnswer"]


def test_new_data_version_invalidates_cached_answer(env):
    _, monkeypatch = env
    agent = FakeAgent(_payload())
    monkeypatch.setattr(server, "agent_executor", agent)

    _chat()
    monkeypatch.setattr(server, "get_data_version", lambda: "2026-10-21 01:00:00")
    _chat()

    assert agent.calls == 2


def test_text_only_answers_are_not_cached(env):
    fake_cache, monkeypatch = env
    agent = FakeAgent(_payload(display_type="text", columns=None, rows=None, summary="Nope"))
    monkeypatch.setattr(server, "agent_executor", agent)

    _chat()
    _chat()

    assert agent.calls == 2
    assert not any(k.startswith("chat:answer:") for k in fake_cache.store)


def test_unknown_data_version_disables_answer_cache(env):
    fake_cache, monkeypatch = env
    agent = FakeAgent(_payload())
    monkeypatch.setattr(server, "agent_executor", agent)
    monkeypatch.setattr(server, "get_data_version", lambda: None)

    _chat()
    _chat()

    assert agent.calls == 2
