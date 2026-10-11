from pathlib import Path
import sys

from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import rag.router as router


def test_router_uses_supported_haiku_model_by_default(monkeypatch):
    models = []

    class FakeChatAnthropic:
        def __init__(self, *, model, temperature):
            models.append(model)

        def with_structured_output(self, schema):
            return RunnableLambda(lambda value: value)

    monkeypatch.delenv("ANTHROPIC_ROUTER_MODEL", raising=False)
    monkeypatch.setattr(router, "ChatAnthropic", FakeChatAnthropic)

    router.PreProcessingRouter()

    assert models == ["claude-haiku-5-5"]


def test_router_model_can_be_overridden_by_environment(monkeypatch):
    models = []

    class FakeChatAnthropic:
        def __init__(self, *, model, temperature):
            models.append(model)

        def with_structured_output(self, schema):
            return RunnableLambda(lambda value: value)

    monkeypatch.setenv("ANTHROPIC_ROUTER_MODEL", "custom-router-model")
    monkeypatch.setattr(router, "ChatAnthropic", FakeChatAnthropic)

    router.PreProcessingRouter()

    assert models == ["custom-router-model"]
