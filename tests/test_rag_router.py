import os
from pathlib import Path
import sys

from langchain_core.runnables import RunnableLambda

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import rag.router as router


def test_router_uses_supported_haiku_model_by_default(monkeypatch):
    models = []

    class FakeChatAnthropic:
        def __init__(self, *, model, default_headers):
            models.append(model)
            assert default_headers == {
                "anthropic-workspace-id": "wrkspc_017ZKZjouz2tme8sy6JZAQuf"
            }

        def with_structured_output(self, schema):
            return RunnableLambda(lambda value: value)

    monkeypatch.delenv("ANTHROPIC_ROUTER_MODEL", raising=False)
    monkeypatch.setattr(router, "ChatAnthropic", FakeChatAnthropic)

    router.PreProcessingRouter()

    assert models == ["claude-haiku-5-5"]


def test_router_model_can_be_overridden_by_environment(monkeypatch):
    models = []

    class FakeChatAnthropic:
        def __init__(self, *, model, default_headers):
            models.append(model)
            assert default_headers == {
                "anthropic-workspace-id": "wrkspc_017ZKZjouz2tme8sy6JZAQuf"
            }

        def with_structured_output(self, schema):
            return RunnableLambda(lambda value: value)

    monkeypatch.setenv("ANTHROPIC_ROUTER_MODEL", "custom-router-model")
    monkeypatch.setattr(router, "ChatAnthropic", FakeChatAnthropic)

    router.PreProcessingRouter()

    assert models == ["custom-router-model"]


def test_project_environment_loader_reads_requested_dotenv_without_override(
    tmp_path, monkeypatch
):
    from scripts.rag import environment

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("ANTHROPIC_API_KEY=dotenv-test-key\n", encoding="utf-8")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    environment.load_project_environment(dotenv_path)

    assert os.environ["ANTHROPIC_API_KEY"] == "dotenv-test-key"


def test_project_environment_loader_prefers_project_dotenv_api_key(
    tmp_path, monkeypatch
):
    from scripts.rag import environment

    dotenv_path = tmp_path / ".env"
    dotenv_path.write_text("ANTHROPIC_API_KEY=dotenv-test-key\n", encoding="utf-8")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "explicit-test-key")

    environment.load_project_environment(dotenv_path)

    assert os.environ["ANTHROPIC_API_KEY"] == "dotenv-test-key"


def test_anthropic_workspace_header_can_be_overridden(monkeypatch):
    from scripts.rag.environment import anthropic_workspace_headers

    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "custom-workspace")

    assert anthropic_workspace_headers() == {
        "anthropic-workspace-id": "custom-workspace"
    }
