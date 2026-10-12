from pathlib import Path

from dotenv import dotenv_values, load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANTHROPIC_WORKSPACE_ID = "wrkspc_017ZKZjouz2tme8sy6JZAQuf"


def load_project_environment(path: Path | None = None) -> None:
    """Load project settings and make the project .env API key authoritative."""
    env_path = path or PROJECT_ROOT / ".env"
    load_dotenv(env_path, override=False)
    project_api_key = dotenv_values(env_path).get("ANTHROPIC_API_KEY")
    if project_api_key:
        import os

        os.environ["ANTHROPIC_API_KEY"] = project_api_key


def anthropic_workspace_headers() -> dict[str, str]:
    """Return the workspace header required by the configured Anthropic key."""
    import os

    workspace_id = os.environ.get(
        "ANTHROPIC_WORKSPACE_ID", DEFAULT_ANTHROPIC_WORKSPACE_ID
    ).strip()
    return {"anthropic-workspace-id": workspace_id} if workspace_id else {}
