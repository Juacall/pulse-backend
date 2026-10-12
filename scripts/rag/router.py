# scripts/rag/router.py
import os
from typing import Literal, Optional

from scripts.rag.environment import (
    anthropic_workspace_headers,
    load_project_environment,
)

load_project_environment()

from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate


# 1. Define Structured Output Schema
class QueryAnalysis(BaseModel):
    """Analyzes user intent and normalizes parameters for legislative queries."""

    intent: Literal["SQL_AGGREGATION", "SEMANTIC_SEARCH", "HYBRID"] = Field(
        description="SQL_AGGREGATION for exact counts/lists/status. SEMANTIC_SEARCH for general topic lookups. HYBRID for summarizing specific subset of bills."
    )
    state_code: Optional[str] = Field(
        default=None,
        description="2-letter postal code (e.g., 'WA', 'CA', 'NY', 'DC'). Always convert full state names to upper-case 2-letter postal codes. Set to None if no state is specified."
    )
    chamber: Optional[Literal["House", "Senate"]] = Field(
        default=None,
        description="Specific legislative chamber if mentioned."
    )
    rewritten_prompt: str = Field(
        description="Refined and explicit version of the user prompt incorporating normalized state codes, exact column targets, and specific instructions for tool sequencing."
    )




# 2. Build Query Router Class
class PreProcessingRouter:
    def __init__(self, model_name: Optional[str] = None):
        model_name = model_name or os.environ.get(
            "ANTHROPIC_ROUTER_MODEL", "claude-haiku-5-5"
        )
        llm = ChatAnthropic(
            model=model_name,
            default_headers=anthropic_workspace_headers(),
        )
        self.structured_llm = llm.with_structured_output(QueryAnalysis)

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a query pre-processor for a legislative database assistant.
Your task is to convert raw user questions into clean, structured parameters and augmented execution instructions.

Rules:
    "1. Map full state/territory names to 2-letter uppercase postal abbreviations (e.g., Washington -> WA, California -> CA, District of Columbia -> DC).",
    "2. For SQL metrics (counts, bills sponsored, status checks), table `mart_legislative_activity` uses `target_member_state_code` for 2-letter codes ('WA', 'NY') and `target_member_state` for full state names ('WASHINGTON', 'NEW YORK'). Always filter 2-letter codes against `target_member_state_code`.",
    "3. Generate a `rewritten_prompt` that gives clear, step-by-step instructions to the agent based on the query intent.",
"""),
            ("user", "{user_input}")
        ])

        self.chain = self.prompt | self.structured_llm

    def process(self, raw_user_input: str) -> QueryAnalysis:
        return self.chain.invoke({"user_input": raw_user_input})


if __name__ == "__main__":
    # Test router directly
    router = PreProcessingRouter()

    test_queries = [
        "How many bills did members from Washington introduce?",
        "What recent legislation focuses on AI safety and water rights?",
        "Summarize the healthcare bills sponsored by California representatives."
    ]

    for q in test_queries:
        print(f"\nRaw Input: {q}")
        print("=" * 60)
        result = router.process(q)
        print(f"Intent:           {result.intent}")
        print(f"State Code:       {result.state_code}")
        print(f"Rewritten Prompt: {result.rewritten_prompt}")
