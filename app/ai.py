from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent, RunContext

from .schemas import RagAnswer
from .vector_store import ChromaVectorStore


@dataclass
class RagDeps:
    store: ChromaVectorStore
    default_top_k: int = 5


RAG_INSTRUCTIONS = """You are a helpful assistant.
Use the `retrieve` tool to look up relevant context in the vector database before answering.
If the retrieved context is not relevant or not sufficient, say you don't know.
Always return a JSON object matching the schema:
{ "answer": "<string>", "used_document_ids": ["<id>", "..."] }
Include only document ids you actually used to write the answer.
"""


def create_rag_agent(model: str) -> Agent[RagDeps, RagAnswer]:
    try:
        agent: Agent[RagDeps, RagAnswer] = Agent(
            model,
            deps_type=RagDeps,
            output_type=RagAnswer,
            instructions=RAG_INSTRUCTIONS,
        )
    except TypeError:
        agent = Agent(
            model,
            deps_type=RagDeps,
            output_type=RagAnswer,
            system_prompt=RAG_INSTRUCTIONS,
        )

    @agent.tool
    def retrieve(ctx: RunContext[RagDeps], query: str, top_k: int | None = None) -> list[dict[str, Any]]:
        return ctx.deps.store.query(query, top_k=top_k or ctx.deps.default_top_k)

    return agent
