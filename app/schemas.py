from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DocumentIn(BaseModel):
    id: str | None = Field(default=None, description="Optional document id. If omitted, server generates one.")
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpsertResponse(BaseModel):
    ids: list[str]


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class QueryResult(BaseModel):
    id: str
    text: str
    distance: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    results: list[QueryResult]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class RagAnswer(BaseModel):
    answer: str
    used_document_ids: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    answer: RagAnswer
    retrieved: list[QueryResult]
