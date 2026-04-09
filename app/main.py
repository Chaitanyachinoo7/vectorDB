from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from .ai import RagDeps, create_rag_agent
from .config import settings
from .schemas import (
    ChatRequest,
    ChatResponse,
    DocumentIn,
    QueryRequest,
    QueryResponse,
    QueryResult,
    UpsertResponse,
)
from .vector_store import ChromaVectorStore, VectorDocument


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = ChromaVectorStore(
        path=settings.chroma_path,
        collection=settings.chroma_collection,
        embedding_dim=settings.embedding_dim,
    )
    agent = create_rag_agent(settings.llm_model)
    app.state.vector_store = store
    app.state.rag_agent = agent

    try:
        yield
    finally:
        if hasattr(app.state, "vector_store"):
            delattr(app.state, "vector_store")
        if hasattr(app.state, "rag_agent"):
            delattr(app.state, "rag_agent")


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/documents", response_model=UpsertResponse)
def upsert_documents(documents: list[DocumentIn]) -> UpsertResponse:
    store: ChromaVectorStore = app.state.vector_store
    ids = store.upsert(VectorDocument(id=d.id, text=d.text, metadata=d.metadata) for d in documents)
    return UpsertResponse(ids=ids)


@app.post("/query", response_model=QueryResponse)
def query_documents(req: QueryRequest) -> QueryResponse:
    store: ChromaVectorStore = app.state.vector_store
    results = [QueryResult(**r) for r in store.query(req.query, top_k=req.top_k)]
    return QueryResponse(results=results)


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    store: ChromaVectorStore = app.state.vector_store
    agent = app.state.rag_agent

    retrieved = [QueryResult(**r) for r in store.query(req.message, top_k=req.top_k)]
    deps = RagDeps(store=store, default_top_k=req.top_k)
    run_result = await agent.run(req.message, deps=deps)
    return ChatResponse(answer=run_result.output, retrieved=retrieved)
