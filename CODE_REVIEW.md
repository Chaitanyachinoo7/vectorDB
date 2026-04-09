# Code Review / Codebase Walkthrough

This document gives a step-by-step, big-picture understanding of the project, then drills down into how each request flows through the code.

## 1) What this app is

This is a small RAG-style API:

- You store text documents in a persistent vector database (ChromaDB).
- You can query similar documents with a text query.
- You can ask a chat question; the app retrieves relevant documents and an LLM (via Pydantic AI) answers.

Key idea: documents and queries are turned into vectors using a deterministic local embedding function (no embedding model download).

## 2) “Map” of the repository

- API entrypoint: [main.py](file:///c:/Users/Chimappa/vector/app/main.py)
- Configuration: [config.py](file:///c:/Users/Chimappa/vector/app/config.py)
- Request/response models: [schemas.py](file:///c:/Users/Chimappa/vector/app/schemas.py)
- Vector database wrapper: [vector_store.py](file:///c:/Users/Chimappa/vector/app/vector_store.py)
- Embeddings implementation: [embeddings.py](file:///c:/Users/Chimappa/vector/app/embeddings.py)
- LLM agent definition (Pydantic AI): [ai.py](file:///c:/Users/Chimappa/vector/app/ai.py)
- Docker runtime: [Dockerfile](file:///c:/Users/Chimappa/vector/Dockerfile), [docker-compose.yml](file:///c:/Users/Chimappa/vector/docker-compose.yml)
- Local developer docs: [README.md](file:///c:/Users/Chimappa/vector/README.md)

## 3) Startup sequence (what happens when the server boots)

1. FastAPI app is created with a lifespan function in [main.py](file:///c:/Users/Chimappa/vector/app/main.py).
2. On startup, `lifespan(...)` runs:
   - Creates a `ChromaVectorStore` (persistent Chroma client + collection).
   - Creates a Pydantic AI agent with a `retrieve` tool.
3. Both objects are stored on `app.state`:
   - `app.state.vector_store`
   - `app.state.rag_agent`

Why this design: these are “expensive-ish” objects (DB client, agent configuration) and are meant to be reused rather than recreated per request.

## 4) Configuration (how environment variables control behavior)

All configuration is defined in [config.py](file:///c:/Users/Chimappa/vector/app/config.py#L7-L20) and loaded as `settings`.

Important fields:

- `chroma_path` (default `data/chroma`): where Chroma persists vectors on disk.
- `chroma_collection` (default `documents`): collection name.
- `embedding_dim` (default `768`): dimension for the hashing-based embedding vector.
- `llm_model` (default `openai:gpt-4o-mini`): model string used by Pydantic AI.

Note: settings load from `.env` by default via `pydantic-settings` (see `env_file=".env"`).

## 5) Data model (what the API accepts/returns)

Request/response schemas are in [schemas.py](file:///c:/Users/Chimappa/vector/app/schemas.py).

Core types:

- `DocumentIn`: incoming document `{id?, text, metadata}`.
- `QueryRequest`: `{query, top_k}`.
- `QueryResult`: `{id, text, distance?, metadata}`.
- `ChatRequest`: `{message, top_k}`.
- `RagAnswer`: `{answer, used_document_ids}` (this is also the structured output required from the agent).

## 6) How vector search works

### 6.1 Embeddings (text -> vector)

Implemented in [embeddings.py](file:///c:/Users/Chimappa/vector/app/embeddings.py#L9-L24).

Approach:

- Uses `HashingVectorizer` to create a fixed-size sparse vector of dimension `embedding_dim`.
- Converts it to dense float32 and L2-normalizes it.

Tradeoff:

- Pros: deterministic, fast, no external model downloads, works offline.
- Cons: semantic quality is weaker than modern embedding models.

### 6.2 Chroma wrapper

Implemented in [vector_store.py](file:///c:/Users/Chimappa/vector/app/vector_store.py).

What it does:

- Creates/opens a persistent Chroma collection ([vector_store.py](file:///c:/Users/Chimappa/vector/app/vector_store.py#L19-L27)).
- `upsert()` embeds and stores documents ([vector_store.py](file:///c:/Users/Chimappa/vector/app/vector_store.py#L28-L45)).
- `query()` embeds the query and runs similarity search ([vector_store.py](file:///c:/Users/Chimappa/vector/app/vector_store.py#L47-L70)).

Output of `query()` is normalized into a list of dicts with `id/text/metadata/distance`.

## 7) How Pydantic AI is used (LLM + tools + structured output)

Agent setup is in [ai.py](file:///c:/Users/Chimappa/vector/app/ai.py).

Conceptually:

- The agent is created with:
  - `deps_type=RagDeps` (dependencies injected at run time)
  - `output_type=RagAnswer` (structured output validated by Pydantic)
  - instructions that tell the model to call the `retrieve` tool and output JSON
- The `retrieve` tool calls `ctx.deps.store.query(...)` ([ai.py](file:///c:/Users/Chimappa/vector/app/ai.py#L43-L46)).

Dependency injection:

- `RagDeps` contains the vector store and `default_top_k` ([ai.py](file:///c:/Users/Chimappa/vector/app/ai.py#L12-L16)).
- In the chat endpoint, a `RagDeps` instance is created per request with `top_k` from the request.

## 8) Endpoint-by-endpoint request flow

### 8.1 GET /health

File: [main.py](file:///c:/Users/Chimappa/vector/app/main.py#L32-L35)

Purpose:

- Quick liveness check.

### 8.2 POST /documents (upsert documents)

File: [main.py](file:///c:/Users/Chimappa/vector/app/main.py#L37-L41)

Step-by-step:

1. FastAPI validates the body as `list[DocumentIn]`.
2. The handler converts each document to `VectorDocument`.
3. `store.upsert(...)`:
   - generates IDs when missing
   - embeds texts
   - writes to Chroma
4. Returns `{ids: [...]}`.

### 8.3 POST /query (vector search)

File: [main.py](file:///c:/Users/Chimappa/vector/app/main.py#L44-L48)

Step-by-step:

1. Body is validated as `QueryRequest`.
2. `store.query(...)` runs embedding + Chroma query.
3. Results are converted into `QueryResult` objects.
4. Returns `{results: [...]}`.

### 8.4 POST /chat (RAG chat)

File: [main.py](file:///c:/Users/Chimappa/vector/app/main.py#L51-L59)

Step-by-step:

1. Body is validated as `ChatRequest`.
2. The endpoint runs a direct retrieval pass:
   - `retrieved = store.query(req.message, top_k=req.top_k)`
   - This list is returned in the response so you can debug what the vector DB matched.
3. The endpoint runs the agent:
   - Builds `deps = RagDeps(store=store, default_top_k=req.top_k)`
   - Calls `await agent.run(req.message, deps=deps)`
4. The agent may call `retrieve(...)` as a tool during its run.
5. Returns:
   - `answer` as a validated `RagAnswer`
   - `retrieved` results as a list of `QueryResult`

## 9) Running the app (Docker-first)

Docker artifacts:

- [Dockerfile](file:///c:/Users/Chimappa/vector/Dockerfile) builds a Python 3.10 image, installs dependencies from Pipenv, and runs Uvicorn.
- [docker-compose.yml](file:///c:/Users/Chimappa/vector/docker-compose.yml) runs the API with `--reload` and persists Chroma to a named volume.

Notes:

- `.env` is loaded by docker compose (env_file) and also by the app (pydantic-settings).
- `CHROMA_PATH` is overridden in compose to `/data/chroma` so persistence goes to the Docker volume.

## 10) Practical limitations (current behavior)

- No chunking: long documents are stored as a single vector; retrieval may be weaker than chunked indexing.
- Hashing embeddings: good for demos and offline use, but not comparable to real embedding models.
- No auth/rate limiting: all endpoints are open.
- Chat retrieval is split: the endpoint retrieves once for `retrieved` output, and the agent may retrieve again via tool calls.

## 11) Suggested improvements (next steps)

- Add document chunking + metadata linking (doc_id, chunk_id).
- Replace hashing embeddings with real embeddings (OpenAI, local model, etc.) behind an interface.
- Add tests for:
  - upsert/query consistency
  - schema validation
  - chat response shape (mock the agent)
- Add a basic evaluation script (golden Q/A against a fixed doc set).
- Add authentication (API key header) if exposing publicly.

## 12) Security notes

- Keep secrets only in `.env` (not committed). If a key ever gets pasted into a file or logs, rotate it immediately.
