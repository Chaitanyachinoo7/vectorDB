# FastAPI + Vector DB + Pydantic AI (RAG)

This project is a minimal FastAPI app that:

- Stores text documents in a local persistent vector database (ChromaDB)
- Retrieves top-k relevant documents for a query
- Uses Pydantic AI to answer questions with a `retrieve` tool (RAG-style)

## WSL notes

- Open this folder from WSL (recommended) rather than running from `/mnt/c/...` for better filesystem performance.
- Install Python 3.10 and pipenv (Ubuntu 22.04 usually works with apt):

```bash
sudo apt update
sudo apt install -y python3-pip
sudo apt install -y python3.10 python3.10-venv python3.10-distutils
python3.10 -V
pip3 install --user pipenv
```

If your distro repo does not provide `python3.10` (common on Ubuntu 24.04), install via pyenv:

```bash
sudo apt update
sudo apt install -y build-essential curl git libssl-dev zlib1g-dev libbz2-dev \
  libreadline-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev
curl https://pyenv.run | bash
export PYENV_ROOT="$HOME/.pyenv"
export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
pyenv install 3.10.14
pyenv local 3.10.14
python -V
pip3 install --user pipenv
```

## Setup

```bash
export PIPENV_VENV_IN_PROJECT=1
pipenv --rm || true
pipenv --python python3.10 || pipenv --python "$(pyenv which python)"
pipenv lock --clear
pipenv sync --dev
pipenv run python -V
```

## Configure the LLM

Pydantic AI uses a model string like `openai:gpt-4o-mini` or `ollama:llama3.1`.

Set environment variables:

- `LLM_MODEL` (optional) e.g. `openai:gpt-4o-mini`
- `OPENAI_API_KEY` (required if you use an OpenAI model)

You can also create a `.env` file:

```env
LLM_MODEL=openai:gpt-4o-mini
OPENAI_API_KEY=...
```

## Run

```bash
pipenv run dev
```

Open: http://127.0.0.1:8000/docs

## Docker

Create a `.env` file (or copy from [.env.example](file:///c:/Users/Chimappa/vector/.env.example)) and set `OPENAI_API_KEY` if you're using OpenAI.

Build and run:

```bash
docker compose up --build
```

Open: http://127.0.0.1:8000/docs

## Dev commands

```bash
pipenv run test
pipenv run lint
pipenv run format
```

## API usage

Add documents:

```bash
curl -X POST http://127.0.0.1:8000/documents \
  -H "Content-Type: application/json" \
  -d '[{"text":"FastAPI is a Python web framework."},{"text":"Chroma is a local vector database."}]'
```

Query documents:

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query":"What is FastAPI?","top_k":5}'
```

Chat (Pydantic AI + retrieve tool):

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"Explain FastAPI in one sentence.","top_k":5}'
```

## Notes

- Vectors are generated locally using a deterministic hashing embedding (no model download needed).
- Chroma persists data under `data/chroma` by default.
- This repo uses `pydantic-ai-slim[openai]` to avoid installing model provider SDKs you aren't using.
