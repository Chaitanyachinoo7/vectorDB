from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "Vector RAG API"
    chroma_path: str = Field(default="data/chroma", description="Filesystem path for Chroma persistence")
    chroma_collection: str = Field(default="documents", description="Chroma collection name")
    embedding_dim: int = Field(default=768, ge=32, le=4096)
    llm_model: str = Field(
        default="openai:gpt-4o-mini",
        description="Pydantic AI model string, e.g. 'openai:gpt-4o-mini' or 'ollama:llama3.1'",
    )


settings = Settings()
