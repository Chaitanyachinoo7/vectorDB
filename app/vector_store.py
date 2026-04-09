from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Iterable
from uuid import uuid4

os.environ.setdefault("ANONYMIZED_TELEMETRY", "FALSE")
os.environ.setdefault("CHROMA_TELEMETRY", "false")

import chromadb
from chromadb.config import Settings as ChromaSettings

from .embeddings import embed_texts


@dataclass(frozen=True)
class VectorDocument:
    id: str | None
    text: str
    metadata: dict[str, Any]


class ChromaVectorStore:
    def __init__(self, *, path: str, collection: str, embedding_dim: int):
        self._path = path
        self._collection_name = collection
        self._embedding_dim = embedding_dim

        try:
            self._client = chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
        except TypeError:
            self._client = chromadb.PersistentClient(path=path)
        self._collection = self._client.get_or_create_collection(name=collection)

    def upsert(self, docs: Iterable[VectorDocument]) -> list[str]:
        docs_list = list(docs)
        if not docs_list:
            return []

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for d in docs_list:
            doc_id = d.id or uuid4().hex
            ids.append(doc_id)
            documents.append(d.text)
            metadatas.append(d.metadata)

        embeddings = embed_texts(documents, dim=self._embedding_dim)
        self._collection.upsert(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
        return ids

    def query(self, query: str, *, top_k: int) -> list[dict[str, Any]]:
        query_embedding = embed_texts([query], dim=self._embedding_dim)[0]
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        out: list[dict[str, Any]] = []
        for i in range(min(len(ids), len(documents))):
            out.append(
                {
                    "id": ids[i],
                    "text": documents[i],
                    "metadata": metadatas[i] or {},
                    "distance": distances[i] if i < len(distances) else None,
                }
            )
        return out
