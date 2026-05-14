"""
Grounded retrieval over processed OCR documents.
Uses ChromaDB (local) + sentence-transformers for embeddings.
"""

from __future__ import annotations

import json
import os
import uuid
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

EMBED_MODEL = "all-MiniLM-L6-v2"


def _load_chunks(json_path: str) -> tuple[list[dict], str, str]:
    with open(json_path, encoding="utf-8") as f:
        payload = json.load(f)

    chunks = payload.get("chunks", [])
    source_md = payload.get("source_md") or ""
    source_file = os.path.basename(source_md) if source_md else os.path.basename(json_path)

    normalized = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        normalized.append(
            {
                "id": chunk.get("id") or str(uuid.uuid4()),
                "text": text,
                "metadata": {
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "chunk_type": chunk.get("chunk_type"),
                    "source_file": source_file,
                    "source_md": source_md,
                },
            }
        )

    return normalized, source_file, source_md


class DocumentRetriever:
    """
    Vector store over processed OCR chunks.
    Supports semantic search and returns evidence with source metadata.
    """

    def __init__(self, persist_dir: str = "./data/chroma_db"):
        self._ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection = self._client.get_or_create_collection(
            name="legal_docs",
            embedding_function=self._ef,
            metadata={"hnsw:space": "cosine"},
        )

    def index(self, ocr_json_path: str) -> int:
        """Load OCR JSON, embed, and store. Returns chunk count."""
        chunks, source_file, _ = _load_chunks(ocr_json_path)
        if not chunks:
            return 0

        try:
            existing = self._collection.get(where={"source_file": source_file})
            existing_ids = existing.get("ids") or []
            if existing_ids:
                self._collection.delete(ids=existing_ids)
        except Exception:
            pass

        self._collection.add(
            ids=[c["id"] for c in chunks],
            documents=[c["text"] for c in chunks],
            metadatas=[c["metadata"] for c in chunks],
        )

        return len(chunks)

    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        source_file: Optional[str] = None,
    ) -> list[dict]:
        """
        Semantic search. Returns passages with source metadata.
        Each result has text, score, page, and section.
        """
        count = self._collection.count()
        if count == 0:
            return []

        where = {"source_file": source_file} if source_file else None
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
            where=where,
        )

        passages = []
        for i, doc in enumerate(results["documents"][0]):
            meta = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            passages.append(
                {
                    "text": doc,
                    "relevance_score": round(1 - distance, 3),
                    "page": meta.get("page"),
                    "section": meta.get("section"),
                    "chunk_type": meta.get("chunk_type"),
                    "source_file": meta.get("source_file"),
                    "source_md": meta.get("source_md"),
                }
            )

        return passages
