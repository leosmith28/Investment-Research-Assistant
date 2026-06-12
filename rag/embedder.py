"""
rag/embedder.py

LangChain-compatible wrapper around ChromaDB's built-in DefaultEmbeddingFunction.

DefaultEmbeddingFunction runs all-MiniLM-L6-v2 via onnxruntime (included in the
chromadb package). No PyTorch, sentence-transformers, or CUDA required.
The ~22 MB ONNX model is downloaded once on first use.
"""

import numpy as np
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.embeddings import Embeddings

_ef = DefaultEmbeddingFunction()


class ChromaDefaultEmbeddings(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(e) for e in _ef(texts)]

    def embed_query(self, text: str) -> np.ndarray:
        return np.array(_ef([text])[0])


def get_embeddings() -> ChromaDefaultEmbeddings:
    return ChromaDefaultEmbeddings()
