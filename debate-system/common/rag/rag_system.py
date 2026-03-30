"""RAG (Retrieval-Augmented Generation) system implementation."""

import faiss
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from typing import List, Optional
import os
from ..configs.settings import Config
from .context_sanitizer import sanitize_context, truncate_context


class RAGSystem:
    """RAG system for retrieving relevant context from a knowledge base."""

    def __init__(self,
                 rag_data_path: Optional[str] = None,
                 rag_model_path: Optional[str] = None,
                 k: int = 3,
                 max_context_length: int = 2000):
        """
        Initialize RAG system.

        Args:
            rag_data_path: Path to RAG dataset
            rag_model_path: Path to sentence transformer model
            k: Number of documents to retrieve
            max_context_length: Maximum length of retrieved context per document
        """
        self.rag_data_path = rag_data_path or Config.RAG_DATA_PATH
        self.rag_model_path = rag_model_path or Config.RAG_MODEL_PATH
        self.k = k
        self.max_context_length = max_context_length

        # Initialize components
        self.corpus = None
        self.embedder = None
        self.index = None

        self._load_data()
        self._load_model()
        self._build_index()

    def _load_data(self):
        """Load the corpus from disk."""
        if not os.path.exists(self.rag_data_path):
            raise FileNotFoundError(f"RAG data path not found: {self.rag_data_path}")

        print("Loading dataset from disk...")
        dataset = load_from_disk(self.rag_data_path)
        self.corpus = dataset["text"]
        print(f"Corpus loaded with {len(self.corpus)} documents.")

    def _load_model(self):
        """Load the sentence transformer model."""
        if not os.path.exists(self.rag_model_path):
            raise FileNotFoundError(f"RAG model path not found: {self.rag_model_path}")

        print("Loading sentence transformer model...")
        self.embedder = SentenceTransformer(self.rag_model_path, device=Config.EMBEDDING_DEVICE)

    def _build_index(self):
        """Build FAISS index for similarity search."""
        print("Generating embeddings for corpus...")
        corpus_embeddings = self.embedder.encode(
            self.corpus,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True
        )

        print("Building FAISS index...")
        dim = corpus_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(corpus_embeddings)
        print(f"Index created with {self.index.ntotal} documents.")

    def retrieve_context(self, query: str, k: Optional[int] = None) -> str:
        """
        Retrieve relevant context for a given query.

        Args:
            query: Query string
            k: Number of documents to retrieve (overrides default)

        Returns:
            Concatenated context string
        """
        if k is None:
            k = self.k

        # Generate query embedding
        query_emb = self.embedder.encode([query], normalize_embeddings=True)

        # Search for similar documents
        distances, indices = self.index.search(query_emb, k=k)
        idxs = indices[0].tolist()

        # Retrieve and sanitize contexts
        contexts = []
        for i in idxs:
            raw = self.corpus[i]
            clean = sanitize_context(raw)
            clean = truncate_context(clean, self.max_context_length)
            contexts.append(clean)

        return "\n".join(contexts)

    def get_stats(self) -> dict:
        """Get statistics about the RAG system."""
        return {
            "corpus_size": len(self.corpus) if self.corpus else 0,
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dimension": self.embedder.get_sentence_embedding_dimension() if self.embedder else 0,
            "model_path": self.rag_model_path,
            "data_path": self.rag_data_path,
            "k": self.k,
            "max_context_length": self.max_context_length
        }