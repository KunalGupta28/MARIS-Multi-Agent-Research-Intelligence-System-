"""
MARIS Vector Store — Qdrant local disk mode with hybrid search (BM25 + Vector + RRF).

Runs entirely in-process using qdrant-client's local path mode (no Docker, no server).
Supports:
    - Batch embedding via OpenAI API or local HuggingFace models
    - Semantic vector search
    - BM25 keyword search with improved tokenization
    - Reciprocal Rank Fusion (RRF) for hybrid retrieval
    - Section-filtered search
    - Collection lifecycle management (create, delete, stats)
"""

from __future__ import annotations

import logging
import pickle
import re
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)
from rank_bm25 import BM25Okapi

from src.config import get_settings
from src.ingestion.parser import TextChunk
from src.storage.database import MARISDatabase

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Manages embedding, indexing, and hybrid retrieval of paper chunks.

    Uses Qdrant in local disk mode (zero RAM overhead) for vector storage,
    and BM25 in-memory for keyword search. Results are fused using RRF.
    """

    def __init__(
        self,
        db: Optional[MARISDatabase] = None,
    ):
        self.settings = get_settings()
        self.db = db or MARISDatabase()

        # Initialize Qdrant client in local disk mode
        qdrant_path = str(self.settings.get_qdrant_abs_path())
        self.client = QdrantClient(path=qdrant_path)
        self.collection_name = self.settings.qdrant_collection_name

        # BM25 index (rebuilt on-demand from stored texts)
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_chunk_ids: list[str] = []

        # Lazy-loaded embedding model
        self._embeddings = None

        self._ensure_collection()

    # Embedding dimension lookup for supported providers/models
    _VECTOR_DIMS = {
        "openai": 1536,        # text-embedding-3-small
        "huggingface": 384,    # all-MiniLM-L6-v2
    }

    def _get_vector_dim(self) -> int:
        """Return the expected embedding dimension for the configured provider."""
        return self._VECTOR_DIMS.get(self.settings.embedding_provider, 384)

    def _ensure_collection(self) -> None:
        """Create the Qdrant collection if it doesn't exist, or validate dimensions."""
        dim = self._get_vector_dim()
        collections = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(
                f"Created Qdrant collection '{self.collection_name}' "
                f"(dim={dim}, provider={self.settings.embedding_provider})"
            )
        else:
            # Validate that the existing collection matches expected dimensions
            try:
                info = self.client.get_collection(self.collection_name)
                existing_dim = info.config.params.vectors.size
                if existing_dim != dim:
                    logger.warning(
                        f"Collection '{self.collection_name}' has dimension {existing_dim} "
                        f"but configured provider expects {dim}. "
                        f"Consider recreating the collection if embeddings are mismatched."
                    )
            except Exception as e:
                logger.debug(f"Could not validate collection dimensions: {e}")

    def _get_embeddings(self):
        """Lazy-load the embedding model to avoid import overhead at startup."""
        if self._embeddings is None:
            provider = self.settings.embedding_provider
            model = self.settings.embedding_model

            if provider == "openai":
                from langchain_openai import OpenAIEmbeddings

                self._embeddings = OpenAIEmbeddings(
                    model=model,
                    openai_api_key=self.settings.openai_api_key,
                )
            elif provider == "huggingface":
                from langchain_huggingface import HuggingFaceEmbeddings

                self._embeddings = HuggingFaceEmbeddings(
                    model_name=model,
                    model_kwargs={"device": "cpu"},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info(f"Loaded local HuggingFace embedding model: {model}")
            else:
                raise ValueError(f"Unsupported embedding provider: {provider}")

        return self._embeddings

    # ── Indexing ──────────────────────────────────────────────────

    def index_chunks(self, chunks: list[TextChunk], batch_size: int = 50) -> int:
        """
        Embed and index chunks into Qdrant in batches.

        Args:
            chunks: List of TextChunk objects to embed.
            batch_size: Number of chunks per embedding API call.

        Returns:
            Number of chunks successfully indexed.
        """
        if not chunks:
            return 0

        embeddings_model = self._get_embeddings()
        total_indexed = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]

            try:
                # Batch embed — sends one API call for up to batch_size chunks
                vectors = embeddings_model.embed_documents(texts)

                points = []
                for chunk, vector in zip(batch, vectors):
                    points.append(
                        PointStruct(
                            id=int(chunk.chunk_id, 16) & 0x7FFFFFFFFFFFFFFF,  # positive int64, stable across restarts
                            vector=vector,
                            payload={
                                "chunk_id": chunk.chunk_id,
                                "paper_id": chunk.paper_id,
                                "text": chunk.text,
                                "section": chunk.section,
                                "page_number": chunk.page_number,
                                "chunk_index": chunk.chunk_index,
                                "title": chunk.title,
                                "authors": chunk.authors,
                                "pdf_url": chunk.pdf_url,
                            },
                        )
                    )

                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )

                # Mark chunks as embedded in SQLite
                for chunk in batch:
                    self.db.mark_chunk_embedded(chunk.chunk_id)

                total_indexed += len(batch)
                logger.info(
                    f"Indexed batch {i // batch_size + 1}: "
                    f"{len(batch)} chunks ({total_indexed}/{len(chunks)} total)"
                )

            except Exception as e:
                logger.error(f"Failed to index batch starting at {i}: {e}")
                continue

        # Invalidate BM25 cache so it rebuilds on next search
        self._bm25 = None
        return total_indexed

    # ── Search ────────────────────────────────────────────────────

    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        section_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        Perform semantic vector search.

        Args:
            query: Natural language query string.
            top_k: Number of results to return.
            section_filter: Optional section name to filter by.

        Returns:
            List of result dicts with 'text', 'score', and metadata.
        """
        embeddings_model = self._get_embeddings()
        query_vector = embeddings_model.embed_query(query)

        search_filter = None
        if section_filter:
            search_filter = Filter(
                must=[
                    FieldCondition(
                        key="section",
                        match=MatchValue(value=section_filter),
                    )
                ]
            )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k,
            query_filter=search_filter,
        ).points

        return [
            {
                "text": hit.payload.get("text", ""),
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id", ""),
                "paper_id": hit.payload.get("paper_id", ""),
                "section": hit.payload.get("section", ""),
                "page_number": hit.payload.get("page_number", 0),
                "title": hit.payload.get("title", ""),
                "authors": hit.payload.get("authors", []),
                "pdf_url": hit.payload.get("pdf_url", ""),
            }
            for hit in results
        ]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 indexing using word-boundary regex.

        Uses ``re.findall`` instead of naive ``.split()`` for better
        handling of punctuation, hyphens, and special characters.
        """
        return re.findall(r'\b\w+\b', text.lower())

    def bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Perform BM25 keyword search over all stored chunks.

        Rebuilds the BM25 index from Qdrant if not already cached.
        """
        self._ensure_bm25_index()

        if self._bm25 is None or not self._bm25_chunk_ids:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Get top-k indices
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk_id = self._bm25_chunk_ids[idx]
            chunk_data = self.db.get_chunk(chunk_id)
            if chunk_data:
                results.append(
                    {
                        "text": chunk_data["text"],
                        "score": float(scores[idx]),
                        "chunk_id": chunk_id,
                        "paper_id": chunk_data["paper_id"],
                        "section": chunk_data["section"],
                        "page_number": chunk_data["page_number"],
                        "title": chunk_data.get("title", ""),
                        "authors": chunk_data.get("authors", []),
                        "pdf_url": chunk_data.get("pdf_url", ""),
                    }
                )

        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        rrf_k: int = 60,
    ) -> list[dict]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).

        Combines semantic vector search and BM25 keyword search scores
        using the RRF formula: score = Σ (weight / (k + rank_i))

        Args:
            query: Search query string.
            top_k: Number of final results to return.
            vector_weight: Weight for vector search in RRF.
            bm25_weight: Weight for BM25 search in RRF.
            rrf_k: RRF constant (higher = more uniform blending).

        Returns:
            Ranked list of result dicts.
        """
        # Fetch more candidates from each source to improve fusion quality
        fetch_k = top_k * 3

        vector_results = []
        try:
            vector_results = self.vector_search(query, top_k=fetch_k)
        except Exception as e:
            logger.error(f"Vector search failed (falling back to BM25 search only): {e}")

        bm25_results = []
        try:
            bm25_results = self.bm25_search(query, top_k=fetch_k)
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")

        # Build RRF score map
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, result in enumerate(vector_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + vector_weight / (rrf_k + rank + 1)
            result_map[cid] = result

        for rank, result in enumerate(bm25_results):
            cid = result["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + bm25_weight / (rrf_k + rank + 1)
            if cid not in result_map:
                result_map[cid] = result

        # Sort by fused RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda c: rrf_scores[c], reverse=True)

        final_results = []
        for cid in sorted_ids[:top_k]:
            result = result_map[cid]
            result["rrf_score"] = rrf_scores[cid]
            final_results.append(result)

        return final_results

    # ── Internal ──────────────────────────────────────────────────

    def _ensure_bm25_index(self) -> None:
        """Build or rebuild the BM25 index from SQLite chunks database with disk caching."""
        if self._bm25 is not None:
            return

        import pickle

        # Determine index cache path
        qdrant_path = Path(self.settings.get_qdrant_abs_path())
        cache_path = qdrant_path.parent / "bm25_index.pkl"

        # Check SQLite chunks collection size first
        db_count = 0
        try:
            with self.db._get_conn() as conn:
                db_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to check SQLite chunks count: {e}")

        # Try to load cached index from disk
        if cache_path.exists() and db_count > 0:
            try:
                with open(cache_path, "rb") as f:
                    cached_data = pickle.load(f)
                
                # Check if cache is up-to-date
                if (
                    isinstance(cached_data, tuple)
                    and len(cached_data) == 2
                    and len(cached_data[1]) == db_count
                ):
                    self._bm25, self._bm25_chunk_ids = cached_data
                    logger.info(f"Loaded cached BM25 index from disk ({db_count} docs)")
                    return
            except Exception as e:
                logger.warning(f"Failed to load BM25 cache: {e}. Rebuilding...")

        # Fallback: rebuild index from SQLite chunks database
        logger.info("Building BM25 index from SQLite stored chunks...")

        try:
            with self.db._get_conn() as conn:
                rows = conn.execute("SELECT chunk_id, text FROM chunks").fetchall()
        except Exception as e:
            logger.error(f"Failed to query chunks from SQLite for BM25: {e}")
            return

        if not rows:
            return

        corpus = []
        chunk_ids = []

        for r in rows:
            text = r["text"]
            cid = r["chunk_id"]
            if text and cid:
                corpus.append(self._tokenize(text))
                chunk_ids.append(cid)

        if corpus:
            self._bm25 = BM25Okapi(corpus)
            self._bm25_chunk_ids = chunk_ids
            logger.info(f"BM25 index built with {len(corpus)} documents")

            # Save newly built index to cache
            try:
                # Ensure parent directory exists
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_path, "wb") as f:
                    pickle.dump((self._bm25, self._bm25_chunk_ids), f)
                logger.info(f"Cached BM25 index to disk: {cache_path}")
            except Exception as e:
                logger.error(f"Failed to save BM25 cache: {e}")

    def get_collection_stats(self) -> dict:
        """Return stats about the Qdrant collection."""
        try:
            info = self.client.get_collection(self.collection_name)
            v_count = getattr(info, "vectors_count", getattr(info, "indexed_vectors_count", 0))
            return {
                "vectors_count": v_count,
                "points_count": info.points_count,
                "status": info.status.value if hasattr(info.status, "value") else str(info.status),
            }
        except Exception as e:
            logger.debug(f"Error fetching collection stats: {e}")
            return {"vectors_count": 0, "points_count": 0, "status": "empty"}

    def delete_collection(self) -> None:
        """Delete the Qdrant collection (useful for test cleanup)."""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(f"Deleted Qdrant collection '{self.collection_name}'")
        except Exception as e:
            logger.debug(f"Could not delete collection: {e}")

    def close(self) -> None:
        """Close the Qdrant client connection and release resources."""
        try:
            self.client.close()
            logger.debug("Qdrant client closed")
        except Exception as e:
            logger.debug(f"Error closing Qdrant client: {e}")
