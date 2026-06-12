"""
MARIS Tests — Vector store unit tests.
"""

import pytest
from tempfile import TemporaryDirectory
from pathlib import Path

from src.storage.database import MARISDatabase
from src.storage.vector_store import VectorStore
from src.ingestion.parser import TextChunk
from src.config import get_settings


class FakeEmbeddings:
    """Mock embeddings provider to avoid downloading models or calling APIs."""
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384


@pytest.fixture
def mock_settings(monkeypatch):
    """Override settings to use HuggingFace and local temp path."""
    settings = get_settings()
    monkeypatch.setattr(settings, "embedding_provider", "huggingface")
    monkeypatch.setattr(settings, "embedding_model", "all-MiniLM-L6-v2")
    return settings


@pytest.fixture
def vs(tmp_path, mock_settings):
    """Create a VectorStore with a mock DB and local temp Qdrant path."""
    db_path = tmp_path / "test_maris.db"
    db = MARISDatabase(db_path=db_path)
    
    # Set Qdrant path to temporary directory
    qdrant_dir = tmp_path / "qdrant_test_db"
    mock_settings.qdrant_path = str(qdrant_dir)
    mock_settings.qdrant_collection_name = "test_collection"
    
    vs_instance = VectorStore(db=db)
    # Inject fake embeddings
    vs_instance._embeddings = FakeEmbeddings()
    
    return vs_instance


class TestVectorStore:
    def test_ensure_collection(self, vs):
        stats = vs.get_collection_stats()
        assert stats["status"] != "empty"
        assert stats["points_count"] == 0

    def test_index_chunks(self, vs):
        chunks = [
            TextChunk(
                chunk_id="chunk_1",
                paper_id="2301.12345",
                text="Attention is all you need.",
                section="Abstract",
                page_number=1,
                chunk_index=0,
                title="Attention Is All You Need",
                authors=["Vaswani"],
                pdf_url="http://arxiv.org/pdf/2301.12345"
            ),
            TextChunk(
                chunk_id="chunk_2",
                paper_id="2301.12345",
                text="Feed forward networks are simple.",
                section="Methods",
                page_number=3,
                chunk_index=1,
                title="Attention Is All You Need",
                authors=["Vaswani"],
                pdf_url="http://arxiv.org/pdf/2301.12345"
            )
        ]
        
        # Populate SQLite with papers and chunks so marks work
        vs.db.upsert_paper(arxiv_id="2301.12345", title="Attention Is All You Need", authors=["Vaswani"])
        for c in chunks:
            vs.db.insert_chunk(c.chunk_id, c.paper_id, c.section, c.page_number, c.chunk_index, c.text)
            
        indexed = vs.index_chunks(chunks)
        assert indexed == 2
        
        stats = vs.get_collection_stats()
        assert stats["points_count"] == 2

    def test_hybrid_search(self, vs):
        # Index a mock chunk
        chunk = TextChunk(
            chunk_id="test_chunk",
            paper_id="2401.00001",
            text="State space models are great.",
            section="Discussion",
            page_number=5,
            chunk_index=0,
            title="SSM Survey",
            authors=["Mamba"],
            pdf_url="http://arxiv.org/pdf/2401.00001"
        )
        vs.db.upsert_paper(arxiv_id="2401.00001", title="SSM Survey", authors=["Mamba"])
        vs.db.insert_chunk(chunk.chunk_id, chunk.paper_id, chunk.section, chunk.page_number, chunk.chunk_index, chunk.text)
        vs.index_chunks([chunk])
        
        # Test vector search
        results = vs.vector_search("state space models", top_k=1)
        assert len(results) == 1
        assert results[0]["chunk_id"] == "test_chunk"
        assert results[0]["text"] == "State space models are great."
        
        # Test hybrid search
        hybrid_results = vs.hybrid_search("state space", top_k=1)
        assert len(hybrid_results) == 1
        assert hybrid_results[0]["chunk_id"] == "test_chunk"
