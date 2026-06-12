"""
MARIS Tests — Database module unit tests.

Tests the SQLite database layer independently of any external services.
"""

import pytest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.storage.database import MARISDatabase


@pytest.fixture
def db(tmp_path):
    """Create a fresh database in a temporary directory."""
    db_path = tmp_path / "test_maris.db"
    return MARISDatabase(db_path=db_path)


class TestPapers:
    def test_upsert_and_get_paper(self, db):
        db.upsert_paper(
            arxiv_id="2301.12345",
            title="Test Paper on Transformers",
            authors=["Alice", "Bob"],
            abstract="This paper explores...",
            published_date="2023-01-15",
            primary_category="cs.CL",
            pdf_url="https://arxiv.org/pdf/2301.12345",
        )

        paper = db.get_paper("2301.12345")
        assert paper is not None
        assert paper["title"] == "Test Paper on Transformers"
        assert paper["authors"] == ["Alice", "Bob"]
        assert paper["primary_category"] == "cs.CL"

    def test_get_nonexistent_paper(self, db):
        paper = db.get_paper("9999.99999")
        assert paper is None

    def test_upsert_updates_existing(self, db):
        db.upsert_paper(
            arxiv_id="2301.12345",
            title="Original Title",
            authors=["Alice"],
        )
        db.upsert_paper(
            arxiv_id="2301.12345",
            title="Updated Title",
            authors=["Alice", "Bob"],
        )

        paper = db.get_paper("2301.12345")
        assert paper["title"] == "Updated Title"
        assert paper["authors"] == ["Alice", "Bob"]

    def test_get_all_papers(self, db):
        for i in range(3):
            db.upsert_paper(
                arxiv_id=f"2301.{i:05d}",
                title=f"Paper {i}",
                authors=[f"Author {i}"],
            )

        papers = db.get_all_papers()
        assert len(papers) == 3


class TestChunks:
    def test_insert_and_get_chunk(self, db):
        db.upsert_paper(arxiv_id="2301.12345", title="Test", authors=["A"])
        db.insert_chunk(
            chunk_id="abc123",
            paper_id="2301.12345",
            section="Methods",
            page_number=3,
            chunk_index=0,
            text="We propose a novel approach...",
        )

        chunk = db.get_chunk("abc123")
        assert chunk is not None
        assert chunk["section"] == "Methods"
        assert chunk["page_number"] == 3

    def test_mark_chunk_embedded(self, db):
        db.upsert_paper(arxiv_id="2301.12345", title="Test", authors=["A"])
        db.insert_chunk(
            chunk_id="abc123",
            paper_id="2301.12345",
            section="Abstract",
            page_number=1,
            chunk_index=0,
            text="Test text",
        )

        chunk = db.get_chunk("abc123")
        assert chunk["embedded"] == 0

        db.mark_chunk_embedded("abc123")
        chunk = db.get_chunk("abc123")
        assert chunk["embedded"] == 1


class TestCitations:
    def test_add_and_get_citations(self, db):
        db.upsert_paper(arxiv_id="paper_a", title="Paper A", authors=["A"])
        db.upsert_paper(arxiv_id="paper_b", title="Paper B", authors=["B"])

        db.add_citation("paper_a", "paper_b", context="As shown by B et al.")

        citations = db.get_citations_for_paper("paper_a")
        assert len(citations) == 1
        assert citations[0]["target_paper_id"] == "paper_b"

    def test_citation_graph(self, db):
        db.upsert_paper(arxiv_id="p1", title="P1", authors=["A"])
        db.upsert_paper(arxiv_id="p2", title="P2", authors=["B"])
        db.upsert_paper(arxiv_id="p3", title="P3", authors=["C"])

        db.add_citation("p1", "p2")
        db.add_citation("p1", "p3")
        db.add_citation("p2", "p3")

        graph = db.get_citation_graph()
        assert len(graph) == 3


class TestSessions:
    def test_create_and_get_session(self, db):
        db.create_session("sess_001", "Compare BERT and GPT")

        session = db.get_session("sess_001")
        assert session is not None
        assert session["query"] == "Compare BERT and GPT"
        assert session["status"] == "active"

    def test_update_session(self, db):
        db.create_session("sess_002", "Test query")
        db.update_session(
            "sess_002",
            literature_review="# Review\nContent here...",
            paper_ids=["p1", "p2"],
            status="completed",
        )

        session = db.get_session("sess_002")
        assert session["status"] == "completed"
        assert session["paper_ids"] == ["p1", "p2"]
        assert "# Review" in session["literature_review"]


class TestStats:
    def test_get_stats(self, db):
        stats = db.get_stats()
        assert stats["papers"] == 0
        assert stats["chunks"] == 0
        assert stats["citations"] == 0
        assert stats["sessions"] == 0

        db.upsert_paper(arxiv_id="test", title="Test", authors=["A"])
        stats = db.get_stats()
        assert stats["papers"] == 1
