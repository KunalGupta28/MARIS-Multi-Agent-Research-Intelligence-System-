"""
MARIS Tests — PDF parser and Ingestion Pipeline unit tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path

from src.ingestion.parser import PDFParser, TextChunk
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.arxiv_client import ArxivPaper
from src.storage.database import MARISDatabase
from src.storage.vector_store import VectorStore


@pytest.fixture
def db(tmp_path):
    """Create a fresh SQLite database in a temporary directory."""
    db_path = tmp_path / "test_maris.db"
    return MARISDatabase(db_path=db_path)


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore to avoid any local disk locking or vector actions."""
    vs = MagicMock(spec=VectorStore)
    vs.index_chunks.return_value = 5
    return vs


class TestPDFParser:
    @patch("fitz.open")
    def test_parse_pdf_basic(self, mock_fitz_open, db):
        # Setup mock PyMuPDF document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        
        # Setup mock page blocks according to the actual dict structure
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "A Survey on Deep Learning",
                                    "size": 18.0,
                                    "font": "Helvetica-Bold"
                                }
                            ]
                        }
                    ]
                },
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "1. Introduction",
                                    "size": 14.0,
                                    "font": "Helvetica-Bold"
                                }
                            ]
                        }
                    ]
                },
                {
                    "type": 0,
                    "lines": [
                        {
                            "spans": [
                                {
                                    "text": "Deep learning has revolutionized brain signal analysis. We study EEG-BCI.",
                                    "size": 10.0,
                                    "font": "Helvetica"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        
        # Make the document iterable so the enumerate(doc) loop executes
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        # Populate SQLite with the paper first to satisfy the foreign key constraint
        db.upsert_paper(arxiv_id="2301.12345", title="A Survey on Deep Learning", authors=["Alice"])

        parser = PDFParser(db=db)
        chunks = parser.parse_pdf(
            pdf_path=Path("dummy.pdf"),
            paper_id="2301.12345",
            title="A Survey on Deep Learning",
            authors=["Alice"],
            pdf_url="http://arxiv.org/pdf/2301.12345"
        )

        assert len(chunks) > 0
        # Check that it extracted the preamble / section structure correctly
        assert chunks[0].paper_id == "2301.12345"
        assert chunks[0].title == "A Survey on Deep Learning"
        assert any("brain signal analysis" in c.text for c in chunks)


class TestIngestionPipeline:
    def test_ingest_paper(self, db, mock_vector_store, tmp_path):
        pipeline = IngestionPipeline(vector_store=mock_vector_store)
        pipeline.db = db
        
        # Create a real empty file to pass the .exists() check
        fake_pdf = tmp_path / "dummy.pdf"
        fake_pdf.touch()
        
        # Mock arxiv client download to return the real temporary path
        pipeline.arxiv_client.download_pdf_sync = MagicMock(return_value=fake_pdf)
        
        # Mock parser
        pipeline.parser.parse_pdf = MagicMock(return_value=[
            TextChunk(
                chunk_id="chunk_a",
                paper_id="2012.06753",
                text="Proposal for Neurohaptics.",
                section="Abstract",
                page_number=1,
                chunk_index=0,
                title="Neurohaptics",
                authors=["Bob"],
                pdf_url="http://arxiv.org/pdf/2012.06753"
            )
        ])

        paper = ArxivPaper(
            arxiv_id="2012.06753",
            title="Towards Neurohaptics",
            authors=["Bob"],
            published="2020-12-10",
            primary_category="cs.HC",
            pdf_url="http://arxiv.org/pdf/2012.06753",
            abstract="Neurohaptics...",
            categories=["cs.HC"]
        )

        # Indexing count should match mock vector store return value
        indexed_count = pipeline.ingest_paper(paper)
        assert indexed_count == 5
        pipeline.arxiv_client.download_pdf_sync.assert_called_once_with(paper)
        pipeline.parser.parse_pdf.assert_called_once()
        mock_vector_store.index_chunks.assert_called_once()
