"""
MARIS Tests — Integration and End-to-End RAG pipeline tests.
"""

import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile

from src.ingestion.parser import PDFParser, TextChunk
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.arxiv_client import ArxivPaper
from src.storage.database import MARISDatabase
from src.storage.vector_store import VectorStore
from src.agents.state import ResearchState, RetrievedChunk
from src.eval.run_eval import run_benchmark_suite


@pytest.fixture
def temp_db():
    """Create a fresh SQLite database in a temporary directory."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "integration_test.db"
        db = MARISDatabase(db_path=db_path)
        yield db


class TestIntegrationRAG:
    @patch("fitz.open")
    def test_parser_to_database_integration(self, mock_fitz_open, temp_db):
        """Test integration between PDFParser and MARIS SQLite relational tables."""
        # Setup mock PyMuPDF document
        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 1
        
        # Setup mock page blocks according to fitz dict structure
        mock_page = MagicMock()
        mock_page.get_text.return_value = {
            "blocks": [
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
                                    "text": "Mamba and SSM sequence models are linear-time alternatives to attention.",
                                    "size": 10.0,
                                    "font": "Helvetica"
                                }
                            ]
                        }
                    ]
                }
            ]
        }
        mock_doc.__iter__.return_value = [mock_page]
        mock_doc.__getitem__.return_value = mock_page
        mock_fitz_open.return_value = mock_doc

        # 1. First upsert paper record to enforce SQLite foreign keys
        arxiv_id = "2312.00752"
        temp_db.upsert_paper(
            arxiv_id=arxiv_id,
            title="Mamba: Linear-Time Sequence Modeling",
            authors=["Albert Gu", "Tri Dao"],
            published_date="2023-12-04",
            primary_category="cs.LG",
            pdf_url="https://arxiv.org/pdf/2312.00752"
        )

        # 2. Run PDF Parser which parses and inserts into relational tables
        parser = PDFParser(db=temp_db)
        chunks = parser.parse_pdf(
            pdf_path=Path("dummy_mamba.pdf"),
            paper_id=arxiv_id,
            title="Mamba: Linear-Time Sequence Modeling",
            authors=["Albert Gu", "Tri Dao"],
            pdf_url="https://arxiv.org/pdf/2312.00752"
        )

        # 3. Assertions on returned chunk structures
        assert len(chunks) > 0
        assert chunks[0].paper_id == arxiv_id
        assert "Mamba and SSM sequence models" in chunks[0].text

        # 4. Assertions on SQLite tables
        paper_record = temp_db.get_paper(arxiv_id)
        assert paper_record is not None
        assert paper_record["chunk_count"] == len(chunks)

        with temp_db._get_conn() as conn:
            rows = conn.execute("SELECT * FROM chunks WHERE paper_id = ?", (arxiv_id,)).fetchall()
            all_chunks = [dict(r) for r in rows]
            
        assert len(all_chunks) == len(chunks)
        assert all_chunks[0]["section"] == "1. Introduction"

    @patch("src.ingestion.arxiv_client.ArxivClient")
    def test_pipeline_ingestion_integration(self, mock_client_cls, temp_db):
        """Test full ingestion pipeline with mocked network and vector store."""
        mock_vector_store = MagicMock(spec=VectorStore)
        mock_vector_store.index_chunks.return_value = 1
        
        # Instantiate pipeline and inject our temporary database and mock vector store
        pipeline = IngestionPipeline(vector_store=mock_vector_store)
        pipeline.db = temp_db
        pipeline.arxiv_client.db = temp_db
        pipeline.parser.db = temp_db

        # Mock pdf download file creation
        with tempfile.TemporaryDirectory() as pdf_dir:
            temp_pdf = Path(pdf_dir) / "downloaded.pdf"
            temp_pdf.touch()

            pipeline.arxiv_client.download_pdf_sync = MagicMock(return_value=temp_pdf)
            pipeline.parser.parse_pdf = MagicMock(return_value=[
                TextChunk(
                    chunk_id="chunk_1",
                    paper_id="2401.00001",
                    text="State Space Models are fast.",
                    section="Abstract",
                    page_number=1,
                    chunk_index=0,
                    title="Mamba Fast",
                    authors=["Albert"],
                    pdf_url="https://arxiv.org/pdf/2401.00001"
                )
            ])

            paper = ArxivPaper(
                arxiv_id="2401.00001",
                title="Mamba Fast",
                authors=["Albert"],
                published="2024-01-01",
                primary_category="cs.LG",
                pdf_url="https://arxiv.org/pdf/2401.00001",
                abstract="State Space Models...",
                categories=["cs.LG"]
            )

            # First, insert the paper in database to allow parser foreign keys
            temp_db.upsert_paper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                published_date=paper.published,
                primary_category=paper.primary_category,
                pdf_url=paper.pdf_url
            )

            # Ingest paper and check vector index and relational database inserts
            indexed_count = pipeline.ingest_paper(paper)
            assert indexed_count == 1
            
            # Check SQL paper insert
            paper_db = temp_db.get_paper("2401.00001")
            assert paper_db is not None
            assert paper_db["title"] == "Mamba Fast"
            
            # Verify Mock calls
            pipeline.arxiv_client.download_pdf_sync.assert_called_once_with(paper)
            pipeline.parser.parse_pdf.assert_called_once()
            mock_vector_store.index_chunks.assert_called_once()

    def test_end_to_end_eval_runner_offline(self):
        """Test evaluation benchmark runner in offline/mock mode."""
        # Run offline benchmark suite which simulates multi-agent runs
        res = run_benchmark_suite(offline=True)
        
        # Verify result reports are populated
        assert "results" in res
        assert "report_path" in res
        results = res["results"]
        assert len(results) == 2
    
        # Verify quantitative metric bounds
        for metric in results:
            assert metric["overall_latency_sec"] > 0.0
            assert metric["precision_at_3"] >= 0.0
            assert metric["grounding_score"] == 1.0  # Perfect mock grounding
            assert metric["faithfulness_score"] > 0.5
            assert len(metric["ungrounded_citations"]) == 0

        # Verify physical report generated on disk
        report_path = Path(res["report_path"])
        assert report_path.exists()
        assert report_path.stat().st_size > 500
