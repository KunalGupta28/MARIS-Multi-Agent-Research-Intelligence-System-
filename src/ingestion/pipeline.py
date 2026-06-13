"""
MARIS Ingestion Pipeline — End-to-end paper ingestion orchestrator.

Ties together: ArXiv search → PDF download → PDF parsing → Vector indexing.
Can be used standalone (CLI) or called from the agent pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Callable, Optional

from pydantic import BaseModel, Field

from src.ingestion.arxiv_client import ArxivClient, ArxivPaper
from src.ingestion.parser import PDFParser, TextChunk
from src.storage.vector_store import VectorStore
from src.storage.database import MARISDatabase

logger = logging.getLogger(__name__)


class IngestionResult(BaseModel):
    """Structured result of an ingestion pipeline run."""

    papers_found: int = Field(default=0, description="Number of papers found on arXiv")
    pdfs_downloaded: int = Field(default=0, description="Number of PDFs successfully downloaded")
    chunks_parsed: int = Field(default=0, description="Total text chunks parsed from PDFs")
    chunks_indexed: int = Field(default=0, description="Chunks successfully embedded and indexed")


class IngestionPipeline:
    """
    Orchestrates the full ingestion flow:
        1. Search arXiv for papers
        2. Download PDFs asynchronously
        3. Parse PDFs into section-aware chunks
        4. Embed and index chunks in Qdrant
    """

    def __init__(self, vector_store: Optional[VectorStore] = None):
        self.db = MARISDatabase()
        self.arxiv_client = ArxivClient(db=self.db)
        self.parser = PDFParser(db=self.db)
        self.vector_store = vector_store or VectorStore(db=self.db)

    def ingest_query(
        self,
        query: str,
        max_papers: int = 5,
        on_progress: Optional[Callable[[str], None]] = None,
    ) -> IngestionResult:
        """
        Full pipeline: search → download → parse → embed.

        Args:
            query: Search query for arXiv.
            max_papers: Maximum number of papers to process.
            on_progress: Optional callback for progress updates.

        Returns:
            IngestionResult with ingestion statistics.
        """
        logger.info(f"Starting ingestion for: '{query}'")

        # Step 1: Search arXiv
        papers = self.arxiv_client.search_papers(query, max_results=max_papers)
        logger.info(f"Found {len(papers)} papers")
        if on_progress:
            on_progress(f"Found {len(papers)} papers on arXiv")

        # Step 2: Download PDFs
        pdf_paths = asyncio.run(self.arxiv_client.download_all_pdfs(papers))
        logger.info(f"Downloaded {len(pdf_paths)} PDFs")
        if on_progress:
            on_progress(f"Downloaded {len(pdf_paths)} PDFs")

        # Step 3: Parse PDFs into chunks
        all_chunks: list[TextChunk] = []
        for paper, pdf_path in zip(papers, pdf_paths):
            if pdf_path and pdf_path.exists():
                chunks = self.parser.parse_pdf(
                    pdf_path=pdf_path,
                    paper_id=paper.arxiv_id,
                    title=paper.title,
                    authors=paper.authors,
                    pdf_url=paper.pdf_url,
                )
                all_chunks.extend(chunks)

        logger.info(f"Parsed {len(all_chunks)} chunks from {len(pdf_paths)} PDFs")
        if on_progress:
            on_progress(f"Parsed {len(all_chunks)} text chunks")

        # Step 4: Embed and index
        indexed = self.vector_store.index_chunks(all_chunks)

        result = IngestionResult(
            papers_found=len(papers),
            pdfs_downloaded=len(pdf_paths),
            chunks_parsed=len(all_chunks),
            chunks_indexed=indexed,
        )

        logger.info(f"Ingestion complete: {result.model_dump()}")
        return result

    def ingest_paper(self, paper: ArxivPaper) -> int:
        """
        Ingest a single paper: download → parse → embed.

        Uses synchronous download to avoid asyncio.run() conflicts
        when called from Streamlit (which has its own event loop).

        Returns the number of chunks indexed.
        """
        # Download (sync — safe inside Streamlit's event loop)
        pdf_path = self.arxiv_client.download_pdf_sync(paper)
        if not pdf_path or not pdf_path.exists():
            logger.warning(f"Could not download PDF for {paper.arxiv_id}")
            return 0

        # Parse
        chunks = self.parser.parse_pdf(
            pdf_path=pdf_path,
            paper_id=paper.arxiv_id,
            title=paper.title,
            authors=paper.authors,
            pdf_url=paper.pdf_url,
        )

        if not chunks:
            logger.warning(f"No chunks parsed from {paper.arxiv_id}")
            return 0

        # Index
        indexed = self.vector_store.index_chunks(chunks)
        logger.info(f"Ingested {paper.arxiv_id}: {len(chunks)} chunks parsed, {indexed} indexed")
        return indexed


# ── CLI Entry Point ───────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion.pipeline 'your search query'")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    pipeline = IngestionPipeline()
    result = pipeline.ingest_query(query)

    print(f"\n{'='*50}")
    print(f"Ingestion Results:")
    for k, v in result.items():
        print(f"  {k}: {v}")
    print(f"{'='*50}")
