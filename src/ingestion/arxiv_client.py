
"""
MARIS ArXiv Client — Asynchronous paper search and PDF downloader.

Features:
    - Search arXiv by keyword queries with configurable max results
    - Download PDFs to local storage asynchronously
    - Rate limiting to respect arXiv's 3 req/sec policy
    - Structured metadata extraction (title, authors, abstract, categories)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Optional

import arxiv
import aiohttp

from src.config import get_settings, DATA_DIR
from src.storage.database import MARISDatabase

logger = logging.getLogger(__name__)

# Directory for downloaded PDFs
PDF_DIR = DATA_DIR / "pdfs"
PDF_DIR.mkdir(parents=True, exist_ok=True)


class ArxivPaper:
    """Structured representation of an arXiv paper's metadata."""

    def __init__(
        self,
        arxiv_id: str,
        title: str,
        authors: list[str],
        abstract: str,
        published: str,
        primary_category: str,
        pdf_url: str,
        categories: list[str],
    ):
        self.arxiv_id = arxiv_id
        self.title = title
        self.authors = authors
        self.abstract = abstract
        self.published = published
        self.primary_category = primary_category
        self.pdf_url = pdf_url
        self.categories = categories

    def __repr__(self) -> str:
        return f"ArxivPaper(id={self.arxiv_id}, title='{self.title[:50]}...')"


class ArxivClient:
    """Async-compatible arXiv paper search and download client."""

    def __init__(self, db: Optional[MARISDatabase] = None):
        self.settings = get_settings()
        self.db = db or MARISDatabase()
        self._semaphore = asyncio.Semaphore(self.settings.arxiv_rate_limit)

    def search_papers(
        self,
        query: str,
        max_results: Optional[int] = None,
        sort_by: arxiv.SortCriterion = arxiv.SortCriterion.Relevance,
    ) -> list[ArxivPaper]:
        """
        Search arXiv for papers matching a query.

        Args:
            query: Search query string (supports arXiv query syntax).
            max_results: Maximum number of results to return.
            sort_by: Sort criterion (Relevance, LastUpdatedDate, SubmittedDate).

        Returns:
            List of ArxivPaper objects with full metadata.
        """
        if max_results is None:
            max_results = self.settings.arxiv_max_results

        logger.info(f"Searching arXiv for: '{query}' (max {max_results} results)")

        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_by,
        )

        papers = []
        client = arxiv.Client()

        for result in client.results(search):
            # Extract a clean arXiv ID (e.g., "2301.12345")
            raw_id = result.entry_id.split("/abs/")[-1]
            # Remove version suffix if present
            arxiv_id = raw_id.split("v")[0] if "v" in raw_id else raw_id

            paper = ArxivPaper(
                arxiv_id=arxiv_id,
                title=result.title.strip().replace("\n", " "),
                authors=[a.name for a in result.authors],
                abstract=result.summary.strip().replace("\n", " "),
                published=result.published.strftime("%Y-%m-%d") if result.published else "",
                primary_category=result.primary_category or "",
                pdf_url=result.pdf_url or "",
                categories=[c for c in result.categories] if result.categories else [],
            )
            papers.append(paper)

            # Persist metadata to SQLite
            self.db.upsert_paper(
                arxiv_id=paper.arxiv_id,
                title=paper.title,
                authors=paper.authors,
                abstract=paper.abstract,
                published_date=paper.published,
                primary_category=paper.primary_category,
                pdf_url=paper.pdf_url,
            )

        logger.info(f"Found {len(papers)} papers for query: '{query}'")
        return papers

    def download_pdf_sync(self, paper: ArxivPaper) -> Optional[Path]:
        """
        Download a paper's PDF synchronously using httpx.

        This method is safe to call from Streamlit or any context where
        asyncio.run() would fail due to a running event loop.
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        import httpx

        # Sanitize filename
        safe_id = paper.arxiv_id.replace("/", "_").replace(".", "_")
        pdf_path = PDF_DIR / f"{safe_id}.pdf"

        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            logger.info(f"PDF already exists: {pdf_path}")
            return pdf_path

        try:
            logger.info(f"Downloading PDF (sync) for {paper.arxiv_id}...")
            with httpx.Client(timeout=60.0, follow_redirects=True) as client:
                resp = client.get(paper.pdf_url)
                if resp.status_code == 200:
                    pdf_path.write_bytes(resp.content)
                    logger.info(
                        f"Downloaded {paper.arxiv_id} ({len(resp.content) / 1024:.1f} KB)"
                    )
                    self.db.upsert_paper(
                        arxiv_id=paper.arxiv_id,
                        title=paper.title,
                        authors=paper.authors,
                        pdf_local_path=str(pdf_path),
                    )
                    return pdf_path
                else:
                    logger.error(
                        f"HTTP {resp.status_code} downloading {paper.arxiv_id}"
                    )
                    return None
        except Exception as e:
            logger.error(f"Failed to download (sync) {paper.arxiv_id}: {e}")
            return None

    async def download_pdf(self, paper: ArxivPaper) -> Optional[Path]:
        """
        Download a paper's PDF to local storage asynchronously.

        Args:
            paper: ArxivPaper object with a valid pdf_url.

        Returns:
            Path to the downloaded PDF file, or None on failure.
        """
        if not paper.pdf_url:
            logger.warning(f"No PDF URL for paper {paper.arxiv_id}")
            return None

        # Sanitize filename
        safe_id = paper.arxiv_id.replace("/", "_").replace(".", "_")
        pdf_path = PDF_DIR / f"{safe_id}.pdf"

        # Skip if already downloaded
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            logger.info(f"PDF already exists: {pdf_path}")
            return pdf_path

        async with self._semaphore:
            try:
                logger.info(f"Downloading PDF for {paper.arxiv_id}...")
                async with aiohttp.ClientSession() as session:
                    async with session.get(paper.pdf_url, timeout=aiohttp.ClientTimeout(total=60)) as resp:
                        if resp.status == 200:
                            content = await resp.read()
                            pdf_path.write_bytes(content)
                            logger.info(
                                f"Downloaded {paper.arxiv_id} ({len(content) / 1024:.1f} KB)"
                            )

                            # Update database with local path
                            self.db.upsert_paper(
                                arxiv_id=paper.arxiv_id,
                                title=paper.title,
                                authors=paper.authors,
                                pdf_local_path=str(pdf_path),
                            )
                            return pdf_path
                        else:
                            logger.error(
                                f"HTTP {resp.status} downloading {paper.arxiv_id}"
                            )
                            return None

            except Exception as e:
                logger.error(f"Failed to download {paper.arxiv_id}: {e}")
                return None

    async def download_all_pdfs(self, papers: list[ArxivPaper]) -> list[Path]:
        """
        Download PDFs for multiple papers concurrently (rate-limited).

        Args:
            papers: List of ArxivPaper objects.

        Returns:
            List of successfully downloaded PDF paths.
        """
        tasks = [self.download_pdf(p) for p in papers]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        paths = []
        for r in results:
            if isinstance(r, Path):
                paths.append(r)
            elif isinstance(r, Exception):
                logger.error(f"Download task failed: {r}")
        logger.info(f"Successfully downloaded {len(paths)}/{len(papers)} PDFs")
        return paths
