"""
MARIS PDF Parser — High-fidelity text extraction with section-level grounding.

Uses PyMuPDF (fitz) for fast, RAM-efficient PDF parsing. Implements structural
section detection via font-size heuristics and header pattern matching.

Every chunk produced carries rich provenance metadata so that every generated
sentence can be traced back to its exact origin in the source paper.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF

from src.storage.database import MARISDatabase

logger = logging.getLogger(__name__)

# Common section headers in scientific papers (case-insensitive matching)
_SECTION_PATTERNS = [
    r"^(?:\d+\.?\s*)?abstract",
    r"^(?:\d+\.?\s*)?introduction",
    r"^(?:\d+\.?\s*)?related\s+work",
    r"^(?:\d+\.?\s*)?background",
    r"^(?:\d+\.?\s*)?method(?:s|ology)?",
    r"^(?:\d+\.?\s*)?approach",
    r"^(?:\d+\.?\s*)?model",
    r"^(?:\d+\.?\s*)?experiment(?:s|al)?",
    r"^(?:\d+\.?\s*)?result(?:s)?",
    r"^(?:\d+\.?\s*)?evaluation",
    r"^(?:\d+\.?\s*)?discussion",
    r"^(?:\d+\.?\s*)?conclusion(?:s)?",
    r"^(?:\d+\.?\s*)?limitation(?:s)?",
    r"^(?:\d+\.?\s*)?future\s+work",
    r"^(?:\d+\.?\s*)?reference(?:s)?",
    r"^(?:\d+\.?\s*)?appendix",
    r"^(?:\d+\.?\s*)?acknowledgment(?:s)?",
]

_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), re.IGNORECASE)


@dataclass
class TextChunk:
    """A grounded text chunk with full provenance metadata."""

    chunk_id: str
    paper_id: str
    text: str
    section: str
    page_number: int
    chunk_index: int
    # Additional metadata for citation grounding
    title: str = ""
    authors: list[str] = field(default_factory=list)
    pdf_url: str = ""

    def to_metadata(self) -> dict:
        """Convert to a flat metadata dict for vector store payload."""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "section": self.section,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "title": self.title,
            "authors": self.authors,
            "pdf_url": self.pdf_url,
        }


@dataclass
class PageBlock:
    """Intermediate: a block of text from a specific page with font info."""

    text: str
    page_number: int
    font_size: float = 0.0
    is_bold: bool = False


class PDFParser:
    """
    Extracts structured, section-aware text chunks from scientific PDFs.

    Pipeline:
        1. Extract raw text blocks with font metadata per page
        2. Detect section boundaries using header heuristics
        3. Split sections into overlapping chunks
        4. Attach provenance metadata to each chunk
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        db: Optional[MARISDatabase] = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.db = db or MARISDatabase()

    def parse_pdf(
        self,
        pdf_path: Path,
        paper_id: str,
        title: str = "",
        authors: Optional[list[str]] = None,
        pdf_url: str = "",
    ) -> list[TextChunk]:
        """
        Parse a PDF file into grounded text chunks.

        Args:
            pdf_path: Path to the PDF file.
            paper_id: arXiv ID of the paper.
            title: Paper title for provenance.
            authors: List of author names.
            pdf_url: URL of the PDF for citation links.

        Returns:
            List of TextChunk objects with full provenance metadata.
        """
        if authors is None:
            authors = []

        logger.info(f"Parsing PDF: {pdf_path.name} ({paper_id})")

        # Step 1: Extract raw page blocks
        page_blocks = self._extract_page_blocks(pdf_path)

        if not page_blocks:
            logger.warning(f"No text extracted from {pdf_path.name}")
            return []

        # Step 2: Detect sections
        sections = self._detect_sections(page_blocks)

        # Step 3: Chunk each section
        chunks = self._chunk_sections(
            sections=sections,
            paper_id=paper_id,
            title=title,
            authors=authors,
            pdf_url=pdf_url,
        )

        # Step 4: Persist chunks to SQLite for provenance tracking
        for chunk in chunks:
            self.db.insert_chunk(
                chunk_id=chunk.chunk_id,
                paper_id=chunk.paper_id,
                section=chunk.section,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                text=chunk.text,
            )
        self.db.update_chunk_count(paper_id, len(chunks))

        # Step 5: Automatically extract citation edges between papers in the DB
        try:
            all_papers = self.db.get_all_papers()
            for other_paper in all_papers:
                other_id = other_paper["arxiv_id"]
                if other_id == paper_id:
                    continue
                # Search for arXiv ID (e.g. "1910.07295" or "2007.07214") in PDF text blocks
                id_pattern = other_id.strip()
                title_keywords = other_paper["title"][:25].lower()
                
                for block in page_blocks:
                    # Match by arXiv ID with word boundaries (prevents partial matches)
                    if re.search(rf'\b{re.escape(id_pattern)}\b', block.text) or (len(title_keywords) > 6 and title_keywords in block.text.lower()):
                        self.db.add_citation(paper_id, other_id, context=block.text[:300])
                        logger.info(f"[Parser] Automatically detected citation relationship: {paper_id} -> {other_id}")
                        break
        except Exception as e:
            logger.error(f"[Parser] Citation auto-extraction failed: {e}")

        logger.info(
            f"Extracted {len(chunks)} chunks from {len(sections)} sections in {pdf_path.name}"
        )
        return chunks

    def _extract_page_blocks(self, pdf_path: Path) -> list[PageBlock]:
        """Extract text blocks with font metadata from each page."""
        blocks = []

        try:
            doc = fitz.open(str(pdf_path))
        except Exception as e:
            logger.error(f"Failed to open PDF {pdf_path}: {e}")
            return []

        try:
            for page_num, page in enumerate(doc, start=1):
                # Extract text blocks: (x0, y0, x1, y1, "text", block_no, block_type)
                raw_blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)

                for block in raw_blocks.get("blocks", []):
                    if block.get("type") != 0:  # skip image blocks
                        continue

                    block_text_parts = []
                    max_font_size = 0.0
                    has_bold = False

                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            text = span.get("text", "").strip()
                            if text:
                                block_text_parts.append(text)
                                font_size = span.get("size", 0)
                                if font_size > max_font_size:
                                    max_font_size = font_size
                                font_name = span.get("font", "").lower()
                                if "bold" in font_name:
                                    has_bold = True

                    full_text = " ".join(block_text_parts).strip()
                    if len(full_text) > 10:  # skip trivially short blocks
                        blocks.append(
                            PageBlock(
                                text=full_text,
                                page_number=page_num,
                                font_size=max_font_size,
                                is_bold=has_bold,
                            )
                        )
        finally:
            doc.close()

        return blocks

    def _detect_sections(
        self, blocks: list[PageBlock]
    ) -> list[dict]:
        """
        Detect section boundaries based on header-like blocks.

        Returns a list of dicts:
            {"section": str, "text": str, "start_page": int}
        """
        # Calculate median font size to identify headers
        font_sizes = [b.font_size for b in blocks if b.font_size > 0]
        if not font_sizes:
            # Fallback: treat everything as one section
            full_text = "\n".join(b.text for b in blocks)
            return [
                {
                    "section": "Full Text",
                    "text": full_text,
                    "start_page": blocks[0].page_number if blocks else 1,
                }
            ]

        median_size = sorted(font_sizes)[len(font_sizes) // 2]
        header_threshold = median_size * 1.15  # headers are usually 15%+ bigger

        sections = []
        current_section = "Preamble"
        current_text_parts = []
        current_start_page = blocks[0].page_number if blocks else 1

        for block in blocks:
            # Check if this block looks like a section header
            is_header = False

            # Condition 1: larger font or bold text
            if block.font_size >= header_threshold or block.is_bold:
                # Condition 2: matches a known section pattern
                clean_line = block.text.strip()
                if _SECTION_RE.match(clean_line) and len(clean_line) < 80:
                    is_header = True

            if is_header:
                # Save the current section
                if current_text_parts:
                    sections.append(
                        {
                            "section": current_section,
                            "text": "\n".join(current_text_parts),
                            "start_page": current_start_page,
                        }
                    )

                # Start a new section
                current_section = block.text.strip()
                current_text_parts = []
                current_start_page = block.page_number
            else:
                current_text_parts.append(block.text)

        # Don't forget the last section
        if current_text_parts:
            sections.append(
                {
                    "section": current_section,
                    "text": "\n".join(current_text_parts),
                    "start_page": current_start_page,
                }
            )

        return sections

    def _chunk_sections(
        self,
        sections: list[dict],
        paper_id: str,
        title: str,
        authors: list[str],
        pdf_url: str,
    ) -> list[TextChunk]:
        """Split section text into overlapping chunks with provenance."""
        all_chunks = []
        global_index = 0

        for sec in sections:
            text = sec["text"]
            section_name = sec["section"]
            start_page = sec["start_page"]

            # Skip reference sections (we parse citations separately)
            if re.match(r"(?:references?|bibliography)", section_name, re.IGNORECASE):
                continue

            # Character-level chunking with overlap
            words = text.split()
            if not words:
                continue

            # Approximate token count by word count (1 word ≈ 1.3 tokens)
            words_per_chunk = int(self.chunk_size / 1.3)
            overlap_words = int(self.chunk_overlap / 1.3)

            i = 0
            while i < len(words):
                chunk_words = words[i : i + words_per_chunk]
                chunk_text = " ".join(chunk_words).strip()

                if len(chunk_text) < 50:  # skip tiny trailing chunks
                    break

                # Generate deterministic chunk ID
                chunk_id = self._generate_chunk_id(paper_id, global_index)

                all_chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        paper_id=paper_id,
                        text=chunk_text,
                        section=section_name,
                        page_number=start_page,
                        chunk_index=global_index,
                        title=title,
                        authors=authors,
                        pdf_url=pdf_url,
                    )
                )

                global_index += 1
                i += words_per_chunk - overlap_words

        return all_chunks

    @staticmethod
    def _generate_chunk_id(paper_id: str, chunk_index: int) -> str:
        """Generate a deterministic, unique chunk ID."""
        raw = f"{paper_id}::{chunk_index}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]
