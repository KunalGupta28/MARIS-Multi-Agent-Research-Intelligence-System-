"""
MARIS Database — SQLite-backed storage for papers, citations, and workspace sessions.

This module provides a lightweight, zero-overhead persistence layer using Python's
built-in sqlite3. No Docker or external database servers are required.

Thread safety: Each call creates a short-lived connection (safe for Streamlit's
multi-threaded model). WAL mode is enabled for concurrent read performance.

Tables:
    papers       — Stores paper metadata (arXiv ID, title, authors, etc.)
    citations    — Directed citation edges between papers
    chunks       — Tracks which chunks were embedded (for deduplication)
    sessions     — Saved research workspace sessions
"""

from __future__ import annotations

import sqlite3
import json
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import get_settings

logger = logging.getLogger(__name__)

# ── Schema Definitions ───────────────────────────────────────────────
# All queries use parameterized placeholders (?), providing protection
# against SQL injection by design.

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS papers (
    arxiv_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         TEXT NOT NULL,       -- JSON array of author names
    abstract        TEXT,
    published_date  TEXT,
    primary_category TEXT,
    pdf_url         TEXT,
    pdf_local_path  TEXT,               -- local file path after download
    ingested_at     TEXT NOT NULL DEFAULT (datetime('now')),
    chunk_count     INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS citations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    source_paper_id TEXT NOT NULL,       -- the paper that cites
    target_paper_id TEXT NOT NULL,       -- the paper being cited
    context         TEXT,               -- surrounding sentence where citation appeared
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(source_paper_id, target_paper_id)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id        TEXT PRIMARY KEY,    -- deterministic hash of paper_id + chunk_index
    paper_id        TEXT NOT NULL,
    section         TEXT,               -- e.g. "Abstract", "Methods", "Results"
    page_number     INTEGER,
    chunk_index     INTEGER NOT NULL,
    text            TEXT NOT NULL,
    embedded        INTEGER DEFAULT 0,  -- 1 if successfully embedded in Qdrant
    FOREIGN KEY (paper_id) REFERENCES papers(arxiv_id)
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    query           TEXT NOT NULL,
    status          TEXT DEFAULT 'active',  -- active | completed | archived
    literature_review TEXT,             -- final generated markdown
    paper_ids       TEXT,               -- JSON array of paper IDs used
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_paper ON chunks(paper_id);
CREATE INDEX IF NOT EXISTS idx_citations_source ON citations(source_paper_id);
CREATE INDEX IF NOT EXISTS idx_citations_target ON citations(target_paper_id);
CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status);
"""


class MARISDatabase:
    """
    Thread-safe SQLite database manager for MARIS.

    Each public method creates a short-lived connection via the ``_connect()``
    context manager, ensuring safe usage across Streamlit's multi-threaded
    execution model. WAL journaling is enabled for concurrent read performance.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = get_settings().get_sqlite_abs_path()
        self.db_path = db_path
        self._init_db()

    @contextmanager
    def _connect(self):
        """
        Context manager that yields a short-lived SQLite connection.

        Ensures the connection is properly closed after use, preventing
        connection leaks in long-running Streamlit processes.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # better concurrent read performance
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _get_conn(self) -> sqlite3.Connection:
        """Create a new connection (legacy — prefer ``_connect()`` context manager)."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def close(self) -> None:
        """Explicit cleanup (no-op for per-call connection model, but signals intent)."""
        pass  # Each method manages its own connection lifecycle

    # ── Papers ────────────────────────────────────────────────────

    def upsert_paper(
        self,
        arxiv_id: str,
        title: str,
        authors: list[str],
        abstract: str = "",
        published_date: str = "",
        primary_category: str = "",
        pdf_url: str = "",
        pdf_local_path: str = "",
    ) -> None:
        """Insert or update a paper record."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO papers (arxiv_id, title, authors, abstract, published_date,
                                    primary_category, pdf_url, pdf_local_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(arxiv_id) DO UPDATE SET
                    title = excluded.title,
                    authors = excluded.authors,
                    abstract = excluded.abstract,
                    pdf_local_path = COALESCE(excluded.pdf_local_path, papers.pdf_local_path)
                """,
                (
                    arxiv_id,
                    title,
                    json.dumps(authors),
                    abstract,
                    published_date,
                    primary_category,
                    pdf_url,
                    pdf_local_path,
                ),
            )

    def get_paper(self, arxiv_id: str) -> Optional[dict]:
        """Retrieve a paper by arXiv ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM papers WHERE arxiv_id = ?", (arxiv_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["authors"] = json.loads(d["authors"])
                return d
            return None

    def get_all_papers(self) -> list[dict]:
        """Return all ingested papers."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM papers ORDER BY ingested_at DESC"
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["authors"] = json.loads(d["authors"])
                results.append(d)
            return results

    def get_paper_count(self) -> int:
        """Return the total number of papers in the database."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]

    def update_chunk_count(self, arxiv_id: str, count: int) -> None:
        """Update the number of chunks generated for a paper."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE papers SET chunk_count = ? WHERE arxiv_id = ?",
                (count, arxiv_id),
            )

    # ── Chunks ────────────────────────────────────────────────────

    def insert_chunk(
        self,
        chunk_id: str,
        paper_id: str,
        section: str,
        page_number: int,
        chunk_index: int,
        text: str,
    ) -> None:
        """Store a text chunk for provenance tracking."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO chunks
                    (chunk_id, paper_id, section, page_number, chunk_index, text)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chunk_id, paper_id, section, page_number, chunk_index, text),
            )

    def mark_chunk_embedded(self, chunk_id: str) -> None:
        """Mark a chunk as successfully embedded in the vector store."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE chunks SET embedded = 1 WHERE chunk_id = ?", (chunk_id,)
            )

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        """Retrieve a chunk by its ID with paper metadata for citation grounding."""
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT c.*, p.title, p.authors, p.published_date, p.pdf_url
                FROM chunks c
                LEFT JOIN papers p ON c.paper_id = p.arxiv_id
                WHERE c.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
            if row:
                d = dict(row)
                if d.get("authors"):
                    try:
                        d["authors"] = json.loads(d["authors"])
                    except (json.JSONDecodeError, TypeError):
                        d["authors"] = []
                else:
                    d["authors"] = []
                return d
            return None

    def get_chunk_count(self) -> int:
        """Return the total number of chunks in the database."""
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]

    # ── Citations ─────────────────────────────────────────────────

    def add_citation(
        self, source_paper_id: str, target_paper_id: str, context: str = ""
    ) -> None:
        """Record a citation edge from one paper to another."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO citations (source_paper_id, target_paper_id, context)
                VALUES (?, ?, ?)
                """,
                (source_paper_id, target_paper_id, context),
            )

    def get_citations_for_paper(self, arxiv_id: str) -> list[dict]:
        """Get all papers cited by a given paper."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, p.title as target_title
                FROM citations c
                LEFT JOIN papers p ON c.target_paper_id = p.arxiv_id
                WHERE c.source_paper_id = ?
                """,
                (arxiv_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_citation_graph(self) -> list[dict]:
        """Return all citation edges for graph visualization."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.source_paper_id, c.target_paper_id,
                       p1.title as source_title, p2.title as target_title
                FROM citations c
                LEFT JOIN papers p1 ON c.source_paper_id = p1.arxiv_id
                LEFT JOIN papers p2 ON c.target_paper_id = p2.arxiv_id
                """
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Sessions ──────────────────────────────────────────────────

    def create_session(self, session_id: str, query: str) -> None:
        """Create a new research session."""
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id, query, paper_ids)
                VALUES (?, ?, ?)
                """,
                (session_id, query, json.dumps([])),
            )

    def update_session(
        self,
        session_id: str,
        literature_review: Optional[str] = None,
        paper_ids: Optional[list[str]] = None,
        status: Optional[str] = None,
    ) -> None:
        """Update a research session with results."""
        updates = []
        params = []
        if literature_review is not None:
            updates.append("literature_review = ?")
            params.append(literature_review)
        if paper_ids is not None:
            updates.append("paper_ids = ?")
            params.append(json.dumps(paper_ids))
        if status is not None:
            updates.append("status = ?")
            params.append(status)
        updates.append("updated_at = datetime('now')")
        params.append(session_id)

        if updates:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE sessions SET {', '.join(updates)} WHERE session_id = ?",
                    params,
                )

    def get_session(self, session_id: str) -> Optional[dict]:
        """Retrieve a session by ID."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
            if row:
                d = dict(row)
                d["paper_ids"] = json.loads(d["paper_ids"]) if d["paper_ids"] else []
                return d
            return None

    def get_recent_sessions(self, limit: int = 20) -> list[dict]:
        """Return the most recent research sessions."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM sessions ORDER BY updated_at DESC LIMIT ?", (limit,)
            ).fetchall()
            results = []
            for row in rows:
                d = dict(row)
                d["paper_ids"] = json.loads(d["paper_ids"]) if d["paper_ids"] else []
                results.append(d)
            return results

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return high-level database statistics."""
        with self._connect() as conn:
            papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
            chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            citations = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
            sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            return {
                "papers": papers,
                "chunks": chunks,
                "citations": citations,
                "sessions": sessions,
            }
