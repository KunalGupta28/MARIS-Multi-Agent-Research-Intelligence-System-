"""
MARIS Agent State — Typed state schema for LangGraph orchestration.

This module defines the shared state that flows through all agent nodes
in the LangGraph StateGraph. Every field is strongly typed with Pydantic
for validation and serialization.
"""

from __future__ import annotations

from typing import Annotated, Optional
from pydantic import BaseModel, Field
import operator


class RetrievedChunk(BaseModel):
    """A single retrieved text chunk with source grounding metadata."""

    chunk_id: str = ""
    paper_id: str = ""
    text: str = ""
    section: str = ""
    page_number: int = 0
    title: str = ""
    authors: list[str] = Field(default_factory=list)
    pdf_url: str = ""
    score: float = 0.0


class ExtractedFact(BaseModel):
    """A structured fact extracted from a paper."""

    paper_id: str = ""
    title: str = ""
    problem_statement: str = ""
    methods: list[str] = Field(default_factory=list)
    datasets: list[str] = Field(default_factory=list)
    key_results: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    benchmarks: list[dict] = Field(default_factory=list)
    citations: list[dict] = Field(default_factory=list)



class AgentStatus(BaseModel):
    """Status update from an agent node (for streaming to UI)."""

    node_name: str = ""
    status: str = ""  # "started" | "completed" | "error"
    message: str = ""
    tokens_used: int = 0


class ResearchState(BaseModel):
    """
    The global state object that flows through the LangGraph pipeline.

    Each agent node reads from and writes to this state. LangGraph handles
    state persistence and checkpointing automatically.
    """

    # ── Input ─────────────────────────────────────────────────────
    research_query: str = ""
    session_id: str = ""

    # ── Planner Output ────────────────────────────────────────────
    sub_queries: Annotated[list[str], operator.add] = Field(default_factory=list)
    research_plan: str = ""

    # ── Retriever Output ──────────────────────────────────────────
    retrieved_chunks: Annotated[list[RetrievedChunk], operator.add] = Field(
        default_factory=list
    )
    papers_found: Annotated[list[str], operator.add] = Field(
        default_factory=list
    )  # list of arXiv IDs

    # ── Extractor Output ──────────────────────────────────────────
    extracted_facts: Annotated[list[ExtractedFact], operator.add] = Field(
        default_factory=list
    )

    # ── Synthesizer Output ────────────────────────────────────────
    literature_review: str = ""

    # ── Observability ─────────────────────────────────────────────
    agent_trace: Annotated[list[AgentStatus], operator.add] = Field(
        default_factory=list
    )
    total_tokens_used: int = 0
    errors: Annotated[list[str], operator.add] = Field(default_factory=list)

    # ── Control Flow ──────────────────────────────────────────────
    current_step: str = "planner"
    iteration_count: int = 0
    max_iterations: int = 3
