"""
MARIS Graph — LangGraph StateGraph orchestration.

Wires all agent nodes into a directed acyclic graph with conditional routing.
Supports:
    - Sequential pipeline: Planner → Retriever → Extractor → Synthesizer
    - Retry loop: if Retriever finds too few chunks, re-plan with refined queries
    - Streaming callbacks: emit node status updates for the Streamlit UI
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Callable, Optional

from langgraph.graph import StateGraph, END

from src.agents.state import ResearchState
from src.agents.nodes import (
    planner_node,
    retriever_node,
    extractor_node,
    synthesizer_node,
)
from src.storage.database import MARISDatabase

logger = logging.getLogger(__name__)


def _should_retry_retrieval(state: ResearchState) -> str:
    """
    Conditional edge: decide whether to retry retrieval or proceed.

    Routes back to planner if we found too few chunks (< 3) and haven't
    exceeded the max iteration count.
    """
    if (
        len(state.retrieved_chunks) < 3
        and state.iteration_count < state.max_iterations
    ):
        logger.info(
            f"[Router] Too few chunks ({len(state.retrieved_chunks)}), "
            f"retry {state.iteration_count + 1}/{state.max_iterations}"
        )
        return "retry"
    return "proceed"


def _increment_iteration(state: ResearchState) -> dict[str, Any]:
    """Increment the iteration counter before retrying."""
    return {"iteration_count": state.iteration_count + 1}


def build_research_graph() -> StateGraph:
    """
    Construct the MARIS research agent graph.

    Graph topology:
        START → planner → retriever → [conditional] → extractor → synthesizer → END
                   ↑                      |
                   └──── retry ───────────┘

    Returns:
        A compiled LangGraph StateGraph ready for invocation.
    """
    graph = StateGraph(ResearchState)

    # ── Add nodes ─────────────────────────────────────────────────
    graph.add_node("planner", planner_node)
    graph.add_node("retriever", retriever_node)
    graph.add_node("retry_gate", _increment_iteration)
    graph.add_node("extractor", extractor_node)
    graph.add_node("synthesizer", synthesizer_node)

    # ── Define edges ──────────────────────────────────────────────
    graph.set_entry_point("planner")

    graph.add_edge("planner", "retriever")

    # Conditional: after retrieval, check if we have enough data
    graph.add_conditional_edges(
        "retriever",
        _should_retry_retrieval,
        {
            "retry": "retry_gate",
            "proceed": "extractor",
        },
    )

    graph.add_edge("retry_gate", "planner")
    graph.add_edge("extractor", "synthesizer")
    graph.add_edge("synthesizer", END)

    return graph


def compile_graph():
    """Build and compile the research graph for execution."""
    graph = build_research_graph()
    return graph.compile()


# ═══════════════════════════════════════════════════════════════════
# Runner — High-level API for executing research queries
# ═══════════════════════════════════════════════════════════════════


def merge_state(current_state: dict, update: dict) -> dict:
    """Merge a partial state update into the current state following LangGraph reducer logic."""
    for key, value in update.items():
        if key in ("sub_queries", "retrieved_chunks", "papers_found", "extracted_facts", "agent_trace", "errors"):
            # Concatenate list fields (operator.add)
            if key not in current_state or current_state[key] is None:
                current_state[key] = []
            merged = current_state[key] + list(value)

            # Deduplicate certain fields to prevent inflation during retry loops
            if key == "retrieved_chunks":
                seen = set()
                deduped = []
                for item in merged:
                    cid = item.chunk_id if hasattr(item, "chunk_id") else item.get("chunk_id", id(item))
                    if cid not in seen:
                        seen.add(cid)
                        deduped.append(item)
                merged = deduped
            elif key == "extracted_facts":
                seen = set()
                deduped = []
                for item in merged:
                    pid = item.paper_id if hasattr(item, "paper_id") else item.get("paper_id", id(item))
                    if pid not in seen:
                        seen.add(pid)
                        deduped.append(item)
                merged = deduped
            elif key == "papers_found":
                seen = set()
                deduped = []
                for item in merged:
                    if item not in seen:
                        seen.add(item)
                        deduped.append(item)
                merged = deduped

            current_state[key] = merged
        else:
            # Overwrite other fields
            current_state[key] = value
    return current_state


class ResearchRunner:
    """
    High-level runner for executing research queries through the agent pipeline.

    Provides a clean API for the Streamlit UI and CLI to trigger research runs,
    with support for streaming status updates.
    """

    def __init__(self):
        self.graph = compile_graph()
        self.db = MARISDatabase()

    def run(
        self,
        query: str,
        session_id: Optional[str] = None,
        on_status: Optional[Callable[[str, str], None]] = None,
    ) -> ResearchState:
        """
        Execute a full research pipeline for a query.

        Args:
            query: The research question or topic.
            session_id: Optional session ID (generated if not provided).
            on_status: Optional callback(node_name, message) for streaming updates.

        Returns:
            The final ResearchState with literature review and all metadata.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        # Create session in database
        self.db.create_session(session_id=session_id, query=query)

        logger.info(f"Starting research run [{session_id}]: '{query}'")

        # Build initial state
        initial_state = ResearchState(
            research_query=query,
            session_id=session_id,
        )

        # Run the graph via streaming and accumulate state
        state_dict = initial_state.model_dump()
        for step_output in self.graph.stream(state_dict):
            # step_output is {node_name: state_update}
            for node_name, state_update in step_output.items():
                logger.info(f"Completed node: {node_name}")
                state_dict = merge_state(state_dict, state_update)

                # Stream status updates to UI callback
                if on_status and "agent_trace" in state_update:
                    for trace in state_update["agent_trace"]:
                        if isinstance(trace, dict):
                            on_status(
                                trace.get("node_name", node_name),
                                trace.get("message", ""),
                            )
                        else:
                            on_status(trace.node_name, trace.message)

        # Build final state object directly from accumulated state_dict
        final = ResearchState(**state_dict)

        # Save results to database
        self.db.update_session(
            session_id=session_id,
            literature_review=final.literature_review,
            paper_ids=final.papers_found,
            status="completed",
        )

        logger.info(
            f"Research run [{session_id}] completed. "
            f"Papers: {len(final.papers_found)}, "
            f"Review: {len(final.literature_review)} chars"
        )

        return final

    def run_streaming(
        self,
        query: str,
        session_id: Optional[str] = None,
    ):
        """
        Execute research pipeline with streaming output for Streamlit.

        Yields (node_name, status_message) tuples as the graph executes.
        The final yield is ("__result__", ResearchState).
        """
        if session_id is None:
            session_id = str(uuid.uuid4())[:8]

        self.db.create_session(session_id=session_id, query=query)

        initial_state = ResearchState(
            research_query=query,
            session_id=session_id,
        )

        state_dict = initial_state.model_dump()
        for step_output in self.graph.stream(state_dict):
            for node_name, state_update in step_output.items():
                state_dict = merge_state(state_dict, state_update)
                # Yield status updates
                if "agent_trace" in state_update:
                    for trace in state_update["agent_trace"]:
                        if isinstance(trace, dict):
                            yield (
                                trace.get("node_name", node_name),
                                trace.get("message", ""),
                            )
                        else:
                            yield (trace.node_name, trace.message)

        # Build final state object directly from accumulated state_dict
        final = ResearchState(**state_dict)

        # Persist
        self.db.update_session(
            session_id=session_id,
            literature_review=final.literature_review,
            paper_ids=final.papers_found,
            status="completed",
        )

        yield ("__result__", final)
