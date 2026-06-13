"""
MARIS Tests — Agent nodes and graph unit tests.
"""

import pytest
from unittest.mock import MagicMock

from src.agents.state import ResearchState, RetrievedChunk, ExtractedFact
from src.agents.nodes import planner_node, extractor_node, synthesizer_node, retriever_node
from src.agents.graph import _should_retry_retrieval, build_research_graph, compile_graph
from src.agents.llm import get_llm


from langchain_core.runnables import RunnableLambda


from tests.conftest import FakeResponse
class TestAgentNodes:
    def test_planner_node(self, mock_llm):
        state = ResearchState(research_query="EEG analysis")
        output = planner_node(state)
        
        assert "sub_queries" in output
        assert len(output["sub_queries"]) == 2
        assert output["sub_queries"][0] == "attention in state space models"
        assert output["current_step"] == "retriever"

    def test_extractor_node(self, mock_llm):
        chunk = RetrievedChunk(
            chunk_id="c1",
            paper_id="arXiv:1905.04149",
            text="SSM is very fast.",
            title="SSM Survey",
            authors=["Alice"]
        )
        state = ResearchState(retrieved_chunks=[chunk])
        output = extractor_node(state)
        
        assert "extracted_facts" in output
        assert len(output["extracted_facts"]) == 1
        fact = output["extracted_facts"][0]
        assert fact.paper_id == "arXiv:1905.04149"
        assert fact.problem_statement == "Temporal alignment"
        assert output["current_step"] == "synthesizer"

    def test_synthesizer_node(self, mock_llm):
        fact = ExtractedFact(
            paper_id="arXiv:1905.04149",
            title="SSM Survey",
            problem_statement="Temporal alignment"
        )
        chunk = RetrievedChunk(
            chunk_id="c1",
            paper_id="arXiv:1905.04149",
            text="SSM is very fast.",
            title="SSM Survey",
            authors=["Alice"]
        )
        state = ResearchState(
            research_query="SSMs",
            retrieved_chunks=[chunk],
            extracted_facts=[fact]
        )
        output = synthesizer_node(state)
        
        assert "literature_review" in output
        assert "This is a mock synthesis review" in output["literature_review"]
        assert output["current_step"] == "complete"

    def test_planner_node_empty_query(self, mock_llm):
        state = ResearchState(research_query="")
        output = planner_node(state)
        assert "sub_queries" in output
        assert len(output["sub_queries"]) > 0

    def test_planner_node_invalid_json_fallback(self, mock_llm, monkeypatch):
        from src.agents.nodes import get_llm
        mock_chat_model = get_llm()
        
        # Override the return value of invoke for this test to return invalid JSON
        mock_chat_model.invoke.side_effect = lambda prompt_val, *args, **kw: FakeResponse("not a json array")
        mock_chat_model.side_effect = lambda prompt_val, *args, **kw: FakeResponse("not a json array")
        
        state = ResearchState(research_query="Quantum Computing")
        output = planner_node(state)
        
        assert "sub_queries" in output
        assert output["sub_queries"] == ["Quantum Computing"]
        assert "Fallback" in output["research_plan"]
        assert any(trace.status == "error" for trace in output["agent_trace"])


class TestRetryLogic:
    def test_should_retry_retrieval(self):
        # Case 1: Enough chunks -> Proceed
        state_proceed = ResearchState(
            retrieved_chunks=[
                RetrievedChunk(chunk_id="1"),
                RetrievedChunk(chunk_id="2"),
                RetrievedChunk(chunk_id="3")
            ],
            iteration_count=0
        )
        assert _should_retry_retrieval(state_proceed) == "proceed"

        # Case 2: Too few chunks, iteration available -> Retry
        state_retry = ResearchState(
            retrieved_chunks=[RetrievedChunk(chunk_id="1")],
            iteration_count=0,
            max_iterations=3
        )
        assert _should_retry_retrieval(state_retry) == "retry"

        # Case 3: Too few chunks, but iteration exhausted -> Proceed
        state_exhausted = ResearchState(
            retrieved_chunks=[RetrievedChunk(chunk_id="1")],
            iteration_count=3,
            max_iterations=3
        )
        assert _should_retry_retrieval(state_exhausted) == "proceed"


class TestGraphStructure:
    def test_build_research_graph(self):
        graph = build_research_graph()
        assert graph is not None
        # Verify planner node is inside
        assert "planner" in graph.nodes
        assert "retriever" in graph.nodes
        assert "extractor" in graph.nodes
        assert "synthesizer" in graph.nodes
