"""
MARIS Tests — Shared fixtures and configuration.
"""

import pytest
import math
from unittest.mock import MagicMock
from pathlib import Path
from langchain_core.runnables import RunnableLambda

from src.storage.database import MARISDatabase
from src.storage.vector_store import VectorStore
from src.config import get_settings, Settings
from src.ingestion.parser import TextChunk

# ── Marker Registration ──────────────────────────────────────────────

def pytest_configure(config):
    """Register custom markers for the test suite."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "slow: mark test as slow running")

# ── Shared Storage Fixtures ──────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Create a fresh database in a temporary directory."""
    db_path = tmp_path / "test_maris.db"
    database = MARISDatabase(db_path=db_path)
    yield database
    database.close()

class FakeEmbeddings:
    """Mock embeddings provider to avoid downloading models or calling APIs."""
    def embed_documents(self, texts):
        return [[0.1] * 384 for _ in texts]
    def embed_query(self, text):
        return [0.1] * 384

@pytest.fixture
def mock_settings(monkeypatch, tmp_path):
    """Override settings to use HuggingFace and local temp path."""
    settings = get_settings()
    monkeypatch.setattr(settings, "embedding_provider", "huggingface")
    monkeypatch.setattr(settings, "embedding_model", "all-MiniLM-L6-v2")
    
    # Set Qdrant path to temporary directory
    qdrant_dir = tmp_path / "qdrant_test_db"
    settings.qdrant_path = str(qdrant_dir)
    settings.qdrant_collection_name = "test_collection"
    return settings

@pytest.fixture
def vs(db, mock_settings):
    """Create a VectorStore with a mock DB and local temp Qdrant path."""
    vs_instance = VectorStore(db=db)
    # Inject fake embeddings
    vs_instance._embeddings = FakeEmbeddings()
    yield vs_instance
    vs_instance.close()

# ── Shared LLM Fixtures ──────────────────────────────────────────────

class FakeResponse:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {"total_tokens": 120}

class FakeLLM:
    """Mock LLM to return predetermined outputs for agent prompts."""
    def __init__(self):
        self.invocations = []

    def invoke(self, messages, *args, **kwargs):
        prompt = str(messages)
        self.invocations.append(prompt)
        prompt_lower = prompt.lower()
        
        if "research planning agent" in prompt_lower or "planner" in prompt_lower:
            return FakeResponse('["attention in state space models", "transformer limitations"]')
        elif "research extraction agent" in prompt_lower or "extractor" in prompt_lower:
            return FakeResponse('{"problem_statement": "Temporal alignment", "methods": ["SSM"], "datasets": ["LRA"], "key_results": ["SOTA"], "limitations": ["Quadratic"]}')
        else:
            return FakeResponse("## Literature Review\n\nThis is a mock synthesis review [1].\n\n### References\n[1] Author, 'Title', 2020. https://arxiv.org/abs/1905.04149")

@pytest.fixture
def mock_llm(monkeypatch):
    from src.agents.nodes import ScientificExtractionSchema, BenchmarkMetric, StructuredCitation
    fake = FakeLLM()
    
    # Create a mock chat model
    mock_chat_model = MagicMock()
    
    # Define with_structured_output behavior to return Pydantic schema
    mock_chat_model.with_structured_output.return_value = RunnableLambda(
        lambda inputs: ScientificExtractionSchema(
            problem_statement="Temporal alignment",
            methods=["SSM"],
            datasets=["LRA"],
            key_results=["SOTA"],
            limitations=["Quadratic"],
            benchmarks=[
                BenchmarkMetric(
                    dataset="LRA",
                    metric_name="Accuracy",
                    value="84.2%",
                    baseline="81.0%",
                    context="Mamba-3B"
                )
            ],
            citations=[
                StructuredCitation(
                    citation_key="[^arXiv:1905.04149]",
                    title="SSM Survey",
                    authors="Alice",
                    year="2020"
                )
            ]
        )
    )
    
    # Define standard invoke behavior (for planner and synthesizer)
    mock_chat_model.invoke.side_effect = lambda prompt_val, *args, **kw: fake.invoke(prompt_val)
    mock_chat_model.side_effect = lambda prompt_val, *args, **kw: fake.invoke(prompt_val)
    
    # Patch get_llm to return our mock_chat_model
    monkeypatch.setattr("src.agents.nodes.get_llm", lambda *a, **kw: mock_chat_model)
    monkeypatch.setattr("src.agents.graph.compile_graph", lambda *a, **kw: MagicMock())
    return fake
