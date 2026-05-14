"""Shared pytest fixtures for the FinRAG test suite.

Provides lightweight, reusable fixtures that multiple test modules
can share without duplicating boilerplate. All fixtures avoid heavy
I/O (no network, no disk writes) to keep tests fast.
"""

import os

import pytest

# ---------------------------------------------------------------------------
# Environment fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True, scope="session")
def set_required_env_vars():
    """Ensure required environment variables are set for all tests.

    EDGAR_USER_AGENT is required by Settings (pydantic-settings validates it).
    Sets a CI-appropriate value if not already present so tests run without
    a .env file.
    """
    env_defaults = {
        "EDGAR_USER_AGENT": "CI ci@finrag.internal",
        "FINRAG_INIT_PIPELINE": "false",
    }
    for key, value in env_defaults.items():
        os.environ.setdefault(key, value)


# ---------------------------------------------------------------------------
# Chunker fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def chunker():
    """A default SectionChunker instance shared across tests."""
    from finrag.ingestion.chunker import SectionChunker

    return SectionChunker(max_tokens=512, overlap_tokens=64)


# ---------------------------------------------------------------------------
# Golden dataset fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def golden_dataset():
    """Full golden dataset loaded once per session."""
    from finrag.evaluation.golden_dataset import load_golden_dataset

    return load_golden_dataset()


@pytest.fixture(scope="session")
def golden_items_numerical(golden_dataset):
    """Only numerical category items."""
    from finrag.evaluation.golden_dataset import Category

    return [i for i in golden_dataset if i.category == Category.NUMERICAL]


@pytest.fixture(scope="session")
def golden_items_out_of_scope(golden_dataset):
    """Only out-of-scope category items."""
    from finrag.evaluation.golden_dataset import Category

    return [i for i in golden_dataset if i.category == Category.OUT_OF_SCOPE]


# ---------------------------------------------------------------------------
# RAGAS evaluator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def ragas_evaluator():
    """A default RAGASEvaluator shared across tests."""
    from finrag.evaluation.ragas_evaluator import RAGASEvaluator

    return RAGASEvaluator(pass_threshold=0.7)


# ---------------------------------------------------------------------------
# Sample chunks (used in multiple test modules)
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_chunks():
    """Minimal reranked chunks suitable for testing generation nodes."""
    return [
        {
            "chunk_id": "aapl_rev_001",
            "text": "Apple reported total revenue of $383.3 billion for fiscal year 2024.",
            "metadata": {
                "ticker": "AAPL",
                "form_type": "10-K",
                "fiscal_period": "FY2024",
                "section_name": "Item 7 - MD&A",
            },
            "reranker_score": 0.95,
            "reranker_rank": 1,
        },
        {
            "chunk_id": "aapl_svc_002",
            "text": "Services revenue increased 13 percent year over year to $96.2 billion.",
            "metadata": {
                "ticker": "AAPL",
                "form_type": "10-K",
                "fiscal_period": "FY2024",
                "section_name": "Item 7 - MD&A",
            },
            "reranker_score": 0.82,
            "reranker_rank": 2,
        },
    ]
