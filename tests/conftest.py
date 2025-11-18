"""
Test configuration and fixtures for AI Project Management Agent.

Provides reusable fixtures for testing with mocked dependencies,
TestModel integration, and common test data.
"""

import pytest
from unittest.mock import Mock, AsyncMock, MagicMock
from typing import Dict, List, Any
from datetime import datetime

from pydantic_ai.models.test import TestModel
from ai_pm_agent.agent import pm_agent
from ai_pm_agent.dependencies import AgentDependencies
from ai_pm_agent.config import (
    get_work_item_templates,
    get_launch_readiness_categories,
    get_gtm_phase_definitions,
    get_current_sprint_info
)


@pytest.fixture
def mock_supabase_client():
    """Create a mocked Supabase client for testing."""
    mock_client = AsyncMock()

    # Mock table operations with AsyncMock
    mock_table = AsyncMock()
    mock_table.insert = Mock(return_value=mock_table)
    mock_table.select = Mock(return_value=mock_table)
    mock_table.eq = Mock(return_value=mock_table)
    mock_table.upsert = Mock(return_value=mock_table)

    # Mock execute with default successful response
    mock_execute = Mock()
    mock_execute.data = [{"id": "test-id", "score": 3}]
    mock_table.execute = AsyncMock(return_value=mock_execute)

    mock_client.table = Mock(return_value=mock_table)

    # Mock RPC function for work item matching
    mock_rpc = AsyncMock()
    mock_rpc.execute = AsyncMock(return_value=Mock(data=[
        {
            "id": 1,
            "ado_id": 12345,
            "title": "Sample Work Item",
            "work_item_type": "Feature",
            "similarity": 0.92
        }
    ]))
    mock_client.rpc = Mock(return_value=mock_rpc)

    return mock_client


@pytest.fixture
def mock_openai_client():
    """Create a mocked OpenAI client for testing."""
    mock_client = Mock()

    # Mock embeddings API
    mock_embedding_response = Mock()
    mock_embedding_response.data = [Mock(embedding=[0.1] * 1536)]

    mock_embeddings = Mock()
    mock_embeddings.create = AsyncMock(return_value=mock_embedding_response)

    mock_client.embeddings = mock_embeddings

    return mock_client


@pytest.fixture
def test_dependencies(mock_supabase_client, mock_openai_client):
    """Create test dependencies with all required configuration."""
    return AgentDependencies(
        supabase_client=mock_supabase_client,
        openai_client=mock_openai_client,
        current_sprint_info=get_current_sprint_info(),
        work_item_templates=get_work_item_templates(),
        launch_readiness_categories=get_launch_readiness_categories(),
        gtm_phase_definitions=get_gtm_phase_definitions(),
        artifact_registry_path="/tmp/test_artifacts",
        ado_project_name="TestProject",
        session_id="test-session-123",
        debug=True
    )


@pytest.fixture
def test_model():
    """Create a TestModel instance for agent testing."""
    return TestModel()


# Test data fixtures

@pytest.fixture
def sample_input_texts():
    """Provide sample input texts for testing classification."""
    return {
        "technical": "Fix authentication bug in login flow causing timeout errors",
        "product": "Add new recipe search feature with ingredient matching",
        "operational": "Update support documentation for new deployment process",
        "commercial": "Implement pricing tiers and subscription billing",
        "customer": "Create onboarding tutorial for new users",
        "security": "Conduct security audit for GDPR compliance",
        "vague": "Something needs to be done about that thing",
        "complex": "Jay mentioned users are confused by the Sous loading screen. Wants progress indicators and food-related animation."
    }


@pytest.fixture
def sample_work_items():
    """Provide sample work item data for testing."""
    return [
        {
            "title": "Implement User Authentication",
            "work_item_type": "Feature",
            "priority": "High",
            "dependencies": []
        },
        {
            "title": "Fix Recipe Search Bug",
            "work_item_type": "User Story",
            "priority": "Medium",
            "dependencies": ["Authentication Feature"]
        },
        {
            "title": "Platform Launch Preparation",
            "work_item_type": "Epic",
            "priority": "Critical",
            "dependencies": ["Authentication", "Search", "UI"]
        }
    ]


@pytest.fixture
def sample_classification_result():
    """Provide a sample classification result for testing."""
    return {
        "category": "Technical Readiness",
        "confidence_score": 87.5,
        "gtm_phase": "Foundation",
        "suggested_action": "create_work_item",
        "rationale": "Technical work detected with high confidence"
    }


@pytest.fixture
def sample_work_item_matches():
    """Provide sample work item matches for testing."""
    return [
        {
            "ado_id": 12345,
            "title": "Implement User Authentication",
            "work_item_type": "Feature",
            "similarity_score": 0.95,
            "recommendation": "link"
        },
        {
            "ado_id": 12346,
            "title": "Add Login Flow",
            "work_item_type": "User Story",
            "similarity_score": 0.82,
            "recommendation": "update"
        }
    ]


@pytest.fixture
def temp_artifact_file(tmp_path):
    """Create a temporary artifact file for testing."""
    artifact_path = tmp_path / "test_wireframe.png"
    artifact_path.write_text("Mock wireframe content")
    return str(artifact_path)


# Confidence threshold test data

@pytest.fixture
def confidence_test_cases():
    """Provide test cases for confidence threshold validation."""
    return {
        "low_confidence": {
            "input": "something vague and unclear",
            "expected_range": (0, 70),
            "expected_action": "ask_for_clarification"
        },
        "medium_confidence": {
            "input": "might need some technical improvements",
            "expected_range": (70, 85),
            "expected_action": "suggest_with_caveat"
        },
        "high_confidence": {
            "input": "implement user authentication feature with OAuth2 support",
            "expected_range": (85, 100),
            "expected_action": "create_work_item"
        }
    }


# Template validation fixtures

@pytest.fixture
def epic_template_fields():
    """Expected fields in Epic template."""
    return [
        "Vision Statement",
        "Business Value",
        "Success Metrics",
        "Tags"
    ]


@pytest.fixture
def feature_template_fields():
    """Expected fields in Feature template."""
    return [
        "Description",
        "Acceptance Criteria",
        "Technical Notes",
        "Tags"
    ]


@pytest.fixture
def user_story_template_fields():
    """Expected fields in User Story template."""
    return [
        "User Story",
        "As a",
        "I want",
        "So that",
        "Acceptance Criteria (Gherkin format)",
        "Given",
        "When",
        "Then",
        "Tags"
    ]


# Async helper functions

@pytest.fixture
def run_async():
    """Helper to run async functions in tests."""
    import asyncio

    def _run(coro):
        """Run coroutine and return result."""
        return asyncio.get_event_loop().run_until_complete(coro)

    return _run
