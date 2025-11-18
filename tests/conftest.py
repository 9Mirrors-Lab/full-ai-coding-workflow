"""
Test configuration and fixtures for AI Project Management Agent.

Provides reusable fixtures for testing with mocked dependencies,
TestModel integration, and common test data.
"""

import pytest
import contextlib
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
            "similarity": 0.92,
            "similarity_score": 0.92  # For comprehensive search ranking
        },
        {
            "id": 2,
            "ado_id": 12346,
            "title": "OAuth2 authentication feature",
            "work_item_type": "Feature",
            "similarity": 0.88,
            "similarity_score": 0.88
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
    # Use Mock (not AsyncMock) for sync embeddings.create
    mock_embeddings.create = Mock(return_value=mock_embedding_response)

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


# ===== Phase 2 Fixtures: Knowledge Graph & Hybrid Search =====


@pytest.fixture
def mock_graphiti_client():
    """Create a mocked GraphitiClient for testing."""
    # Create pure mock without importing GraphitiClient (avoids graphiti_core dependency)
    mock_client = Mock()

    # Mock get_related_entities method (what tools actually call)
    async def mock_get_related_entities(entity_name, relationship_types=None, depth=2):
        """Mock the get_related_entities method."""
        return {
            "entities": [
                {"name": "Epic #1234", "type": "Epic", "ado_id": 1234},
                {"name": "Feature #5678", "type": "Feature", "ado_id": 5678},
                {"name": "User Story #9012", "type": "User Story", "ado_id": 9012}
            ],
            "related_facts": [
                {
                    "fact": "Epic #1234 CONTAINS Feature #5678",
                    "uuid": "graph-uuid-1",
                    "valid_at": datetime(2025, 1, 15).isoformat(),
                    "invalid_at": None,
                    "created_at": datetime(2025, 1, 10).isoformat()
                },
                {
                    "fact": "Feature #5678 IMPLEMENTS User Story #9012",
                    "uuid": "graph-uuid-2",
                    "valid_at": datetime(2025, 1, 16).isoformat(),
                    "invalid_at": None,
                    "created_at": datetime(2025, 1, 11).isoformat()
                }
            ],
            "search_method": "graphiti_semantic_search"
        }

    mock_client.get_related_entities = AsyncMock(side_effect=mock_get_related_entities)

    # Mock search method for timeline queries
    async def mock_search(query, depth=2):
        """Mock the search method for timeline queries."""
        return [
            {
                "fact": f"Work item 1234 updated at {datetime(2025, 1, 15).isoformat()}",
                "uuid": "timeline-1",
                "valid_at": datetime(2025, 1, 15).isoformat(),
                "created_at": datetime(2025, 1, 15).isoformat()
            },
            {
                "fact": f"Work item 1234 status changed at {datetime(2025, 1, 16).isoformat()}",
                "uuid": "timeline-2",
                "valid_at": datetime(2025, 1, 16).isoformat(),
                "created_at": datetime(2025, 1, 16).isoformat()
            }
        ]

    mock_client.search = AsyncMock(side_effect=mock_search)

    # Mock get_entity_timeline method for timeline queries
    async def mock_get_entity_timeline(entity_name, start_date=None, end_date=None):
        """Mock the get_entity_timeline method."""
        return [
            {
                "fact": "Work item 1234 created",
                "uuid": "timeline-1",
                "timestamp": datetime(2025, 1, 10, 10, 0).isoformat(),
                "valid_at": datetime(2025, 1, 10, 10, 0).isoformat(),
                "source": "ADO",
                "confidence": 1.0
            },
            {
                "fact": "Work item 1234 status changed to Active",
                "uuid": "timeline-2",
                "timestamp": datetime(2025, 1, 15, 14, 30).isoformat(),
                "valid_at": datetime(2025, 1, 15, 14, 30).isoformat(),
                "source": "ADO",
                "confidence": 1.0
            },
            {
                "fact": "Work item 1234 assigned to User A",
                "uuid": "timeline-3",
                "timestamp": datetime(2025, 1, 16, 9, 15).isoformat(),
                "valid_at": datetime(2025, 1, 16, 9, 15).isoformat(),
                "source": "ADO",
                "confidence": 1.0
            }
        ]

    mock_client.get_entity_timeline = AsyncMock(side_effect=mock_get_entity_timeline)

    # Use AsyncMock for close method
    mock_client.close = AsyncMock()

    return mock_client


@pytest.fixture
def mock_database_pool():
    """Create a mocked DatabasePool for testing."""
    # Create pure mock without importing DatabasePool (avoids asyncpg dependency during tests)
    mock_pool = Mock()

    # Mock data for hybrid search
    async def mock_fetch(*args, **kwargs):
        """Mock fetchrow/fetch for hybrid search."""
        return [
            {
                "work_item_id": 1,
                "ado_id": 12345,
                "title": "Fix API timeout bug",
                "description": "Timeout errors in payment processing API",
                "work_item_type": "Bug",
                "combined_score": 0.89,
                "vector_similarity": 0.85,
                "text_similarity": 0.92
            },
            {
                "work_item_id": 2,
                "ado_id": 12346,
                "title": "Implement OAuth2 authentication",
                "description": "Add OAuth2 flow for secure authentication",
                "work_item_type": "Feature",
                "combined_score": 0.78,
                "vector_similarity": 0.80,
                "text_similarity": 0.75
            }
        ]

    # Mock connection that will be yielded by acquire()
    mock_connection = AsyncMock()
    mock_connection.fetch = AsyncMock(side_effect=mock_fetch)

    # Create an async context manager for database_pool.acquire()
    @contextlib.asynccontextmanager
    async def mock_acquire():
        """Mock async context manager for acquire()."""
        yield mock_connection

    # Set the acquire method on the mock_pool to return our context manager
    mock_pool.acquire = mock_acquire

    # Mock close method
    mock_pool.close = AsyncMock()

    return mock_pool


@pytest.fixture
def test_dependencies_phase2(mock_supabase_client, mock_openai_client,
                             mock_graphiti_client, mock_database_pool):
    """Create test dependencies with Phase 2 clients included."""
    return AgentDependencies(
        # Phase 1 dependencies
        supabase_client=mock_supabase_client,
        openai_client=mock_openai_client,
        current_sprint_info=get_current_sprint_info(),
        work_item_templates=get_work_item_templates(),
        launch_readiness_categories=get_launch_readiness_categories(),
        gtm_phase_definitions=get_gtm_phase_definitions(),
        artifact_registry_path="/tmp/test_artifacts",
        ado_project_name="TestProject",
        session_id="test-session-123",
        debug=True,

        # Phase 2 dependencies
        graphiti_client=mock_graphiti_client,
        database_pool=mock_database_pool
    )


@pytest.fixture
def sample_graph_relationships():
    """Provide sample graph relationship data for testing."""
    return [
        {
            "source_work_item": "Epic #1234",
            "relationship_type": "CONTAINS",
            "target_work_item": "Feature #5678",
            "valid_from": "2025-01-15T00:00:00Z",
            "valid_until": None
        },
        {
            "source_work_item": "Feature #5678",
            "relationship_type": "IMPLEMENTS",
            "target_work_item": "User Story #9012",
            "valid_from": "2025-01-16T00:00:00Z",
            "valid_until": None
        },
        {
            "source_work_item": "User Story #9012",
            "relationship_type": "BLOCKS",
            "target_work_item": "User Story #9013",
            "valid_from": "2025-01-17T00:00:00Z",
            "valid_until": None
        }
    ]


@pytest.fixture
def sample_timeline_events():
    """Provide sample timeline event data for testing."""
    return [
        {
            "fact": "WorkItem#1234 status changed from New to In Progress",
            "timestamp": "2025-01-10T09:00:00Z",
            "event_type": "status_change",
            "metadata": {
                "previous_value": "New",
                "new_value": "In Progress",
                "user": "ryan@codeveloper.com"
            }
        },
        {
            "fact": "WorkItem#1234 assigned to ryan@codeveloper.com",
            "timestamp": "2025-01-10T09:05:00Z",
            "event_type": "assignment_update",
            "metadata": {
                "previous_value": None,
                "new_value": "ryan@codeveloper.com"
            }
        },
        {
            "fact": "WorkItem#1234 moved to Sprint 42",
            "timestamp": "2025-01-11T10:00:00Z",
            "event_type": "sprint_movement",
            "metadata": {
                "previous_value": "Backlog",
                "new_value": "Sprint 42"
            }
        }
    ]


@pytest.fixture
def sample_hybrid_search_results():
    """Provide sample hybrid search results for testing."""
    return [
        {
            "work_item_id": 1,
            "ado_id": 12345,
            "title": "Fix API timeout bug",
            "description": "Timeout errors in payment processing API",
            "work_item_type": "Bug",
            "combined_score": 0.89,
            "vector_similarity": 0.85,
            "text_similarity": 0.92
        },
        {
            "work_item_id": 2,
            "ado_id": 12346,
            "title": "Implement OAuth2 authentication",
            "description": "Add OAuth2 flow for secure authentication",
            "work_item_type": "Feature",
            "combined_score": 0.78,
            "vector_similarity": 0.80,
            "text_similarity": 0.75
        }
    ]
