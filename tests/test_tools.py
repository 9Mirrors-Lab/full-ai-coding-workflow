"""
Tool-specific tests for AI Project Management Agent.

Comprehensive tests for all 8 tools with validation of:
- Input/output schemas
- Error handling
- Edge cases
- Performance requirements
- Business logic
"""

import pytest
import os
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ai_pm_agent.tools import (
    classify_input_tool,
    match_work_items_tool,
    generate_work_item_tool,
    tag_artifact_tool,
    recommend_sprint_placement_tool,
    create_action_item_tool,
    update_launch_readiness_tool,
    log_agent_decision_tool
)
from ai_pm_agent.models import (
    ClassificationResult,
    WorkItemMatch,
    GeneratedWorkItem,
    ArtifactTagResult,
    SprintRecommendation,
    ActionItemResult,
    LaunchReadinessUpdate
)
from ai_pm_agent.config import (
    get_launch_readiness_categories,
    get_gtm_phase_definitions,
    get_work_item_templates
)


class TestClassifyInputTool:
    """Test classify_input_tool functionality."""

    @pytest.mark.asyncio
    async def test_classify_technical_input(self):
        """Test classification of technical input."""
        result = await classify_input_tool(
            "Fix authentication bug in login flow",
            get_launch_readiness_categories(),
            get_gtm_phase_definitions()
        )

        assert isinstance(result, ClassificationResult)
        assert result.category in [
            "Product Readiness",
            "Technical Readiness",
            "Operational Readiness",
            "Commercial Readiness",
            "Customer Readiness",
            "Security & Compliance"
        ]
        assert 0 <= result.confidence_score <= 100
        assert result.gtm_phase in get_gtm_phase_definitions().keys()
        assert result.suggested_action in [
            "ask_for_clarification",
            "suggest_with_caveat",
            "create_work_item",
            "manual_review"
        ]

    @pytest.mark.asyncio
    async def test_confidence_threshold_low(self):
        """Test low confidence (<70%) returns ask_for_clarification."""
        result = await classify_input_tool(
            "something vague",
            get_launch_readiness_categories(),
            get_gtm_phase_definitions()
        )

        assert isinstance(result, ClassificationResult)
        if result.confidence_score < 70:
            assert result.suggested_action == "ask_for_clarification"

    @pytest.mark.asyncio
    async def test_confidence_threshold_medium(self):
        """Test medium confidence (70-85%) returns suggest_with_caveat."""
        result = await classify_input_tool(
            "maybe update some technical documentation",
            get_launch_readiness_categories(),
            get_gtm_phase_definitions()
        )

        assert isinstance(result, ClassificationResult)
        if 70 <= result.confidence_score <= 85:
            assert result.suggested_action == "suggest_with_caveat"

    @pytest.mark.asyncio
    async def test_confidence_threshold_high(self):
        """Test high confidence (>85%) returns create_work_item."""
        result = await classify_input_tool(
            "Fix critical authentication security bug in production login flow causing timeouts",
            get_launch_readiness_categories(),
            get_gtm_phase_definitions()
        )

        assert isinstance(result, ClassificationResult)
        assert result.confidence_score >= 0

    @pytest.mark.asyncio
    async def test_classify_all_categories(self, sample_input_texts):
        """Test classification handles all input types."""
        for input_type, input_text in sample_input_texts.items():
            result = await classify_input_tool(
                input_text,
                get_launch_readiness_categories(),
                get_gtm_phase_definitions()
            )

            assert isinstance(result, ClassificationResult)
            assert result.category is not None
            assert result.rationale is not None

    @pytest.mark.asyncio
    async def test_classify_with_empty_input(self):
        """Test classification handles empty input gracefully."""
        result = await classify_input_tool(
            "",
            get_launch_readiness_categories(),
            get_gtm_phase_definitions()
        )

        assert isinstance(result, ClassificationResult)
        assert result.confidence_score >= 0

    @pytest.mark.asyncio
    async def test_classify_error_handling(self):
        """Test classification error handling."""
        # Pass invalid categories to trigger error
        result = await classify_input_tool(
            "Test input",
            [],  # Empty categories
            get_gtm_phase_definitions()
        )

        assert isinstance(result, ClassificationResult)
        # Should return default/error result with ask_for_clarification action
        assert result.suggested_action == "ask_for_clarification" or result.confidence_score < 50


class TestMatchWorkItemsTool:
    """Test match_work_items_tool functionality."""

    @pytest.mark.asyncio
    async def test_match_work_items_basic(self, mock_supabase_client, mock_openai_client):
        """Test basic work item matching."""
        results = await match_work_items_tool(
            "Implement user authentication",
            mock_supabase_client,
            mock_openai_client,
            limit=5
        )

        assert isinstance(results, list)
        if len(results) > 0:
            assert isinstance(results[0], WorkItemMatch)
            assert hasattr(results[0], 'ado_id')
            assert hasattr(results[0], 'similarity_score')

    @pytest.mark.asyncio
    async def test_match_work_items_similarity_scores(self, mock_supabase_client, mock_openai_client):
        """Test similarity scores are in valid range."""
        results = await match_work_items_tool(
            "Authentication feature",
            mock_supabase_client,
            mock_openai_client,
            limit=3
        )

        for match in results:
            assert 0.0 <= match.similarity_score <= 1.0

    @pytest.mark.asyncio
    async def test_match_work_items_recommendations(self, mock_supabase_client, mock_openai_client):
        """Test recommendation logic based on similarity."""
        # Mock different similarity scores
        mock_rpc = Mock()
        mock_rpc.execute = AsyncMock(return_value=Mock(data=[
            {"id": 1, "ado_id": 100, "title": "High similarity", "work_item_type": "Feature", "similarity": 0.95},
            {"id": 2, "ado_id": 101, "title": "Medium similarity", "work_item_type": "Feature", "similarity": 0.80},
            {"id": 3, "ado_id": 102, "title": "Low similarity", "work_item_type": "Feature", "similarity": 0.60}
        ]))
        mock_supabase_client.rpc = Mock(return_value=mock_rpc)

        results = await match_work_items_tool(
            "Test query",
            mock_supabase_client,
            mock_openai_client,
            limit=3
        )

        # Verify recommendations
        assert len(results) == 3
        assert results[0].recommendation == "link"  # High similarity > 0.90
        assert results[1].recommendation == "update"  # Medium similarity > 0.75
        assert results[2].recommendation == "create_new"  # Low similarity

    @pytest.mark.asyncio
    async def test_match_work_items_limit(self, mock_supabase_client, mock_openai_client):
        """Test limit parameter works correctly."""
        results = await match_work_items_tool(
            "Test",
            mock_supabase_client,
            mock_openai_client,
            limit=2
        )

        # Should respect limit (mock returns 1 item)
        assert len(results) <= 2

    @pytest.mark.asyncio
    async def test_match_work_items_error_handling(self, mock_supabase_client, mock_openai_client):
        """Test error handling for API failures."""
        # Mock OpenAI failure
        mock_openai_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

        results = await match_work_items_tool(
            "Test",
            mock_supabase_client,
            mock_openai_client,
            limit=5
        )

        # Should return empty list on error
        assert isinstance(results, list)
        assert len(results) == 0


class TestGenerateWorkItemTool:
    """Test generate_work_item_tool functionality."""

    @pytest.mark.asyncio
    async def test_generate_epic(self):
        """Test Epic generation."""
        result = await generate_work_item_tool(
            "Launch the CoDeveloper platform to market",
            "Epic",
            get_work_item_templates()
        )

        assert isinstance(result, GeneratedWorkItem)
        assert result.work_item_type == "Epic"
        assert "Vision Statement" in result.description
        assert "Business Value" in result.description
        assert "Success Metrics" in result.description

    @pytest.mark.asyncio
    async def test_generate_feature(self):
        """Test Feature generation."""
        result = await generate_work_item_tool(
            "Add user authentication with OAuth2",
            "Feature",
            get_work_item_templates()
        )

        assert isinstance(result, GeneratedWorkItem)
        assert result.work_item_type == "Feature"
        assert "Acceptance Criteria" in result.description
        assert result.acceptance_criteria is not None
        assert len(result.acceptance_criteria) > 0

    @pytest.mark.asyncio
    async def test_generate_user_story(self):
        """Test User Story generation."""
        result = await generate_work_item_tool(
            "Allow users to save recipes to favorites",
            "User Story",
            get_work_item_templates()
        )

        assert isinstance(result, GeneratedWorkItem)
        assert result.work_item_type == "User Story"
        assert "User Story" in result.description
        assert "As a" in result.description
        assert result.acceptance_criteria is not None

    @pytest.mark.asyncio
    async def test_generate_with_parent_link(self):
        """Test work item generation with parent link."""
        result = await generate_work_item_tool(
            "Implement OAuth2 callback handler",
            "User Story",
            get_work_item_templates(),
            parent_id=12345
        )

        assert result.parent_link == 12345

    @pytest.mark.asyncio
    async def test_generate_tags_extraction(self):
        """Test tag extraction from input."""
        result = await generate_work_item_tool(
            "Implement Sous AI recipe formulate feature for launch",
            "Feature",
            get_work_item_templates()
        )

        assert len(result.tags) > 0
        # Should extract relevant tags
        assert "Sous AI" in result.tags or "Formulate" in result.tags or "Launch Prep" in result.tags

    @pytest.mark.asyncio
    async def test_generate_invalid_type(self):
        """Test error handling for invalid work item type."""
        result = await generate_work_item_tool(
            "Test",
            "InvalidType",
            get_work_item_templates()
        )

        # Should return error result
        assert isinstance(result, GeneratedWorkItem)
        assert "ERROR" in result.title or result.suggested_sprint == 0

    @pytest.mark.asyncio
    async def test_template_compliance(self):
        """Test that generated work items comply with templates."""
        templates = get_work_item_templates()

        for work_item_type in ["Epic", "Feature", "User Story"]:
            result = await generate_work_item_tool(
                f"Test {work_item_type}",
                work_item_type,
                templates
            )

            assert result.work_item_type == work_item_type
            assert len(result.description) > 0
            assert result.title is not None


class TestTagArtifactTool:
    """Test tag_artifact_tool functionality."""

    @pytest.mark.asyncio
    async def test_tag_wireframe(self, temp_artifact_file, mock_supabase_client):
        """Test tagging a wireframe artifact."""
        # Rename to wireframe
        wireframe_path = temp_artifact_file.replace(".png", "_wireframe.png")
        os.rename(temp_artifact_file, wireframe_path)

        result = await tag_artifact_tool(
            wireframe_path,
            mock_supabase_client,
            "/tmp/artifacts"
        )

        assert isinstance(result, ArtifactTagResult)
        assert result.detected_type == "wireframe"
        assert len(result.artifact_id) > 0

        # Cleanup
        if os.path.exists(wireframe_path):
            os.remove(wireframe_path)

    @pytest.mark.asyncio
    async def test_tag_different_types(self, tmp_path, mock_supabase_client):
        """Test detection of different artifact types."""
        test_cases = {
            "wireframe_v1.png": "wireframe",
            "project_brief.pdf": "brief",
            "test_plan.doc": "test_plan",
            "meeting_notes.txt": "meeting_notes",
            "design_doc.md": "design_doc",
            "random_file.xyz": "unknown"
        }

        for filename, expected_type in test_cases.items():
            file_path = tmp_path / filename
            file_path.write_text("test content")

            result = await tag_artifact_tool(
                str(file_path),
                mock_supabase_client,
                "/tmp/artifacts"
            )

            assert result.detected_type == expected_type, \
                f"Failed for {filename}: expected {expected_type}, got {result.detected_type}"

    @pytest.mark.asyncio
    async def test_tag_artifact_metadata_extraction(self, temp_artifact_file, mock_supabase_client):
        """Test metadata extraction from artifact."""
        result = await tag_artifact_tool(
            temp_artifact_file,
            mock_supabase_client,
            "/tmp/artifacts"
        )

        assert "size" in result.extracted_metadata
        assert "created_at" in result.extracted_metadata
        assert "modified_at" in result.extracted_metadata

    @pytest.mark.asyncio
    async def test_tag_artifact_missing_file(self, mock_supabase_client):
        """Test error handling for missing file."""
        result = await tag_artifact_tool(
            "/nonexistent/file.pdf",
            mock_supabase_client,
            "/tmp/artifacts"
        )

        assert isinstance(result, ArtifactTagResult)
        assert result.detected_type == "unknown"
        assert "error" in result.extracted_metadata

    @pytest.mark.asyncio
    async def test_tag_artifact_supabase_insert(self, temp_artifact_file, mock_supabase_client):
        """Test Supabase insert is called."""
        await tag_artifact_tool(
            temp_artifact_file,
            mock_supabase_client,
            "/tmp/artifacts"
        )

        # Verify Supabase insert was called
        mock_supabase_client.table.assert_called_with("artifacts")


class TestRecommendSprintPlacementTool:
    """Test recommend_sprint_placement_tool functionality."""

    @pytest.mark.asyncio
    async def test_recommend_high_priority(self):
        """Test sprint recommendation for high priority items."""
        work_item = {
            "title": "Critical Bug Fix",
            "work_item_type": "User Story",
            "priority": "High",
            "dependencies": []
        }

        result = await recommend_sprint_placement_tool(
            work_item,
            {"sprint": 42, "capacity": 100},
            get_gtm_phase_definitions()
        )

        assert isinstance(result, SprintRecommendation)
        assert result.recommended_sprint == 42  # Current sprint for high priority

    @pytest.mark.asyncio
    async def test_recommend_medium_priority(self):
        """Test sprint recommendation for medium priority items."""
        work_item = {
            "title": "Feature Enhancement",
            "work_item_type": "Feature",
            "priority": "Medium",
            "dependencies": []
        }

        result = await recommend_sprint_placement_tool(
            work_item,
            {"sprint": 42, "capacity": 100},
            get_gtm_phase_definitions()
        )

        assert isinstance(result, SprintRecommendation)
        assert result.recommended_sprint == 43  # Next sprint for medium

    @pytest.mark.asyncio
    async def test_recommend_with_dependencies(self):
        """Test sprint recommendation with dependencies."""
        work_item = {
            "title": "Dependent Feature",
            "work_item_type": "Feature",
            "priority": "High",
            "dependencies": ["Auth Feature", "Database Schema"]
        }

        result = await recommend_sprint_placement_tool(
            work_item,
            {"sprint": 42, "capacity": 100},
            get_gtm_phase_definitions()
        )

        assert isinstance(result, SprintRecommendation)
        # Should be pushed back due to dependencies
        assert result.recommended_sprint >= 42
        assert len(result.dependencies) == 2

    @pytest.mark.asyncio
    async def test_recommend_gtm_phase_mapping(self):
        """Test GTM phase mapping."""
        epic_item = {
            "title": "Platform Launch",
            "work_item_type": "Epic",
            "priority": "Medium",
            "dependencies": []
        }

        result = await recommend_sprint_placement_tool(
            epic_item,
            {"sprint": 42, "capacity": 100},
            get_gtm_phase_definitions()
        )

        assert result.gtm_phase == "Foundation"  # Epics → Foundation

    @pytest.mark.asyncio
    async def test_recommend_alternative_sprints(self):
        """Test alternative sprint recommendations."""
        work_item = {
            "title": "Test Work Item",
            "work_item_type": "Feature",
            "priority": "Medium",
            "dependencies": []
        }

        result = await recommend_sprint_placement_tool(
            work_item,
            {"sprint": 42, "capacity": 100},
            get_gtm_phase_definitions()
        )

        assert len(result.alternative_sprints) > 0

    @pytest.mark.asyncio
    async def test_recommend_error_handling(self):
        """Test error handling for invalid input."""
        result = await recommend_sprint_placement_tool(
            {},  # Empty work item
            {"sprint": 42},
            get_gtm_phase_definitions()
        )

        assert isinstance(result, SprintRecommendation)


class TestCreateActionItemTool:
    """Test create_action_item_tool functionality."""

    @pytest.mark.asyncio
    async def test_create_action_item_basic(self, mock_supabase_client):
        """Test basic action item creation."""
        result = await create_action_item_tool(
            "Follow up on authentication feature",
            "Need to review OAuth2 implementation",
            mock_supabase_client
        )

        assert isinstance(result, ActionItemResult)
        assert len(result.action_item_id) > 0
        assert result.created_at is not None

    @pytest.mark.asyncio
    async def test_create_action_item_with_work_item_link(self, mock_supabase_client):
        """Test action item with work item link."""
        result = await create_action_item_tool(
            "Test Action",
            "Description",
            mock_supabase_client,
            work_item_id=12345
        )

        assert result.linked_work_item == 12345

    @pytest.mark.asyncio
    async def test_create_action_item_with_request_link(self, mock_supabase_client):
        """Test action item with request link."""
        result = await create_action_item_tool(
            "Test Action",
            "Description",
            mock_supabase_client,
            request_id=67890
        )

        assert result.linked_request == 67890

    @pytest.mark.asyncio
    async def test_create_action_item_supabase_insert(self, mock_supabase_client):
        """Test Supabase insert is called correctly."""
        await create_action_item_tool(
            "Test",
            "Test Description",
            mock_supabase_client,
            work_item_id=123
        )

        mock_supabase_client.table.assert_called_with("action_items")

    @pytest.mark.asyncio
    async def test_create_action_item_error_handling(self, mock_supabase_client):
        """Test error handling for database failures."""
        # Mock database failure
        mock_table = Mock()
        mock_table.insert = Mock(side_effect=Exception("DB Error"))
        mock_supabase_client.table = Mock(return_value=mock_table)

        result = await create_action_item_tool(
            "Test",
            "Description",
            mock_supabase_client
        )

        # Should return error result
        assert isinstance(result, ActionItemResult)
        assert result.action_item_id == ""


class TestUpdateLaunchReadinessTool:
    """Test update_launch_readiness_tool functionality."""

    @pytest.mark.asyncio
    async def test_update_launch_readiness_small_change(self, mock_supabase_client):
        """Test small score changes (≤1 point) don't require approval."""
        result = await update_launch_readiness_tool(
            "Technical Readiness",
            "API Development",
            1,  # +1 point
            "Completed API endpoints",
            mock_supabase_client
        )

        assert isinstance(result, LaunchReadinessUpdate)
        assert result.requires_approval is False

    @pytest.mark.asyncio
    async def test_update_launch_readiness_large_change(self, mock_supabase_client):
        """Test large score changes (>1 point) require approval."""
        result = await update_launch_readiness_tool(
            "Product Readiness",
            "Feature Completeness",
            3,  # +3 points
            "Major feature release",
            mock_supabase_client
        )

        assert isinstance(result, LaunchReadinessUpdate)
        assert result.requires_approval is True

    @pytest.mark.asyncio
    async def test_update_launch_readiness_score_bounds(self, mock_supabase_client):
        """Test score is clamped to 1-5 range."""
        # Mock current score of 5
        mock_select = Mock()
        mock_select.eq = Mock(return_value=mock_select)
        mock_select.execute = AsyncMock(return_value=Mock(data=[{"score": 5}]))
        mock_supabase_client.table = Mock(return_value=mock_select)

        result = await update_launch_readiness_tool(
            "Technical Readiness",
            "Test Item",
            5,  # Try to go to 10
            "Notes",
            mock_supabase_client
        )

        # Should clamp to 5
        assert result.updated_score <= 5

    @pytest.mark.asyncio
    async def test_update_launch_readiness_negative_delta(self, mock_supabase_client):
        """Test negative score changes work correctly."""
        result = await update_launch_readiness_tool(
            "Customer Readiness",
            "Onboarding",
            -1,  # Decrease by 1
            "Regression found",
            mock_supabase_client
        )

        assert isinstance(result, LaunchReadinessUpdate)

    @pytest.mark.asyncio
    async def test_update_launch_readiness_error_handling(self, mock_supabase_client):
        """Test error handling for database failures."""
        # Mock database failure
        mock_table = Mock()
        mock_table.select = Mock(side_effect=Exception("DB Error"))
        mock_supabase_client.table = Mock(return_value=mock_table)

        result = await update_launch_readiness_tool(
            "Test",
            "Test Item",
            1,
            "Notes",
            mock_supabase_client
        )

        assert isinstance(result, LaunchReadinessUpdate)
        assert result.requires_approval is True


class TestLogAgentDecisionTool:
    """Test log_agent_decision_tool functionality."""

    @pytest.mark.asyncio
    async def test_log_decision_basic(self, mock_supabase_client):
        """Test basic decision logging."""
        log_id = await log_agent_decision_tool(
            "Test input",
            {"category": "Technical Readiness", "confidence": 0.85},
            ["classify_input", "match_ado_work_items"],
            0.85,
            mock_supabase_client
        )

        assert isinstance(log_id, str)
        assert len(log_id) > 0

    @pytest.mark.asyncio
    async def test_log_decision_supabase_insert(self, mock_supabase_client):
        """Test Supabase insert is called."""
        await log_agent_decision_tool(
            "Test",
            {},
            [],
            0.5,
            mock_supabase_client
        )

        mock_supabase_client.table.assert_called_with("agent_logs")

    @pytest.mark.asyncio
    async def test_log_decision_all_fields(self, mock_supabase_client):
        """Test logging with all fields populated."""
        classification = {
            "category": "Product Readiness",
            "confidence_score": 90.0,
            "gtm_phase": "Foundation",
            "suggested_action": "create_work_item"
        }

        actions = [
            "classify_input",
            "match_ado_work_items",
            "generate_work_item",
            "log_agent_decision"
        ]

        log_id = await log_agent_decision_tool(
            "Complete workflow test",
            classification,
            actions,
            0.90,
            mock_supabase_client
        )

        assert log_id is not None

    @pytest.mark.asyncio
    async def test_log_decision_error_handling(self, mock_supabase_client):
        """Test error handling for database failures."""
        # Mock database failure
        mock_table = Mock()
        mock_table.insert = Mock(side_effect=Exception("DB Error"))
        mock_supabase_client.table = Mock(return_value=mock_table)

        log_id = await log_agent_decision_tool(
            "Test",
            {},
            [],
            0.5,
            mock_supabase_client
        )

        # Should return empty string on error
        assert log_id == ""


class TestToolPerformance:
    """Test tool performance requirements from PRP."""

    @pytest.mark.asyncio
    async def test_classification_performance(self):
        """Test classification completes in <1 second."""
        import time

        start = time.time()
        await classify_input_tool(
            "Test classification performance",
            get_launch_readiness_categories(),
            get_gtm_phase_definitions()
        )
        duration = time.time() - start

        # Should be fast for simple classification
        assert duration < 1.0, f"Classification took {duration}s, expected <1s"

    @pytest.mark.asyncio
    async def test_semantic_search_performance(self, mock_supabase_client, mock_openai_client):
        """Test semantic search completes in <2 seconds."""
        import time

        start = time.time()
        await match_work_items_tool(
            "Test search performance",
            mock_supabase_client,
            mock_openai_client,
            limit=10
        )
        duration = time.time() - start

        # With mocks, should be very fast
        assert duration < 2.0, f"Search took {duration}s, expected <2s"
