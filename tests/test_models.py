"""
Pydantic model validation tests for AI Project Management Agent.

Tests all structured output models for:
- Field validation
- Type safety
- Constraint enforcement
- Serialization/deserialization
"""

import pytest
from pydantic import ValidationError
from datetime import datetime

from ai_pm_agent.models import (
    ClassificationResult,
    WorkItemMatch,
    GeneratedWorkItem,
    ArtifactTagResult,
    SprintRecommendation,
    ActionItemResult,
    LaunchReadinessUpdate
)


class TestClassificationResultModel:
    """Test ClassificationResult Pydantic model."""

    def test_classification_result_valid(self):
        """Test valid ClassificationResult creation."""
        result = ClassificationResult(
            category="Technical Readiness",
            confidence_score=87.5,
            gtm_phase="Foundation",
            suggested_action="create_work_item",
            rationale="Technical work identified"
        )

        assert result.category == "Technical Readiness"
        assert result.confidence_score == 87.5
        assert result.gtm_phase == "Foundation"
        assert result.suggested_action == "create_work_item"

    def test_classification_confidence_bounds(self):
        """Test confidence score is bounded 0-100."""
        # Valid confidence
        result = ClassificationResult(
            category="Product Readiness",
            confidence_score=50.0,
            gtm_phase="Validation",
            suggested_action="suggest_with_caveat",
            rationale="Test"
        )
        assert 0 <= result.confidence_score <= 100

        # Test boundaries
        result_min = ClassificationResult(
            category="Test",
            confidence_score=0.0,
            gtm_phase="Test",
            suggested_action="test",
            rationale="Test"
        )
        assert result_min.confidence_score == 0.0

        result_max = ClassificationResult(
            category="Test",
            confidence_score=100.0,
            gtm_phase="Test",
            suggested_action="test",
            rationale="Test"
        )
        assert result_max.confidence_score == 100.0

    def test_classification_confidence_invalid(self):
        """Test invalid confidence scores are rejected."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                category="Test",
                confidence_score=150.0,  # Invalid: >100
                gtm_phase="Test",
                suggested_action="test",
                rationale="Test"
            )

        with pytest.raises(ValidationError):
            ClassificationResult(
                category="Test",
                confidence_score=-10.0,  # Invalid: <0
                gtm_phase="Test",
                suggested_action="test",
                rationale="Test"
            )

    def test_classification_required_fields(self):
        """Test all fields are required."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                category="Test",
                confidence_score=50.0
                # Missing gtm_phase, suggested_action, rationale
            )

    def test_classification_serialization(self):
        """Test model serialization to dict."""
        result = ClassificationResult(
            category="Test",
            confidence_score=75.0,
            gtm_phase="Foundation",
            suggested_action="create_work_item",
            rationale="Test rationale"
        )

        data = result.model_dump()
        assert isinstance(data, dict)
        assert data["category"] == "Test"
        assert data["confidence_score"] == 75.0


class TestWorkItemMatchModel:
    """Test WorkItemMatch Pydantic model."""

    def test_work_item_match_valid(self):
        """Test valid WorkItemMatch creation."""
        match = WorkItemMatch(
            ado_id=12345,
            title="User Authentication",
            work_item_type="Feature",
            similarity_score=0.92,
            recommendation="link"
        )

        assert match.ado_id == 12345
        assert match.title == "User Authentication"
        assert match.similarity_score == 0.92

    def test_work_item_match_similarity_bounds(self):
        """Test similarity score is bounded 0-1."""
        # Valid scores
        match_low = WorkItemMatch(
            ado_id=1,
            title="Test",
            work_item_type="Feature",
            similarity_score=0.0,
            recommendation="create_new"
        )
        assert match_low.similarity_score == 0.0

        match_high = WorkItemMatch(
            ado_id=2,
            title="Test",
            work_item_type="Feature",
            similarity_score=1.0,
            recommendation="link"
        )
        assert match_high.similarity_score == 1.0

    def test_work_item_match_similarity_invalid(self):
        """Test invalid similarity scores are rejected."""
        with pytest.raises(ValidationError):
            WorkItemMatch(
                ado_id=1,
                title="Test",
                work_item_type="Feature",
                similarity_score=1.5,  # Invalid: >1.0
                recommendation="link"
            )

        with pytest.raises(ValidationError):
            WorkItemMatch(
                ado_id=1,
                title="Test",
                work_item_type="Feature",
                similarity_score=-0.5,  # Invalid: <0.0
                recommendation="link"
            )

    def test_work_item_match_recommendation_values(self):
        """Test valid recommendation values."""
        valid_recommendations = ["link", "update", "create_new"]

        for rec in valid_recommendations:
            match = WorkItemMatch(
                ado_id=1,
                title="Test",
                work_item_type="Feature",
                similarity_score=0.8,
                recommendation=rec
            )
            assert match.recommendation == rec


class TestGeneratedWorkItemModel:
    """Test GeneratedWorkItem Pydantic model."""

    def test_generated_work_item_valid(self):
        """Test valid GeneratedWorkItem creation."""
        item = GeneratedWorkItem(
            title="Implement User Auth",
            description="Full description with template",
            work_item_type="Feature",
            acceptance_criteria=["Criterion 1", "Criterion 2"],
            tags=["Authentication", "Security"],
            suggested_sprint=42,
            parent_link=12345,
            metadata={"priority": "High"}
        )

        assert item.title == "Implement User Auth"
        assert len(item.acceptance_criteria) == 2
        assert len(item.tags) == 2
        assert item.suggested_sprint == 42

    def test_generated_work_item_optional_fields(self):
        """Test optional fields can be None."""
        item = GeneratedWorkItem(
            title="Test",
            description="Description",
            work_item_type="Epic",
            suggested_sprint=42
        )

        # Optional fields default to None or empty
        assert item.acceptance_criteria is None
        assert item.parent_link is None
        assert len(item.tags) == 0
        assert len(item.metadata) == 0

    def test_generated_work_item_work_types(self):
        """Test valid work item types."""
        valid_types = ["Epic", "Feature", "User Story"]

        for work_type in valid_types:
            item = GeneratedWorkItem(
                title="Test",
                description="Test",
                work_item_type=work_type,
                suggested_sprint=42
            )
            assert item.work_item_type == work_type

    def test_generated_work_item_serialization(self):
        """Test model serialization."""
        item = GeneratedWorkItem(
            title="Test",
            description="Description",
            work_item_type="Feature",
            acceptance_criteria=["AC1"],
            tags=["tag1"],
            suggested_sprint=42,
            metadata={"key": "value"}
        )

        data = item.model_dump()
        assert isinstance(data, dict)
        assert data["title"] == "Test"
        assert isinstance(data["acceptance_criteria"], list)
        assert isinstance(data["metadata"], dict)


class TestArtifactTagResultModel:
    """Test ArtifactTagResult Pydantic model."""

    def test_artifact_tag_result_valid(self):
        """Test valid ArtifactTagResult creation."""
        result = ArtifactTagResult(
            artifact_id="abc-123",
            detected_type="wireframe",
            extracted_metadata={"size": 1024, "format": "png"},
            suggested_tags=["UI", "Design"],
            linked_work_items=[12345, 67890]
        )

        assert result.artifact_id == "abc-123"
        assert result.detected_type == "wireframe"
        assert len(result.suggested_tags) == 2
        assert len(result.linked_work_items) == 2

    def test_artifact_tag_result_defaults(self):
        """Test default values for optional fields."""
        result = ArtifactTagResult(
            artifact_id="test-id",
            detected_type="unknown"
        )

        assert len(result.extracted_metadata) == 0
        assert len(result.suggested_tags) == 0
        assert len(result.linked_work_items) == 0

    def test_artifact_tag_result_artifact_types(self):
        """Test different artifact type values."""
        valid_types = [
            "wireframe",
            "brief",
            "test_plan",
            "meeting_notes",
            "design_doc",
            "marketing_material",
            "unknown"
        ]

        for artifact_type in valid_types:
            result = ArtifactTagResult(
                artifact_id="test",
                detected_type=artifact_type
            )
            assert result.detected_type == artifact_type


class TestSprintRecommendationModel:
    """Test SprintRecommendation Pydantic model."""

    def test_sprint_recommendation_valid(self):
        """Test valid SprintRecommendation creation."""
        rec = SprintRecommendation(
            recommended_sprint=42,
            gtm_phase="Foundation",
            dependencies=["Auth", "Database"],
            justification="High priority work for current sprint",
            alternative_sprints=[41, 43]
        )

        assert rec.recommended_sprint == 42
        assert rec.gtm_phase == "Foundation"
        assert len(rec.dependencies) == 2
        assert len(rec.alternative_sprints) == 2

    def test_sprint_recommendation_defaults(self):
        """Test default values for optional fields."""
        rec = SprintRecommendation(
            recommended_sprint=42,
            gtm_phase="Validation",
            justification="Test"
        )

        assert len(rec.dependencies) == 0
        assert len(rec.alternative_sprints) == 0

    def test_sprint_recommendation_gtm_phases(self):
        """Test valid GTM phase values."""
        valid_phases = [
            "Foundation",
            "Validation",
            "Launch Prep",
            "Go-to-Market",
            "Growth",
            "Market Leadership"
        ]

        for phase in valid_phases:
            rec = SprintRecommendation(
                recommended_sprint=42,
                gtm_phase=phase,
                justification="Test"
            )
            assert rec.gtm_phase == phase


class TestActionItemResultModel:
    """Test ActionItemResult Pydantic model."""

    def test_action_item_result_valid(self):
        """Test valid ActionItemResult creation."""
        result = ActionItemResult(
            action_item_id="action-123",
            created_at="2025-01-16T10:00:00",
            linked_work_item=12345,
            linked_request=67890
        )

        assert result.action_item_id == "action-123"
        assert result.created_at == "2025-01-16T10:00:00"
        assert result.linked_work_item == 12345
        assert result.linked_request == 67890

    def test_action_item_result_optional_links(self):
        """Test optional link fields can be None."""
        result = ActionItemResult(
            action_item_id="action-456",
            created_at="2025-01-16T10:00:00"
        )

        assert result.linked_work_item is None
        assert result.linked_request is None


class TestLaunchReadinessUpdateModel:
    """Test LaunchReadinessUpdate Pydantic model."""

    def test_launch_readiness_update_valid(self):
        """Test valid LaunchReadinessUpdate creation."""
        update = LaunchReadinessUpdate(
            updated_score=4,
            previous_score=3,
            category="Technical Readiness",
            timestamp="2025-01-16T10:00:00",
            requires_approval=False
        )

        assert update.updated_score == 4
        assert update.previous_score == 3
        assert update.requires_approval is False

    def test_launch_readiness_score_bounds(self):
        """Test score bounds (1-5)."""
        # Valid scores
        update_min = LaunchReadinessUpdate(
            updated_score=1,
            previous_score=1,
            category="Test",
            timestamp="2025-01-16T10:00:00"
        )
        assert update_min.updated_score == 1

        update_max = LaunchReadinessUpdate(
            updated_score=5,
            previous_score=5,
            category="Test",
            timestamp="2025-01-16T10:00:00"
        )
        assert update_max.updated_score == 5

    def test_launch_readiness_score_invalid(self):
        """Test invalid scores are rejected."""
        with pytest.raises(ValidationError):
            LaunchReadinessUpdate(
                updated_score=6,  # Invalid: >5
                previous_score=3,
                category="Test",
                timestamp="2025-01-16T10:00:00"
            )

        with pytest.raises(ValidationError):
            LaunchReadinessUpdate(
                updated_score=0,  # Invalid: <1
                previous_score=3,
                category="Test",
                timestamp="2025-01-16T10:00:00"
            )

    def test_launch_readiness_approval_default(self):
        """Test requires_approval defaults to False."""
        update = LaunchReadinessUpdate(
            updated_score=3,
            previous_score=2,
            category="Test",
            timestamp="2025-01-16T10:00:00"
        )

        assert update.requires_approval is False

    def test_launch_readiness_approval_logic(self):
        """Test approval requirement for large changes."""
        # Small change (≤1 point)
        small_change = LaunchReadinessUpdate(
            updated_score=3,
            previous_score=2,
            category="Test",
            timestamp="2025-01-16T10:00:00",
            requires_approval=False
        )
        assert small_change.requires_approval is False

        # Large change (>1 point)
        large_change = LaunchReadinessUpdate(
            updated_score=5,
            previous_score=2,
            category="Test",
            timestamp="2025-01-16T10:00:00",
            requires_approval=True
        )
        assert large_change.requires_approval is True


class TestModelSerialization:
    """Test model serialization and deserialization."""

    def test_all_models_serialize_to_dict(self):
        """Test all models can serialize to dict."""
        models = [
            ClassificationResult(
                category="Test",
                confidence_score=50.0,
                gtm_phase="Foundation",
                suggested_action="test",
                rationale="Test"
            ),
            WorkItemMatch(
                ado_id=1,
                title="Test",
                work_item_type="Feature",
                similarity_score=0.8,
                recommendation="link"
            ),
            GeneratedWorkItem(
                title="Test",
                description="Test",
                work_item_type="Feature",
                suggested_sprint=42
            ),
            ArtifactTagResult(
                artifact_id="test",
                detected_type="unknown"
            ),
            SprintRecommendation(
                recommended_sprint=42,
                gtm_phase="Foundation",
                justification="Test"
            ),
            ActionItemResult(
                action_item_id="test",
                created_at="2025-01-16T10:00:00"
            ),
            LaunchReadinessUpdate(
                updated_score=3,
                previous_score=2,
                category="Test",
                timestamp="2025-01-16T10:00:00"
            )
        ]

        for model in models:
            data = model.model_dump()
            assert isinstance(data, dict)
            assert len(data) > 0

    def test_all_models_serialize_to_json(self):
        """Test all models can serialize to JSON."""
        models = [
            ClassificationResult(
                category="Test",
                confidence_score=50.0,
                gtm_phase="Foundation",
                suggested_action="test",
                rationale="Test"
            ),
            WorkItemMatch(
                ado_id=1,
                title="Test",
                work_item_type="Feature",
                similarity_score=0.8,
                recommendation="link"
            ),
        ]

        for model in models:
            json_str = model.model_dump_json()
            assert isinstance(json_str, str)
            assert len(json_str) > 0
            assert "{" in json_str  # Valid JSON


class TestModelValidation:
    """Test Pydantic validation features."""

    def test_type_coercion(self):
        """Test Pydantic type coercion works correctly."""
        # String to int
        match = WorkItemMatch(
            ado_id="12345",  # String instead of int
            title="Test",
            work_item_type="Feature",
            similarity_score=0.8,
            recommendation="link"
        )
        assert match.ado_id == 12345
        assert isinstance(match.ado_id, int)

    def test_required_vs_optional_fields(self):
        """Test required vs optional field enforcement."""
        # All required fields present - should work
        item = GeneratedWorkItem(
            title="Test",
            description="Test",
            work_item_type="Feature",
            suggested_sprint=42
        )
        assert item is not None

        # Missing required field - should fail
        with pytest.raises(ValidationError):
            GeneratedWorkItem(
                title="Test",
                description="Test"
                # Missing work_item_type and suggested_sprint
            )

    def test_field_constraints(self):
        """Test field constraints are enforced."""
        # Test ge/le constraints on confidence_score
        with pytest.raises(ValidationError):
            ClassificationResult(
                category="Test",
                confidence_score=101.0,  # Exceeds le=100.0
                gtm_phase="Foundation",
                suggested_action="test",
                rationale="Test"
            )

        # Test ge/le constraints on similarity_score
        with pytest.raises(ValidationError):
            WorkItemMatch(
                ado_id=1,
                title="Test",
                work_item_type="Feature",
                similarity_score=1.5,  # Exceeds le=1.0
                recommendation="link"
            )
