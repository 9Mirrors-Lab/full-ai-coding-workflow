"""
Pydantic models for structured outputs.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class ClassificationResult(BaseModel):
    """Result from classify_input tool."""
    category: str = Field(..., description="Launch readiness category")
    confidence_score: float = Field(..., ge=0.0, le=100.0, description="Confidence percentage")
    gtm_phase: str = Field(..., description="GTM phase assignment")
    suggested_action: str = Field(..., description="Recommended next action")
    rationale: str = Field(..., description="Explanation of classification")


class WorkItemMatch(BaseModel):
    """A single work item match from semantic search."""
    ado_id: int = Field(..., description="Azure DevOps work item ID")
    title: str = Field(..., description="Work item title")
    work_item_type: str = Field(..., description="Type: Epic, Feature, or User Story")
    similarity_score: float = Field(..., ge=0.0, le=1.0, description="Cosine similarity score")
    recommendation: str = Field(..., description="Action recommendation: link, update, or create_new")


class GeneratedWorkItem(BaseModel):
    """A generated work item ready for approval."""
    title: str = Field(..., description="Work item title")
    description: str = Field(..., description="Full description following template")
    work_item_type: str = Field(..., description="Epic, Feature, or User Story")
    acceptance_criteria: Optional[List[str]] = Field(default=None, description="Acceptance criteria list")
    tags: List[str] = Field(default_factory=list, description="Suggested tags")
    suggested_sprint: int = Field(..., description="Recommended sprint number")
    parent_link: Optional[int] = Field(default=None, description="Parent work item ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ArtifactTagResult(BaseModel):
    """Result from tagging an artifact."""
    artifact_id: str = Field(..., description="UUID for artifact record")
    detected_type: str = Field(..., description="Detected artifact type")
    extracted_metadata: Dict[str, Any] = Field(default_factory=dict, description="File metadata")
    suggested_tags: List[str] = Field(default_factory=list, description="Auto-generated tags")
    linked_work_items: List[int] = Field(default_factory=list, description="Linked ADO work item IDs")


class SprintRecommendation(BaseModel):
    """Sprint placement recommendation."""
    recommended_sprint: int = Field(..., description="Recommended sprint number")
    gtm_phase: str = Field(..., description="GTM phase alignment")
    dependencies: List[str] = Field(default_factory=list, description="Dependency work items")
    justification: str = Field(..., description="Reasoning for recommendation")
    alternative_sprints: List[int] = Field(default_factory=list, description="Alternative options")


class ActionItemResult(BaseModel):
    """Result from creating an action item."""
    action_item_id: str = Field(..., description="UUID of created action item")
    created_at: str = Field(..., description="ISO timestamp")
    linked_work_item: Optional[int] = Field(default=None, description="Linked work item ID")
    linked_request: Optional[int] = Field(default=None, description="Linked request ID")


class LaunchReadinessUpdate(BaseModel):
    """Result from updating launch readiness score."""
    updated_score: int = Field(..., ge=1, le=5, description="New score (1-5)")
    previous_score: int = Field(..., ge=1, le=5, description="Previous score (1-5)")
    category: str = Field(..., description="Category name")
    timestamp: str = Field(..., description="ISO timestamp of update")
    requires_approval: bool = Field(default=False, description="True if change >1 point")
