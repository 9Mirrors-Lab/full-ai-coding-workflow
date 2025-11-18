"""
AI Project Management Agent - Intelligent project management automation.
"""

from .agent import pm_agent, run_pm_agent, create_agent_with_deps
from .dependencies import AgentDependencies
from .models import (
    ClassificationResult,
    WorkItemMatch,
    GeneratedWorkItem,
    ArtifactTagResult,
    SprintRecommendation,
    ActionItemResult,
    LaunchReadinessUpdate
)

__version__ = "1.0.0"
__all__ = [
    "pm_agent",
    "run_pm_agent",
    "create_agent_with_deps",
    "AgentDependencies",
    "ClassificationResult",
    "WorkItemMatch",
    "GeneratedWorkItem",
    "ArtifactTagResult",
    "SprintRecommendation",
    "ActionItemResult",
    "LaunchReadinessUpdate",
]
