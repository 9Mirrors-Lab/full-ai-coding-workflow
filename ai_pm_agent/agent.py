"""
AI Project Management Agent - Main agent definition.
"""

import logging
from typing import Dict, List, Any, Optional
from pydantic_ai import Agent, RunContext

from .providers import get_llm_model
from .dependencies import AgentDependencies
from .prompts import SYSTEM_PROMPT
from .settings import settings
from .tools import (
    classify_input_tool,
    match_work_items_tool,
    generate_work_item_tool,
    tag_artifact_tool,
    recommend_sprint_placement_tool,
    create_action_item_tool,
    update_launch_readiness_tool,
    log_agent_decision_tool
)

logger = logging.getLogger(__name__)

# Initialize the agent
pm_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
    retries=settings.max_retries
)


# Register Tool 1: classify_input
@pm_agent.tool
async def classify_input(
    ctx: RunContext[AgentDependencies],
    input_text: str
) -> Dict[str, Any]:
    """
    Classify freeform input into launch readiness categories.

    Args:
        input_text: The freeform input to classify

    Returns:
        Classification result with category, confidence, GTM phase, and suggested action
    """
    result = await classify_input_tool(
        input_text,
        ctx.deps.launch_readiness_categories,
        ctx.deps.gtm_phase_definitions
    )
    return result.model_dump()


# Register Tool 2: match_ado_work_items
@pm_agent.tool
async def match_ado_work_items(
    ctx: RunContext[AgentDependencies],
    input_text: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search for existing ADO work items using semantic similarity.

    Args:
        input_text: Text to search for similar work items
        limit: Maximum number of matches to return (1-20)

    Returns:
        List of matching work items with similarity scores
    """
    limit = min(max(limit, 1), 20)
    results = await match_work_items_tool(
        input_text,
        ctx.deps.supabase_client,
        ctx.deps.openai_client,
        limit
    )
    return [r.model_dump() for r in results]


# Register Tool 3: generate_work_item
@pm_agent.tool
async def generate_work_item(
    ctx: RunContext[AgentDependencies],
    input_description: str,
    work_item_type: str,
    parent_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate ADO-ready work item using templates.

    Args:
        input_description: Description of the work to be done
        work_item_type: "Epic", "Feature", or "User Story"
        parent_id: Optional parent work item ID for linking

    Returns:
        Generated work item ready for approval (NEVER creates in ADO directly)
    """
    result = await generate_work_item_tool(
        input_description,
        work_item_type,
        ctx.deps.work_item_templates,
        parent_id
    )
    return result.model_dump()


# Register Tool 4: tag_artifact
@pm_agent.tool
async def tag_artifact(
    ctx: RunContext[AgentDependencies],
    file_path: str
) -> Dict[str, Any]:
    """
    Auto-tag uploaded documents and link to work items.

    Args:
        file_path: Path to the uploaded artifact file

    Returns:
        Artifact tagging result with metadata and suggested tags
    """
    result = await tag_artifact_tool(
        file_path,
        ctx.deps.supabase_client,
        ctx.deps.artifact_registry_path
    )
    return result.model_dump()


# Register Tool 5: recommend_sprint_placement
@pm_agent.tool
async def recommend_sprint_placement(
    ctx: RunContext[AgentDependencies],
    work_item: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recommend sprint placement based on priority, dependencies, and capacity.

    Args:
        work_item: Work item details (title, type, priority, dependencies)

    Returns:
        Sprint recommendation with GTM phase, dependencies, and justification
    """
    result = await recommend_sprint_placement_tool(
        work_item,
        ctx.deps.current_sprint_info,
        ctx.deps.gtm_phase_definitions
    )
    return result.model_dump()


# Register Tool 6: create_action_item
@pm_agent.tool
async def create_action_item(
    ctx: RunContext[AgentDependencies],
    title: str,
    description: str,
    work_item_id: Optional[int] = None,
    request_id: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create action item in dashboard linked to work items or requests.

    Args:
        title: Action item title
        description: Action item description
        work_item_id: Optional linked ADO work item ID
        request_id: Optional linked request ID

    Returns:
        Action item result with ID and timestamp
    """
    result = await create_action_item_tool(
        title,
        description,
        ctx.deps.supabase_client,
        work_item_id,
        request_id
    )
    return result.model_dump()


# Register Tool 7: update_launch_readiness
@pm_agent.tool
async def update_launch_readiness(
    ctx: RunContext[AgentDependencies],
    category: str,
    item_title: str,
    score_delta: int,
    notes: str
) -> Dict[str, Any]:
    """
    Update launch readiness score for a category (prevents >1 point changes without approval).

    Args:
        category: Launch readiness category name
        item_title: Title of the item being updated
        score_delta: Change in score (-5 to +5)
        notes: Justification for the update

    Returns:
        Launch readiness update result with approval flag
    """
    result = await update_launch_readiness_tool(
        category,
        item_title,
        score_delta,
        notes,
        ctx.deps.supabase_client
    )
    return result.model_dump()


# Register Tool 8: log_agent_decision
@pm_agent.tool
async def log_agent_decision(
    ctx: RunContext[AgentDependencies],
    input_text: str,
    classification: Dict[str, Any],
    actions_taken: List[str],
    confidence: float
) -> str:
    """
    Log AI decision for transparency and continuous learning.

    Args:
        input_text: Original input text
        classification: Classification results
        actions_taken: List of actions/tools used
        confidence: Overall confidence score (0.0-1.0)

    Returns:
        Log ID for feedback tracking
    """
    return await log_agent_decision_tool(
        input_text,
        classification,
        actions_taken,
        confidence,
        ctx.deps.supabase_client
    )


# Convenience functions
async def run_pm_agent(
    prompt: str,
    session_id: Optional[str] = None,
    **dependency_overrides
) -> str:
    """
    Run the PM agent with automatic dependency injection.

    Args:
        prompt: User prompt/query
        session_id: Optional session identifier
        **dependency_overrides: Override default dependencies

    Returns:
        Agent response as string
    """
    deps = AgentDependencies.from_settings(
        settings,
        session_id=session_id,
        **dependency_overrides
    )

    result = await pm_agent.run(prompt, deps=deps)
    return result.data


def create_agent_with_deps(**dependency_overrides):
    """
    Create agent instance with custom dependencies.

    Args:
        **dependency_overrides: Custom dependency values

    Returns:
        Tuple of (agent, dependencies)
    """
    deps = AgentDependencies.from_settings(settings, **dependency_overrides)
    return pm_agent, deps
