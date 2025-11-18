"""
Tool implementations for AI Project Management Agent.
"""

import os
import uuid
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .models import (
    ClassificationResult,
    WorkItemMatch,
    GeneratedWorkItem,
    ArtifactTagResult,
    SprintRecommendation,
    ActionItemResult,
    LaunchReadinessUpdate
)

logger = logging.getLogger(__name__)


async def classify_input_tool(
    input_text: str,
    categories: List[Dict[str, str]],
    gtm_phases: Dict[str, Dict[str, str]]
) -> ClassificationResult:
    """
    Classify input into launch readiness categories.

    Args:
        input_text: The freeform input to classify
        categories: List of launch readiness category definitions
        gtm_phases: Dictionary of GTM phase definitions

    Returns:
        Classification result with category, confidence, phase, and action
    """
    try:
        input_lower = input_text.lower()

        # Simple keyword-based classification
        category_scores = {}
        for cat in categories:
            name = cat["name"]
            description = cat["description"].lower()
            examples = cat["examples"].lower()

            # Calculate score based on keyword matches
            score = 0.0
            desc_words = description.split()
            example_words = examples.split()

            for word in desc_words + example_words:
                if len(word) > 3 and word in input_lower:
                    score += 10.0

            category_scores[name] = min(score, 100.0)

        # Select category with highest score
        if not category_scores or max(category_scores.values()) == 0:
            category = "Product Readiness"  # Default
            confidence = 30.0
        else:
            category = max(category_scores, key=category_scores.get)
            confidence = category_scores[category]

        # Map to GTM phase (simplified logic)
        if "bug" in input_lower or "fix" in input_lower:
            gtm_phase = "Validation"
        elif "feature" in input_lower or "new" in input_lower:
            gtm_phase = "Foundation"
        elif "launch" in input_lower or "pricing" in input_lower:
            gtm_phase = "Launch Prep"
        elif "marketing" in input_lower or "sales" in input_lower:
            gtm_phase = "Go-to-Market"
        elif "scale" in input_lower or "optimize" in input_lower:
            gtm_phase = "Growth"
        else:
            gtm_phase = "Foundation"

        # Determine suggested action
        if confidence < 70:
            suggested_action = "ask_for_clarification"
        elif confidence < 85:
            suggested_action = "suggest_with_caveat"
        else:
            suggested_action = "create_work_item"

        rationale = f"Classified as {category} with {confidence:.1f}% confidence based on keyword analysis"

        return ClassificationResult(
            category=category,
            confidence_score=confidence,
            gtm_phase=gtm_phase,
            suggested_action=suggested_action,
            rationale=rationale
        )

    except Exception as e:
        logger.error(f"Classification failed: {e}")
        return ClassificationResult(
            category="Unknown",
            confidence_score=0.0,
            gtm_phase="Unknown",
            suggested_action="manual_review",
            rationale=f"Classification error: {str(e)}"
        )


async def match_work_items_tool(
    input_text: str,
    supabase_client,
    openai_client,
    limit: int = 5
) -> List[WorkItemMatch]:
    """
    Match input against existing work items using semantic similarity.

    Args:
        input_text: Text to search for similar work items
        supabase_client: Supabase client for database queries
        openai_client: OpenAI client for embeddings
        limit: Maximum number of matches to return

    Returns:
        List of work item matches with similarity scores
    """
    try:
        # Generate embedding
        response = await openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=input_text
        )
        embedding = response.data[0].embedding

        # Query Supabase for matches
        result = await supabase_client.rpc(
            "match_work_items",
            {"query_embedding": embedding, "match_count": limit}
        ).execute()

        matches = []
        for item in result.data[:limit]:
            similarity = item.get("similarity", 0.0)

            # Determine recommendation
            if similarity > 0.90:
                recommendation = "link"
            elif similarity > 0.75:
                recommendation = "update"
            else:
                recommendation = "create_new"

            matches.append(WorkItemMatch(
                ado_id=item.get("ado_id", 0),
                title=item.get("title", ""),
                work_item_type=item.get("work_item_type", "Unknown"),
                similarity_score=similarity,
                recommendation=recommendation
            ))

        return matches

    except Exception as e:
        logger.error(f"Work item matching failed: {e}")
        return []


async def generate_work_item_tool(
    input_description: str,
    work_item_type: str,
    templates: Dict[str, str],
    parent_id: Optional[int] = None
) -> GeneratedWorkItem:
    """
    Generate ADO-ready work item using templates.

    Args:
        input_description: Description of work to be done
        work_item_type: "Epic", "Feature", or "User Story"
        templates: Dictionary of work item templates
        parent_id: Optional parent work item ID

    Returns:
        Generated work item ready for approval
    """
    try:
        template = templates.get(work_item_type)
        if not template:
            raise ValueError(f"Unknown work item type: {work_item_type}")

        # Simple title extraction (first sentence or first 60 chars)
        title = input_description.split(".")[0][:60]

        # Fill template with description
        description = template.replace("[Clear, concise epic name]", title)
        description = description.replace("[Brief user story summary]", title)
        description = description.replace("[User-facing capability name]", title)

        # Extract tags from input
        tags = []
        if "sous" in input_description.lower():
            tags.append("Sous AI")
        if "formulate" in input_description.lower():
            tags.append("Formulate")
        if "launch" in input_description.lower():
            tags.append("Launch Prep")

        # Generate acceptance criteria for Features/Stories
        acceptance_criteria = None
        if work_item_type in ["Feature", "User Story"]:
            acceptance_criteria = [
                "Functionality works as described",
                "User experience is intuitive",
                "Tests pass successfully"
            ]

        return GeneratedWorkItem(
            title=title,
            description=description,
            work_item_type=work_item_type,
            acceptance_criteria=acceptance_criteria,
            tags=tags if tags else ["General"],
            suggested_sprint=42,  # Default to current sprint
            parent_link=parent_id,
            metadata={"generated_at": datetime.utcnow().isoformat()}
        )

    except Exception as e:
        logger.error(f"Work item generation failed: {e}")
        return GeneratedWorkItem(
            title="ERROR: Generation failed",
            description=f"Error: {str(e)}",
            work_item_type=work_item_type,
            acceptance_criteria=[],
            tags=["error"],
            suggested_sprint=0,
            parent_link=parent_id,
            metadata={"error": str(e)}
        )


async def tag_artifact_tool(
    file_path: str,
    supabase_client,
    artifact_registry_path: str
) -> ArtifactTagResult:
    """
    Auto-tag uploaded documents and link to work items.

    Args:
        file_path: Path to the uploaded artifact
        supabase_client: Supabase client for database operations
        artifact_registry_path: Base path for artifact storage

    Returns:
        Artifact tagging result with metadata
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Artifact not found: {file_path}")

        # Detect artifact type from filename
        filename = Path(file_path).name.lower()
        if "wireframe" in filename or "mockup" in filename:
            detected_type = "wireframe"
        elif "brief" in filename or "spec" in filename:
            detected_type = "brief"
        elif "test" in filename:
            detected_type = "test_plan"
        elif "notes" in filename or "meeting" in filename:
            detected_type = "meeting_notes"
        elif "design" in filename:
            detected_type = "design_doc"
        else:
            detected_type = "unknown"

        # Extract metadata
        file_stats = os.stat(file_path)
        metadata = {
            "size": file_stats.st_size,
            "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        }

        # Generate artifact ID
        artifact_id = str(uuid.uuid4())

        # Store in Supabase
        await supabase_client.table("artifacts").insert({
            "id": artifact_id,
            "file_path": file_path,
            "artifact_type": detected_type,
            "metadata": metadata,
            "created_at": datetime.utcnow().isoformat()
        }).execute()

        # Generate suggested tags
        suggested_tags = [detected_type]
        if "sous" in filename:
            suggested_tags.append("Sous AI")

        return ArtifactTagResult(
            artifact_id=artifact_id,
            detected_type=detected_type,
            extracted_metadata=metadata,
            suggested_tags=suggested_tags,
            linked_work_items=[]
        )

    except Exception as e:
        logger.error(f"Artifact tagging failed: {e}")
        return ArtifactTagResult(
            artifact_id="",
            detected_type="unknown",
            extracted_metadata={"error": str(e)},
            suggested_tags=["unprocessed"],
            linked_work_items=[]
        )


async def recommend_sprint_placement_tool(
    work_item: Dict[str, Any],
    current_sprint_info: Dict[str, Any],
    gtm_phases: Dict[str, Dict[str, str]]
) -> SprintRecommendation:
    """
    Recommend sprint placement based on priority and capacity.

    Args:
        work_item: Work item details
        current_sprint_info: Current sprint number and capacity
        gtm_phases: GTM phase definitions

    Returns:
        Sprint recommendation with justification
    """
    try:
        current_sprint = current_sprint_info.get("sprint", 42)
        priority = work_item.get("priority", "Medium")
        dependencies = work_item.get("dependencies", [])

        # Calculate recommendation based on priority
        if priority == "High" or priority == "Critical":
            recommended_sprint = current_sprint
        elif priority == "Medium":
            recommended_sprint = current_sprint + 1
        else:
            recommended_sprint = current_sprint + 2

        # Adjust for dependencies
        if dependencies:
            recommended_sprint = max(recommended_sprint, current_sprint + len(dependencies))

        # Determine GTM phase
        work_type = work_item.get("work_item_type", "Feature")
        if work_type == "Epic":
            gtm_phase = "Foundation"
        elif work_type == "Feature":
            gtm_phase = "Validation"
        else:
            gtm_phase = "Foundation"

        justification = f"{priority} priority work item recommended for Sprint {recommended_sprint}"
        if dependencies:
            justification += f" (considering {len(dependencies)} dependencies)"

        alternative_sprints = [
            recommended_sprint - 1 if recommended_sprint > current_sprint else current_sprint,
            recommended_sprint + 1
        ]

        return SprintRecommendation(
            recommended_sprint=recommended_sprint,
            gtm_phase=gtm_phase,
            dependencies=dependencies,
            justification=justification,
            alternative_sprints=alternative_sprints
        )

    except Exception as e:
        logger.error(f"Sprint recommendation failed: {e}")
        return SprintRecommendation(
            recommended_sprint=0,
            gtm_phase="Unknown",
            dependencies=[],
            justification=f"Error: {str(e)}",
            alternative_sprints=[]
        )


async def create_action_item_tool(
    title: str,
    description: str,
    supabase_client,
    work_item_id: Optional[int] = None,
    request_id: Optional[int] = None
) -> ActionItemResult:
    """
    Create action item in dashboard.

    Args:
        title: Action item title
        description: Action item description
        supabase_client: Supabase client for database operations
        work_item_id: Optional linked work item ID
        request_id: Optional linked request ID

    Returns:
        Action item result with ID and timestamp
    """
    try:
        action_item_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        await supabase_client.table("action_items").insert({
            "id": action_item_id,
            "title": title,
            "description": description,
            "work_item_id": work_item_id,
            "request_id": request_id,
            "created_at": created_at,
            "completed": False
        }).execute()

        return ActionItemResult(
            action_item_id=action_item_id,
            created_at=created_at,
            linked_work_item=work_item_id,
            linked_request=request_id
        )

    except Exception as e:
        logger.error(f"Action item creation failed: {e}")
        return ActionItemResult(
            action_item_id="",
            created_at="",
            linked_work_item=work_item_id,
            linked_request=request_id
        )


async def update_launch_readiness_tool(
    category: str,
    item_title: str,
    score_delta: int,
    notes: str,
    supabase_client
) -> LaunchReadinessUpdate:
    """
    Update launch readiness score for a category.

    Args:
        category: Launch readiness category name
        item_title: Title of item being updated
        score_delta: Change in score (-5 to +5)
        notes: Justification for update
        supabase_client: Supabase client for database operations

    Returns:
        Launch readiness update result
    """
    try:
        # Fetch current score
        current = await supabase_client.table("launch_readiness").select("score").eq("category", category).eq("item_title", item_title).execute()

        previous_score = current.data[0]["score"] if current.data else 3

        # Validate score_delta
        if abs(score_delta) > 5:
            raise ValueError(f"Score delta {score_delta} exceeds maximum allowed (±5)")

        # Calculate new score (clamp to 1-5)
        new_score = max(1, min(5, previous_score + score_delta))

        # Check if approval required
        requires_approval = abs(score_delta) > 1

        if requires_approval:
            logger.warning(f"Large score change requires approval: {score_delta}")
            return LaunchReadinessUpdate(
                updated_score=new_score,
                previous_score=previous_score,
                category=category,
                timestamp=datetime.utcnow().isoformat(),
                requires_approval=True
            )

        # Update database
        await supabase_client.table("launch_readiness").upsert({
            "category": category,
            "item_title": item_title,
            "score": new_score,
            "notes": notes,
            "updated_at": datetime.utcnow().isoformat()
        }).execute()

        return LaunchReadinessUpdate(
            updated_score=new_score,
            previous_score=previous_score,
            category=category,
            timestamp=datetime.utcnow().isoformat(),
            requires_approval=False
        )

    except Exception as e:
        logger.error(f"Launch readiness update failed: {e}")
        return LaunchReadinessUpdate(
            updated_score=1,
            previous_score=1,
            category=category,
            timestamp=datetime.utcnow().isoformat(),
            requires_approval=True
        )


async def log_agent_decision_tool(
    input_text: str,
    classification: Dict[str, Any],
    actions_taken: List[str],
    confidence: float,
    supabase_client
) -> str:
    """
    Log AI decision for transparency and learning.

    Args:
        input_text: Original input text
        classification: Classification results
        actions_taken: List of actions/tools used
        confidence: Overall confidence score
        supabase_client: Supabase client for database operations

    Returns:
        Log ID for feedback tracking
    """
    try:
        log_id = str(uuid.uuid4())
        created_at = datetime.utcnow().isoformat()

        await supabase_client.table("agent_logs").insert({
            "id": log_id,
            "input_text": input_text,
            "classification": classification,
            "actions_taken": actions_taken,
            "confidence": confidence,
            "created_at": created_at
        }).execute()

        logger.info(f"Decision logged: {log_id} (confidence: {confidence:.2%})")

        return log_id

    except Exception as e:
        logger.error(f"Decision logging failed: {e}")
        # Fallback to local file logging
        try:
            with open("/tmp/agent_decisions.log", "a") as f:
                f.write(f"{datetime.utcnow().isoformat()} | ERROR | {str(e)}\n")
        except:
            pass
        return ""
