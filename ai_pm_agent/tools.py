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
    LaunchReadinessUpdate,
    # Phase 2 models
    GraphRelationship,
    GraphSearchResult,
    TimelineEvent,
    HybridSearchMatch,
    ComprehensiveSearchResult
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


# ===== Phase 2 Tools: Knowledge Graph & Hybrid Search =====


async def search_work_item_graph_tool(
    entity_name: str,
    relationship_types: Optional[List[str]],
    depth: int,
    graphiti_client
) -> GraphSearchResult:
    """
    Search knowledge graph for work item relationships.

    Queries Neo4j/Graphiti to find related work items and their connections.
    Useful for questions like "Show all Features in Epic #1234" or
    "What work items block User Story #5678?"

    Args:
        entity_name: Work item identifier (e.g., "Epic #1234", "Feature #5678")
        relationship_types: Filter by specific relationships (CONTAINS, IMPLEMENTS, BLOCKS, DEPENDS_ON)
        depth: Maximum graph traversal depth (1-5)
        graphiti_client: GraphitiClient instance for knowledge graph access

    Returns:
        Graph search results with related entities and relationships

    Raises:
        ValueError: If depth not in range 1-5 or graphiti_client is None
    """
    try:
        # Validate inputs
        if not graphiti_client:
            raise ValueError("Phase 2 not configured: graphiti_client is None")

        if not 1 <= depth <= 5:
            raise ValueError("depth must be between 1 and 5")

        # Call Graphiti to get related entities
        results = await graphiti_client.get_related_entities(
            entity_name=entity_name,
            relationship_types=relationship_types,
            depth=depth
        )

        # Parse Graphiti facts into structured relationships
        relationships = []
        for fact_dict in results.get("related_facts", []):
            fact_text = fact_dict.get("fact", "")

            # Simple relationship extraction from fact text
            # Example: "Epic #1234 CONTAINS Feature #5678"
            rel = parse_graphiti_fact(fact_text, fact_dict)
            if rel:
                relationships.append(rel)

        return GraphSearchResult(
            central_entity=entity_name,
            related_entities=results.get("entities", []),
            relationships=relationships,
            search_method=results.get("search_method", "graphiti_semantic_search")
        )

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Graph search failed for {entity_name}: {e}")
        # Return empty result for graceful degradation
        return GraphSearchResult(
            central_entity=entity_name,
            related_entities=[],
            relationships=[],
            search_method="graphiti_semantic_search"
        )


def parse_graphiti_fact(fact_text: str, fact_dict: Dict[str, Any]) -> Optional[GraphRelationship]:
    """
    Parse Graphiti fact text into GraphRelationship.

    Examples:
        "Epic #1234 CONTAINS Feature #5678"
        "Feature #5678 IMPLEMENTS User Story #9012"
        "User Story #9012 BLOCKS User Story #9013"

    Args:
        fact_text: Fact description from Graphiti
        fact_dict: Full fact dictionary with metadata

    Returns:
        GraphRelationship or None if parsing fails
    """
    try:
        # Extract relationship keywords
        relationship_keywords = ["CONTAINS", "IMPLEMENTS", "BLOCKS", "DEPENDS_ON"]

        found_rel_type = None
        for rel_type in relationship_keywords:
            if rel_type in fact_text.upper():
                found_rel_type = rel_type
                break

        if not found_rel_type:
            return None

        # Simple pattern extraction (can be enhanced with regex)
        parts = fact_text.split(found_rel_type)
        if len(parts) != 2:
            return None

        source = parts[0].strip()
        target = parts[1].strip()

        return GraphRelationship(
            source_work_item=source,
            relationship_type=found_rel_type,
            target_work_item=target,
            valid_from=fact_dict.get("valid_at"),
            valid_until=fact_dict.get("invalid_at"),
            metadata={"uuid": fact_dict.get("uuid")}
        )
    except Exception as e:
        logger.debug(f"Failed to parse fact '{fact_text}': {e}")
        return None


async def get_work_item_timeline_tool(
    work_item_id: int,
    start_date: Optional[str],
    end_date: Optional[str],
    graphiti_client
) -> List[TimelineEvent]:
    """
    Retrieve temporal history of work item changes from knowledge graph.

    Tracks status changes, assignments, sprint movements, and other lifecycle
    events over time using Graphiti's bi-temporal model.

    Args:
        work_item_id: Azure DevOps work item ID (e.g., 1234)
        start_date: Start of time range (ISO format: "2025-01-01T00:00:00Z") or None for all
        end_date: End of time range (ISO format) or None for up to present
        graphiti_client: GraphitiClient instance

    Returns:
        Chronologically ordered list of timeline events

    Raises:
        ValueError: If graphiti_client is None or invalid date format
    """
    try:
        if not graphiti_client:
            raise ValueError("Phase 2 not configured: graphiti_client is None")

        # Format entity name for Graphiti
        entity_name = f"WorkItem#{work_item_id}"

        # Parse date strings to datetime objects
        from datetime import datetime, timezone

        start_dt = None
        end_dt = None

        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            except ValueError as e:
                raise ValueError(f"Invalid start_date format: {e}")

        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            except ValueError as e:
                raise ValueError(f"Invalid end_date format: {e}")

        # Query Graphiti timeline
        timeline_facts = await graphiti_client.get_entity_timeline(
            entity_name=entity_name,
            start_date=start_dt,
            end_date=end_dt
        )

        # Convert facts to TimelineEvent objects
        events = []
        for fact in timeline_facts:
            event = TimelineEvent(
                fact=fact.get("fact", "Unknown event"),
                timestamp=fact.get("timestamp") or fact.get("valid_at") or datetime.now(timezone.utc).isoformat(),
                event_type=classify_event_type(fact.get("fact", "")),
                metadata={
                    "uuid": fact.get("uuid"),
                    "source": fact.get("source", "graphiti"),
                    "confidence": fact.get("confidence", 1.0)
                }
            )
            events.append(event)

        # Sort chronologically (oldest first)
        events.sort(key=lambda e: e.timestamp)

        return events

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Timeline retrieval failed for work item {work_item_id}: {e}")
        return []


def classify_event_type(fact: str) -> str:
    """
    Classify event type based on fact text.

    Examples:
        "Status changed from New to In Progress" -> status_change
        "Assigned to John Doe" -> assignment_update
        "Moved to Sprint 42" -> sprint_movement
        "Description updated" -> description_change

    Args:
        fact: Event description

    Returns:
        Event type category
    """
    fact_lower = fact.lower()

    if "status changed" in fact_lower or "moved to" in fact_lower and "sprint" not in fact_lower:
        return "status_change"
    elif "assigned" in fact_lower:
        return "assignment_update"
    elif "sprint" in fact_lower:
        return "sprint_movement"
    elif "description" in fact_lower:
        return "description_change"
    elif "priority" in fact_lower:
        return "priority_change"
    elif "tag" in fact_lower or "label" in fact_lower:
        return "tag_update"
    else:
        return "other"


async def hybrid_work_item_search_tool(
    query: str,
    limit: int,
    keyword_weight: float,
    openai_client,
    database_pool
) -> List[HybridSearchMatch]:
    """
    Combined semantic (pgvector) + keyword (TSVector) search.

    Ideal for queries with specific terminology like "API timeout bug" or
    "security issue in login flow" where both conceptual meaning and exact
    keywords matter.

    Args:
        query: Search query (natural language and/or keywords)
        limit: Maximum results to return (1-100)
        keyword_weight: Balance between semantic and keyword (0.0-1.0)
                       0.0 = pure semantic, 1.0 = pure keyword, 0.3 = 70% semantic + 30% keyword
        openai_client: OpenAI client for embedding generation
        database_pool: DatabasePool instance for direct database access

    Returns:
        Work item matches ordered by combined_score (highest first)

    Raises:
        ValueError: If invalid parameters or database_pool is None
    """
    try:
        if not database_pool:
            raise ValueError("Phase 2 not configured: database_pool is None")

        # Validate parameters
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")

        if not 0.0 <= keyword_weight <= 1.0:
            raise ValueError("keyword_weight must be between 0.0 and 1.0")

        # Generate embedding using OpenAI
        from .providers import generate_embedding
        embedding = await generate_embedding(query, openai_client)

        # Import hybrid search function
        from .db_utils import hybrid_work_item_search

        # Execute hybrid search
        results = await hybrid_work_item_search(
            database_pool,
            embedding,
            query,
            limit,
            keyword_weight
        )

        # Convert dict results to HybridSearchMatch objects
        matches = [HybridSearchMatch(**result) for result in results]

        return matches

    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Hybrid search failed for query '{query}': {e}")
        return []


async def comprehensive_work_item_search_tool(
    query: str,
    use_semantic: bool,
    use_graph: bool,
    use_hybrid: bool,
    limit: int,
    openai_client,
    supabase_client,
    graphiti_client,
    database_pool
) -> ComprehensiveSearchResult:
    """
    Master search tool executing all three strategies in parallel.

    Combines semantic search (pgvector), knowledge graph search (Graphiti),
    and hybrid search (TSVector + pgvector) for complete context.

    Args:
        query: Search query
        use_semantic: Enable semantic search (Phase 1 pgvector)
        use_graph: Enable knowledge graph search (Graphiti)
        use_hybrid: Enable hybrid search (TSVector + pgvector)
        limit: Maximum results per strategy (1-50)
        openai_client: OpenAI client for embeddings
        supabase_client: Supabase client for semantic search
        graphiti_client: GraphitiClient for graph search
        database_pool: DatabasePool for hybrid search

    Returns:
        Comprehensive search results with all enabled strategies

    Note:
        - Executes enabled strategies in parallel using asyncio.gather()
        - Returns partial results if any strategy fails (graceful degradation)
        - Empty lists for disabled strategies
    """
    import asyncio

    try:
        # Validate limit
        if not 1 <= limit <= 50:
            raise ValueError("limit must be between 1 and 50")

        # Build task list dynamically
        tasks = []
        strategies_used = []

        if use_semantic:
            # Use Phase 1 tool - semantic search with pgvector
            from .providers import search_work_items_by_embedding

            # Generate embedding once for efficiency
            from .providers import generate_embedding
            embedding = await generate_embedding(query, openai_client)

            tasks.append(search_work_items_by_embedding(embedding, supabase_client, limit))
            strategies_used.append("semantic")

        if use_graph:
            # Use Graphiti knowledge graph search
            tasks.append(graphiti_client.search(query) if graphiti_client else asyncio.sleep(0))
            strategies_used.append("graph")

        if use_hybrid:
            # Use hybrid TSVector + pgvector search
            if not database_pool:
                logger.warning("Hybrid search requested but database_pool is None")
                tasks.append(asyncio.sleep(0))
            else:
                tasks.append(hybrid_work_item_search_tool(
                    query, limit, 0.3, openai_client, database_pool
                ))
            strategies_used.append("hybrid")

        # Execute all searches in parallel
        if not tasks:
            # No strategies enabled
            return ComprehensiveSearchResult(
                semantic_matches=[],
                graph_relationships=[],
                hybrid_matches=[],
                total_results=0,
                search_strategies_used=[],
                combined_ranking=[]
            )

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Extract results for each strategy
        idx = 0
        semantic_matches = []
        graph_results = []
        hybrid_matches = []

        if use_semantic:
            if not isinstance(results[idx], Exception):
                semantic_matches = results[idx]
            idx += 1

        if use_graph:
            if not isinstance(results[idx], Exception):
                graph_results = results[idx]
            idx += 1

        if use_hybrid:
            if not isinstance(results[idx], Exception):
                hybrid_matches = results[idx]
            idx += 1

        # Convert graph results to relationships
        graph_relationships = []
        if graph_results:
            for result in graph_results:
                if isinstance(result, dict):
                    fact_text = result.get("fact", "")
                    rel = parse_graphiti_fact(fact_text, result)
                    if rel:
                        graph_relationships.append(rel)

        # Merge and rank results (simple implementation - can be enhanced)
        combined_ranking = merge_and_rank_results(
            semantic_matches,
            graph_relationships,
            hybrid_matches
        )

        return ComprehensiveSearchResult(
            semantic_matches=semantic_matches,
            graph_relationships=graph_relationships,
            hybrid_matches=hybrid_matches,
            total_results=len(semantic_matches) + len(graph_relationships) + len(hybrid_matches),
            search_strategies_used=strategies_used,
            combined_ranking=combined_ranking[:limit]
        )

    except Exception as e:
        logger.error(f"Comprehensive search failed for query '{query}': {e}")
        return ComprehensiveSearchResult(
            semantic_matches=[],
            graph_relationships=[],
            hybrid_matches=[],
            total_results=0,
            search_strategies_used=[],
            combined_ranking=[]
        )


def merge_and_rank_results(
    semantic_matches: List[WorkItemMatch],
    graph_relationships: List[GraphRelationship],
    hybrid_matches: List[HybridSearchMatch]
) -> List[WorkItemMatch]:
    """
    Merge and rank results from all three search strategies.

    Simple weighted ranking algorithm that prioritizes hybrid matches
    (which already combine semantic + keyword), then semantic matches,
    then work items mentioned in graph relationships.

    Args:
        semantic_matches: From semantic search
        graph_relationships: From graph search
        hybrid_matches: From hybrid search

    Returns:
        Merged and ranked list of WorkItemMatch objects
    """
    combined = []

    # Add hybrid matches first (highest priority - already optimized)
    for match in hybrid_matches:
        combined.append(WorkItemMatch(
            ado_id=match.ado_id,
            title=match.title,
            work_item_type=match.work_item_type,
            similarity_score=match.combined_score,
            recommendation="highly_relevant"
        ))

    # Add semantic matches (medium priority)
    for match in semantic_matches:
        # Avoid duplicates
        if not any(c.ado_id == match.ado_id for c in combined):
            combined.append(match)

    # Extract work items from graph relationships (lower priority)
    graph_work_items = set()
    for rel in graph_relationships:
        # Extract ADO IDs from relationship source/target
        # Example: "Epic #1234" -> 1234
        try:
            source_id = int(rel.source_work_item.split('#')[-1])
            target_id = int(rel.target_work_item.split('#')[-1])
            graph_work_items.add(source_id)
            graph_work_items.add(target_id)
        except:
            pass

    # Sort by similarity_score (descending)
    combined.sort(key=lambda x: x.similarity_score, reverse=True)

    return combined
