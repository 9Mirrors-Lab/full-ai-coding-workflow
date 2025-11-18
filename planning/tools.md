# AI Project Management Agent - Tools Specification

**Version:** 1.0
**Created:** 2025-01-16
**Agent:** AI Project Management Agent
**Purpose:** Tool specifications for building an intelligent project management agent with Pydantic AI

---

## Overview

This document specifies 8 essential tools for the AI Project Management Agent. Each tool has a single, focused purpose with simple parameters (1-4 per tool) and clear error handling. These tools enable the agent to automate project management workflows including classification, semantic search, work item generation, and decision logging.

**Design Philosophy:**
- Simple, focused tools with clear single purpose
- Minimal parameters (1-4 per tool)
- Structured responses using Pydantic models
- Graceful error handling with error dictionaries
- Async/await patterns throughout

---

## Tool 1: classify_input

### Purpose
Classifies freeform input text into one of 6 launch readiness categories with confidence scoring. Determines the GTM phase and suggests appropriate next actions.

### Parameters
1. **input_text** (str, required): The freeform input to classify (meeting notes, ideas, bug reports, etc.)

### Return Type
```python
ClassificationResult(BaseModel):
    category: str  # One of: "Product Readiness", "Technical Readiness", "Operational Readiness", "Commercial Readiness", "Customer Readiness", "Security & Compliance"
    confidence_score: float  # 0-100% confidence
    gtm_phase: str  # One of: "Foundation", "Validation", "Launch Prep", "Go-to-Market", "Growth", "Market Leadership"
    suggested_action: str  # Recommended next action (e.g., "create_work_item", "link_to_existing", "create_action_item")
    rationale: str  # Explanation of classification reasoning
```

### Implementation Strategy
- Use keyword matching against category definitions
- Analyze content for technical vs. product vs. operational signals
- Calculate confidence based on match strength and ambiguity
- Map category to GTM phase based on content analysis
- Return structured classification with transparent reasoning

### Error Handling
```python
try:
    # Classification logic
    return ClassificationResult(...)
except Exception as e:
    logger.error(f"Classification failed: {e}")
    return ClassificationResult(
        category="Unknown",
        confidence_score=0.0,
        gtm_phase="Unknown",
        suggested_action="manual_review",
        rationale=f"Classification error: {str(e)}"
    )
```

### Usage Context
- Accessed via `ctx.deps.launch_readiness_categories` (list of category definitions)
- Accessed via `ctx.deps.gtm_phase_definitions` (dict of phase definitions)
- Used as first step in agent workflow to route inputs

---

## Tool 2: match_ado_work_items

### Purpose
Performs semantic similarity search against existing Azure DevOps work items to find related work and prevent duplicates.

### Parameters
1. **input_text** (str, required): Text to search for similar work items
2. **limit** (int, optional, default=5): Maximum number of matches to return (1-20)

### Return Type
```python
List[WorkItemMatch(BaseModel)]:
    ado_id: int  # Azure DevOps work item ID
    title: str  # Work item title
    work_item_type: str  # "Epic", "Feature", or "User Story"
    similarity_score: float  # 0.0-1.0 cosine similarity score
    recommendation: str  # "link", "update", or "create_new"
```

### Implementation Strategy
1. Generate embedding for input_text using OpenAI embeddings API
2. Query Supabase `match_work_items()` RPC function with embedding vector
3. Return top N matches sorted by similarity score
4. Provide recommendation based on similarity threshold:
   - >0.90: "link" (highly similar, link to existing)
   - 0.75-0.90: "update" (similar, consider updating existing)
   - <0.75: "create_new" (not similar enough, create new)

### Error Handling
```python
try:
    # Generate embedding
    embedding = await openai_client.embeddings.create(
        model="text-embedding-3-small",
        input=input_text
    )

    # Query Supabase
    results = await supabase_client.rpc(
        "match_work_items",
        {"query_embedding": embedding.data[0].embedding, "match_count": limit}
    ).execute()

    return [WorkItemMatch(...) for r in results.data]

except Exception as e:
    logger.error(f"Work item matching failed: {e}")
    return []  # Return empty list on failure for graceful degradation
```

### Usage Context
- Accessed via `ctx.deps.supabase_client` for database queries
- Accessed via `ctx.deps.openai_client` for embeddings
- Implements exponential backoff for rate limit errors (429)
- Caches embeddings to avoid regeneration

---

## Tool 3: generate_work_item

### Purpose
Generates properly formatted Azure DevOps work items (Epic, Feature, User Story) following ADO templates. Returns structured work item for approval - NEVER creates directly in ADO.

### Parameters
1. **input_description** (str, required): Description of the work to be done
2. **work_item_type** (str, required): "Epic", "Feature", or "User Story"
3. **parent_id** (int, optional): Parent work item ID for linking

### Return Type
```python
GeneratedWorkItem(BaseModel):
    title: str  # Clear, concise work item title
    description: str  # Full description following template format
    work_item_type: str  # "Epic", "Feature", or "User Story"
    acceptance_criteria: Optional[List[str]]  # List of acceptance criteria (Features/Stories)
    tags: List[str]  # Suggested tags (e.g., ["Sous AI", "Launch Prep"])
    suggested_sprint: int  # Recommended sprint number
    parent_link: Optional[int]  # Parent work item ID if provided
    metadata: Dict[str, Any]  # Additional metadata (priority, effort estimate, etc.)
```

### Template Formats

**Epic Template:**
```
Title: [Clear, concise epic name]

Vision Statement:
[What is this epic trying to achieve?]

Business Value:
[Why is this important? What value does it deliver?]

Success Metrics:
- [Measurable metric 1]
- [Measurable metric 2]

Tags: [Relevant tags]
```

**Feature Template:**
```
Title: [User-facing capability name]

Description:
[What capability does this provide to users?]

Acceptance Criteria:
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

Technical Notes:
[Implementation considerations, dependencies, etc.]

Tags: [Relevant tags]
```

**User Story Template:**
```
Title: [Brief user story summary]

User Story:
As a [user type]
I want [capability]
So that [benefit]

Acceptance Criteria (Gherkin format):

Scenario 1: [Scenario name]
Given [initial context]
When [action taken]
Then [expected outcome]

Scenario 2: [Another scenario name]
Given [initial context]
When [action taken]
Then [expected outcome]

Tags: [Relevant tags]
```

### Implementation Strategy
- Select appropriate template based on work_item_type
- Parse input_description to extract key components
- Fill template with extracted information
- Validate completeness (all required fields populated)
- Suggest sprint based on priority and capacity
- Return structured work item for Ryan's approval

### Error Handling
```python
try:
    template = templates.get(work_item_type)
    if not template:
        raise ValueError(f"Unknown work item type: {work_item_type}")

    # Generate work item
    work_item = GeneratedWorkItem(...)

    # Validate completeness
    if not work_item.title or not work_item.description:
        raise ValueError("Generated work item missing required fields")

    return work_item

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
```

### Usage Context
- Accessed via `ctx.deps.work_item_templates` (dict of template strings)
- Accessed via `ctx.deps.current_sprint_info` for sprint recommendations
- CRITICAL: Never creates work items in ADO directly - only generates for approval

---

## Tool 4: tag_artifact

### Purpose
Auto-tags uploaded documents by detecting type, extracting metadata, and linking to relevant work items.

### Parameters
1. **file_path** (str, required): Path to the uploaded artifact file

### Return Type
```python
ArtifactTagResult(BaseModel):
    artifact_id: str  # UUID for the artifact record
    detected_type: str  # One of: "wireframe", "brief", "test_plan", "meeting_notes", "design_doc", "marketing_material"
    extracted_metadata: Dict[str, Any]  # File metadata (size, created_at, author, etc.)
    suggested_tags: List[str]  # Auto-generated tags based on content
    linked_work_items: List[int]  # ADO work item IDs to link to
```

### Implementation Strategy
1. Detect artifact type from filename and extension
2. Extract file metadata (size, modified date, etc.)
3. Read file content for keyword extraction
4. Generate suggested tags based on content analysis
5. Query Supabase for related work items using semantic search
6. Store artifact record in Supabase `artifacts` table
7. Create correlations in `correlations` table

### Error Handling
```python
try:
    import os
    import uuid
    from pathlib import Path

    # Validate file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Artifact not found: {file_path}")

    # Detect type and extract metadata
    artifact_type = detect_artifact_type(file_path)
    metadata = extract_file_metadata(file_path)

    # Store in Supabase
    artifact_id = str(uuid.uuid4())
    await supabase_client.table("artifacts").insert({
        "id": artifact_id,
        "file_path": file_path,
        "artifact_type": artifact_type,
        "metadata": metadata,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    return ArtifactTagResult(...)

except Exception as e:
    logger.error(f"Artifact tagging failed: {e}")
    return ArtifactTagResult(
        artifact_id="",
        detected_type="unknown",
        extracted_metadata={"error": str(e)},
        suggested_tags=["unprocessed"],
        linked_work_items=[]
    )
```

### Usage Context
- Accessed via `ctx.deps.supabase_client` for database operations
- Accessed via `ctx.deps.artifact_registry_path` for file storage location
- Stores artifacts and creates correlations for tracking

---

## Tool 5: recommend_sprint_placement

### Purpose
Recommends sprint placement for a work item based on priority, dependencies, capacity, and GTM phase alignment.

### Parameters
1. **work_item** (Dict[str, Any], required): Work item details (title, type, priority, dependencies)
2. **suggested_sprint** (int, optional): Override sprint suggestion

### Return Type
```python
SprintRecommendation(BaseModel):
    recommended_sprint: int  # Sprint number recommendation
    gtm_phase: str  # GTM phase alignment
    dependencies: List[str]  # List of dependency work item IDs/titles
    justification: str  # Reasoning for recommendation
    alternative_sprints: List[int]  # Alternative sprint options
```

### Implementation Strategy
1. Analyze work item priority (High → earlier sprint)
2. Check current sprint capacity via `ctx.deps.current_sprint_info`
3. Identify dependencies that must be completed first
4. Map work item to GTM phase based on type and description
5. Calculate recommended sprint considering all factors
6. Provide 2-3 alternative sprint options
7. Include transparent justification

### Priority-Based Logic
- **Critical/High Priority**: Current sprint or next sprint
- **Medium Priority**: Next 2-3 sprints
- **Low Priority**: Backlog (sprint TBD)

### Error Handling
```python
try:
    current_sprint = ctx.deps.current_sprint_info.get("sprint", 1)
    capacity = ctx.deps.current_sprint_info.get("capacity", 100)

    # Calculate recommendation
    priority = work_item.get("priority", "Medium")
    dependencies = work_item.get("dependencies", [])

    if priority == "High":
        recommended_sprint = current_sprint
    elif priority == "Medium":
        recommended_sprint = current_sprint + 1
    else:
        recommended_sprint = current_sprint + 2

    # Adjust for dependencies
    if dependencies:
        recommended_sprint = max(recommended_sprint, current_sprint + len(dependencies))

    return SprintRecommendation(...)

except Exception as e:
    logger.error(f"Sprint recommendation failed: {e}")
    return SprintRecommendation(
        recommended_sprint=0,
        gtm_phase="Unknown",
        dependencies=[],
        justification=f"Error calculating recommendation: {str(e)}",
        alternative_sprints=[]
    )
```

### Usage Context
- Accessed via `ctx.deps.current_sprint_info` (current sprint number and capacity)
- Accessed via `ctx.deps.gtm_phase_definitions` for phase alignment
- Considers team velocity and historical data

---

## Tool 6: create_action_item

### Purpose
Creates action items in the Supabase dashboard linked to work items or requests for tracking follow-ups.

### Parameters
1. **title** (str, required): Action item title
2. **description** (str, required): Action item description
3. **work_item_id** (int, optional): Linked ADO work item ID
4. **request_id** (int, optional): Linked request ID

### Return Type
```python
ActionItemResult(BaseModel):
    action_item_id: str  # UUID of created action item
    created_at: str  # ISO timestamp
    linked_work_item: Optional[int]  # Linked work item ID
    linked_request: Optional[int]  # Linked request ID
```

### Implementation Strategy
1. Validate that either work_item_id or request_id is provided
2. Create UUID for action item
3. Insert into Supabase `action_items` table
4. Return structured result with ID and timestamp

### Error Handling
```python
try:
    import uuid
    from datetime import datetime

    action_item_id = str(uuid.uuid4())
    created_at = datetime.utcnow().isoformat()

    result = await supabase_client.table("action_items").insert({
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
```

### Usage Context
- Accessed via `ctx.deps.supabase_client` for database operations
- Creates trackable action items in dashboard
- Links to work items for context

---

## Tool 7: update_launch_readiness

### Purpose
Updates launch readiness scores (1-5 scale) for categories. Prevents automatic updates >1 point without explicit approval.

### Parameters
1. **category** (str, required): Launch readiness category name
2. **item_title** (str, required): Title of the item being updated
3. **score_delta** (int, required): Change in score (-5 to +5)
4. **notes** (str, required): Justification for the update

### Return Type
```python
LaunchReadinessUpdate(BaseModel):
    updated_score: int  # New score (1-5)
    previous_score: int  # Previous score (1-5)
    category: str  # Category name
    timestamp: str  # ISO timestamp of update
    requires_approval: bool  # True if score_delta > 1
```

### Implementation Strategy
1. Fetch current score for category from Supabase
2. Validate score_delta is within bounds (-5 to +5)
3. Check if abs(score_delta) > 1 → requires_approval = True
4. Calculate new score (clamp to 1-5 range)
5. Update Supabase `launch_readiness` table
6. Return update result with approval flag

### Safety Constraints
- **CRITICAL**: Prevent auto-updates >1 point without approval
- Validate category exists in predefined categories
- Ensure score stays within 1-5 range
- Require justification notes for all updates

### Error Handling
```python
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
        # Return result but don't update database
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

    return LaunchReadinessUpdate(...)

except Exception as e:
    logger.error(f"Launch readiness update failed: {e}")
    return LaunchReadinessUpdate(
        updated_score=0,
        previous_score=0,
        category=category,
        timestamp="",
        requires_approval=True
    )
```

### Usage Context
- Accessed via `ctx.deps.supabase_client` for database operations
- Accessed via `ctx.deps.launch_readiness_categories` for validation
- Implements safety check for large score changes

---

## Tool 8: log_agent_decision

### Purpose
Logs all agent decisions, classifications, and actions for transparency, debugging, and continuous learning.

### Parameters
1. **input_text** (str, required): Original input text
2. **classification** (Dict[str, Any], required): Classification results
3. **actions_taken** (List[str], required): List of actions/tools used
4. **confidence** (float, required): Overall confidence score (0.0-1.0)

### Return Type
```python
str  # Log ID (UUID) for feedback tracking
```

### Implementation Strategy
1. Create UUID for decision log
2. Capture timestamp
3. Store comprehensive log in Supabase `agent_logs` table
4. Include all decision details for auditability
5. Return log ID for feedback linking

### Error Handling
```python
try:
    import uuid
    from datetime import datetime

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
    # Log to local file as fallback
    with open("/tmp/agent_decisions.log", "a") as f:
        f.write(f"{datetime.utcnow().isoformat()} | ERROR | {str(e)}\n")
    return ""
```

### Usage Context
- Accessed via `ctx.deps.supabase_client` for database operations
- Logs EVERY agent run for transparency
- Enables feedback loop for improvement
- Fallback to local file logging on database failure

---

## Tool Registration Pattern

All tools should be registered using the `@agent.tool` decorator pattern:

```python
from pydantic_ai import Agent, RunContext
from .dependencies import AgentDependencies
from .tools import classify_input_tool  # Pure function

pm_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT
)

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
```

**Key Pattern Elements:**
- Pure tool functions in `tools.py` (testable independently)
- Agent decorators in `agent.py` (wrap pure functions)
- Access dependencies via `ctx.deps`
- Comprehensive docstrings for LLM understanding
- Return Pydantic models from pure functions
- Convert to dict with `.model_dump()` for agent responses

---

## Dependencies Configuration

Tools access external services and configuration through the `AgentDependencies` dataclass:

```python
from dataclasses import dataclass
from typing import Dict, Any, List
from supabase import Client
from openai import OpenAI

@dataclass
class AgentDependencies:
    """Dependencies for agent execution"""
    supabase_client: Client  # Supabase client for database operations
    openai_client: OpenAI  # OpenAI client for embeddings
    current_sprint_info: Dict[str, Any]  # Current sprint number and capacity
    work_item_templates: Dict[str, str]  # ADO templates (Epic, Feature, User Story)
    launch_readiness_categories: List[Dict[str, str]]  # Category definitions
    gtm_phase_definitions: Dict[str, Dict[str, str]]  # GTM phase definitions
    artifact_registry_path: str  # File storage location
    ado_project_name: str  # Azure DevOps project name
```

---

## Error Handling Strategy

All tools implement consistent error handling:

1. **Try/Except Wrapper**: Wrap all tool logic in try/except
2. **Error Logging**: Log errors with context for debugging
3. **Graceful Degradation**: Return error dictionaries instead of raising exceptions
4. **Fallback Values**: Provide safe defaults (empty lists, zero scores, etc.)
5. **Transparency**: Include error details in return values when appropriate

**Example Error Pattern:**
```python
try:
    # Tool implementation
    return SuccessResult(...)
except ValueError as e:
    logger.error(f"Validation error in tool_name: {e}")
    return ErrorResult(error=str(e), error_type="validation")
except Exception as e:
    logger.error(f"Unexpected error in tool_name: {e}")
    return ErrorResult(error=str(e), error_type="unexpected")
```

---

## Testing Strategy

Each tool should have comprehensive tests:

1. **Unit Tests**: Test pure functions independently with mock dependencies
2. **Integration Tests**: Test tools with TestModel agent
3. **Edge Cases**: Test error conditions, empty inputs, invalid parameters
4. **Validation Tests**: Verify Pydantic model validation works correctly

**Example Test Pattern:**
```python
import pytest
from ai_pm_agent.tools import classify_input_tool
from ai_pm_agent.models import ClassificationResult

@pytest.mark.asyncio
async def test_classify_input_technical():
    """Test classification of technical input"""
    result = await classify_input_tool(
        "Fix authentication bug in login flow",
        launch_readiness_categories=[...],
        gtm_phase_definitions={...}
    )

    assert isinstance(result, ClassificationResult)
    assert result.category == "Technical Readiness"
    assert result.confidence_score > 70.0
    assert result.gtm_phase in ["Foundation", "Validation"]
```

---

## Performance Requirements

- **Semantic Search**: <2 seconds for match_ado_work_items
- **Classification**: <1 second for classify_input
- **Work Item Generation**: <2 seconds for generate_work_item
- **Artifact Tagging**: <3 seconds for tag_artifact
- **Database Operations**: <500ms for CRUD operations

**Optimization Strategies:**
- Cache embeddings to avoid regeneration
- Use connection pooling for Supabase
- Implement rate limiting for OpenAI API
- Batch operations where possible

---

## Security Considerations

1. **API Keys**: All API keys via environment variables (never hardcoded)
2. **Input Validation**: Validate all parameters with Pydantic models
3. **ADO Read-Only**: CRITICAL - Never create/update ADO work items directly
4. **Path Validation**: Sanitize file paths to prevent directory traversal
5. **SQL Injection**: Use parameterized queries for all database operations
6. **Rate Limiting**: Implement exponential backoff for external APIs

---

## Implementation Checklist

- [ ] All 8 tools implemented as pure functions in `tools.py`
- [ ] All 8 Pydantic output models defined in `models.py`
- [ ] All tools registered with `@agent.tool` decorators in `agent.py`
- [ ] AgentDependencies dataclass configured in `dependencies.py`
- [ ] Error handling implemented for all tools
- [ ] Logging added for debugging and monitoring
- [ ] Test suite created with >80% coverage
- [ ] Performance requirements validated
- [ ] Security measures implemented
- [ ] Documentation complete with usage examples

---

## Success Metrics

- **Classification Accuracy**: >90% correct category assignments
- **Search Relevance**: >85% of top matches are truly relevant
- **Template Compliance**: 100% of generated work items follow templates
- **Error Rate**: <2% of tool calls result in errors
- **Performance**: 95% of operations meet latency requirements
- **User Satisfaction**: 60% reduction in manual work item creation

---

**End of Tools Specification**

This specification provides complete guidance for implementing the 8 essential tools for the AI Project Management Agent. Each tool is focused, simple, and follows Pydantic AI best practices for production-ready AI applications.
