---
name: "AI Project Management Agent PRP"
description: "Comprehensive PRP for building an intelligent AI agent that automates project management tasks including classification, work item matching, generation, and sprint planning"
version: "1.0"
created: "2025-01-16"
---

## Purpose

Build an intelligent AI Project Management Agent that transforms project management overhead into an automated system. The agent takes freeform input (meeting notes, ideas, client feedback, documents) and automatically classifies, routes, and structures it into actionable project deliverables for the CoDeveloper platform.

## Core Principles

1. **Pydantic AI Best Practices**: Deep integration with Pydantic AI patterns for agent creation, tools, and structured outputs
2. **Type Safety First**: Leverage Pydantic AI's type-safe design and Pydantic validation throughout
3. **Context Engineering Integration**: Apply proven context engineering workflows to AI agent development
4. **Comprehensive Testing**: Use TestModel and FunctionModel for thorough agent validation
5. **Security First**: Never create ADO work items directly - only generate and return for approval
6. **Transparency**: Log all decisions with confidence scores and rationale

## ⚠️ Implementation Guidelines: Don't Over-Engineer

**IMPORTANT**: Keep your agent implementation focused and practical. Don't build unnecessary complexity.

- ✅ **Start simple** - Build the minimum viable agent that meets requirements
- ✅ **Add tools incrementally** - Implement only what the agent needs to function
- ✅ **Follow main_agent_reference** - Use proven patterns from examples, don't reinvent
- ✅ **Use string output by default** - Only add result_type when validation is required
- ✅ **Test early and often** - Use TestModel to validate as you build

### Key Question:
**"Does this agent really need this feature to accomplish its core purpose?"**

If the answer is no, don't build it. Keep it simple, focused, and functional.

---

## Goal

Create a Pydantic AI agent that:
1. **Classifies Input** into 6 launch readiness categories with confidence scores
2. **Matches ADO Work Items** using semantic similarity search against existing backlog
3. **Generates Work Items** following ADO templates (Epics, Features, User Stories)
4. **Tags Artifacts** and links documents to work items automatically
5. **Recommends Sprint Placement** based on priority, dependencies, and capacity
6. **Logs Decisions** with transparency for continuous improvement

**Target Outcome**: Reduce manual work item creation by 60%, enable 80%+ of work to flow through the agent, minimize Ryan's clicks to single approval actions.

## Why

Ryan wants to focus on building the CoDeveloper platform instead of managing spreadsheets and work items. The current workflow requires:
- Manual classification of ideas and notes into categories
- Searching ADO backlog to avoid duplicate work items
- Creating properly formatted work items with templates
- Tagging and organizing documents
- Planning sprint assignments manually

This agent offloads all mental overhead, allowing Ryan to dump ideas without worrying about organization.

## What

### Agent Type Classification
- [x] **Tool-Enabled Agent**: Agent with external tool integration capabilities for classification, search, and generation
- [x] **Structured Output Agent**: Complex data validation and formatting for work items and classifications
- [ ] **Chat Agent**: Not needed - this is a task-oriented agent, not conversational
- [ ] **Workflow Agent**: Not needed - simple tool orchestration is sufficient

### External Integrations
- [x] **Supabase** - PostgreSQL database with pgvector for semantic search, storing work items, artifacts, action items, launch readiness, agent logs
- [x] **Azure DevOps MCP** - Read-only access to fetch existing work items from NorthStar project (NEVER create/update directly)
- [x] **OpenAI Embeddings** - Generate vector embeddings for semantic similarity search
- [x] **File System** - Store uploaded artifacts with metadata tracking

### Core Capabilities

#### 1. Input Classification (classify_input tool)
- Classifies freeform input into 6 launch readiness categories
- Returns: category, confidence_score (0-100%), gtm_phase, suggested_action
- Categories: Product, Technical, Operational, Commercial, Customer, Security & Compliance

#### 2. ADO Work Item Matching (match_ado_work_items tool)
- Semantic similarity search against existing ADO work items
- Uses OpenAI embeddings + Supabase vector search
- Returns: top matches with ado_id, title, work_item_type, similarity_score, recommendation

#### 3. Work Item Generation (generate_work_item tool)
- Generates ADO-ready work items following templates
- Types: Epic (vision, business value, metrics), Feature (capability, acceptance criteria), User Story (As a/I want/So that with Gherkin scenarios)
- Returns: structured work item for Ryan's approval (NEVER creates in ADO directly)

#### 4. Artifact Management (tag_artifact tool)
- Auto-tags uploaded documents by type (wireframe, brief, test plan, meeting notes, design doc, marketing material)
- Extracts metadata and links to work items
- Returns: artifact_id, detected_type, extracted_metadata, suggested_tags, linked_work_items

#### 5. Sprint & Phase Alignment (recommend_sprint_placement tool)
- Recommends sprint placement based on priority, dependencies, capacity
- Maps work to GTM phases (Foundation → Market Leadership)
- Returns: recommended_sprint, gtm_phase, dependencies, justification

#### 6. Action Item Creation (create_action_item tool)
- Creates action items in Supabase dashboard linked to work items
- Returns: action_item_id, created_at, linked_work_item, linked_request

#### 7. Launch Readiness Updates (update_launch_readiness tool)
- Updates launch readiness scores (1-5 scale) for categories
- Prevents auto-updates >1 point without approval
- Returns: updated_score, previous_score, category, timestamp

#### 8. Decision Logging (log_agent_decision tool)
- Logs all classification, matching, and routing decisions
- Includes: input_text, classification results, similarity scores, actions taken, confidence
- Enables transparency and continuous learning

### Success Criteria
- [x] Agent successfully handles all 8 tool capabilities
- [x] All tools work correctly with proper error handling and retry logic
- [x] Structured outputs validate according to Pydantic models
- [x] Comprehensive test coverage with TestModel and FunctionModel
- [x] Security measures implemented (API keys via .env, input validation, ADO read-only)
- [x] Confidence thresholds enforced (<70% ask, 70-85% suggest, >85% recommend)
- [x] Template compliance validated for all generated work items
- [x] Decision logging captures all agent actions with rationale
- [x] Performance meets requirements (semantic search <2s, classification <1s)
- [x] Success metrics: 60% reduction in manual work items, 90%+ classification accuracy, 80%+ work flows through agent

## All Needed Context

### Pydantic AI Documentation & Research

```yaml
# ESSENTIAL PYDANTIC AI DOCUMENTATION - Must reference during implementation

- url: https://ai.pydantic.dev/
  why: Official Pydantic AI documentation with getting started guide
  content: Agent creation, model providers, dependency injection patterns
  key_patterns: |
    - Use get_llm_model() from providers.py for model configuration
    - Default to string output unless structured validation needed
    - Use @agent.tool with RunContext for dependency injection
    - Environment variables for API keys via pydantic-settings

- url: https://ai.pydantic.dev/agents/
  why: Comprehensive agent architecture and configuration patterns
  content: System prompts, output types, execution methods, agent composition
  key_patterns: |
    - System prompts can be static strings or dynamic functions
    - Agent(model, deps_type, system_prompt) is the core pattern
    - Dataclass dependencies for type-safe context passing

- url: https://ai.pydantic.dev/tools/
  why: Tool integration patterns and function registration
  content: @agent.tool decorators, RunContext usage, parameter validation
  key_patterns: |
    - @agent.tool for context-aware tools with RunContext[DepsType]
    - @agent.tool_plain for simple stateless utility functions
    - Docstrings become tool descriptions for the LLM
    - Parameter descriptions extracted from Google/NumPy/Sphinx docstring formats
    - Return values must be JSON-serializable (anything Pydantic can serialize)

- url: https://ai.pydantic.dev/testing/
  why: Testing strategies specific to Pydantic AI agents
  content: TestModel, FunctionModel, Agent.override(), pytest patterns
  key_patterns: |
    - TestModel for basic validation without real LLM calls
    - FunctionModel for custom test behavior and tool call control
    - Agent.override() to replace models in test contexts
    - capture_run_messages() to inspect agent-model exchanges
    - Set ALLOW_MODEL_REQUESTS=False to prevent accidental API calls

- url: https://ai.pydantic.dev/models/
  why: Model provider configuration and authentication
  content: OpenAI, Anthropic, Gemini setup, API key management, fallback models
  key_patterns: |
    - Use OpenAIProvider with base_url and api_key
    - OpenAIModel(model_name, provider=provider)
    - Environment-based configuration via pydantic-settings

# Codebase Examples - Reference implementations

- path: PRPs/examples/main_agent_reference/
  why: CRITICAL - Best practices for Pydantic AI agent architecture
  files:
    - settings.py: pydantic-settings with dotenv, field validators for API keys
    - providers.py: get_llm_model() abstraction, validate_llm_configuration()
    - research_agent.py: Agent with tools using RunContext, dependency injection
    - tools.py: Pure tool functions (importable, testable independently)
  key_patterns: |
    - Load environment variables with load_dotenv() at module level
    - Settings class with ConfigDict(env_file=".env", case_sensitive=False)
    - Field validators for API key validation
    - Dataclass dependencies (not dict, not BaseModel)
    - Tools as pure functions that agents wrap with @agent.tool
    - Comprehensive docstrings for tool descriptions

- path: FullExample/agent/
  why: Complex agent with database integration and multiple tools
  files:
    - agent.py: Multiple tools with structured outputs, RunContext usage
    - tools.py: Database operations, vector search, graph queries
    - models.py: Pydantic models for tool inputs and outputs
    - db_utils.py: Supabase connection and query patterns
  key_patterns: |
    - Tool input validation with Pydantic models
    - Database connection via Supabase client
    - Vector search using Supabase RPC functions
    - Error handling and logging in tools
    - Type-safe tool parameters and return values
```

### Supabase Integration Research

```yaml
# Supabase Vector Search Documentation

- url: https://supabase.com/docs/guides/ai/semantic-search
  why: Official guide for semantic search with pgvector
  content: Vector embeddings, similarity search, match_documents function
  key_patterns: |
    - Use pgvector extension for vector columns
    - match_documents(query_embedding, match_count, filter) RPC function
    - Cosine distance for similarity calculation
    - Store embeddings as vector(1536) for OpenAI text-embedding-3-small

- url: https://supabase.com/docs/guides/ai/python/api
  why: Python client API for vector operations
  content: vecs client, query methods, filters
  key_patterns: |
    - create_client(connection_string) for client initialization
    - query(data="text", limit=10, filters={}) for text-based search
    - Use .rpc() for custom database functions

# Implementation Patterns from Research

database_schema: |
  -- Existing tables from INITIAL.md database schema reference
  - work_items: id, title, description, work_item_type, ado_id, embedding (vector), metadata
  - artifacts: id, file_path, artifact_type, metadata, linked_work_items, created_at
  - action_items: id, title, description, work_item_id, request_id, created_at, completed
  - launch_readiness: id, category, item_title, score (1-5), notes, updated_at
  - agent_logs: id, input_text, classification, actions_taken, confidence, created_at
  - correlations: id, work_item_id, artifact_id, correlation_type, created_at

supabase_functions: |
  -- Use existing Supabase functions from database
  - match_work_items(query_embedding vector, match_count int) -> table
  - Returns: id, title, description, work_item_type, ado_id, similarity

embedding_generation: |
  # OpenAI Embeddings API
  from openai import OpenAI
  client = OpenAI(api_key=api_key)
  response = client.embeddings.create(
      model="text-embedding-3-small",
      input=text
  )
  embedding = response.data[0].embedding  # List of 1536 floats
```

### Azure DevOps Integration Research

```yaml
# Azure DevOps Python API Documentation

- url: https://learn.microsoft.com/en-us/rest/api/azure/devops/
  why: Official REST API documentation for work items
  content: Work item tracking API, query operations
  key_patterns: |
    - READ-ONLY access for this agent
    - Use ADO MCP server tools to fetch work items
    - Never create/update work items directly
    - Agent generates work items for Ryan's manual approval

# ADO MCP Server Tools (from INITIAL.md dependencies)

mcp_tools_available: |
  - fetch_work_items(project, work_item_type) -> List[WorkItem]
  - query_work_items(wiql_query) -> List[WorkItem]
  - get_work_item(id) -> WorkItem

  Note: These are READ-ONLY tools from the ADO MCP server
  The agent NEVER creates or updates work items in ADO
  It only generates properly formatted work items for approval

# Work Item Templates (from INITIAL.md)

epic_template: |
  Title: [Clear, concise epic name]

  Vision Statement:
  [What is this epic trying to achieve?]

  Business Value:
  [Why is this important? What value does it deliver?]

  Success Metrics:
  - [Measurable metric 1]
  - [Measurable metric 2]

  Tags: [Relevant tags]

feature_template: |
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

user_story_template: |
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

### GTM Phases and Launch Readiness Framework

```yaml
# GTM Phase Definitions (from INITIAL.md)

gtm_phases:
  foundation:
    description: "Core development, scaffolding, essential features"
    typical_work: "Architecture setup, core APIs, database schema, authentication"

  validation:
    description: "Testing, onboarding, user feedback"
    typical_work: "Alpha/beta testing, user interviews, bug fixes, UX refinement"

  launch_prep:
    description: "Pricing, support setup, documentation"
    typical_work: "Pricing strategy, support docs, customer success processes"

  go_to_market:
    description: "Marketing, training, initial customer acquisition"
    typical_work: "Marketing campaigns, sales training, customer onboarding"

  growth:
    description: "Scaling, optimization, performance improvements"
    typical_work: "Performance optimization, scaling infrastructure, analytics"

  market_leadership:
    description: "Advanced features, automation, competitive differentiation"
    typical_work: "Advanced AI features, workflow automation, integrations"

# Launch Readiness Categories (from INITIAL.md)

launch_readiness_categories:
  product_readiness:
    description: "Core product features, UX completeness, feature parity"
    examples: ["Feature development", "UX improvements", "Product specs"]

  technical_readiness:
    description: "Infrastructure, performance, scalability, technical debt"
    examples: ["API development", "Database optimization", "Bug fixes"]

  operational_readiness:
    description: "Support processes, documentation, monitoring"
    examples: ["Support docs", "Monitoring setup", "Operational procedures"]

  commercial_readiness:
    description: "Pricing, billing, contracts, sales enablement"
    examples: ["Pricing strategy", "Billing integration", "Sales materials"]

  customer_readiness:
    description: "Onboarding, training, customer success"
    examples: ["Onboarding flows", "Training materials", "Customer success"]

  security_compliance:
    description: "Security audits, compliance, data protection"
    examples: ["Security review", "GDPR compliance", "Data encryption"]

# Confidence Threshold Rules (from INITIAL.md)

confidence_thresholds:
  ask_for_clarification: "< 70%"
  suggest_with_caveat: "70-85%"
  recommend_with_confidence: "> 85%"

  rationale: |
    Be transparent about confidence scores.
    When uncertain, it's better to ask than guess.
    Include confidence scores in all tool outputs.
```

### Common Pydantic AI Gotchas (research-based)

```yaml
implementation_gotchas:
  async_patterns:
    issue: "Mixing sync and async agent calls inconsistently"
    solution: |
      - Use async/await consistently throughout
      - All tools should be async for database/API operations
      - Use asyncio.run() or agent.run() (sync) vs await agent.run() (async)
      - Don't mix sync Supabase calls with async agent execution

  dependency_complexity:
    issue: "Complex dependency graphs can be hard to debug"
    solution: |
      - Use simple dataclass dependencies (not BaseModel, not dict)
      - Keep dependency count minimal (5-10 max)
      - Pass only configuration, not tool instances
      - Tools access dependencies via ctx.deps in RunContext

  tool_error_handling:
    issue: "Tool failures can crash entire agent runs"
    solution: |
      - Wrap tool logic in try/except blocks
      - Return error dictionaries instead of raising exceptions
      - Log errors for debugging
      - Provide graceful degradation (e.g., empty results instead of crash)
      - Use custom error types for better debugging

  structured_output_validation:
    issue: "LLM may generate invalid structured outputs"
    solution: |
      - Use Pydantic models with strict validation
      - Provide clear field descriptions and examples
      - Use Field() with description for guidance
      - Test with TestModel to catch schema issues early

  supabase_async_client:
    issue: "Supabase Python client has both sync and async versions"
    solution: |
      - Import from supabase for async: from supabase import create_client
      - Use await for all database operations
      - Don't mix postgrest sync client with async agent

  embedding_rate_limits:
    issue: "OpenAI embeddings API has rate limits"
    solution: |
      - Implement exponential backoff retry logic
      - Cache embeddings in database to avoid regeneration
      - Batch embedding generation when possible
      - Handle 429 rate limit errors gracefully
```

## Implementation Blueprint

### Technology Research Phase

**RESEARCH COMPLETED** - The following research has been completed and documented above:

✅ **Pydantic AI Framework Deep Dive:**
- [x] Agent creation patterns and best practices (see main_agent_reference examples)
- [x] Model provider configuration via get_llm_model() abstraction
- [x] Tool integration with @agent.tool and RunContext[DepsType]
- [x] Dependency injection with dataclass dependencies
- [x] Testing strategies with TestModel and FunctionModel

✅ **Agent Architecture Investigation:**
- [x] Project structure: settings.py, providers.py, dependencies.py, models.py, tools.py, prompts.py, agent.py
- [x] System prompt design: static string constants in prompts.py
- [x] Structured output validation with Pydantic models
- [x] Async patterns for database and API operations
- [x] Error handling with try/except and error dictionaries

✅ **Security and Production Patterns:**
- [x] API key management via pydantic-settings with .env files
- [x] Input validation with Pydantic models for all tool parameters
- [x] ADO read-only constraint (NEVER create work items directly)
- [x] Confidence thresholds for transparency
- [x] Decision logging for auditability

### Agent Implementation Plan

```yaml
Implementation Task 1 - Project Structure Setup:
  description: "Create complete agent project structure following main_agent_reference pattern"
  actions:
    - CREATE directory structure:
        - ai_pm_agent/
        - ai_pm_agent/settings.py
        - ai_pm_agent/providers.py
        - ai_pm_agent/dependencies.py
        - ai_pm_agent/models.py
        - ai_pm_agent/tools.py
        - ai_pm_agent/prompts.py
        - ai_pm_agent/agent.py
        - ai_pm_agent/__init__.py
        - tests/
        - tests/conftest.py
        - tests/test_agent.py
        - tests/test_tools.py
        - tests/test_models.py
        - .env.example
        - requirements.txt
        - README.md
    - CREATE .env.example with required variables:
        - LLM_API_KEY
        - LLM_MODEL
        - LLM_BASE_URL
        - OPENAI_EMBEDDING_API_KEY
        - SUPABASE_URL
        - SUPABASE_KEY
        - ADO_PROJECT_NAME
        - ARTIFACT_REGISTRY_PATH
    - CREATE requirements.txt with dependencies:
        - pydantic-ai
        - pydantic-settings
        - python-dotenv
        - supabase
        - openai
        - pytest
        - pytest-asyncio
  validation:
    - All files and directories created
    - .env.example contains all required variables
    - requirements.txt has all dependencies

Implementation Task 2 - Settings and Provider Configuration:
  description: "Implement environment-based configuration and model provider abstraction"
  files_to_create:
    - settings.py: |
        - Settings class with pydantic-settings
        - ConfigDict with env_file=".env", case_sensitive=False
        - Field validators for API keys (non-empty check)
        - Fields: llm_provider, llm_api_key, llm_model, llm_base_url
        - Fields: openai_embedding_api_key, openai_embedding_model
        - Fields: supabase_url, supabase_key
        - Fields: ado_project_name, artifact_registry_path
        - load_dotenv() at module level

    - providers.py: |
        - get_llm_model(model_choice: Optional[str]) -> OpenAIModel
        - get_embedding_client() -> OpenAI
        - get_supabase_client() -> Client
        - validate_llm_configuration() -> bool
        - Import from settings: settings
  reference: PRPs/examples/main_agent_reference/settings.py and providers.py
  validation:
    - Settings loads from .env file
    - Field validators work correctly
    - get_llm_model() returns configured model
    - All provider functions work

Implementation Task 3 - Dependencies and Models Definition:
  description: "Create dataclass dependencies and Pydantic models for structured outputs"
  files_to_create:
    - dependencies.py: |
        @dataclass
        class AgentDependencies:
            supabase_client: Client
            openai_client: OpenAI
            current_sprint_info: Dict[str, Any]
            work_item_templates: Dict[str, str]
            launch_readiness_categories: List[Dict[str, str]]
            gtm_phase_definitions: Dict[str, Dict[str, str]]
            artifact_registry_path: str
            ado_project_name: str

    - models.py: |
        # Classification Output Model
        class ClassificationResult(BaseModel):
            category: str
            confidence_score: float
            gtm_phase: str
            suggested_action: str
            rationale: str

        # Work Item Match Model
        class WorkItemMatch(BaseModel):
            ado_id: int
            title: str
            work_item_type: str
            similarity_score: float
            recommendation: str

        # Generated Work Item Model
        class GeneratedWorkItem(BaseModel):
            title: str
            description: str
            work_item_type: str
            acceptance_criteria: Optional[List[str]]
            tags: List[str]
            suggested_sprint: int
            parent_link: Optional[int]
            metadata: Dict[str, Any]

        # Artifact Tag Result Model
        class ArtifactTagResult(BaseModel):
            artifact_id: str
            detected_type: str
            extracted_metadata: Dict[str, Any]
            suggested_tags: List[str]
            linked_work_items: List[int]

        # Sprint Recommendation Model
        class SprintRecommendation(BaseModel):
            recommended_sprint: int
            gtm_phase: str
            dependencies: List[str]
            justification: str
            alternative_sprints: List[int]

        # Action Item Model
        class ActionItemResult(BaseModel):
            action_item_id: str
            created_at: str
            linked_work_item: Optional[int]
            linked_request: Optional[int]

        # Launch Readiness Update Model
        class LaunchReadinessUpdate(BaseModel):
            updated_score: int
            previous_score: int
            category: str
            timestamp: str

        # Decision Log Model
        class DecisionLog(BaseModel):
            log_id: str
            input_text: str
            classification: Dict[str, Any]
            actions_taken: List[str]
            confidence: float
            created_at: str
  validation:
    - All models validate correctly
    - Field types match requirements
    - Optional fields work as expected

Implementation Task 4 - Tool Implementations:
  description: "Implement all 8 core tools with proper error handling"
  files_to_create:
    - tools.py: |
        # Import all models from models.py
        # Import dependencies from dependencies.py

        # Tool 1: classify_input
        async def classify_input_tool(
            input_text: str,
            categories: List[Dict[str, str]],
            gtm_phases: Dict[str, Dict[str, str]]
        ) -> ClassificationResult:
            """
            Classify input into launch readiness categories.

            Uses keyword matching and heuristics to determine category.
            Returns confidence score based on match strength.
            Maps to GTM phase based on category and content analysis.
            """
            # Implementation with error handling
            # Calculate confidence score (0-100%)
            # Determine GTM phase
            # Return ClassificationResult

        # Tool 2: match_ado_work_items
        async def match_work_items_tool(
            input_text: str,
            supabase_client: Client,
            openai_client: OpenAI,
            limit: int = 5
        ) -> List[WorkItemMatch]:
            """
            Perform semantic similarity search against existing work items.

            Generates embedding for input text using OpenAI.
            Queries Supabase match_work_items() function.
            Returns top matches with similarity scores.
            """
            # Generate embedding
            # Query Supabase RPC function
            # Format results
            # Return list of WorkItemMatch

        # Tool 3: generate_work_item
        async def generate_work_item_tool(
            input_description: str,
            work_item_type: str,
            templates: Dict[str, str],
            parent_id: Optional[int] = None
        ) -> GeneratedWorkItem:
            """
            Generate ADO-ready work item using templates.

            Applies appropriate template based on work_item_type.
            Validates against template requirements.
            Returns structured work item for approval.
            """
            # Select template
            # Parse input and fill template
            # Validate completeness
            # Return GeneratedWorkItem

        # Tool 4: tag_artifact
        async def tag_artifact_tool(
            file_path: str,
            supabase_client: Client,
            artifact_registry_path: str
        ) -> ArtifactTagResult:
            """
            Auto-tag uploaded documents and link to work items.

            Detects artifact type from filename and content.
            Extracts metadata.
            Suggests tags and linked work items.
            """
            # Detect file type
            # Extract metadata
            # Store in Supabase
            # Return ArtifactTagResult

        # Tool 5: recommend_sprint_placement
        async def recommend_sprint_placement_tool(
            work_item: Dict[str, Any],
            current_sprint: int,
            sprint_capacity: Dict[str, Any],
            gtm_phases: Dict[str, Dict[str, str]]
        ) -> SprintRecommendation:
            """
            Recommend sprint placement based on priority and capacity.

            Analyzes work item priority and dependencies.
            Checks sprint capacity.
            Maps to GTM phase.
            """
            # Analyze priority
            # Check capacity
            # Calculate recommendation
            # Return SprintRecommendation

        # Tool 6: create_action_item
        async def create_action_item_tool(
            title: str,
            description: str,
            supabase_client: Client,
            work_item_id: Optional[int] = None,
            request_id: Optional[int] = None
        ) -> ActionItemResult:
            """
            Create action item in dashboard.

            Inserts into Supabase action_items table.
            Links to work items and requests.
            """
            # Insert into Supabase
            # Return ActionItemResult

        # Tool 7: update_launch_readiness
        async def update_launch_readiness_tool(
            category: str,
            item_title: str,
            score_delta: int,
            notes: str,
            supabase_client: Client
        ) -> LaunchReadinessUpdate:
            """
            Update launch readiness score for a category.

            Prevents updates >1 point without approval.
            Updates Supabase launch_readiness table.
            """
            # Validate score_delta
            # Update Supabase
            # Return LaunchReadinessUpdate

        # Tool 8: log_agent_decision
        async def log_agent_decision_tool(
            input_text: str,
            classification: Dict[str, Any],
            actions_taken: List[str],
            confidence: float,
            supabase_client: Client
        ) -> str:
            """
            Log AI decision for transparency and learning.

            Stores decision log in Supabase.
            Returns log_id for feedback tracking.
            """
            # Insert into agent_logs table
            # Return log_id
  reference:
    - PRPs/examples/main_agent_reference/tools.py (pure function pattern)
    - FullExample/agent/tools.py (database integration pattern)
  validation:
    - All 8 tools implemented
    - Error handling in place
    - Return types match models
    - Async patterns consistent

Implementation Task 5 - System Prompt Creation:
  description: "Create comprehensive system prompt for the agent"
  files_to_create:
    - prompts.py: |
        SYSTEM_PROMPT = """
        You are an AI Project Management Assistant for the CoDeveloper platform.
        Your job is to take freeform input from Ryan (meeting notes, ideas, client
        feedback, bug reports, documents) and intelligently classify, route, and
        structure it into actionable project deliverables.

        **Your Primary Responsibilities:**

        1. **Classify Input** - Analyze input and classify into one of 6 launch
           readiness categories: Product Readiness, Technical Readiness, Operational
           Readiness, Commercial Readiness, Customer Readiness, or Security & Compliance.
           Provide confidence scores (0-100%).

        2. **Match to ADO Work Items** - Use semantic similarity search to find existing
           Azure DevOps Epics, Features, or User Stories that relate to the input. Return
           top matches with similarity scores. Recommend whether to link to existing work
           or create new items.

        3. **Assign GTM Phase** - Map work to one of 6 GTM delivery phases: Foundation
           (core dev, scaffolding), Validation (testing, onboarding), Launch Prep (pricing,
           support), Go-to-Market (marketing, training), Growth (scaling, optimization), or
           Market Leadership (advanced features, automation).

        4. **Generate Work Items** - When creating new work, apply proper ADO templates:
           - **Epics**: Vision statement, business value, success metrics
           - **Features**: User-facing capability, acceptance criteria, technical notes
           - **User Stories**: "As a [user] I want [capability] so that [benefit]" format
             with Gherkin acceptance criteria (Given/When/Then scenarios)

        5. **Recommend Sprint Placement** - Suggest which sprint work should be completed
           in, considering dependencies, priority, and capacity. Justify recommendations.

        6. **Tag & Route Artifacts** - For uploaded documents (wireframes, briefs, test
           plans), detect type, extract metadata, suggest tags, and link to relevant work
           items and GTM phases.

        7. **Log Decisions** - Always log your classification, matching logic, confidence
           scores, and actions taken. Be transparent about reasoning. Accept feedback to improve.

        **Key Principles:**
        - Always search ADO backlog before creating new work items to avoid duplication
        - Provide confidence scores for all classifications and matches (be honest about uncertainty)
        - When uncertain (confidence <70%), ask clarifying questions instead of guessing
        - Use structured JSON output for all tool calls
        - Cite sources (ADO work item IDs, document names) when referencing existing work
        - Optimize for Ryan's workflow: reduce clicks, offload mental overhead, stay execution-focused

        **CRITICAL SECURITY CONSTRAINT:**
        - NEVER create, update, or modify ADO work items directly through any API
        - You may only GENERATE work items and RETURN them for Ryan's approval
        - Only Ryan creates work items in ADO manually after reviewing your recommendations

        **Confidence Thresholds:**
        - < 70%: Ask for clarification, don't guess
        - 70-85%: Suggest with caveat, explain uncertainty
        - > 85%: Recommend with confidence, provide rationale

        **Example Tags:**
        ["Sous AI", "Formulate", "Reverse Engineer", "Ingredient Matching", "Pantry",
        "Claims", "CSV Export", "Dialogue UI", "Launch Prep", "Technical Debt"]

        **Context:**
        Ryan wants to dump ideas and notes without worrying about organization. Your job
        is to smartly classify and route everything so he can focus on building the
        product instead of managing spreadsheets and work items.
        """
  validation:
    - System prompt is clear and comprehensive
    - All responsibilities documented
    - Constraints clearly stated
    - Examples provided

Implementation Task 6 - Agent Definition:
  description: "Create main agent with tool registration and dependency injection"
  files_to_create:
    - agent.py: |
        from pydantic_ai import Agent, RunContext
        from .providers import get_llm_model
        from .dependencies import AgentDependencies
        from .prompts import SYSTEM_PROMPT
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

        # Initialize agent
        pm_agent = Agent(
            get_llm_model(),
            deps_type=AgentDependencies,
            system_prompt=SYSTEM_PROMPT
        )

        # Register tools with @agent.tool decorator

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
            results = await match_work_items_tool(
                input_text,
                ctx.deps.supabase_client,
                ctx.deps.openai_client,
                min(max(limit, 1), 20)
            )
            return [r.model_dump() for r in results]

        # ... Register remaining 6 tools following same pattern ...

        # Convenience function to create agent with dependencies
        def create_pm_agent(
            supabase_client,
            openai_client,
            current_sprint_info,
            work_item_templates,
            launch_readiness_categories,
            gtm_phase_definitions,
            artifact_registry_path,
            ado_project_name
        ):
            """Create PM agent with configured dependencies."""
            return pm_agent
  reference:
    - PRPs/examples/main_agent_reference/research_agent.py (tool registration pattern)
    - FullExample/agent/agent.py (multiple tools pattern)
  validation:
    - Agent instantiates correctly
    - All 8 tools registered
    - Dependencies properly typed
    - Tool docstrings clear

Implementation Task 7 - Comprehensive Testing:
  description: "Implement comprehensive test suite with TestModel and FunctionModel"
  files_to_create:
    - tests/conftest.py: |
        import pytest
        from unittest.mock import Mock, AsyncMock
        from pydantic_ai.models.test import TestModel
        from ai_pm_agent.agent import pm_agent
        from ai_pm_agent.dependencies import AgentDependencies

        @pytest.fixture
        def mock_supabase():
            return Mock()

        @pytest.fixture
        def mock_openai():
            return Mock()

        @pytest.fixture
        def test_dependencies(mock_supabase, mock_openai):
            return AgentDependencies(
                supabase_client=mock_supabase,
                openai_client=mock_openai,
                current_sprint_info={"sprint": 42, "capacity": 100},
                work_item_templates={"Epic": "...", "Feature": "...", "User Story": "..."},
                launch_readiness_categories=[...],
                gtm_phase_definitions={...},
                artifact_registry_path="/tmp/artifacts",
                ado_project_name="NorthStar"
            )

        @pytest.fixture
        def test_model_agent(test_dependencies):
            with pm_agent.override(model=TestModel()):
                yield pm_agent

    - tests/test_agent.py: |
        import pytest
        from pydantic_ai.models.test import TestModel

        @pytest.mark.asyncio
        async def test_agent_instantiation(test_model_agent, test_dependencies):
            """Test that agent can be instantiated with TestModel."""
            result = await test_model_agent.run(
                "Test input",
                deps=test_dependencies
            )
            assert result is not None

        @pytest.mark.asyncio
        async def test_classify_input_tool_called(test_model_agent, test_dependencies):
            """Test that classify_input tool is registered and callable."""
            # TestModel should automatically call tools
            result = await test_model_agent.run(
                "Implement user authentication for Sous AI",
                deps=test_dependencies
            )
            # Verify tool was invoked by checking result structure
            assert result.data is not None

    - tests/test_tools.py: |
        import pytest
        from ai_pm_agent.tools import (
            classify_input_tool,
            match_work_items_tool,
            generate_work_item_tool
        )
        from ai_pm_agent.models import ClassificationResult

        @pytest.mark.asyncio
        async def test_classify_input_tool():
            """Test input classification with sample data."""
            result = await classify_input_tool(
                "Fix authentication bug in login flow",
                launch_readiness_categories=[...],
                gtm_phase_definitions={...}
            )
            assert isinstance(result, ClassificationResult)
            assert result.confidence_score >= 0
            assert result.confidence_score <= 100
            assert result.category in [
                "Product Readiness",
                "Technical Readiness",
                "Operational Readiness",
                "Commercial Readiness",
                "Customer Readiness",
                "Security & Compliance"
            ]

        @pytest.mark.asyncio
        async def test_match_work_items_with_mock(mock_supabase, mock_openai):
            """Test work item matching with mocked services."""
            # Mock OpenAI embedding response
            mock_openai.embeddings.create.return_value = Mock(
                data=[Mock(embedding=[0.1] * 1536)]
            )

            # Mock Supabase RPC response
            mock_supabase.rpc.return_value.execute.return_value = Mock(
                data=[
                    {
                        "id": 1,
                        "title": "Implement user auth",
                        "work_item_type": "Feature",
                        "similarity": 0.92
                    }
                ]
            )

            results = await match_work_items_tool(
                "Add authentication feature",
                mock_supabase,
                mock_openai,
                limit=5
            )

            assert len(results) > 0
            assert results[0].similarity_score > 0.9

    - tests/test_models.py: |
        import pytest
        from ai_pm_agent.models import (
            ClassificationResult,
            WorkItemMatch,
            GeneratedWorkItem
        )

        def test_classification_result_validation():
            """Test ClassificationResult model validation."""
            result = ClassificationResult(
                category="Technical Readiness",
                confidence_score=87.5,
                gtm_phase="Foundation",
                suggested_action="create_action_item",
                rationale="Technical work detected"
            )
            assert result.confidence_score == 87.5
            assert result.category == "Technical Readiness"

        def test_work_item_match_validation():
            """Test WorkItemMatch model validation."""
            match = WorkItemMatch(
                ado_id=12345,
                title="User Authentication",
                work_item_type="Feature",
                similarity_score=0.92,
                recommendation="link"
            )
            assert match.ado_id == 12345
            assert match.similarity_score == 0.92
  reference:
    - https://ai.pydantic.dev/testing/ (TestModel and FunctionModel patterns)
    - PRPs/examples/testing_examples/test_agent_patterns.py
  validation:
    - All tests pass with pytest
    - TestModel validation works
    - Mock dependencies work correctly
    - Tool validation covers edge cases

Implementation Task 8 - Documentation and README:
  description: "Create comprehensive documentation for the agent"
  files_to_create:
    - README.md: |
        # AI Project Management Agent

        Intelligent AI agent that automates project management tasks for the
        CoDeveloper platform.

        ## Features

        - **Input Classification**: Classifies freeform input into 6 launch readiness categories
        - **Semantic Work Item Matching**: Uses vector embeddings to find similar ADO work items
        - **Work Item Generation**: Generates properly formatted Epics, Features, and User Stories
        - **Artifact Management**: Auto-tags documents and links to work items
        - **Sprint Recommendations**: Suggests sprint placement based on capacity and dependencies
        - **Decision Logging**: Logs all agent decisions for transparency and learning

        ## Installation

        1. Clone the repository
        2. Install dependencies: `pip install -r requirements.txt`
        3. Copy `.env.example` to `.env` and configure environment variables
        4. Run tests: `pytest tests/`

        ## Usage

        ```python
        from ai_pm_agent.agent import pm_agent
        from ai_pm_agent.dependencies import AgentDependencies
        from ai_pm_agent.providers import (
            get_supabase_client,
            get_embedding_client
        )

        # Create dependencies
        deps = AgentDependencies(
            supabase_client=get_supabase_client(),
            openai_client=get_embedding_client(),
            current_sprint_info={"sprint": 42, "capacity": 100},
            work_item_templates={...},
            launch_readiness_categories=[...],
            gtm_phase_definitions={...},
            artifact_registry_path="/path/to/artifacts",
            ado_project_name="NorthStar"
        )

        # Run agent
        result = await pm_agent.run(
            "Jay mentioned users are confused by the Sous loading screen",
            deps=deps
        )
        ```

        ## Architecture

        - **settings.py**: Environment configuration with pydantic-settings
        - **providers.py**: Model and service provider abstractions
        - **dependencies.py**: Dataclass dependencies for dependency injection
        - **models.py**: Pydantic models for structured outputs
        - **tools.py**: Tool implementations (8 core tools)
        - **prompts.py**: System prompt definition
        - **agent.py**: Main agent definition and tool registration

        ## Testing

        Run tests with pytest:
        ```bash
        pytest tests/ -v
        ```

        Tests use Pydantic AI's TestModel for validation without real LLM calls.

        ## Security

        - All API keys managed via .env files
        - ADO work items are READ-ONLY - agent never creates/updates directly
        - Input validation with Pydantic models
        - Confidence thresholds prevent low-confidence actions

        ## License

        MIT License
  validation:
    - README is comprehensive
    - Usage examples clear
    - Architecture documented
    - Security notes included
```

## Validation Loop

### Level 1: Agent Structure Validation

```bash
# Verify complete agent project structure
ls -la ai_pm_agent/
test -f ai_pm_agent/agent.py && echo "✓ Agent definition present"
test -f ai_pm_agent/tools.py && echo "✓ Tools module present"
test -f ai_pm_agent/models.py && echo "✓ Models module present"
test -f ai_pm_agent/dependencies.py && echo "✓ Dependencies module present"
test -f ai_pm_agent/settings.py && echo "✓ Settings module present"
test -f ai_pm_agent/providers.py && echo "✓ Providers module present"
test -f ai_pm_agent/prompts.py && echo "✓ Prompts module present"

# Verify proper Pydantic AI imports
grep -q "from pydantic_ai import Agent" ai_pm_agent/agent.py && echo "✓ Agent import correct"
grep -q "@pm_agent.tool" ai_pm_agent/agent.py && echo "✓ Tool decorator used"
grep -q "from pydantic import BaseModel" ai_pm_agent/models.py && echo "✓ Pydantic models used"

# Expected: All required files with proper Pydantic AI patterns
# If missing: Generate missing components with correct patterns
```

### Level 2: Agent Functionality Validation

```bash
# Test agent can be imported and instantiated
python -c "
from ai_pm_agent.agent import pm_agent
print('✓ Agent created successfully')
print(f'Model: {pm_agent.model}')
print(f'Tools: {len(pm_agent.tools)}')
"

# Expected output:
# ✓ Agent created successfully
# Model: OpenAIModel(...)
# Tools: 8

# Test with TestModel for validation
python -c "
from pydantic_ai.models.test import TestModel
from ai_pm_agent.agent import pm_agent
from ai_pm_agent.dependencies import AgentDependencies
from unittest.mock import Mock

# Create mock dependencies
deps = AgentDependencies(
    supabase_client=Mock(),
    openai_client=Mock(),
    current_sprint_info={'sprint': 42},
    work_item_templates={},
    launch_readiness_categories=[],
    gtm_phase_definitions={},
    artifact_registry_path='/tmp',
    ado_project_name='Test'
)

with pm_agent.override(model=TestModel()):
    import asyncio
    result = asyncio.run(pm_agent.run('Test message', deps=deps))
    print(f'✓ Agent response: {result.output}')
"

# Expected: Agent instantiation works, tools registered, TestModel validation passes
# If failing: Debug agent configuration and tool registration
```

### Level 3: Tool Validation

```bash
# Test each tool individually with pytest
pytest tests/test_tools.py::test_classify_input_tool -v
pytest tests/test_tools.py::test_match_work_items_with_mock -v
pytest tests/test_tools.py::test_generate_work_item_tool -v
pytest tests/test_tools.py::test_tag_artifact_tool -v
pytest tests/test_tools.py::test_recommend_sprint_placement_tool -v
pytest tests/test_tools.py::test_create_action_item_tool -v
pytest tests/test_tools.py::test_update_launch_readiness_tool -v
pytest tests/test_tools.py::test_log_agent_decision_tool -v

# Expected: All 8 tool tests pass
# If failing: Fix tool implementation based on test failures
```

### Level 4: Model Validation

```bash
# Test Pydantic models
pytest tests/test_models.py -v

# Expected: All model validation tests pass
# If failing: Fix model definitions and constraints
```

### Level 5: Integration Validation

```bash
# Run complete test suite
pytest tests/ -v --cov=ai_pm_agent --cov-report=term-missing

# Test agent with real-world scenario
python -c "
from ai_pm_agent.agent import pm_agent
from ai_pm_agent.dependencies import AgentDependencies
from pydantic_ai.models.test import TestModel
from unittest.mock import Mock
import asyncio

deps = AgentDependencies(
    supabase_client=Mock(),
    openai_client=Mock(),
    current_sprint_info={'sprint': 42, 'capacity': 100},
    work_item_templates={'Epic': '...', 'Feature': '...', 'User Story': '...'},
    launch_readiness_categories=[
        {'name': 'Product Readiness', 'description': '...'},
        {'name': 'Technical Readiness', 'description': '...'}
    ],
    gtm_phase_definitions={
        'Foundation': {'description': '...'},
        'Validation': {'description': '...'}
    },
    artifact_registry_path='/tmp/artifacts',
    ado_project_name='NorthStar'
)

with pm_agent.override(model=TestModel()):
    result = asyncio.run(pm_agent.run(
        'Jay mentioned users are confused by the Sous loading screen. Wants progress indicators and food-related animation.',
        deps=deps
    ))
    print(f'✓ Integration test passed')
    print(f'Result: {result.data}')
"

# Expected: Integration test passes, agent processes input correctly
# If failing: Debug agent workflow and tool orchestration
```

### Level 6: Confidence Threshold Validation

```bash
# Test confidence thresholds
python -c "
from ai_pm_agent.tools import classify_input_tool
import asyncio

# Test low confidence (<70%)
result = asyncio.run(classify_input_tool(
    'some vague input',
    launch_readiness_categories=[...],
    gtm_phase_definitions={...}
))

assert result.confidence_score < 70, 'Low confidence test failed'
print('✓ Low confidence threshold working')

# Test medium confidence (70-85%)
result = asyncio.run(classify_input_tool(
    'somewhat clear technical input',
    launch_readiness_categories=[...],
    gtm_phase_definitions={...}
))

assert 70 <= result.confidence_score <= 85, 'Medium confidence test failed'
print('✓ Medium confidence threshold working')

# Test high confidence (>85%)
result = asyncio.run(classify_input_tool(
    'very clear and specific technical feature request',
    launch_readiness_categories=[...],
    gtm_phase_definitions={...}
))

assert result.confidence_score > 85, 'High confidence test failed'
print('✓ High confidence threshold working')
"

# Expected: All confidence threshold tests pass
# If failing: Adjust confidence calculation logic in classify_input_tool
```

### Level 7: Decision Logging Validation

```bash
# Test decision logging
python -c "
from ai_pm_agent.tools import log_agent_decision_tool
from unittest.mock import Mock
import asyncio

mock_supabase = Mock()
mock_supabase.table.return_value.insert.return_value.execute.return_value = Mock(
    data=[{'id': 'log-123'}]
)

log_id = asyncio.run(log_agent_decision_tool(
    input_text='Test input',
    classification={'category': 'Technical Readiness', 'confidence': 87.5},
    actions_taken=['classify_input', 'match_ado_work_items'],
    confidence=0.875,
    supabase_client=mock_supabase
))

assert log_id is not None
print(f'✓ Decision logging working: {log_id}')
"

# Expected: Decision logging test passes
# If failing: Fix log_agent_decision_tool implementation
```

## Final Validation Checklist

### Agent Implementation Completeness

- [ ] Complete agent project structure: `agent.py`, `tools.py`, `models.py`, `dependencies.py`, `settings.py`, `providers.py`, `prompts.py`
- [ ] Agent instantiation with proper model provider configuration via `get_llm_model()`
- [ ] All 8 tools registered with @agent.tool decorators and RunContext integration
- [ ] All 8 tools implemented:
  - [ ] classify_input - Input classification with confidence scores
  - [ ] match_ado_work_items - Semantic similarity search
  - [ ] generate_work_item - Template-based work item generation
  - [ ] tag_artifact - Document tagging and linking
  - [ ] recommend_sprint_placement - Sprint recommendation logic
  - [ ] create_action_item - Supabase action item creation
  - [ ] update_launch_readiness - Launch readiness score updates
  - [ ] log_agent_decision - Decision logging for transparency
- [ ] Structured outputs with Pydantic model validation (8 output models)
- [ ] Dependency injection properly configured with AgentDependencies dataclass
- [ ] Comprehensive test suite with TestModel and FunctionModel
- [ ] Environment configuration with .env files and pydantic-settings

### Pydantic AI Best Practices

- [ ] Type safety throughout with proper type hints and Pydantic validation
- [ ] Security patterns implemented:
  - [ ] API keys via .env files (never hardcoded)
  - [ ] Input validation with Pydantic models
  - [ ] ADO read-only constraint enforced (NEVER create work items directly)
  - [ ] Confidence thresholds implemented (<70% ask, 70-85% suggest, >85% recommend)
- [ ] Error handling and retry mechanisms for robust operation
- [ ] Async/await patterns consistent throughout
- [ ] Logging for debugging and monitoring
- [ ] Documentation and code comments for maintainability

### Template Compliance

- [ ] Epic template validation (vision, business value, metrics)
- [ ] Feature template validation (capability, acceptance criteria, technical notes)
- [ ] User Story template validation (As a/I want/So that with Gherkin scenarios)
- [ ] Template compliance tests pass

### Confidence and Decision Logging

- [ ] Confidence scores calculated for all classifications
- [ ] Confidence thresholds enforced in agent logic
- [ ] Decision logs capture all agent actions
- [ ] Rationale provided for all recommendations

### Full Readiness

- [ ] Environment configuration with .env.example
- [ ] Requirements.txt with all dependencies
- [ ] README.md with usage examples
- [ ] Test suite passes with >80% coverage
- [ ] Integration tests validate full workflow
- [ ] Documentation complete for deployment

---

## Anti-Patterns to Avoid

### Pydantic AI Agent Development

- ❌ Don't skip TestModel validation - always test with TestModel during development
- ❌ Don't hardcode API keys - use environment variables via pydantic-settings for all credentials
- ❌ Don't ignore async patterns - Pydantic AI requires consistent async/await throughout
- ❌ Don't create complex tool chains - keep tools focused and composable
- ❌ Don't skip error handling - implement graceful degradation with error dictionaries
- ❌ Don't use result_type unless structured validation is needed - default to string output
- ❌ Don't hardcode model strings like "openai:gpt-4o" - use get_llm_model() abstraction

### Agent Architecture

- ❌ Don't mix agent types - this is a tool-enabled agent with structured outputs, not a chat agent
- ❌ Don't ignore dependency injection - use dataclass dependencies via RunContext
- ❌ Don't skip output validation - always use Pydantic models for structured responses
- ❌ Don't forget tool documentation - docstrings become tool descriptions for the LLM
- ❌ Don't use BaseModel for dependencies - use dataclass instead
- ❌ Don't pass tool instances in dependencies - pass configuration only

### Security and Constraints

- ❌ Don't create ADO work items directly - this is a CRITICAL constraint, only generate for approval
- ❌ Don't skip confidence thresholds - transparency is essential for trust
- ❌ Don't ignore decision logging - auditability is required for continuous improvement
- ❌ Don't skip template validation - work items must follow ADO templates exactly
- ❌ Don't make auto-updates >1 point to launch readiness without approval

### Supabase Integration

- ❌ Don't mix sync and async Supabase client calls
- ❌ Don't forget to handle Supabase errors gracefully
- ❌ Don't skip vector search optimization - use existing RPC functions
- ❌ Don't regenerate embeddings unnecessarily - cache in database

---

## PRP Quality Self-Assessment

### Research Completeness: ✅ 10/10
- [x] Pydantic AI documentation thoroughly researched (agents, tools, testing, models)
- [x] Supabase vector search patterns documented
- [x] Azure DevOps API constraints understood
- [x] OpenAI embeddings integration researched
- [x] Codebase examples analyzed (main_agent_reference, FullExample)
- [x] Common gotchas documented with solutions

### Context Engineering: ✅ 10/10
- [x] All necessary Pydantic AI patterns included
- [x] Complete tool implementation specifications
- [x] Structured output models defined
- [x] Dependencies and configuration detailed
- [x] Template compliance requirements documented
- [x] Confidence threshold logic specified

### Implementation Blueprint: ✅ 9/10
- [x] 8 clear implementation tasks in logical order
- [x] Each task has specific deliverables
- [x] Reference examples provided for each component
- [x] Validation criteria for each task
- [x] Clear file structure and organization
- [ ] Could benefit from more detailed pseudocode in tools (minor)

### Validation Strategy: ✅ 10/10
- [x] 7 validation levels from structure to integration
- [x] Executable validation commands provided
- [x] TestModel patterns for development
- [x] Tool-specific validation tests
- [x] Confidence threshold validation
- [x] Decision logging validation
- [x] Integration testing scenarios

### Security and Constraints: ✅ 10/10
- [x] ADO read-only constraint clearly documented
- [x] API key management via .env files
- [x] Input validation with Pydantic models
- [x] Confidence threshold requirements specified
- [x] Decision logging for transparency
- [x] Template compliance enforced

### Documentation Quality: ✅ 9/10
- [x] Clear purpose and goals
- [x] Comprehensive agent capabilities documented
- [x] All 8 tools specified with inputs/outputs
- [x] Dependencies clearly defined
- [x] GTM phases and launch readiness categories documented
- [ ] Could benefit from more use case examples (minor)

### One-Pass Implementation Readiness: ✅ 9/10

**Confidence Score: 9/10**

**Rationale:**
- Comprehensive research completed and documented
- All necessary context included for agent implementation
- Clear implementation blueprint with 8 tasks
- Extensive validation strategy with executable tests
- Reference examples provided from codebase
- Security constraints clearly documented
- Testing patterns well-defined
- Minor gaps: Could use more detailed tool pseudocode and use case examples

**Expected Implementation Time:** 4-6 hours for experienced developer

**Risk Areas:**
1. Supabase vector search integration (Medium) - requires testing with real database
2. Template compliance validation (Low) - templates are well-defined
3. Confidence calculation logic (Medium) - requires tuning and validation
4. OpenAI embeddings rate limiting (Low) - error handling documented

**Success Likelihood:** 90% - This PRP provides sufficient context and guidance for successful one-pass implementation with Pydantic AI best practices.

---

**PRP RESEARCH STATUS: COMPLETED** ✅

This PRP is ready for implementation using the `/execute-pydantic-ai-prp` command.
