## PROJECT LINEAGE:

**Phase 1:** See `INITIAL-Phase1.md` → Generated `ai-project-management-agent.md` PRP → Implemented and tested  
**Phase 2:** This document → Will generate new PRP → Adds Hybrid RAG capabilities

---

## FEATURE:

Build an intelligent AI Project Management Agent with **Hybrid RAG capabilities** that transforms project management overhead into an automated system. The agent takes freeform input (meeting notes, ideas, client feedback, documents) and automatically classifies, routes, and structures it into actionable Azure DevOps (ADO) work items.

This is a **Phase 2 Enhancement Project** - Phase 1 (semantic search foundation) is already implemented based on `INITIAL-Phase1.md`. Phase 2 will add Knowledge Graph retrieval (Neo4j/Graphiti) and Hybrid Search (TSVector) to create a complete three-layer search system.

### Already Implemented (Phase 1)
✅ **Core agent structure** - `ai_pm_agent/agent.py` with 8 registered tools  
✅ **Semantic similarity search** - OpenAI embeddings + Supabase pgvector for work item matching  
✅ **8 core tools** - classify_input, match_work_items, generate_work_item, tag_artifact, recommend_sprint, create_action_item, update_launch_readiness, log_agent_decision  
✅ **Pydantic models** - All tool input/output validation in `models.py`  
✅ **Dependencies** - `AgentDependencies` dataclass with Supabase client, OpenAI client, templates  
✅ **Settings management** - python-dotenv configuration in `settings.py`  
✅ **Comprehensive tests** - All tests passing with proper mocking (TestModel, AsyncMock)  
✅ **Supabase integration** - Tables for work_items, action_items, correlations, launch_readiness, artifacts, agent_logs  

### Phase 2 Requirements (To Be Implemented)

**Add Knowledge Graph Retrieval:**
- Integrate Neo4j + Graphiti for mapping ADO work item relationships (Epic → Feature → User Story → Bug)
- Track temporal changes in work item status, assignments, and dependencies
- Enable relationship-based queries like "Show all Features blocked by Epic #1234"

**Add Hybrid Search:**
- Implement TSVector full-text search in PostgreSQL for keyword-based retrieval
- Combine semantic (pgvector) + keyword (TSVector) search with configurable weighting
- Enable queries like "security bug in login API" to match both semantic meaning and exact keywords

**Unified Search Tool:**
- Create `comprehensive_work_item_search` tool that queries all three strategies in parallel:
  1. Semantic search (pgvector) - already implemented
  2. Knowledge graph (Graphiti) - to be added
  3. Hybrid search (TSVector) - to be added
- Merge and rank results by relevance score

**Implementation References:**
- Copy architecture patterns from `FullExample/agent/graph_utils.py` for Neo4j/Graphiti integration
- Copy hybrid search patterns from `FullExample/agent/db_utils.py` for TSVector implementation
- Reuse existing `FullExample/sql/schema.sql` patterns for hybrid search functions

---

## TOOLS:

### Phase 1 Tools (Already Implemented)

- **classify_input(input_text: str) -> ClassificationResult**: Classifies freeform input into launch readiness categories. Returns classification result with category (Product/Technical/Operational/Commercial/Customer/Security), confidence_score (0-100%), gtm_phase (Foundation/Validation/Launch Prep/Go-to-Market/Growth/Market Leadership), and suggested_action (create_action_item, link_to_work_item, update_readiness).

- **match_work_items(input_text: str, limit: int = 5) -> List[WorkItemMatch]**: Performs semantic similarity search against existing ADO work items using OpenAI embeddings and Supabase pgvector. Returns matches with ado_id, title, work_item_type, similarity_score (0-100%), and recommendation (link/create_new).

- **generate_work_item(input_description: str, work_item_type: str, parent_id: Optional[int] = None) -> GeneratedWorkItem**: Generates ADO-ready work item with proper template structure. Returns structured work item with title, description, acceptance_criteria (Gherkin format for User Stories), tags, suggested_sprint, and parent_link. Templates: Epic (vision, business value, metrics), Feature (capability, acceptance criteria, technical notes), User Story (As a/I want/So that with scenarios).

- **tag_artifact(file_path: str, metadata: Dict) -> ArtifactTagResult**: Auto-tags uploaded documents/artifacts by type (wireframe, brief, test plan, meeting notes, design doc, marketing material). Returns artifact_id, detected_type, extracted_metadata, suggested_tags, linked_work_items, and gtm_phase assignment.

- **recommend_sprint_placement(work_item: Dict, current_sprint: int, sprint_capacity: Dict) -> SprintRecommendation**: Recommends sprint placement based on priority, dependencies, and capacity. Returns recommended_sprint, gtm_phase, dependencies, justification, and alternative_sprints.

- **create_action_item(title: str, description: str, work_item_id: Optional[int] = None, request_id: Optional[int] = None) -> ActionItemResult**: Creates action item in dashboard linked to work items and requests. Returns action_item_id, created_at, linked_work_item, and linked_request.

- **update_launch_readiness(category: str, item_title: str, score_delta: int, notes: str) -> LaunchReadinessUpdate**: Updates launch readiness score for a specific category and item. Returns updated_score (1-5), previous_score, category, and timestamp.

- **log_agent_decision(input_text: str, classification: Dict, actions_taken: List[str], confidence: float) -> str**: Logs AI decision with rationale, similarity scores, and suggested actions. Returns log_id for feedback and learning. Enables transparency and continuous improvement.

### Phase 2 Tools (To Be Implemented)

- **comprehensive_work_item_search(query: str, use_semantic: bool = True, use_graph: bool = True, use_hybrid: bool = False, limit: int = 10) -> Dict**: Master search function that queries all three retrieval strategies in parallel. Returns combined results with semantic_matches (pgvector), graph_relationships (Neo4j/Graphiti), and hybrid_matches (TSVector). Merges and ranks by relevance.

- **search_work_item_graph(entity_name: str, relationship_types: Optional[List[str]] = None, depth: int = 2) -> Dict**: Searches Neo4j knowledge graph for work item relationships. Returns entity with connected nodes (parent Epics, child Features, linked User Stories, blocking/blocked-by dependencies).

- **get_work_item_timeline(work_item_id: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> List[Dict]**: Retrieves temporal history from knowledge graph (status changes, assignment updates, sprint movements). Returns timeline of events with timestamps.

- **hybrid_work_item_search(query: str, limit: int = 10, keyword_weight: float = 0.3) -> List[Dict]**: Performs combined semantic (pgvector) + keyword (TSVector) search. The keyword_weight parameter controls balance (0=pure semantic, 1=pure keyword). Returns work items ranked by combined score with both vector_similarity and text_similarity scores.

---

## DEPENDENCIES

### Already Configured (Phase 1)
- **supabase_client**: Supabase client for database operations (AsyncMock in tests, real client in production)
- **ado_mcp_tools**: Azure DevOps MCP server tools for fetching/syncing work items from NorthStar project
- **embedding_client**: OpenAI-compatible embedding API client (text-embedding-3-small, 1536 dimensions)
- **current_sprint_info**: Dictionary containing current sprint number, capacity, and active work items
- **work_item_templates**: Dictionary of templates for Epic, Feature, User Story, Bug, and Task creation
- **launch_readiness_categories**: List of 6 launch readiness categories with definitions and scoring criteria
- **gtm_phase_definitions**: Dictionary defining the 6 GTM phases (Foundation, Validation, Launch Prep, Go-to-Market, Growth, Market Leadership)
- **artifact_registry_path**: File system path for storing uploaded artifacts
- **sprint_map**: Lookup table mapping sprints to GTM phases and capacity

### Phase 2 Additions (To Be Implemented)
- **neo4j_client**: Neo4j driver for knowledge graph operations (follow `FullExample/agent/graph_utils.py` pattern)
- **graphiti_client**: Graphiti client for entity/relationship extraction and temporal tracking
- **database_pool**: AsyncPG connection pool for PostgreSQL (for hybrid search TSVector queries)
- **search_config**: Dictionary with search strategy preferences (enable_semantic: bool, enable_graph: bool, enable_hybrid: bool, default_keyword_weight: float)

---

## SYSTEM PROMPT(S)

You are an AI Project Management Assistant for the CoDeveloper platform with **advanced Hybrid RAG capabilities**. Your job is to take freeform input from Ryan (meeting notes, ideas, client feedback, bug reports, documents) and intelligently classify, route, and structure it into actionable Azure DevOps work items.

### Your Search Capabilities:

You have three powerful search strategies that work together:

1. **Semantic Search (pgvector)** - Understands meaning and context. Use when searching for conceptually similar work items.
2. **Knowledge Graph (Neo4j/Graphiti)** - Understands relationships and dependencies. Use when exploring connections between Epics, Features, and User Stories.
3. **Hybrid Search (TSVector + pgvector)** - Combines semantic understanding with exact keyword matching. Use when users need both conceptual relevance and specific terminology matches.

**When to use each search strategy:**
- Use **semantic search** for most work item matching (already your default behavior)
- Use **knowledge graph** when asked about relationships ("What Features belong to Epic X?", "Show me dependencies")
- Use **hybrid search** when keywords matter ("Find bugs with 'API timeout' error", "Security issues in login flow")
- Use **comprehensive search** when you want to leverage all three strategies for complete context

### Your Primary Responsibilities:

1. **Classify Input** - Analyze input and classify into one of 6 launch readiness categories: Product Readiness, Technical Readiness, Operational Readiness, Commercial Readiness, Customer Readiness, or Security & Compliance. Provide confidence scores (0-100%).

2. **Match to ADO Work Items** - Use your search capabilities to find existing Azure DevOps Epics, Features, or User Stories:
   - For general matching: Use semantic search (your default)
   - For relationship queries: Use knowledge graph search
   - For keyword-specific queries: Use hybrid search
   - For comprehensive context: Use all three strategies in parallel

3. **Assign GTM Phase** - Map work to one of 6 GTM delivery phases: Foundation (core dev, scaffolding), Validation (testing, onboarding), Launch Prep (pricing, support), Go-to-Market (marketing, training), Growth (scaling, optimization), or Market Leadership (advanced features, automation).

4. **Generate Work Items** - When creating new work, apply proper ADO templates:
   - **Epics**: Vision statement, business value, success metrics
   - **Features**: User-facing capability, acceptance criteria, technical notes  
   - **User Stories**: "As a [user] I want [capability] so that [benefit]" format with Gherkin acceptance criteria (Given/When/Then scenarios)

5. **Recommend Sprint Placement** - Suggest which sprint work should be completed in, considering dependencies, priority, and capacity. Justify recommendations.

6. **Tag & Route Artifacts** - For uploaded documents (wireframes, briefs, test plans), detect type, extract metadata, suggest tags, and link to relevant work items and GTM phases.

7. **Log Decisions** - Always log your classification, search strategy used, matching logic, confidence scores, and actions taken. Be transparent about reasoning. Accept feedback to improve.

**Key Principles:**
- Always search ADO backlog before creating new work items to avoid duplication
- Choose the right search strategy for each query type
- Provide confidence scores for all classifications and matches (be honest about uncertainty)
- When uncertain, ask clarifying questions instead of guessing
- Use structured JSON output for all tool calls
- Cite sources (ADO work item IDs, document names, relationship types) when referencing existing work
- Optimize for Ryan's workflow: reduce clicks, offload mental overhead, stay execution-focused

**Example Tags:** ["Sous AI", "Formulate", "Reverse Engineer", "Ingredient Matching", "Pantry", "Claims", "CSV Export", "Dialogue UI", "Launch Prep", "Technical Debt"]

**Context:** Ryan wants to dump ideas and notes without worrying about organization. Your job is to smartly classify and route everything, leveraging all your search capabilities to understand both content and relationships, so he can focus on building the product instead of managing spreadsheets and work items.

---

## EXAMPLES:

### Use Case Examples:

1. **Meeting Notes Classification (Uses Semantic Search)**
   - Input: "Jay mentioned users are confused by the Sous loading screen. Wants progress indicators and food-related animation."
   - Agent classifies as: Customer Readiness (87%), uses semantic search to match Feature #1587 "Sous Loading Experience" (91% similarity), creates action item, suggests Sprint 42

2. **Technical Debt Discovery (Uses Hybrid Search for Keywords)**
   - Input: "Formula validation endpoint timing out on large requests. Need pagination and caching."
   - Agent classifies as: Technical Readiness (94%), uses hybrid search for "timeout endpoint" keywords, no strong match (42%), generates draft Bug work item with acceptance criteria, suggests Priority: High

3. **Relationship Query (Uses Knowledge Graph)**
   - Input: "Show me all Features linked to the 'AI Sous Chef' Epic and their current status"
   - Agent uses knowledge graph search to traverse Epic → Features relationships, returns structured hierarchy with status metadata and temporal changes

4. **Product Strategy Document (Uses Comprehensive Search)**
   - Input: [Uploaded PDF: "Q1 2026 Product Roadmap.pdf"]
   - Agent uses comprehensive search to match against existing work items (semantic), check for related Epics (graph), and find keyword matches in descriptions (hybrid). Extracts 5 features, matches 2 to existing Epics, suggests 3 new Epics, stores in Artifact Library with tags ["Product Strategy", "Roadmap", "Q1 2026"]

### Pydantic AI Reference Examples:

- examples/basic_chat_agent - Basic chat agent with conversation memory
- examples/tool_enabled_agent - Tool-enabled agent with web search capabilities  
- examples/structured_output_agent - Structured output agent for data validation
- examples/testing_examples - Testing examples with TestModel and FunctionModel
- examples/main_agent_reference - Best practices for building Pydantic AI agents
- **FullExample/** - Complete Hybrid RAG reference implementation with graph_utils.py, db_utils.py, and unified search tools

---

## DOCUMENTATION:

### Project-Specific Documentation:

- AI Project Management Vision: `PROJECTS/dashboard/docs/AI-Project-Management-Vision.md`
- Dashboard Enhancement Plan: `archive/dashboard-enhancement-plan.md` (Phase 4 details)
- Launch Readiness Framework: `PROJECTS/work-items/knowledge/launch_readiness.md`
- Work Item Templates: `PROJECTS/work-items/tracker/feature-template.md` and `user-story-template.md`
- User Story Writing Standard: Follow "As a/I want/So that" format with Gherkin acceptance criteria
- Database Schema: `dashboard/supabase/migrations/001_initial_schema.sql`

### External Documentation:

- Pydantic AI Official Documentation: https://ai.pydantic.dev/
- Agent Creation Guide: https://ai.pydantic.dev/agents/
- Tool Integration: https://ai.pydantic.dev/tools/
- Testing Patterns: https://ai.pydantic.dev/testing/
- Model Providers: https://ai.pydantic.dev/models/
- Azure DevOps REST API: https://learn.microsoft.com/en-us/rest/api/azure/devops/
- Supabase Documentation: https://supabase.com/docs
- Neo4j Documentation: https://neo4j.com/docs/
- Graphiti Documentation: https://github.com/getzep/graphiti

### Implementation Architecture Reference:

**CRITICAL: Use `FullExample/` as your architectural blueprint for Phase 2 implementation:**

- `FullExample/agent/graph_utils.py` - Neo4j/Graphiti integration patterns, episode management, entity search
- `FullExample/agent/db_utils.py` - AsyncPG connection pooling, vector_search, hybrid_search functions
- `FullExample/agent/tools.py` - Unified search tools that combine all three retrieval strategies
- `FullExample/sql/schema.sql` - PostgreSQL schema with pgvector, TSVector, and hybrid search SQL functions
- `FullExample/ingestion/graph_builder.py` - Episode creation and knowledge graph population patterns

**Key Implementation Notes:**
- Use AsyncPG for PostgreSQL connections (not Supabase SDK for hybrid search queries)
- Use official Neo4j Python driver with Graphiti wrapper for knowledge graph
- Store work item relationships as Graphiti episodes (Epic contains Features, Feature implements User Story, etc.)
- Implement hybrid_search SQL function combining pgvector similarity with TSVector keyword ranking
- Use configurable weighting for hybrid search (default: 70% semantic, 30% keyword)

---

## OTHER CONSIDERATIONS:

### General Best Practices:
- Use environment variables for API key configuration instead of hardcoded model strings
- Keep agents simple - default to string output unless structured output is specifically needed
- Follow the main_agent_reference patterns for configuration and providers
- Always include comprehensive testing with TestModel for development

### Project-Specific Gotchas:

- **ADO Access Policy**: NEVER create, update, or modify ADO work items directly through the API. The agent generates work items and returns them for Ryan's approval. Only Ryan creates work items in ADO manually.

- **Semantic Search (Phase 1 ✅)**: OpenAI embeddings (text-embedding-3-small, 1536 dimensions) for work item matching. Database has `embedding` column in `work_items` table and `match_work_items()` Supabase RPC function. Already fully implemented and tested.

- **Knowledge Graph (Phase 2 🔧)**: Use Neo4j with Graphiti to store work item relationships. Model Epics/Features/User Stories as entities with relationships (CONTAINS, IMPLEMENTS, BLOCKS, DEPENDS_ON). Track temporal changes (status updates, assignment changes, sprint movements). Initialize Graphiti client in `dependencies.py` similar to `FullExample/agent/graph_utils.py`.

- **Hybrid Search (Phase 2 🔧)**: Implement TSVector full-text search in PostgreSQL. Create `hybrid_search` SQL function that combines pgvector cosine similarity with TSVector ts_rank. Use AsyncPG for direct PostgreSQL access (Supabase SDK doesn't support TSVector operations well). Follow `FullExample/sql/schema.sql` patterns.

- **Template Compliance**: All generated work items MUST follow the templates in `work-items/tracker/`. User Stories require "As a/I want/So that" format with multiple Gherkin scenarios (Given/When/Then). Features need acceptance criteria and technical notes.

- **Confidence Thresholds**: Be transparent about confidence scores. <70% = ask for clarification, 70-85% = suggest with caveat, >85% = recommend with confidence.

- **Launch Readiness Scoring**: Scores are 1-5 scale. Don't auto-update scores >1 point without explicit approval. Small updates (±1) are acceptable when work items complete.

- **Sprint Capacity**: Check current sprint capacity before recommending placement. Don't overload sprints. Query `current_sprint_info` dependency for real-time data.

- **Artifact Storage**: Store artifacts in file system at `artifact_registry_path`, but metadata goes in Supabase `artifacts` table with links to work items.

- **Decision Logs**: Every classification and routing decision should be logged for transparency and learning. Include input_text, search_strategy_used, classification results, similarity scores, and actions taken.

- **Search Strategy Selection**: Choose the right search for each query:
  - General work item matching → Semantic search (default)
  - Relationship exploration → Knowledge graph
  - Keyword-specific searches → Hybrid search
  - Need complete context → Comprehensive search (all three)

### Phase 2 Implementation Order:

1. **Add Neo4j/Graphiti Integration**
   - Copy `FullExample/agent/graph_utils.py` patterns
   - Initialize `GraphitiClient` in `dependencies.py`
   - Create `search_work_item_graph` and `get_work_item_timeline` tools
   - Test with existing work item data

2. **Add Hybrid Search (TSVector)**
   - Implement `hybrid_search` SQL function in Supabase
   - Add AsyncPG connection pool to `dependencies.py`
   - Create `hybrid_work_item_search` tool in `tools.py`
   - Test keyword + semantic combinations

3. **Create Comprehensive Search Tool**
   - Implement `comprehensive_work_item_search` that calls all three strategies in parallel
   - Merge and rank results by relevance
   - Add caching for frequently-searched work items
   - Update agent system prompt to guide search strategy selection

4. **Enhance Existing Tools**
   - Update `match_work_items` to optionally use comprehensive search
   - Add graph relationship context to work item generation
   - Include temporal history in decision logs
   - Add search strategy metadata to all tool outputs

### Success Metrics:
- Reduce manual work item creation by 60%
- Achieve 90%+ classification accuracy (with >85% confidence)
- Enable 80%+ of new work to flow through agent
- Minimize Ryan's clicks - aim for single approval action per input
- **Phase 2**: Improve work item matching accuracy by 15% with hybrid search
- **Phase 2**: Enable relationship-based queries (Epic → Feature → Story navigation)
- **Phase 2**: Track temporal work item history for better sprint predictions

