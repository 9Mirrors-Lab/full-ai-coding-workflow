## FEATURE:

Build an intelligent AI Project Management Agent that transforms project management overhead into an automated system. The agent takes freeform input (meeting notes, ideas, client feedback, documents) and automatically classifies, routes, and structures it into actionable project deliverables.

**Core Capabilities:**
1. **Input Classification** - Classify input into 6 launch readiness categories (Product, Technical, Operational, Commercial, Customer, Security & Compliance)
2. **ADO Work Item Matching** - Match input to existing Azure DevOps Epics, Features, or User Stories using semantic similarity
3. **Work Item Generation** - Auto-generate ADO-ready work items (Epics, Features, User Stories) with proper templates and acceptance criteria
4. **Artifact Management** - Auto-tag and organize uploaded documents (wireframes, briefs, test plans) and link them to work items
5. **Sprint & Phase Alignment** - Recommend sprint placement and map work to GTM delivery phases (Foundation → Market Leadership)
6. **Decision Logging** - Log all AI decisions with rationale, confidence scores, and allow feedback for continuous learning

**Goal:** Offload mental overhead of organizing work, allow Ryan to focus on building instead of categorizing, routing, and manually creating ADO work items.

## TOOLS:

- **classify_input(input_text: str) -> Dict**: Classifies freeform input into launch readiness categories. Returns classification result with category (Product/Technical/Operational/Commercial/Customer/Security), confidence_score (0-100%), gtm_phase (Foundation/Validation/Launch Prep/Go-to-Market/Growth/Market Leadership), and suggested_action (create_action_item, link_to_work_item, update_readiness).

- **match_ado_work_items(input_text: str, limit: int = 5) -> List[Dict]**: Performs semantic similarity search against existing ADO work items (Epics, Features, User Stories). Returns matches with ado_id, title, work_item_type, similarity_score (0-100%), and recommendation (link/create_new).

- **generate_work_item(input_description: str, work_item_type: str, parent_id: Optional[int] = None) -> Dict**: Generates ADO-ready work item with proper template structure. Returns structured work item with title, description, acceptance_criteria (Gherkin format for User Stories), tags, suggested_sprint, and parent_link. Templates: Epic (vision, business value, metrics), Feature (capability, acceptance criteria, technical notes), User Story (As a/I want/So that with scenarios).

- **tag_artifact(file_path: str, metadata: Dict) -> Dict**: Auto-tags uploaded documents/artifacts by type (wireframe, brief, test plan, meeting notes, design doc, marketing material). Returns artifact_id, detected_type, extracted_metadata, suggested_tags, linked_work_items, and gtm_phase assignment.

- **recommend_sprint_placement(work_item: Dict, current_sprint: int, sprint_capacity: Dict) -> Dict**: Recommends sprint placement based on priority, dependencies, and capacity. Returns recommended_sprint, gtm_phase, dependencies, justification, and alternative_sprints.

- **create_action_item(title: str, description: str, work_item_id: Optional[int] = None, request_id: Optional[int] = None) -> Dict**: Creates action item in dashboard linked to work items and requests. Returns action_item_id, created_at, linked_work_item, and linked_request.

- **update_launch_readiness(category: str, item_title: str, score_delta: int, notes: str) -> Dict**: Updates launch readiness score for a specific category and item. Returns updated_score (1-5), previous_score, category, and timestamp.

- **log_agent_decision(input_text: str, classification: Dict, actions_taken: List[str], confidence: float) -> str**: Logs AI decision with rationale, similarity scores, and suggested actions. Returns log_id for feedback and learning. Enables transparency and continuous improvement.

## DEPENDENCIES

- **supabase_client**: Supabase client for database operations (action items, work items, correlations, launch readiness, artifacts, agent logs tables)
- **ado_mcp_tools**: Azure DevOps MCP server tools for fetching/syncing work items from NorthStar project
- **embedding_client**: OpenAI-compatible embedding API client for generating vector embeddings for semantic search
- **current_sprint_info**: Dictionary containing current sprint number, capacity, and active work items
- **work_item_templates**: Dictionary of templates for Epic, Feature, User Story, Bug, and Task creation
- **launch_readiness_categories**: List of 6 launch readiness categories with definitions and scoring criteria
- **gtm_phase_definitions**: Dictionary defining the 6 GTM phases (Foundation, Validation, Launch Prep, Go-to-Market, Growth, Market Leadership)
- **artifact_registry_path**: File system path for storing uploaded artifacts
- **sprint_map**: Lookup table mapping sprints to GTM phases and capacity

## SYSTEM PROMPT(S)

You are an AI Project Management Assistant for the CoDeveloper platform. Your job is to take freeform input from Ryan (meeting notes, ideas, client feedback, bug reports, documents) and intelligently classify, route, and structure it into actionable project deliverables.

**Your Primary Responsibilities:**

1. **Classify Input** - Analyze input and classify into one of 6 launch readiness categories: Product Readiness, Technical Readiness, Operational Readiness, Commercial Readiness, Customer Readiness, or Security & Compliance. Provide confidence scores (0-100%).

2. **Match to ADO Work Items** - Use semantic similarity search to find existing Azure DevOps Epics, Features, or User Stories that relate to the input. Return top matches with similarity scores. Recommend whether to link to existing work or create new items.

3. **Assign GTM Phase** - Map work to one of 6 GTM delivery phases: Foundation (core dev, scaffolding), Validation (testing, onboarding), Launch Prep (pricing, support), Go-to-Market (marketing, training), Growth (scaling, optimization), or Market Leadership (advanced features, automation).

4. **Generate Work Items** - When creating new work, apply proper ADO templates:
   - **Epics**: Vision statement, business value, success metrics
   - **Features**: User-facing capability, acceptance criteria, technical notes  
   - **User Stories**: "As a [user] I want [capability] so that [benefit]" format with Gherkin acceptance criteria (Given/When/Then scenarios)

5. **Recommend Sprint Placement** - Suggest which sprint work should be completed in, considering dependencies, priority, and capacity. Justify recommendations.

6. **Tag & Route Artifacts** - For uploaded documents (wireframes, briefs, test plans), detect type, extract metadata, suggest tags, and link to relevant work items and GTM phases.

7. **Log Decisions** - Always log your classification, matching logic, confidence scores, and actions taken. Be transparent about reasoning. Accept feedback to improve.

**Key Principles:**
- Always search ADO backlog before creating new work items to avoid duplication
- Provide confidence scores for all classifications and matches (be honest about uncertainty)
- When uncertain, ask clarifying questions instead of guessing
- Use structured JSON output for all tool calls
- Cite sources (ADO work item IDs, document names) when referencing existing work
- Optimize for Ryan's workflow: reduce clicks, offload mental overhead, stay execution-focused

**Example Tags:** ["Sous AI", "Formulate", "Reverse Engineer", "Ingredient Matching", "Pantry", "Claims", "CSV Export", "Dialogue UI", "Launch Prep", "Technical Debt"]

**Context:** Ryan wants to dump ideas and notes without worrying about organization. Your job is to smartly classify and route everything so he can focus on building the product instead of managing spreadsheets and work items.

## EXAMPLES:

**Use Case Examples:**

1. **Meeting Notes Classification**
   - Input: "Jay mentioned users are confused by the Sous loading screen. Wants progress indicators and food-related animation."
   - Agent classifies as: Customer Readiness (87%), matches Feature #1587 "Sous Loading Experience" (91% similarity), creates action item, suggests Sprint 42

2. **Technical Debt Discovery**
   - Input: "Formula validation endpoint timing out on large requests. Need pagination and caching."
   - Agent classifies as: Technical Readiness (94%), no strong ADO match (42%), generates draft Bug work item with acceptance criteria, suggests Priority: High

3. **Product Strategy Document**
   - Input: [Uploaded PDF: "Q1 2026 Product Roadmap.pdf"]
   - Agent extracts 5 features, matches 2 to existing Epics, suggests 3 new Epics, stores in Artifact Library with tags ["Product Strategy", "Roadmap", "Q1 2026"]

**Pydantic AI Reference Examples:**

- examples/basic_chat_agent - Basic chat agent with conversation memory
- examples/tool_enabled_agent - Tool-enabled agent with web search capabilities  
- examples/structured_output_agent - Structured output agent for data validation
- examples/testing_examples - Testing examples with TestModel and FunctionModel
- examples/main_agent_reference - Best practices for building Pydantic AI agents

## DOCUMENTATION:

**Project-Specific Documentation:**

- AI Project Management Vision: `PROJECTS/dashboard/docs/AI-Project-Management-Vision.md`
- Dashboard Enhancement Plan: `archive/dashboard-enhancement-plan.md` (Phase 4 details)
- Launch Readiness Framework: `PROJECTS/work-items/knowledge/launch_readiness.md`
- Work Item Templates: `PROJECTS/work-items/tracker/feature-template.md` and `user-story-template.md`
- User Story Writing Standard: Follow "As a/I want/So that" format with Gherkin acceptance criteria
- Database Schema: `dashboard/supabase/migrations/001_initial_schema.sql`

**External Documentation:**

- Pydantic AI Official Documentation: https://ai.pydantic.dev/
- Agent Creation Guide: https://ai.pydantic.dev/agents/
- Tool Integration: https://ai.pydantic.dev/tools/
- Testing Patterns: https://ai.pydantic.dev/testing/
- Model Providers: https://ai.pydantic.dev/models/
- Azure DevOps REST API: https://learn.microsoft.com/en-us/rest/api/azure/devops/
- Supabase Documentation: https://supabase.com/docs

## OTHER CONSIDERATIONS:

**General Best Practices:**
- Use environment variables for API key configuration instead of hardcoded model strings
- Keep agents simple - default to string output unless structured output is specifically needed
- Follow the main_agent_reference patterns for configuration and providers
- Always include comprehensive testing with TestModel for development

**Project-Specific Gotchas:**

- **ADO Access Policy**: NEVER create, update, or modify ADO work items directly through the API. The agent generates work items and returns them for Ryan's approval. Only Ryan creates work items in ADO manually.

- **Semantic Search**: Use OpenAI embeddings for work item matching. Database already has `embedding` column in `work_items` table and `match_work_items()` function. Leverage existing Supabase vector similarity functions.

- **Template Compliance**: All generated work items MUST follow the templates in `work-items/tracker/`. User Stories require "As a/I want/So that" format with multiple Gherkin scenarios (Given/When/Then). Features need acceptance criteria and technical notes.

- **Confidence Thresholds**: Be transparent about confidence scores. <70% = ask for clarification, 70-85% = suggest with caveat, >85% = recommend with confidence.

- **Launch Readiness Scoring**: Scores are 1-5 scale. Don't auto-update scores >1 point without explicit approval. Small updates (±1) are acceptable when work items complete.

- **Sprint Capacity**: Check current sprint capacity before recommending placement. Don't overload sprints. Query `current_sprint_info` dependency for real-time data.

- **Artifact Storage**: Store artifacts in file system at `artifact_registry_path`, but metadata goes in Supabase `artifacts` table with links to work items.

- **Decision Logs**: Every classification and routing decision should be logged for transparency and learning. Include input_text, classification results, similarity scores, and actions taken.

**Success Metrics:**
- Reduce manual work item creation by 60%
- Achieve 90%+ classification accuracy (with >85% confidence)
- Enable 80%+ of new work to flow through agent
- Minimize Ryan's clicks - aim for single approval action per input