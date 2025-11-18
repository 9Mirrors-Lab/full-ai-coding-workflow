---
name: "AI Project Management Agent - Phase 2: Hybrid RAG Enhancement"
description: "Comprehensive PRP for adding Knowledge Graph retrieval (Neo4j/Graphiti) and Hybrid Search (TSVector) to the existing AI PM Agent"
version: "2.0"
phase: "Phase 2 Enhancement"
created: "2025-01-18"
base_prp: "ai-project-management-agent.md (Phase 1)"
---

## PROJECT LINEAGE

**Phase 1 (Completed ✅)**:
- PRP: `ai-project-management-agent.md`
- Implementation: `ai_pm_agent/` directory
- Status: All 8 core tools implemented and tested
- Foundation: Semantic search with OpenAI embeddings + Supabase pgvector
- Test Coverage: Comprehensive with TestModel and AsyncMock patterns

**Phase 2 (This Document)**:
- Enhancement: Add Hybrid RAG capabilities to existing agent
- New Capabilities: Knowledge Graph retrieval + Hybrid Search (semantic + keyword)
- Reference Architecture: `FullExample/` - Complete hybrid RAG implementation
- Build On: Phase 1's agent structure, tools, models, dependencies, tests

---

## Purpose

Enhance the existing AI Project Management Agent (Phase 1) with **Hybrid RAG capabilities** by adding two powerful retrieval strategies:

1. **Knowledge Graph Retrieval** - Neo4j + Graphiti for mapping ADO work item relationships (Epic → Feature → User Story → Bug) with temporal tracking
2. **Hybrid Search** - PostgreSQL TSVector + pgvector combining semantic understanding with keyword precision

This creates a **three-layer search system** that intelligently chooses the optimal retrieval strategy based on query type:
- **Semantic Search** (pgvector) - Already implemented in Phase 1 ✅
- **Knowledge Graph** (Graphiti) - Phase 2 addition 🔧
- **Hybrid Search** (TSVector + pgvector) - Phase 2 addition 🔧

## Core Principles

1. **Build on Phase 1 Foundation** - Extend, don't replace. Leverage existing agent structure, tools, and tests
2. **Follow FullExample Patterns** - Copy proven architecture from `FullExample/agent/graph_utils.py` and `db_utils.py`
3. **Pydantic AI Best Practices** - Deep integration with Pydantic AI patterns for agents, tools, and dependencies
4. **Type Safety First** - Leverage Pydantic AI's type-safe design and Pydantic validation throughout
5. **Comprehensive Testing** - Use TestModel and FunctionModel for thorough validation (follow Phase 1 test patterns)
6. **Don't Over-Engineer** - Start simple, add complexity only when needed

## ⚠️ Implementation Guidelines: Keep It Simple

**IMPORTANT**: This is an enhancement project. Don't rebuild what already works.

- ✅ **Reuse Phase 1 code** - Don't rewrite existing tools, models, or tests
- ✅ **Copy FullExample patterns** - Don't reinvent graph or hybrid search implementations
- ✅ **Add incrementally** - Implement one search strategy at a time (Graph → Hybrid → Comprehensive)
- ✅ **Test as you build** - Validate each component before moving to the next
- ✅ **Follow main_agent_reference** - Use proven Pydantic AI patterns

### Key Question:
**"Does this agent really need this feature to accomplish its hybrid RAG purpose?"**

If the answer is no, don't build it. Keep it simple, focused, and functional.

---

## Goal

**Phase 2 adds 4 new capabilities to the existing Phase 1 agent:**

1. **Knowledge Graph Search** - Query work item relationships (Epic → Feature → Story) and dependencies using Graphiti
2. **Temporal History** - Track work item evolution (status changes, assignment updates, sprint movements) over time
3. **Hybrid Search** - Combine semantic similarity (pgvector) with keyword matching (TSVector) for precision+relevance
4. **Comprehensive Search** - Master tool that executes all three strategies in parallel and merges results

**Success Metrics (Phase 2 Additions)**:
- Improve work item matching accuracy by 15% with hybrid search
- Enable relationship-based queries (Epic → Feature → Story navigation)
- Track temporal work item history for better sprint predictions
- Reduce false positives in keyword-specific searches (e.g., "API timeout bug")

**Phase 1 Metrics (Maintained)**:
- Reduce manual work item creation by 60%
- Achieve 90%+ classification accuracy (with >85% confidence)
- Enable 80%+ of new work to flow through agent
- Minimize Ryan's clicks - aim for single approval action per input

---

## Why

Phase 1 implemented semantic search effectively, but it has limitations:

**Semantic Search Alone (Phase 1) Cannot Handle:**
- Relationship queries: "Show all Features blocked by Epic #1234"
- Keyword-specific searches: "Find bugs with 'API timeout' error message"
- Temporal queries: "What changed in Sprint 42?"
- Dependency mapping: "Which User Stories are linked to Feature X?"

**Phase 2 Hybrid RAG Solves:**
- **Knowledge Graph** reveals hidden connections between work items
- **Hybrid Search** combines semantic understanding with exact keyword matching
- **Temporal Intelligence** tracks how work items evolve over time
- **Comprehensive Search** provides complete context by querying all three strategies

**Business Impact:**
- More accurate work item matching reduces duplicate creation
- Relationship awareness prevents orphaned User Stories
- Temporal tracking improves sprint capacity planning
- Keyword precision reduces manual filtering of search results

---

## What

### Agent Type Classification
- [x] **Tool-Enabled Agent** - Existing Phase 1 agent with 8 tools
- [x] **Hybrid RAG Agent** - Phase 2 enhancement with 3-layer retrieval

### External Integrations (Phase 2 Additions)
- [x] **Neo4j** - Knowledge graph database for entity relationships
- [x] **Graphiti** - Temporal knowledge graph framework with bi-temporal model
- [x] **AsyncPG** - Direct PostgreSQL connection for TSVector hybrid search
- [ ] **Supabase** - Already integrated in Phase 1 for pgvector semantic search ✅

### Phase 1 Tools (Already Implemented ✅)

These 8 tools are working and tested. **DO NOT modify them unless absolutely necessary.**

1. **classify_input** - Classifies input into launch readiness categories
2. **match_work_items** - Semantic similarity search using OpenAI embeddings + Supabase pgvector
3. **generate_work_item** - Generates ADO-ready work items with templates
4. **tag_artifact** - Auto-tags documents and links to work items
5. **recommend_sprint_placement** - Recommends sprint placement based on capacity
6. **create_action_item** - Creates action items linked to work items
7. **update_launch_readiness** - Updates launch readiness scores
8. **log_agent_decision** - Logs AI decisions for transparency

**Phase 1 Files (Reference, Don't Modify)**:
- `ai_pm_agent/agent.py` - Main agent with 8 registered tools
- `ai_pm_agent/tools.py` - Tool implementations
- `ai_pm_agent/models.py` - Pydantic models for tool inputs/outputs
- `ai_pm_agent/dependencies.py` - AgentDependencies dataclass
- `ai_pm_agent/settings.py` - Environment configuration with python-dotenv
- `tests/` - Comprehensive test suite with TestModel and AsyncMock

### Phase 2 Tools (To Be Implemented)

**Add these 4 new tools to the existing agent:**

#### 1. comprehensive_work_item_search
**Purpose**: Master search tool that queries all three retrieval strategies in parallel

**Parameters**:
- `query: str` - Search query
- `use_semantic: bool = True` - Enable semantic search (pgvector)
- `use_graph: bool = True` - Enable graph search (Graphiti)
- `use_hybrid: bool = False` - Enable hybrid search (TSVector)
- `limit: int = 10` - Max results per strategy

**Returns**: `ComprehensiveSearchResult`
```python
class ComprehensiveSearchResult(BaseModel):
    semantic_matches: List[WorkItemMatch]  # From Phase 1 tool
    graph_relationships: List[GraphRelationship]  # New
    hybrid_matches: List[HybridSearchMatch]  # New
    total_results: int
    search_strategies_used: List[str]
    combined_ranking: List[WorkItemMatch]  # Merged and ranked
```

**Implementation Pattern**: Copy from `FullExample/agent/tools.py` - parallel async execution with `asyncio.gather()`

---

#### 2. search_work_item_graph
**Purpose**: Search Neo4j knowledge graph for work item relationships

**Parameters**:
- `entity_name: str` - Work item identifier (e.g., "Epic #1234")
- `relationship_types: Optional[List[str]] = None` - Filter by relationship (CONTAINS, IMPLEMENTS, BLOCKS, DEPENDS_ON)
- `depth: int = 2` - Maximum graph traversal depth

**Returns**: `GraphSearchResult`
```python
class GraphSearchResult(BaseModel):
    central_entity: str  # The work item searched
    related_entities: List[Dict[str, Any]]  # Connected work items
    relationships: List[Dict[str, str]]  # Relationship types and directions
    search_method: str = "graphiti_semantic_search"
```

**Implementation Pattern**: Copy from `FullExample/agent/graph_utils.py` - `GraphitiClient.get_related_entities()`

---

#### 3. get_work_item_timeline
**Purpose**: Retrieve temporal history of work item changes from knowledge graph

**Parameters**:
- `work_item_id: int` - ADO work item ID
- `start_date: Optional[str] = None` - Start of time range (ISO format)
- `end_date: Optional[str] = None` - End of time range (ISO format)

**Returns**: `List[TimelineEvent]`
```python
class TimelineEvent(BaseModel):
    fact: str  # Description of the event
    timestamp: str  # When it occurred (ISO format)
    event_type: str  # status_change, assignment_update, sprint_movement, etc.
    metadata: Dict[str, Any]
```

**Implementation Pattern**: Copy from `FullExample/agent/graph_utils.py` - `GraphitiClient.get_entity_timeline()`

---

#### 4. hybrid_work_item_search
**Purpose**: Combined semantic (pgvector) + keyword (TSVector) search

**Parameters**:
- `query: str` - Search query
- `limit: int = 10` - Maximum results
- `keyword_weight: float = 0.3` - Balance between semantic (0) and keyword (1)

**Returns**: `List[HybridSearchMatch]`
```python
class HybridSearchMatch(BaseModel):
    work_item_id: int
    title: str
    description: str
    combined_score: float  # Weighted average
    vector_similarity: float  # Semantic score
    text_similarity: float  # Keyword score
    metadata: Dict[str, Any]
```

**Implementation Pattern**: Copy from `FullExample/agent/db_utils.py` - `hybrid_search()` function

---

### Success Criteria (Phase 2)

- [x] **Neo4j/Graphiti integrated** - GraphitiClient initialized and tested
- [x] **Hybrid search implemented** - TSVector + pgvector queries working
- [x] **4 new tools registered** - Added to agent with @agent.tool decorators
- [x] **Dependencies extended** - neo4j_client, graphiti_client, database_pool added
- [x] **System prompt updated** - Guides agent on search strategy selection
- [x] **All tests passing** - TestModel validation for new tools + mocks for Graph/AsyncPG
- [x] **Phase 1 functionality preserved** - Original 8 tools still work correctly
- [x] **Performance meets requirements** - P95 latency <500ms for comprehensive search

---

## All Needed Context

### Phase 1 Implementation (Foundation - Already Built ✅)

**CRITICAL: Read these files to understand existing architecture before implementing Phase 2**

#### Agent Core
- **`ai_pm_agent/agent.py`** - Main agent definition with 8 registered tools (DON'T modify tool registrations)
- **`ai_pm_agent/tools.py`** - Tool implementations (ADD new tools, don't modify existing)
- **`ai_pm_agent/models.py`** - Pydantic models (ADD new models for Phase 2 tools)
- **`ai_pm_agent/dependencies.py`** - AgentDependencies dataclass (EXTEND with neo4j_client, database_pool)
- **`ai_pm_agent/settings.py`** - Environment configuration (ADD Neo4j and AsyncPG settings)
- **`ai_pm_agent/providers.py`** - LLM and embedding clients (reference for provider patterns)
- **`ai_pm_agent/prompts.py`** - System prompt (UPDATE to guide search strategy selection)

#### Phase 1 Planning Documents (Context for Design Decisions)
- **`planning/tools.md`** - Phase 1 tool design rationale
- **`planning/prompts.md`** - System prompt architecture decisions
- **`planning/dependencies.md`** - Dependency planning notes

#### Phase 1 Tests (Patterns to Follow)
- **`tests/test_models.py`** - Pydantic model validation patterns
- **`tests/conftest.py`** - Test fixtures and mocks (ADD GraphitiClient and DatabasePool mocks)
- **`tests/pytest.ini`** - Pytest configuration

**Key Phase 1 Patterns to Preserve:**
- Tool parameters: 1-4 parameters maximum
- Async/await throughout all tools
- Pydantic models for structured responses
- Error handling with graceful degradation
- TestModel validation before real LLM testing

---

### FullExample Architecture (Phase 2 Blueprint - Copy These Patterns)

**CRITICAL: Use FullExample/ as your architectural reference for Phase 2 implementation**

#### Knowledge Graph Integration
**`FullExample/agent/graph_utils.py`** (448 lines)
- Lines 27-71: `GraphitiClient.__init__()` - Neo4j configuration pattern
- Lines 72-117: `initialize()` - LLMConfig, OpenAIClient, OpenAIEmbedder, Graphiti setup
- Lines 127-161: `add_episode()` - Episode creation with EpisodeType.text pattern
- Lines 163-201: `search()` - Graphiti search with result dict conversion
- Lines 203-245: `get_related_entities()` - Relationship traversal pattern
- Lines 247-282: `get_entity_timeline()` - Temporal query pattern
- Lines 356-448: Global client instance and convenience functions

**Key Implementations to Copy:**
```python
# GraphitiClient initialization (lines 27-117)
class GraphitiClient:
    def __init__(self, neo4j_uri, neo4j_user, neo4j_password):
        # Load Neo4j config from environment
        # Load LLM config for Graphiti
        # Load embedding config

    async def initialize(self):
        # Create LLMConfig
        # Create OpenAIClient
        # Create OpenAIEmbedder
        # Initialize Graphiti with custom clients
        # Build indices and constraints

# Episode management (lines 127-161)
async def add_episode(episode_id, content, source, timestamp, metadata):
    from graphiti_core.nodes import EpisodeType
    await self.graphiti.add_episode(
        name=episode_id,
        episode_body=content,
        source=EpisodeType.text,
        source_description=source,
        reference_time=timestamp
    )
```

---

#### Hybrid Search Implementation
**`FullExample/agent/db_utils.py`** (513 lines)
- Lines 24-80: `DatabasePool` class - AsyncPG connection pooling pattern
- Lines 369-405: `vector_search()` - Pure semantic search with pgvector
- Lines 408-452: `hybrid_search()` - **CRITICAL: Combined TSVector + pgvector search**
- Lines 500-513: `test_connection()` - Database health check pattern

**Key Implementation to Copy (lines 408-452):**
```python
async def hybrid_search(
    embedding: List[float],
    query_text: str,
    limit: int = 10,
    text_weight: float = 0.3
) -> List[Dict[str, Any]]:
    async with db_pool.acquire() as conn:
        # Convert embedding to PostgreSQL vector string format
        embedding_str = '[' + ','.join(map(str, embedding)) + ']'

        # Call SQL function hybrid_search()
        results = await conn.fetch(
            "SELECT * FROM hybrid_search($1::vector, $2, $3, $4)",
            embedding_str,
            query_text,
            limit,
            text_weight
        )

        return [
            {
                "chunk_id": row["chunk_id"],
                "combined_score": row["combined_score"],
                "vector_similarity": row["vector_similarity"],
                "text_similarity": row["text_similarity"],
                # ... metadata
            }
            for row in results
        ]
```

---

#### Database Schema (SQL Functions)
**`FullExample/sql/schema.sql`** (169 lines)
- Lines 1-12: Extensions and index setup (pgvector, TSVector with pg_trgm)
- Lines 28-41: `chunks` table with `embedding vector(1536)` and TSVector index
- Lines 38: **IVFFlat index** for pgvector - `idx_chunks_embedding`
- Lines 41: **GIN index** for TSVector - `idx_chunks_content_trgm`
- Lines 66-97: `match_chunks()` SQL function - Pure semantic search
- Lines 99-169: **`hybrid_search()` SQL function** - **CRITICAL: Copy this entire function**

**Key SQL Function to Copy (lines 99-169):**
```sql
CREATE OR REPLACE FUNCTION hybrid_search(
    query_embedding vector(1536),
    query_text TEXT,
    match_count INT DEFAULT 10,
    text_weight FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    chunk_id UUID,
    combined_score FLOAT,
    vector_similarity FLOAT,
    text_similarity FLOAT,
    -- ... other columns
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    WITH vector_results AS (
        SELECT c.id, 1 - (c.embedding <=> query_embedding) AS vector_sim
        FROM chunks c WHERE c.embedding IS NOT NULL
    ),
    text_results AS (
        SELECT c.id,
            ts_rank_cd(to_tsvector('english', c.content),
                       plainto_tsquery('english', query_text)) AS text_sim
        FROM chunks c
        WHERE to_tsvector('english', c.content) @@ plainto_tsquery('english', query_text)
    )
    SELECT
        COALESCE(v.chunk_id, t.chunk_id),
        (COALESCE(v.vector_sim, 0) * (1 - text_weight) +
         COALESCE(t.text_sim, 0) * text_weight) AS combined_score,
        v.vector_sim AS vector_similarity,
        t.text_sim AS text_similarity
    FROM vector_results v
    FULL OUTER JOIN text_results t ON v.chunk_id = t.chunk_id
    ORDER BY combined_score DESC
    LIMIT match_count;
END;
$$;
```

---

#### Unified Search Tools
**`FullExample/agent/tools.py`** (100+ lines of search tool patterns)
- Lines 39-57: `generate_embedding()` - OpenAI embedding client pattern
- Lines 60-100: Tool input Pydantic models (VectorSearchInput, HybridSearchInput, GraphSearchInput)
- Tool registration patterns with `@agent.tool` decorators
- Parallel async execution with `asyncio.gather()` for comprehensive search

**Key Pattern for comprehensive_work_item_search:**
```python
async def comprehensive_search(query: str, limit: int = 10):
    # Generate embedding once
    embedding = await generate_embedding(query)

    # Execute all searches in parallel
    results = await asyncio.gather(
        vector_search(embedding, limit),
        search_knowledge_graph(query),
        hybrid_search(embedding, query, limit)
    )

    # Merge and rank results
    return combine_and_rank(results)
```

---

### External Documentation (Pydantic AI Patterns)

#### Pydantic AI Official Documentation
- **https://ai.pydantic.dev/** - Getting started guide and agent creation
- **https://ai.pydantic.dev/dependencies/** - **CRITICAL: RunContext and dependency injection patterns**
- **https://ai.pydantic.dev/tools/** - **CRITICAL: @agent.tool and @agent.tool_plain decorators**
- **https://ai.pydantic.dev/testing/** - TestModel and FunctionModel validation strategies
- **https://ai.pydantic.dev/agents/** - Agent configuration and system prompts

**Key Pydantic AI Patterns for Phase 2:**
```python
# Tool with dependencies (RunContext)
@pm_agent.tool
async def search_work_item_graph(
    ctx: RunContext[AgentDependencies],
    entity_name: str,
    depth: int = 2
) -> GraphSearchResult:
    # Access dependencies via ctx.deps
    graphiti_client = ctx.deps.graphiti_client
    results = await graphiti_client.get_related_entities(entity_name, depth=depth)
    return GraphSearchResult(**results)

# Tool without dependencies (tool_plain)
@pm_agent.tool_plain
async def hybrid_work_item_search(
    query: str,
    limit: int = 10,
    keyword_weight: float = 0.3
) -> List[HybridSearchMatch]:
    # No context needed, direct implementation
    embedding = await generate_embedding(query)
    return await db_utils.hybrid_search(embedding, query, limit, keyword_weight)
```

---

### Neo4j and Graphiti Documentation

#### Graphiti Framework
- **https://github.com/getzep/graphiti** - Official Graphiti GitHub repository
- **https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/** - Graphiti integration guide
- **Key Concepts**:
  - Bi-temporal model (t_valid, t_invalid) for temporal tracking
  - Episodes as atomic knowledge units
  - Semantic search across graph nodes
  - Real-time entity extraction and relationship building

#### Neo4j Python Driver
- **https://neo4j.com/docs/python-manual/current/** - Official Python driver documentation
- **Connection Pattern**: Use Graphiti wrapper, not direct Neo4j driver
- **Query Language**: Graphiti abstracts Cypher queries into Python methods

**Graphiti Episode Pattern for Work Items:**
```python
# Example: Adding Epic as episode
await graphiti_client.add_episode(
    episode_id=f"epic_{epic_id}",
    content=f"Epic #{epic_id}: {epic_title}. Status: {status}. Contains Features: {feature_ids}",
    source="Azure DevOps",
    timestamp=datetime.now(timezone.utc),
    metadata={"work_item_type": "Epic", "ado_id": epic_id}
)
```

---

### PostgreSQL Hybrid Search Documentation

#### PostgreSQL Full-Text Search
- **https://www.postgresql.org/docs/current/textsearch.html** - TSVector and TSQuery documentation
- **https://www.postgresql.org/docs/current/pgtrgm.html** - pg_trgm extension for trigram matching

#### Hybrid Search Implementation Guides
- **https://supabase.com/docs/guides/ai/hybrid-search** - Supabase hybrid search guide
- **https://www.tigerdata.com/blog/postgresql-hybrid-search-using-pgvector-and-cohere** - TSVector + pgvector patterns
- **https://jkatz05.com/post/postgres/hybrid-search-postgres-pgvector/** - Production hybrid search strategies

**Key SQL Concepts:**
- `to_tsvector('english', content)` - Converts text to searchable vector
- `ts_rank_cd()` - Ranks documents by relevance (BM25-like scoring)
- `plainto_tsquery()` - Converts plain text to search query
- `@@` operator - TSVector matching operator
- Reciprocal Rank Fusion (RRF) - Combining semantic + keyword scores

---

### Project-Specific Documentation (ADO Templates, Launch Readiness)

**Note**: These are Phase 1 references - already integrated. **DO NOT modify.**

- **Launch Readiness Framework**: `PROJECTS/work-items/knowledge/launch_readiness.md`
- **Work Item Templates**: `PROJECTS/work-items/tracker/feature-template.md` and `user-story-template.md`
- **User Story Standard**: "As a/I want/So that" format with Gherkin acceptance criteria

---

## Implementation Blueprint

### Prerequisites Check

Before starting Phase 2 implementation, verify Phase 1 is complete:

```bash
# 1. Verify Phase 1 tests pass
cd /Volumes/Fulcrum/Develop/full-ai-coding-workflow
source venv/bin/activate
pytest tests/ -v --tb=short

# 2. Verify Phase 1 tools are registered
python -c "
from ai_pm_agent.agent import pm_agent
tool_names = []
for toolset in pm_agent.toolsets:
    tool_names.extend(toolset.tools.keys())
print(f'Phase 1 tools: {len(tool_names)}')
print('Tools:', tool_names)
"
# Expected output: "Phase 1 tools: 8"

# 3. Verify Supabase connection works
python -c "from ai_pm_agent.providers import get_supabase_client; client = get_supabase_client(); print('Supabase connected')"
```

**If any checks fail, DO NOT proceed with Phase 2. Fix Phase 1 first.**

---

### Phase 2 Implementation Steps

**CRITICAL: Follow this order strictly. Do not skip steps.**

---

#### STEP 1: Add Neo4j/Graphiti Integration (graph_utils.py)

**Goal**: Implement knowledge graph client for work item relationship mapping

**Actions**:

1. **Create `ai_pm_agent/graph_utils.py`**
   - Copy entire `GraphitiClient` class from `FullExample/agent/graph_utils.py` (lines 27-353)
   - Copy global client instance pattern (lines 356-367)
   - Copy convenience functions (lines 371-448)

2. **Update `ai_pm_agent/settings.py`**
   - Add Neo4j configuration:
   ```python
   # Neo4j Configuration
   neo4j_uri: str = Field(default="bolt://localhost:7687", env="NEO4J_URI")
   neo4j_user: str = Field(default="neo4j", env="NEO4J_USER")
   neo4j_password: str = Field(..., env="NEO4J_PASSWORD")
   ```

3. **Update `.env.example`**
   ```bash
   # Neo4j Configuration
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USER=neo4j
   NEO4J_PASSWORD=your_password
   ```

4. **Extend `ai_pm_agent/dependencies.py`**
   - Import: `from .graph_utils import GraphitiClient`
   - Add field to `AgentDependencies`:
   ```python
   @dataclass
   class AgentDependencies:
       # ... existing Phase 1 fields ...

       # Phase 2: Knowledge Graph
       graphiti_client: Optional[GraphitiClient] = None
   ```

5. **Test GraphitiClient initialization**
   ```bash
   # Create test file: tests/test_graph_utils.py
   pytest tests/test_graph_utils.py -v
   ```

**Validation Gate**:
```bash
# Verify graph_utils.py is created
ls -la ai_pm_agent/graph_utils.py

# Verify settings updated
grep "neo4j" ai_pm_agent/settings.py

# Run tests
pytest tests/test_graph_utils.py -v
```

**Files Created/Modified**:
- ✅ `ai_pm_agent/graph_utils.py` (NEW)
- ✅ `ai_pm_agent/settings.py` (MODIFIED - add Neo4j config)
- ✅ `ai_pm_agent/dependencies.py` (MODIFIED - add graphiti_client)
- ✅ `.env.example` (MODIFIED - add Neo4j vars)
- ✅ `tests/test_graph_utils.py` (NEW)

---

#### STEP 2: Add AsyncPG + TSVector Hybrid Search (db_utils.py)

**Goal**: Implement direct PostgreSQL connection for TSVector hybrid search

**Actions**:

1. **Create `ai_pm_agent/db_utils.py`**
   - Copy `DatabasePool` class from `FullExample/agent/db_utils.py` (lines 24-66)
   - Copy `hybrid_search()` function (lines 408-452)
   - Copy `test_connection()` function (lines 500-513)
   - **Adapt for work items** (not documents/chunks):
     ```python
     async def hybrid_work_item_search(
         embedding: List[float],
         query_text: str,
         limit: int = 10,
         keyword_weight: float = 0.3
     ) -> List[Dict[str, Any]]:
         # Use work_items table instead of chunks
         # Call hybrid_work_item_search() SQL function
     ```

2. **Create SQL migration in Supabase**
   - Add TSVector generated column to work_items table:
   ```sql
   -- Step 1: Add TSVector column for keyword search
   ALTER TABLE work_items
   ADD COLUMN IF NOT EXISTS search_vector tsvector
   GENERATED ALWAYS AS (
       to_tsvector('english', 
           COALESCE(title, '') || ' ' || COALESCE(description, '')
       )
   ) STORED;

   -- Step 2: Create GIN index for fast text search
   CREATE INDEX IF NOT EXISTS idx_work_items_search_vector
   ON work_items USING GIN (search_vector);
   ```

   - Create `hybrid_work_item_search()` SQL function (adapted from FullExample):
   ```sql
   -- Step 3: Create hybrid search function combining pgvector + TSVector
   CREATE OR REPLACE FUNCTION hybrid_work_item_search(
       query_embedding vector(1536),
       query_text TEXT,
       match_count INT DEFAULT 10,
       text_weight FLOAT DEFAULT 0.3
   )
   RETURNS TABLE (
       work_item_id INTEGER,
       ado_id INTEGER,
       title TEXT,
       description TEXT,
       work_item_type TEXT,
       combined_score FLOAT,
       vector_similarity FLOAT,
       text_similarity FLOAT,
       metadata JSONB
   )
   LANGUAGE plpgsql
   AS $$
   BEGIN
       RETURN QUERY
       WITH vector_results AS (
           SELECT 
               w.id,
               w.ado_id,
               w.title,
               w.description,
               w.work_item_type,
               w.metadata,
               1 - (w.embedding <=> query_embedding) AS vector_sim
           FROM work_items w 
           WHERE w.embedding IS NOT NULL
       ),
       text_results AS (
           SELECT 
               w.id,
               w.ado_id,
               w.title,
               w.description,
               w.work_item_type,
               w.metadata,
               ts_rank_cd(w.search_vector, plainto_tsquery('english', query_text)) AS text_sim
           FROM work_items w
           WHERE w.search_vector @@ plainto_tsquery('english', query_text)
       )
       SELECT
           COALESCE(v.id, t.id) AS work_item_id,
           COALESCE(v.ado_id, t.ado_id) AS ado_id,
           COALESCE(v.title, t.title) AS title,
           COALESCE(v.description, t.description) AS description,
           COALESCE(v.work_item_type, t.work_item_type) AS work_item_type,
           (COALESCE(v.vector_sim, 0) * (1 - text_weight) +
            COALESCE(t.text_sim, 0) * text_weight) AS combined_score,
           v.vector_sim AS vector_similarity,
           t.text_sim AS text_similarity,
           COALESCE(v.metadata, t.metadata) AS metadata
       FROM vector_results v
       FULL OUTER JOIN text_results t ON v.id = t.id
       ORDER BY combined_score DESC
       LIMIT match_count;
   END;
   $$;
   ```

   **Key differences from FullExample (chunks table):**
   - Uses `work_items` table instead of `chunks`
   - Returns `ado_id` (work item identifier) instead of `chunk_id`
   - Returns `work_item_type` (Epic, Feature, User Story) instead of `document_id`
   - Handles NULL descriptions with COALESCE

3. **Update `ai_pm_agent/settings.py`**
   - Add AsyncPG configuration (same as DATABASE_URL):
   ```python
   # Database Configuration (for AsyncPG direct access)
   database_url: str = Field(..., env="DATABASE_URL")
   ```

4. **Extend `ai_pm_agent/dependencies.py`**
   - Import: `from .db_utils import DatabasePool`
   - Add field:
   ```python
   # Phase 2: Direct PostgreSQL Access
   database_pool: Optional[DatabasePool] = None
   ```

5. **Test hybrid_search function**
   ```bash
   pytest tests/test_db_utils.py -v
   ```

**Validation Gate**:
```bash
# Verify db_utils.py created
ls -la ai_pm_agent/db_utils.py

# Verify SQL migration applied (check Supabase dashboard or run query)
# Test the function (replace with actual 1536-dimension embedding):
# SELECT * FROM hybrid_work_item_search(
#     '[0.1,0.2,...]'::vector(1536), 
#     'API timeout bug', 
#     10, 
#     0.3
# );

# Run tests
pytest tests/test_db_utils.py -v
```

**Files Created/Modified**:
- ✅ `ai_pm_agent/db_utils.py` (NEW)
- ✅ `supabase/migrations/002_hybrid_search.sql` (NEW - if using Supabase migrations)
- ✅ `ai_pm_agent/dependencies.py` (MODIFIED - add database_pool)
- ✅ `tests/test_db_utils.py` (NEW)

---

#### STEP 3: Add Phase 2 Pydantic Models

**Goal**: Define structured outputs for new tools

**Actions**:

1. **Update `ai_pm_agent/models.py`**
   - Add Phase 2 models (keep Phase 1 models unchanged):

   ```python
   # Phase 2: Knowledge Graph Models
   class GraphRelationship(BaseModel):
       """Represents a relationship between work items in the knowledge graph."""
       source_work_item: str
       relationship_type: str  # CONTAINS, IMPLEMENTS, BLOCKS, DEPENDS_ON
       target_work_item: str
       valid_from: Optional[str] = None
       valid_until: Optional[str] = None
       metadata: Dict[str, Any] = Field(default_factory=dict)

   class GraphSearchResult(BaseModel):
       """Result from knowledge graph search."""
       central_entity: str
       related_entities: List[Dict[str, Any]]
       relationships: List[GraphRelationship]
       search_method: str = "graphiti_semantic_search"

   # Phase 2: Hybrid Search Models
   class HybridSearchMatch(BaseModel):
       """Work item match from hybrid search."""
       work_item_id: int
       title: str
       description: str
       combined_score: float
       vector_similarity: float
       text_similarity: float
       metadata: Dict[str, Any] = Field(default_factory=dict)

   # Phase 2: Timeline Models
   class TimelineEvent(BaseModel):
       """Event in work item timeline."""
       fact: str
       timestamp: str
       event_type: str  # status_change, assignment_update, sprint_movement
       metadata: Dict[str, Any] = Field(default_factory=dict)

   # Phase 2: Comprehensive Search Result
   class ComprehensiveSearchResult(BaseModel):
       """Combined results from all search strategies."""
       semantic_matches: List[WorkItemMatch]  # Phase 1 model
       graph_relationships: List[GraphRelationship]  # New
       hybrid_matches: List[HybridSearchMatch]  # New
       total_results: int
       search_strategies_used: List[str]
       combined_ranking: List[WorkItemMatch]
   ```

2. **Test models with Pydantic validation**
   ```bash
   pytest tests/test_models.py -v
   ```

**Validation Gate**:
```bash
# Verify models added
grep "class GraphSearchResult" ai_pm_agent/models.py
grep "class HybridSearchMatch" ai_pm_agent/models.py
grep "class ComprehensiveSearchResult" ai_pm_agent/models.py

# Run model tests
pytest tests/test_models.py -v
```

**Files Modified**:
- ✅ `ai_pm_agent/models.py` (MODIFIED - add 5 new models)
- ✅ `tests/test_models.py` (MODIFIED - add Phase 2 model tests)

---

#### STEP 4: Implement Phase 2 Tools

**Goal**: Add 4 new tools to the agent

**Actions**:

1. **Update `ai_pm_agent/tools.py`**
   - Add Phase 2 tool implementations (keep Phase 1 tools unchanged):

   ```python
   # Phase 2: Tool Implementation Functions

   async def comprehensive_work_item_search_tool(
       query: str,
       use_semantic: bool,
       use_graph: bool,
       use_hybrid: bool,
       limit: int,
       openai_client,  # For embeddings
       supabase_client,  # For semantic search
       graphiti_client,  # For graph search
       database_pool  # For hybrid search
   ) -> ComprehensiveSearchResult:
       """Execute all search strategies in parallel and merge results."""

       # Generate embedding once
       embedding = await generate_embedding(query, openai_client)

       # Build task list
       tasks = []
       strategies_used = []

       if use_semantic:
           tasks.append(semantic_search_supabase(embedding, limit, supabase_client))
           strategies_used.append("semantic")

       if use_graph:
           tasks.append(graphiti_client.search(query))
           strategies_used.append("graph")

       if use_hybrid:
           tasks.append(db_utils.hybrid_work_item_search(embedding, query, limit))
           strategies_used.append("hybrid")

       # Execute in parallel
       results = await asyncio.gather(*tasks)

       # Merge and rank
       return ComprehensiveSearchResult(
           semantic_matches=results[0] if use_semantic else [],
           graph_relationships=results[1] if use_graph else [],
           hybrid_matches=results[2] if use_hybrid else [],
           total_results=sum(len(r) for r in results),
           search_strategies_used=strategies_used,
           combined_ranking=merge_and_rank_results(results)
       )

   async def search_work_item_graph_tool(
       entity_name: str,
       relationship_types: Optional[List[str]],
       depth: int,
       graphiti_client
   ) -> GraphSearchResult:
       """Search knowledge graph for work item relationships."""
       results = await graphiti_client.get_related_entities(
           entity_name,
           relationship_types=relationship_types,
           depth=depth
       )
       return GraphSearchResult(**results)

   async def get_work_item_timeline_tool(
       work_item_id: int,
       start_date: Optional[str],
       end_date: Optional[str],
       graphiti_client
   ) -> List[TimelineEvent]:
       """Retrieve temporal history from knowledge graph."""
       timeline = await graphiti_client.get_entity_timeline(
           f"WorkItem#{work_item_id}",
           start_date=datetime.fromisoformat(start_date) if start_date else None,
           end_date=datetime.fromisoformat(end_date) if end_date else None
       )
       return [TimelineEvent(**event) for event in timeline]

   async def hybrid_work_item_search_tool(
       query: str,
       limit: int,
       keyword_weight: float,
       openai_client,
       database_pool
   ) -> List[HybridSearchMatch]:
       """Combined semantic + keyword search."""
       embedding = await generate_embedding(query, openai_client)
       results = await db_utils.hybrid_work_item_search(
           embedding, query, limit, keyword_weight
       )
       return [HybridSearchMatch(**r) for r in results]
   ```

2. **Update `ai_pm_agent/agent.py`**
   - Register 4 new tools with `@pm_agent.tool` decorator (keep Phase 1 tools):

   ```python
   # Phase 2 Tool Registrations

   @pm_agent.tool
   async def comprehensive_work_item_search(
       ctx: RunContext[AgentDependencies],
       query: str,
       use_semantic: bool = True,
       use_graph: bool = True,
       use_hybrid: bool = False,
       limit: int = 10
   ) -> ComprehensiveSearchResult:
       """Master search tool combining all strategies."""
       return await comprehensive_work_item_search_tool(
           query=query,
           use_semantic=use_semantic,
           use_graph=use_graph,
           use_hybrid=use_hybrid,
           limit=limit,
           openai_client=ctx.deps.openai_client,
           supabase_client=ctx.deps.supabase_client,
           graphiti_client=ctx.deps.graphiti_client,
           database_pool=ctx.deps.database_pool
       )

   @pm_agent.tool
   async def search_work_item_graph(
       ctx: RunContext[AgentDependencies],
       entity_name: str,
       relationship_types: Optional[List[str]] = None,
       depth: int = 2
   ) -> GraphSearchResult:
       """Search knowledge graph for relationships."""
       return await search_work_item_graph_tool(
           entity_name=entity_name,
           relationship_types=relationship_types,
           depth=depth,
           graphiti_client=ctx.deps.graphiti_client
       )

   @pm_agent.tool
   async def get_work_item_timeline(
       ctx: RunContext[AgentDependencies],
       work_item_id: int,
       start_date: Optional[str] = None,
       end_date: Optional[str] = None
   ) -> List[TimelineEvent]:
       """Retrieve temporal work item history."""
       return await get_work_item_timeline_tool(
           work_item_id=work_item_id,
           start_date=start_date,
           end_date=end_date,
           graphiti_client=ctx.deps.graphiti_client
       )

   @pm_agent.tool
   async def hybrid_work_item_search(
       ctx: RunContext[AgentDependencies],
       query: str,
       limit: int = 10,
       keyword_weight: float = 0.3
   ) -> List[HybridSearchMatch]:
       """Hybrid semantic + keyword search."""
       return await hybrid_work_item_search_tool(
           query=query,
           limit=limit,
           keyword_weight=keyword_weight,
           openai_client=ctx.deps.openai_client,
           database_pool=ctx.deps.database_pool
       )
   ```

3. **Verify tool count**
   ```python
   # Should now have 12 tools (8 Phase 1 + 4 Phase 2)
   python -c "
from ai_pm_agent.agent import pm_agent
tool_names = []
for toolset in pm_agent.toolsets:
    tool_names.extend(toolset.tools.keys())
print(f'Total tools: {len(tool_names)}')
print('Tools:', tool_names)
"
   # Expected: "Total tools: 12"
   ```

**Validation Gate**:
```bash
# Verify tools implemented
grep "async def comprehensive_work_item_search_tool" ai_pm_agent/tools.py
grep "async def search_work_item_graph_tool" ai_pm_agent/tools.py
grep "async def get_work_item_timeline_tool" ai_pm_agent/tools.py
grep "async def hybrid_work_item_search_tool" ai_pm_agent/tools.py

# Verify tool registrations
grep "@pm_agent.tool" ai_pm_agent/agent.py | wc -l
# Expected: 12 (8 Phase 1 + 4 Phase 2)

# Run syntax check
ruff check ai_pm_agent/ --fix
```

**Files Modified**:
- ✅ `ai_pm_agent/tools.py` (MODIFIED - add 4 tool implementations)
- ✅ `ai_pm_agent/agent.py` (MODIFIED - register 4 new tools)

---

#### STEP 5: Update System Prompt for Search Strategy Guidance

**Goal**: Guide agent to choose the right search strategy for each query type

**Actions**:

1. **Update `ai_pm_agent/prompts.py`**
   - Add Phase 2 search strategy guidance to system prompt:

   ```python
   SYSTEM_PROMPT = """You are an AI Project Management Assistant for the CoDeveloper platform with **advanced Hybrid RAG capabilities**.

   ### Your Search Capabilities:

   You have three powerful search strategies that work together:

   1. **Semantic Search (pgvector)** - Understands meaning and context. Use when searching for conceptually similar work items.
      - Tool: `match_work_items()` (Phase 1)
      - Best for: General work item matching

   2. **Knowledge Graph (Neo4j/Graphiti)** - Understands relationships and dependencies. Use when exploring connections between Epics, Features, and User Stories.
      - Tool: `search_work_item_graph()`
      - Tool: `get_work_item_timeline()`
      - Best for: Relationship queries, dependency mapping, temporal history

   3. **Hybrid Search (TSVector + pgvector)** - Combines semantic understanding with exact keyword matching. Use when users need both conceptual relevance and specific terminology matches.
      - Tool: `hybrid_work_item_search()`
      - Best for: Keyword-specific searches ("API timeout bug", "login security issue")

   4. **Comprehensive Search (All Three Combined)** - Leverages all strategies in parallel for complete context.
      - Tool: `comprehensive_work_item_search()`
      - Best for: Complex queries needing multiple perspectives

   **When to use each search strategy:**
   - Use **semantic search** for most work item matching (your default behavior)
   - Use **knowledge graph** when asked about relationships ("What Features belong to Epic X?", "Show me dependencies")
   - Use **hybrid search** when keywords matter ("Find bugs with 'API timeout' error", "Security issues in login flow")
   - Use **comprehensive search** when you want to leverage all three strategies for complete context

   ### Your Primary Responsibilities:

   [... rest of Phase 1 system prompt unchanged ...]
   """
   ```

**Validation Gate**:
```bash
# Verify system prompt updated
grep "Knowledge Graph" ai_pm_agent/prompts.py
grep "Hybrid Search" ai_pm_agent/prompts.py
grep "comprehensive_work_item_search" ai_pm_agent/prompts.py
```

**Files Modified**:
- ✅ `ai_pm_agent/prompts.py` (MODIFIED - add search strategy guidance)

---

#### STEP 6: Add Comprehensive Testing

**Goal**: Validate all Phase 2 components with TestModel and mocks

**Actions**:

1. **Update `tests/conftest.py`**
   - Add Phase 2 mocks:

   ```python
   @pytest.fixture
   def mock_graphiti_client():
       """Mock GraphitiClient for testing."""
       mock = AsyncMock()
       mock.search.return_value = [
           {"fact": "Epic #1234 contains Feature #5678", "uuid": "test-uuid"}
       ]
       mock.get_related_entities.return_value = {
           "central_entity": "Epic #1234",
           "related_facts": [],
           "search_method": "graphiti_semantic_search"
       }
       mock.get_entity_timeline.return_value = [
           {"fact": "Status changed to In Progress", "timestamp": "2025-01-15T10:00:00Z"}
       ]
       return mock

   @pytest.fixture
   def mock_database_pool():
       """Mock DatabasePool for testing."""
       mock = AsyncMock()
       # Mock hybrid search results
       return mock

   @pytest.fixture
   def phase2_dependencies(
       mock_supabase_client,
       mock_openai_client,
       mock_graphiti_client,
       mock_database_pool
   ):
       """Dependencies with Phase 2 additions."""
       from ai_pm_agent.dependencies import AgentDependencies
       return AgentDependencies(
           # Phase 1 dependencies
           supabase_client=mock_supabase_client,
           openai_client=mock_openai_client,
           current_sprint_info={"sprint": 42, "capacity": 100},
           work_item_templates={},
           launch_readiness_categories=[],
           gtm_phase_definitions={},
           artifact_registry_path="/tmp/artifacts",
           ado_project_name="NorthStar",
           # Phase 2 dependencies
           graphiti_client=mock_graphiti_client,
           database_pool=mock_database_pool
       )
   ```

2. **Create `tests/test_phase2_tools.py`**
   - Test each new tool with TestModel:

   ```python
   import pytest
   from pydantic_ai.models.test import TestModel
   from ai_pm_agent.agent import pm_agent

   @pytest.mark.asyncio
   async def test_comprehensive_work_item_search(phase2_dependencies):
       """Test comprehensive search tool."""
       test_model = TestModel()

       with pm_agent.override(model=test_model):
           result = await pm_agent.run(
               "Find work items related to API timeout bugs",
               deps=phase2_dependencies
           )

           # Verify tool was called
           assert "comprehensive_work_item_search" in str(result.tool_calls)

   @pytest.mark.asyncio
   async def test_search_work_item_graph(phase2_dependencies):
       """Test graph search tool."""
       # Similar pattern...

   @pytest.mark.asyncio
   async def test_get_work_item_timeline(phase2_dependencies):
       """Test timeline tool."""
       # Similar pattern...

   @pytest.mark.asyncio
   async def test_hybrid_work_item_search(phase2_dependencies):
       """Test hybrid search tool."""
       # Similar pattern...
   ```

3. **Create integration test for all 12 tools**
   ```python
   @pytest.mark.asyncio
   async def test_all_tools_registered(phase2_dependencies):
       """Verify all 12 tools are registered."""
       assert len(pm_agent.tools) == 12

       tool_names = [t.name for t in pm_agent.tools]

       # Phase 1 tools (8)
       assert "classify_input" in tool_names
       assert "match_work_items" in tool_names
       # ... etc

       # Phase 2 tools (4)
       assert "comprehensive_work_item_search" in tool_names
       assert "search_work_item_graph" in tool_names
       assert "get_work_item_timeline" in tool_names
       assert "hybrid_work_item_search" in tool_names
   ```

4. **Run full test suite**
   ```bash
   pytest tests/ -v --tb=short --cov=ai_pm_agent --cov-report=term-missing
   ```

**Validation Gate**:
```bash
# Verify test files created
ls -la tests/test_phase2_tools.py
ls -la tests/test_graph_utils.py
ls -la tests/test_db_utils.py

# Run all tests
pytest tests/ -v --tb=short

# Check coverage
pytest tests/ --cov=ai_pm_agent --cov-report=term-missing
# Target: >80% coverage
```

**Files Created/Modified**:
- ✅ `tests/conftest.py` (MODIFIED - add Phase 2 mocks)
- ✅ `tests/test_phase2_tools.py` (NEW)
- ✅ `tests/test_graph_utils.py` (NEW)
- ✅ `tests/test_db_utils.py` (NEW)

---

#### STEP 7: Update Documentation

**Goal**: Document Phase 2 enhancements

**Actions**:

1. **Update `README.md` (if exists)**
   - Add Phase 2 capabilities section
   - Update installation instructions for Neo4j
   - Add hybrid search configuration examples

2. **Create `PHASE2_MIGRATION.md`**
   - Document upgrade path from Phase 1 to Phase 2
   - Environment variables to add
   - Database migrations to run
   - Testing procedures

3. **Update `planning/` documents** (optional)
   - Add `planning/phase2_architecture.md` with graph/hybrid search design decisions

**Validation Gate**:
```bash
# Verify documentation updated
ls -la PHASE2_MIGRATION.md
grep "Phase 2" README.md || echo "No README.md found"
```

**Files Created/Modified**:
- ✅ `PHASE2_MIGRATION.md` (NEW)
- ✅ `README.md` (MODIFIED - if exists)
- ✅ `planning/phase2_architecture.md` (NEW - optional)

---

#### STEP 8: Final Integration Testing

**Goal**: Verify Phase 1 + Phase 2 work together seamlessly

**Actions**:

1. **Run complete test suite**
   ```bash
   pytest tests/ -v --tb=short --cov=ai_pm_agent --cov-report=html
   ```

2. **Verify all 12 tools registered**
   ```bash
   python -c "
from ai_pm_agent.agent import pm_agent
tool_names = []
for toolset in pm_agent.toolsets:
    tool_names.extend(toolset.tools.keys())
print(f'Total tools: {len(tool_names)}')
for name in tool_names:
    print(f'  - {name}')
"
   ```

3. **Test agent with comprehensive query**
   ```python
   # Create manual test script: test_phase2_manual.py
   from ai_pm_agent.agent import pm_agent
   from ai_pm_agent.dependencies import AgentDependencies
   from ai_pm_agent.settings import settings

   async def main():
       deps = AgentDependencies.from_settings(settings)

       # Test comprehensive search
       result = await pm_agent.run(
           "Find all work items related to API timeout bugs and show their dependencies",
           deps=deps
       )

       print("Agent Response:", result.output)
       print("Tools Used:", [t.name for t in result.tool_calls])

   if __name__ == "__main__":
       import asyncio
       asyncio.run(main())
   ```

4. **Performance testing**
   - Measure P95 latency for comprehensive_work_item_search
   - Target: <500ms

**Validation Gate**:
```bash
# All tests pass
pytest tests/ -v

# Coverage >80%
pytest tests/ --cov=ai_pm_agent --cov-report=term

# No linting errors
ruff check ai_pm_agent/
black ai_pm_agent/ --check

# Type checking (if using mypy)
mypy ai_pm_agent/ || echo "mypy not configured"
```

---

## Validation Loop

### Executable Validation Gates

**Phase 2 validation gates must pass before considering implementation complete.**

#### 1. Syntax and Style
```bash
# Activate virtual environment
source venv/bin/activate

# Fix linting issues
ruff check ai_pm_agent/ --fix

# Format code
black ai_pm_agent/

# Verify no errors
ruff check ai_pm_agent/
black ai_pm_agent/ --check
```

#### 2. Type Checking (Optional - if mypy configured)
```bash
mypy ai_pm_agent/ --strict
```

#### 3. Unit Tests (Phase 1 + Phase 2)
```bash
# Run all tests
pytest tests/ -v --tb=short

# Run Phase 2 tests specifically
pytest tests/test_phase2_tools.py -v
pytest tests/test_graph_utils.py -v
pytest tests/test_db_utils.py -v

# Run Phase 1 tests (ensure no regression)
pytest tests/test_models.py -v
```

#### 4. Test Coverage
```bash
# Generate coverage report
pytest tests/ --cov=ai_pm_agent --cov-report=term-missing --cov-report=html

# Minimum coverage: 80%
# Open htmlcov/index.html to view detailed report
```

#### 5. Integration Tests
```bash
# Verify 12 tools registered
python -c "
from ai_pm_agent.agent import pm_agent
tool_names = []
for toolset in pm_agent.toolsets:
    tool_names.extend(toolset.tools.keys())
assert len(tool_names) == 12, f'Expected 12 tools, got {len(tool_names)}'
print('✓ All 12 tools registered')
"

# Verify Neo4j connection (if configured)
python -c "from ai_pm_agent.graph_utils import test_graph_connection; import asyncio; result = asyncio.run(test_graph_connection()); assert result, 'Neo4j connection failed'; print('✓ Neo4j connected')"

# Verify PostgreSQL connection
python -c "from ai_pm_agent.db_utils import test_connection; import asyncio; result = asyncio.run(test_connection()); assert result, 'PostgreSQL connection failed'; print('✓ PostgreSQL connected')"
```

#### 6. Performance Tests
```bash
# Run performance benchmarks (if implemented)
pytest tests/test_performance.py -v
```

#### 7. Manual Agent Testing
```bash
# Test comprehensive search manually
python test_phase2_manual.py
```

---

### Success Criteria Checklist

**Before marking Phase 2 complete, verify ALL items:**

#### Phase 2 Implementation
- [ ] `ai_pm_agent/graph_utils.py` created with GraphitiClient
- [ ] `ai_pm_agent/db_utils.py` created with DatabasePool and hybrid_search
- [ ] `ai_pm_agent/models.py` extended with 5 new Pydantic models
- [ ] `ai_pm_agent/tools.py` extended with 4 new tool implementations
- [ ] `ai_pm_agent/agent.py` updated with 4 new tool registrations
- [ ] `ai_pm_agent/dependencies.py` extended with graphiti_client and database_pool
- [ ] `ai_pm_agent/settings.py` updated with Neo4j configuration
- [ ] `ai_pm_agent/prompts.py` updated with search strategy guidance

#### Testing
- [ ] All Phase 1 tests still pass (no regression)
- [ ] Phase 2 tests created and passing
- [ ] Test coverage >80%
- [ ] GraphitiClient mocked in tests
- [ ] DatabasePool mocked in tests
- [ ] TestModel validation for all 4 new tools

#### Documentation
- [ ] `.env.example` updated with Neo4j variables
- [ ] `PHASE2_MIGRATION.md` created
- [ ] README.md updated (if exists)

#### Integration
- [ ] 12 tools registered (8 Phase 1 + 4 Phase 2)
- [ ] Neo4j connection working
- [ ] PostgreSQL AsyncPG connection working
- [ ] Supabase connection still working (Phase 1)
- [ ] Hybrid search SQL function deployed

#### Performance
- [ ] comprehensive_work_item_search P95 latency <500ms
- [ ] No memory leaks in connection pooling
- [ ] Concurrent tool execution works correctly

---

## Quality Score

### One-Pass Implementation Confidence: **9/10**

**Rationale:**

**Strengths (+9)**:
- ✅ **Complete reference implementation** - FullExample provides exact patterns to copy
- ✅ **Solid Phase 1 foundation** - No need to refactor existing code
- ✅ **Clear file paths and line numbers** - Every pattern referenced with specific locations
- ✅ **Proven architecture** - Hybrid RAG is production-tested in FullExample
- ✅ **Comprehensive context** - All documentation, patterns, and gotchas included
- ✅ **Executable validation gates** - Clear pass/fail criteria
- ✅ **Type-safe design** - Pydantic AI + Pydantic models catch errors early
- ✅ **Mocking patterns established** - Phase 1 tests show how to mock all dependencies
- ✅ **Incremental implementation** - Clear step-by-step order prevents mistakes

**Minor Risks (-1)**:
- ⚠️ **Database schema adaptation** - Need to adapt chunks/documents schema to work_items (low risk, well-documented)
- ⚠️ **Neo4j setup** - Requires external service, but FullExample patterns handle edge cases

**Overall Assessment**: This PRP has exceptionally high one-pass success probability due to complete FullExample reference, solid Phase 1 foundation, and comprehensive context engineering.

---

## Implementation Time Estimate

**Total: 6-8 hours** (with FullExample reference)

- **Step 1** (Graph Utils): 1.5 hours
- **Step 2** (Hybrid Search): 1.5 hours
- **Step 3** (Models): 0.5 hours
- **Step 4** (Tools): 2 hours
- **Step 5** (System Prompt): 0.5 hours
- **Step 6** (Testing): 2 hours
- **Step 7** (Documentation): 0.5 hours
- **Step 8** (Integration): 1 hour

---

## Post-Implementation Checklist

After Phase 2 is complete:

1. **Update CLAUDE.md**
   - Document Phase 2 capabilities
   - Update architecture diagrams
   - Add Phase 2 tool descriptions

2. **Create Phase 3 INITIAL.md** (if applicable)
   - Reference Phase 1 + Phase 2 as foundation
   - Define next enhancements

3. **Production Deployment**
   - Set up Neo4j production instance
   - Configure connection pooling limits
   - Monitor P95 latency
   - Set up alerts for graph/database connection failures

4. **Performance Optimization** (if needed)
   - Add caching for frequent graph queries
   - Optimize hybrid search SQL function
   - Implement result pagination for large result sets

---

**END OF PRP**
