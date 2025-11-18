# Hybrid RAG Agent - Development Rules

> **📚 Generic development rules for building Hybrid RAG agents with Pydantic AI**
>
> **Purpose:** Best practices and patterns for this reference architecture
>
> **Adaptable to:** Any domain requiring semantic + keyword + graph search

## Overview

This reference architecture demonstrates a production-ready Hybrid RAG system combining:
- **Semantic Search** via PostgreSQL + pgvector
- **Keyword Search** via PostgreSQL TSVector  
- **Knowledge Graph** via Neo4j + Graphiti

These rules apply to building and maintaining this type of system.

## Core Development Principles

### 1. Agent-First Development
- **Tools drive functionality** - Each capability should be a well-defined tool
- **Test with TestModel first** - Validate logic before using real LLMs
- **System prompt guides behavior** - Clear instructions for when to use each tool
- **Dependency injection** - Use `RunContext[AgentDependencies]` for external services

### 2. Database-First Design
- **Schema drives implementation** - Design database schema before tools
- **SQL functions for complex queries** - Leverage database capabilities
- **Proper indexing is critical** - IVFFlat for vectors, GIN for text
- **Connection pooling** - Always use AsyncPG connection pools

### 3. Modular Architecture
- **Separate concerns** - agent.py, tools.py, models.py, db_utils.py, graph_utils.py
- **Keep files under 500 lines** - Split when approaching limit
- **Reusable utilities** - Database and graph operations should be isolated
- **Clear interfaces** - Pydantic models for all data exchange

### 4. Testing Strategy
- **Unit tests for utilities** - Test database and graph functions independently
- **Integration tests for tools** - Test agent tools with mocked dependencies
- **Agent tests with TestModel** - Validate agent behavior without API calls
- **Mock external services** - Don't depend on real databases or LLMs in tests

## Pydantic AI Patterns

### Agent Setup

```python
from pydantic_ai import Agent, RunContext
from dataclasses import dataclass

@dataclass
class AgentDependencies:
    """Dependencies injected into agent tools"""
    db_pool: asyncpg.Pool
    graphiti_client: Graphiti
    embedder: Embedder
    config: Settings

agent = Agent(
    model=get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT
)
```

### Tool Registration

```python
# Tool WITH context (needs dependencies)
@agent.tool
async def search_tool(
    ctx: RunContext[AgentDependencies], 
    query: str,
    limit: int = 10
) -> list[SearchResult]:
    """
    Tool description that guides LLM on when to use this.
    Be very specific about the use case.
    """
    return await vector_search(
        ctx.deps.db_pool,
        query,
        limit
    )

# Tool WITHOUT context (pure function)
@agent.tool_plain
def format_result(data: dict) -> str:
    """Simple tool that doesn't need external dependencies"""
    return json.dumps(data, indent=2)
```

### Testing Agents

```python
from pydantic_ai.models.test import TestModel

async def test_agent_behavior():
    test_model = TestModel()
    
    # Override model for testing
    with agent.override(model=test_model):
        result = await agent.run(
            "test query",
            deps=test_dependencies
        )
        assert result.output
```

## Database Patterns

### PostgreSQL with pgvector

```python
# Creating vectors
embedding = await embedder.embed(text)
await conn.execute("""
    INSERT INTO entities (content, embedding)
    VALUES ($1, $2)
""", text, embedding)

# Semantic search with cosine similarity
results = await conn.fetch("""
    SELECT *, 1 - (embedding <=> $1) AS similarity
    FROM entities
    WHERE 1 - (embedding <=> $1) > $2
    ORDER BY embedding <=> $1
    LIMIT $3
""", query_embedding, threshold, limit)
```

### Full-Text Search (TSVector)

```python
# Creating search vectors
await conn.execute("""
    UPDATE entities
    SET search_vector = to_tsvector('english', content)
    WHERE search_vector IS NULL
""")

# Keyword search with ranking
results = await conn.fetch("""
    SELECT *, ts_rank(search_vector, query) AS rank
    FROM entities, plainto_tsquery('english', $1) query
    WHERE search_vector @@ query
    ORDER BY rank DESC
    LIMIT $2
""", query_text, limit)
```

### Hybrid Search

```python
# Combining vector + keyword search
results = await conn.fetch("""
    WITH vector_results AS (
        SELECT id, 1 - (embedding <=> $1) AS vector_sim
        FROM entities
    ),
    text_results AS (
        SELECT id, ts_rank(search_vector, plainto_tsquery($2)) AS text_sim
        FROM entities
        WHERE search_vector @@ plainto_tsquery($2)
    )
    SELECT 
        e.*,
        (COALESCE(v.vector_sim, 0) * (1 - $4) +
         COALESCE(t.text_sim, 0) * $4) AS combined_score
    FROM entities e
    LEFT JOIN vector_results v ON v.id = e.id
    LEFT JOIN text_results t ON t.id = e.id
    ORDER BY combined_score DESC
    LIMIT $3
""", embedding, query_text, limit, keyword_weight)
```

## Knowledge Graph Patterns (Graphiti)

### Graph Client Setup

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client import OpenAIClient

# OpenAI-compatible client (works with Ollama, OpenRouter, etc.)
llm_client = OpenAIClient(
    base_url=settings.llm_base_url,
    api_key=settings.llm_api_key,
    model=settings.llm_choice
)

graphiti = Graphiti(
    neo4j_uri=settings.neo4j_uri,
    neo4j_user=settings.neo4j_user,
    neo4j_password=settings.neo4j_password,
    llm_client=llm_client
)
```

### Adding Episodes

```python
# Group related changes into episodes
await graphiti.add_episode(
    name="Feature #123 development",
    episode_body="Implemented new API endpoint for user authentication",
    source_description="Work item update",
    reference_time=datetime.utcnow()
)
```

### Searching Graph

```python
# Search for entities and relationships
results = await graphiti.search(
    query="How are Epic #123 and Feature #456 related?",
    num_results=10
)

for edge in results.edges:
    print(f"{edge.source.name} -> {edge.relationship} -> {edge.target.name}")
```

## System Prompt Best Practices

### 1. Be Specific About Tools

❌ **Vague:**
```
"You have access to search capabilities."
```

✅ **Specific:**
```
"You have three search tools:
- vector_search: Use for conceptual similarity (e.g., 'find similar work items')
- hybrid_search: Use when combining semantic + keywords (e.g., 'API timeout issues')
- graph_search: Use for relationships (e.g., 'how are X and Y related?')"
```

### 2. Provide Examples

```python
SYSTEM_PROMPT = """
...

## Example Interactions

User: "Find work items similar to Epic #123"
You: *Use vector_search to find semantically similar items*

User: "Show all bugs mentioning 'API timeout'"
You: *Use hybrid_search with high keyword_weight*

User: "How are Epic #123 and Feature #456 related?"
You: *Use graph_search to traverse relationships*
"""
```

### 3. Set Clear Boundaries

```python
SYSTEM_PROMPT = """
...

Your responses must:
- ✅ Be based solely on retrieved data
- ✅ Cite specific entities by ID/name
- ❌ Never invent information
- ❌ Never rely on general knowledge
"""
```

## Ingestion Pipeline Patterns

### Semantic Chunking

```python
class SemanticChunker:
    """Break text into meaningful chunks"""
    
    async def chunk(self, text: str) -> list[str]:
        # Use LLM to identify natural breakpoints
        # Or use fixed-size with sentence boundaries
        # Or use recursive splitting
        pass
```

### Embedding Generation

```python
class Embedder:
    """Generate embeddings with caching"""
    
    def __init__(self, provider: str, model: str):
        self.provider = provider
        self.model = model
        self.cache = {}
    
    async def embed(self, text: str) -> list[float]:
        if text in self.cache:
            return self.cache[text]
        
        # Generate embedding based on provider
        embedding = await self._generate(text)
        self.cache[text] = embedding
        return embedding
```

### Graph Construction

```python
async def build_graph(entities: list[Entity], graphiti: Graphiti):
    """Build knowledge graph from entities"""
    
    for entity in entities:
        # Create episode for each entity
        await graphiti.add_episode(
            name=f"{entity.type} {entity.id}",
            episode_body=entity.content,
            source_description=entity.source,
            reference_time=entity.created_at
        )
```

## Testing Patterns

### Mocking Database

```python
@pytest.fixture
async def mock_db_pool():
    """Mock AsyncPG connection pool"""
    pool = AsyncMock()
    
    async def mock_fetch(*args):
        return [
            {"id": 1, "title": "Test", "similarity": 0.95}
        ]
    
    pool.fetch = mock_fetch
    return pool
```

### Mocking Graph

```python
@pytest.fixture
def mock_graphiti():
    """Mock Graphiti client"""
    client = Mock()
    
    async def mock_search(*args, **kwargs):
        return SearchResults(
            edges=[],
            nodes=[]
        )
    
    client.search = mock_search
    return client
```

### Testing Tools

```python
async def test_search_tool(mock_db_pool, mock_graphiti):
    """Test agent tool with mocked dependencies"""
    
    deps = AgentDependencies(
        db_pool=mock_db_pool,
        graphiti_client=mock_graphiti,
        embedder=Mock(),
        config=Settings()
    )
    
    result = await search_tool(
        RunContext(deps=deps),
        query="test query",
        limit=10
    )
    
    assert len(result) > 0
```

## Common Gotchas

### 1. Embedding Dimension Mismatch

❌ **Problem:** Schema expects 1536, model produces 768

✅ **Solution:** Match vector dimension to your model
```sql
-- OpenAI text-embedding-3-small
vector(1536)

-- Ollama nomic-embed-text
vector(768)
```

### 2. Missing Indexes

❌ **Problem:** Slow vector searches

✅ **Solution:** Create proper indexes
```sql
-- Vector index
CREATE INDEX ON entities USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Text search index
CREATE INDEX ON entities USING gin(search_vector);
```

### 3. Connection Pool Exhaustion

❌ **Problem:** Too many concurrent connections

✅ **Solution:** Configure pool size
```python
pool = await asyncpg.create_pool(
    dsn=settings.database_url,
    min_size=10,
    max_size=20
)
```

### 4. Oversized Chunks

❌ **Problem:** Chunks too large (lost specificity) or too small (lost context)

✅ **Solution:** Test and tune chunk size (typically 200-500 tokens)
```python
CHUNK_SIZE = 300  # tokens
CHUNK_OVERLAP = 50  # tokens
```

### 5. Vague Tool Descriptions

❌ **Problem:** Agent doesn't know when to use which tool

✅ **Solution:** Very specific tool docstrings
```python
@agent.tool
async def semantic_search(ctx: RunContext[Deps], query: str) -> list[Result]:
    """
    Search for semantically similar content using vector embeddings.
    
    USE WHEN:
    - User asks for "similar to X"
    - Query is conceptual or thematic
    - Looking for related ideas
    
    DON'T USE WHEN:
    - User wants exact keywords
    - Query includes specific IDs or error codes
    - Relationships between entities are being asked about
    """
    pass
```

### 6. Not Testing with TestModel

❌ **Problem:** Making expensive API calls during development

✅ **Solution:** Use TestModel for fast iteration
```python
from pydantic_ai.models.test import TestModel

# Development
test_model = TestModel()
with agent.override(model=test_model):
    result = await agent.run("test query")

# Production
result = await agent.run("real query")  # Uses real LLM
```

## Configuration Management

### Environment Variables

```python
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

class Settings(BaseSettings):
    """All configuration from environment"""
    
    model_config = ConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Database
    database_url: str
    
    # Neo4j
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    
    # LLM
    llm_provider: str = "openai"
    llm_base_url: str
    llm_api_key: str
    llm_choice: str = "gpt-4.1-mini"
    
    # Embeddings
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"

def load_settings() -> Settings:
    load_dotenv()
    return Settings()
```

## Performance Optimization

### 1. Batch Processing
```python
# Don't: Process one at a time
for item in items:
    await process(item)

# Do: Batch process
await asyncio.gather(*[process(item) for item in items])
```

### 2. Connection Pooling
```python
# Don't: Create connection each time
async def search(query):
    conn = await asyncpg.connect(dsn)
    result = await conn.fetch("SELECT ...")
    await conn.close()

# Do: Use connection pool
async def search(query, pool: asyncpg.Pool):
    async with pool.acquire() as conn:
        result = await conn.fetch("SELECT ...")
```

### 3. Embedding Caching
```python
# Don't: Regenerate embeddings
embedding = await client.embed(query)

# Do: Cache embeddings
if query not in cache:
    cache[query] = await client.embed(query)
embedding = cache[query]
```

### 4. Lazy Loading
```python
# Don't: Load everything upfront
all_data = await fetch_all_data()

# Do: Load on demand
data = await fetch_data_for_query(query)
```

## Deployment Considerations

### 1. Environment-Specific Configuration
```bash
# Development
LLM_CHOICE=gpt-4.1-nano  # Cheaper model

# Production  
LLM_CHOICE=gpt-4.1-mini  # Better quality
```

### 2. Error Handling
```python
try:
    result = await agent.run(query)
except Exception as e:
    logger.error(f"Agent error: {e}")
    return {"error": "Something went wrong"}
```

### 3. Monitoring
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Tool called: {tool_name} with args: {args}")
```

### 4. Rate Limiting
```python
from aiohttp import ClientSession
import asyncio

class RateLimitedClient:
    def __init__(self, max_per_second: int = 10):
        self.max_per_second = max_per_second
        self.last_call = 0
    
    async def call_api(self, *args):
        now = time.time()
        if now - self.last_call < (1 / self.max_per_second):
            await asyncio.sleep((1 / self.max_per_second) - (now - self.last_call))
        self.last_call = time.time()
        return await self._make_call(*args)
```

## Documentation Standards

- **Every function needs a docstring** - Explain purpose, args, returns
- **Tool docstrings guide the LLM** - Be specific about when to use
- **Comment complex logic** - Explain the "why", not just the "what"
- **Keep README updated** - Installation, configuration, usage examples
- **Document your schema** - What each table/column represents

## Anti-Patterns to Avoid

- ❌ Hardcoding API keys or credentials
- ❌ Mixing sync and async code inconsistently
- ❌ Creating files over 500 lines
- ❌ Skipping tests to save time
- ❌ Using vague tool descriptions
- ❌ Not using connection pooling
- ❌ Ignoring database indexes
- ❌ Oversized or undersized chunks
- ❌ No error handling
- ❌ Committing `.env` files

## Checklist for Production

- [ ] All configuration via environment variables
- [ ] Connection pooling configured
- [ ] Proper database indexes created
- [ ] Error handling throughout
- [ ] Logging set up
- [ ] Tests passing (>80% coverage)
- [ ] Documentation complete
- [ ] `.env.example` provided
- [ ] No hardcoded secrets
- [ ] Rate limiting implemented
- [ ] Monitoring in place

---

**Building your own Hybrid RAG agent?** Start with [PLANNING.md](./PLANNING.md) for the complete adaptation guide, then follow [TASK.md](./TASK.md) for the implementation checklist.
