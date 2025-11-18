# Hybrid RAG Agent - Reference Architecture & Template

> **📚 Production-ready template for building Hybrid RAG agents with Pydantic AI**
>
> **What you get:** Semantic search (pgvector) + Keyword search (TSVector) + Knowledge graphs (Neo4j/Graphiti)
>
> **Adaptable to:** Work items, support tickets, compliance documents, customer data, and more

## What is This?

This is a **complete, working reference implementation** that combines three powerful retrieval strategies into a single AI agent:

| Strategy | Technology | Use Case |
|----------|-----------|----------|
| **Semantic Search** | PostgreSQL + pgvector | Find conceptually similar content |
| **Keyword Search** | PostgreSQL TSVector | Match exact terminology and technical terms |
| **Knowledge Graph** | Neo4j + Graphiti | Understand relationships and temporal context |

## Quick Start (Use This Template)

### Option 1: Quick Test (See It Work)

```bash
# 1. Clone and setup
git clone [your-repo]
cd FullExample
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Set up databases (see Prerequisites below)

# 3. Configure .env (copy from .env.example)

# 4. Ingest sample data
python -m ingestion.ingest docs/

# 5. Try the CLI
python cli.py

# 6. Or start the API
uvicorn agent.api:app --host 0.0.0.0 --port 8058
```

### Option 2: Adapt for Your Domain

See **[PLANNING.md](./PLANNING.md)** for complete adaptation guide.

**Quick adaptation checklist:**
1. ✅ Update `sql/schema.sql` - Replace `documents`/`chunks` with your tables
2. ✅ Update `agent/tools.py` - Rename tools for your domain
3. ✅ Update `agent/prompts.py` - Add domain expertise
4. ✅ Update `agent/models.py` - Your Pydantic models
5. ✅ Update `ingestion/` - How data enters your system

## Prerequisites

### Required Services

| Service | Purpose | Setup Guide |
|---------|---------|-------------|
| **PostgreSQL** | Vector + keyword search | [Neon](https://neon.tech) (easiest) or local Postgres |
| **Neo4j** | Knowledge graph | [Local-AI-Packaged](https://github.com/coleam00/local-ai-packaged) or Neo4j Desktop |
| **LLM API** | Agent intelligence | OpenAI, Ollama, Gemini, or OpenRouter |

### Python Requirements

- Python 3.11 or higher
- See `requirements.txt` for dependencies

## Architecture Overview

```
User Query
    ↓
[Pydantic AI Agent] 
    ↓
┌───────────────────────────────┐
│  Choose Search Strategy       │
│  - Semantic (pgvector)        │
│  - Keyword (TSVector)         │
│  - Graph (Neo4j/Graphiti)     │
│  - Hybrid (all three)         │
└───────────────────────────────┘
    ↓
[Combine & Rank Results]
    ↓
[Stream Response to User]
```

## Configuration

Create `.env` file:

```bash
# Database (PostgreSQL with pgvector)
DATABASE_URL=postgresql://user:pass@host:5432/dbname

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# LLM Configuration
LLM_PROVIDER=openai  # or ollama, openrouter, gemini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-your-key
LLM_CHOICE=gpt-4.1-mini

# Embedding Configuration
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small  # 1536 dimensions

# Ingestion (can use cheaper/faster model)
INGESTION_LLM_CHOICE=gpt-4.1-nano
```

## Usage Examples

### CLI Interface

```bash
python cli.py

# Ask questions:
> What are the main topics in the documents?
> Show me relationships between X and Y
> Find content about [keyword]
```

### API Interface

```bash
# Start server
uvicorn agent.api:app --host 0.0.0.0 --port 8058

# Test endpoint
curl -X POST http://localhost:8058/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What information do you have?"}'
```

### Streaming Example

```python
import httpx

async with httpx.AsyncClient() as client:
    async with client.stream(
        "POST",
        "http://localhost:8058/chat/stream",
        json={"message": "Tell me about X"},
        timeout=60.0
    ) as response:
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                print(line[6:])
```

## Key Features

### 1. Three Search Strategies

**Vector Search** - Semantic similarity using embeddings:
```python
await vector_search(query="API timeout issues", limit=10)
```

**Keyword Search** - Exact term matching:
```python
await hybrid_search(
    query="API timeout", 
    keyword_weight=0.7  # 70% keyword, 30% semantic
)
```

**Graph Search** - Relationship traversal:
```python
await graph_search(query="How are Epic #123 and Feature #456 related?")
```

### 2. Flexible LLM Providers

Switch providers by changing `.env`:

```bash
# OpenAI
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1

# Ollama (local)
LLM_PROVIDER=ollama
LLM_BASE_URL=http://localhost:11434/v1

# OpenRouter
LLM_PROVIDER=openrouter
LLM_BASE_URL=https://openrouter.ai/api/v1

# Gemini
LLM_PROVIDER=gemini
LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta
```

### 3. Comprehensive Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=agent --cov=ingestion --cov-report=html

# All 58 tests passing
```

## Project Structure

```
FullExample/
├── agent/                  # Core agent system
│   ├── agent.py           # Pydantic AI agent
│   ├── tools.py           # ⚠️ CUSTOMIZE for your domain
│   ├── prompts.py         # ⚠️ CUSTOMIZE system prompt
│   ├── models.py          # ⚠️ CUSTOMIZE Pydantic models
│   ├── db_utils.py        # PostgreSQL utilities
│   ├── graph_utils.py     # Neo4j/Graphiti utilities
│   ├── api.py             # FastAPI endpoints
│   └── providers.py       # LLM provider abstraction
├── ingestion/             # Data ingestion pipeline
│   ├── ingest.py          # Main ingestion script
│   ├── chunker.py         # Semantic chunking
│   ├── embedder.py        # Embedding generation
│   └── graph_builder.py   # Knowledge graph construction
├── sql/
│   └── schema.sql         # ⚠️ ADAPT for your schema
├── tests/                 # Comprehensive test suite
├── cli.py                 # Interactive CLI
├── PLANNING.md            # Architecture guide
└── requirements.txt       # Python dependencies
```

## Domain Adaptation Examples

### Example 1: Work Items (ADO, Jira, etc.)

**Schema:**
```sql
work_items (id, ado_id, title, description, work_item_type, embedding, search_vector)
```

**Tools:**
- `search_work_items` - Semantic search
- `search_work_item_graph` - Epic → Feature relationships
- `hybrid_work_item_search` - Semantic + keyword

**System Prompt:** "You are an AI PM assistant managing Azure DevOps work items..."

### Example 2: SOC2 Compliance

**Schema:**
```sql
policies (id, policy_id, title, content, category, embedding, search_vector)
controls (id, control_id, policy_id, description, evidence)
```

**Tools:**
- `search_policies` - Find relevant policies
- `search_controls` - Find controls by policy
- `find_evidence` - Locate supporting evidence

**System Prompt:** "You are a SOC2 compliance assistant with access to security policies..."

### Example 3: Customer Support

**Schema:**
```sql
cases (id, case_id, subject, description, status, embedding, search_vector)
resolutions (id, case_id, solution, applied_at)
```

**Tools:**
- `search_similar_cases` - Find similar past issues
- `get_case_resolution_path` - Show how issue was resolved
- `search_solutions` - Keyword search for technical terms

**System Prompt:** "You are a support assistant with access to historical cases..."

## Common Customization Patterns

### 1. Change Table/Column Names

**In `agent/db_utils.py`:**
```python
# Before (example):
results = await conn.fetch("SELECT * FROM chunks WHERE...")

# After (your domain):
results = await conn.fetch("SELECT * FROM work_items WHERE...")
```

### 2. Add Domain-Specific Fields

**In `agent/models.py`:**
```python
class WorkItemMatch(BaseModel):
    ado_id: int
    title: str
    work_item_type: str  # Your domain field
    similarity_score: float
```

### 3. Customize Search Weighting

**In `agent/tools.py`:**
```python
# For technical content (favor keywords)
await hybrid_search(query, keyword_weight=0.7)

# For conceptual search (favor semantic)
await hybrid_search(query, keyword_weight=0.3)
```

## Troubleshooting

### Issue: "relation 'documents' does not exist"
**Solution:** Run `sql/schema.sql` to create tables

### Issue: "Neo4j connection failed"
**Solution:** Check Neo4j is running (`bolt://localhost:7687`)

### Issue: "Embedding dimension mismatch"
**Solution:** Update `sql/schema.sql` vector dimensions to match your model (1536 for OpenAI, 768 for Ollama nomic-embed-text)

### Issue: "Agent not calling tools"
**Solution:** Check `agent/prompts.py` - System prompt guides tool selection

## Performance Optimization

- **Connection pooling:** AsyncPG handles this automatically
- **Embedding caching:** Embedder class caches repeated queries
- **Batch processing:** Ingestion processes in batches
- **Index optimization:** IVFFlat for vectors, GIN for text

## Testing

```bash
# All tests
pytest tests/ -v

# Specific component
pytest tests/agent/ -v
pytest tests/ingestion/ -v

# With coverage
pytest --cov=agent --cov=ingestion --cov-report=term-missing

# Expected: 58/58 tests passing
```

## Documentation

- **[PLANNING.md](./PLANNING.md)** - Complete architecture guide and adaptation instructions
- **[TASK.md](./TASK.md)** - Development checklist template
- **[CLAUDE.md](./CLAUDE.md)** - Development rules and best practices

## Built With

- **[Pydantic AI](https://ai.pydantic.dev/)** - Agent framework
- **[Graphiti](https://github.com/getzep/graphiti)** - Knowledge graph framework
- **[pgvector](https://github.com/pgvector/pgvector)** - Vector similarity search
- **[Neo4j](https://neo4j.com/)** - Graph database
- **[FastAPI](https://fastapi.tiangolo.com/)** - API framework

## License

MIT License - Use freely for your projects

## Need Help?

1. Check [PLANNING.md](./PLANNING.md) for detailed adaptation guide
2. See `tests/` for usage examples
3. Review `agent/tools.py` for tool patterns
4. Look at `sql/schema.sql` for database design

---

**Ready to adapt this for your domain?** Start with [PLANNING.md](./PLANNING.md)!
