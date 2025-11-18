# Hybrid RAG Agent - Reference Architecture

> **📚 This is a production-ready reference implementation for building Hybrid RAG agents**
>
> **Purpose:** Demonstrates best practices for combining semantic search (pgvector), keyword search (TSVector), and knowledge graphs (Neo4j/Graphiti)
>
> **Adaptable to any domain:** Work items, support tickets, customer data, compliance documents, etc.

## What is Hybrid RAG?

Hybrid RAG combines three powerful retrieval strategies to provide comprehensive, context-aware responses:

1. **Semantic Search (pgvector)** - Finds conceptually similar content using vector embeddings
2. **Keyword Search (TSVector)** - Matches exact terminology and technical terms  
3. **Knowledge Graph (Neo4j/Graphiti)** - Understands relationships and temporal context

This architecture enables agents to:
- Answer relationship queries ("What Features belong to Epic X?")
- Handle keyword-specific searches ("Find bugs mentioning 'API timeout'")
- Provide temporal context ("What changed last sprint?")
- Combine all three for comprehensive results

## How to Adapt This Template

### Step 1: Define Your Domain

**Current Example:** Documents with chunks (generic content retrieval)

**Your Domain Examples:**
- **Work Items:** Epic → Feature → User Story → Bug
- **Support:** Case → Thread → Response → Resolution
- **Compliance:** Policy → Control → Evidence → Audit
- **Customer:** Account → Contact → Interaction → Note

### Step 2: Database Schema Adaptation

**Current Schema (`sql/schema.sql`):**
```sql
documents (id, title, source, content, metadata)
chunks (id, document_id, content, embedding, chunk_index)
```

**Your Schema Example (Work Items):**
```sql
work_items (id, ado_id, title, description, work_item_type, embedding, search_vector)
relationships (source_id, target_id, relationship_type, created_at)
```

### Step 3: Agent Tools Adaptation

**Current Tools:** `vector_search`, `graph_search`, `hybrid_search`, `get_document`

**Your Tools Example (Work Items):**
- `search_work_items` - Semantic search for similar work items
- `search_work_item_graph` - Find Epic → Feature relationships
- `hybrid_work_item_search` - Combine semantic + keyword
- `get_work_item_timeline` - Temporal history

### Step 4: System Prompt Customization

**Current:** Generic knowledge retrieval assistant

**Your Domain:** Add domain expertise (ADO processes, SOC2 compliance, customer service patterns)

### Step 5: Knowledge Graph Design

**Current:** Generic entity/fact model

**Your Domain:** Define your entity relationships (Epic CONTAINS Feature, Case RESOLVES_TO Solution)

## Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    API Layer                             │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │   FastAPI       │        │   Streaming SSE    │     │
│  │   Endpoints     │        │   Responses        │     │
│  └────────┬────────┘        └────────────────────┘     │
│           │                                              │
├───────────┴──────────────────────────────────────────────┤
│                    Agent Layer                           │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │  Pydantic AI    │        │   Agent Tools      │     │
│  │    Agent        │◄──────►│  - Vector Search   │     │
│  └────────┬────────┘        │  - Graph Search    │     │
│           │                 │  - Doc Retrieval   │     │
│           │                 └────────────────────┘     │
├───────────┴──────────────────────────────────────────────┤
│                  Storage Layer                           │
│  ┌─────────────────┐        ┌────────────────────┐     │
│  │   PostgreSQL    │        │      Neo4j         │     │
│  │   + pgvector    │        │   (via Graphiti)   │     │
│  └─────────────────┘        └────────────────────┘     │
└─────────────────────────────────────────────────────────┘
```

## Key Files to Customize

### Critical Files for Domain Adaptation

| File | What to Change | Why |
|------|---------------|-----|
| `sql/schema.sql` | Replace `documents`/`chunks` with your tables | Defines your data structure |
| `agent/tools.py` | Rename and adapt search tools | Domain-specific retrieval logic |
| `agent/prompts.py` | Add domain expertise to system prompt | Guides agent behavior |
| `agent/models.py` | Update Pydantic models for your data | Type safety for your domain |
| `agent/db_utils.py` | Adapt SQL function calls | Match your schema |
| `agent/graph_utils.py` | Define your relationship types | Knowledge graph structure |
| `ingestion/*.py` | Adapt for your data sources | How data enters the system |

### Files That Usually Don't Need Changes

- `agent/agent.py` - Core Pydantic AI setup (generic)
- `agent/providers.py` - LLM provider abstraction (reusable)
- `agent/api.py` - FastAPI endpoints (generic REST/SSE patterns)
- `tests/conftest.py` - Test fixtures (adapt mocks for your models)

## Core Components

### 1. Agent System (`/agent`) - **Adapt these for your domain**
- **agent.py**: Main Pydantic AI agent (usually no changes needed)
- **tools.py**: ⚠️ **CUSTOMIZE** - Rename tools for your domain
- **prompts.py**: ⚠️ **CUSTOMIZE** - Add domain expertise
- **api.py**: FastAPI endpoints (usually no changes needed)
- **db_utils.py**: ⚠️ **ADAPT** - Update SQL function names
- **graph_utils.py**: ⚠️ **ADAPT** - Define relationship types
- **models.py**: ⚠️ **CUSTOMIZE** - Your domain Pydantic models
- **providers.py**: Flexible LLM provider (reusable as-is)

### 2. Ingestion System (`/ingestion`)
- **ingest.py**: Main ingestion script to process markdown files
- **chunker.py**: Semantic chunking implementation
- **embedder.py**: Document embedding generation
- **graph_builder.py**: Knowledge graph construction from documents
- **cleaner.py**: Database cleanup utilities

### 3. Database Schema (`/sql`)
- **schema.sql**: PostgreSQL schema with pgvector
- **migrations/**: Database migration scripts

### 4. Tests (`/tests`)
- Comprehensive unit and integration tests
- Mocked external dependencies
- Test fixtures and utilities

### 5. CLI Interface (`/cli.py`)
- Interactive command-line interface for the agent
- Real-time streaming with Server-Sent Events
- Tool usage visibility showing agent reasoning
- Session management and conversation context

## Technical Stack

### Core Technologies
- **Python 3.11+**: Primary language
- **Pydantic AI**: Agent framework
- **FastAPI**: API framework
- **PostgreSQL + pgvector**: Vector database
- **Neo4j + Graphiti**: Knowledge graph
- **Flexible LLM Providers**: OpenAI, Ollama, OpenRouter, Gemini

### Key Libraries
- **asyncpg**: PostgreSQL async driver
- **httpx**: Async HTTP client
- **python-dotenv**: Environment management
- **pytest + pytest-asyncio**: Testing
- **black + ruff**: Code formatting/linting

## Design Principles

### 1. Modularity
- Clear separation of concerns
- Reusable components
- Clean dependency injection

### 2. Type Safety
- Comprehensive type hints
- Pydantic models for validation
- Dataclasses for dependencies

### 3. Async-First
- All database operations async
- Concurrent processing where applicable
- Proper resource management

### 4. Error Handling
- Graceful degradation
- Comprehensive logging
- User-friendly error messages

### 5. Testing
- Unit tests for all components
- Integration tests for workflows
- Mocked external dependencies

## Key Features

### 1. Hybrid Search
- Vector similarity search for semantic queries
- Knowledge graph traversal for relationship queries
- Combined results with intelligent ranking

### 2. Document Management
- Semantic chunking for optimal retrieval
- Metadata preservation
- Full document retrieval capability

### 3. Knowledge Graph
- Entity and relationship extraction
- Temporal data handling
- Graph-based reasoning

### 4. API Capabilities
- Streaming responses (SSE)
- Session management
- File attachment support

### 5. Flexible Provider System
- Multiple LLM providers (OpenAI, Ollama, OpenRouter, Gemini)
- Environment-based provider switching
- Separate models for different tasks (chat vs ingestion)
- OpenAI-compatible API interface
- Graphiti with custom OpenAI-compatible clients (OpenAIClient, OpenAIEmbedder)

### 6. Agent Transparency
- Tool usage tracking and display in API responses
- CLI with real-time tool visibility
- Configurable agent behavior via system prompt
- Clear reasoning process exposure

## Implementation Strategy

### Phase 1: Foundation
1. Set up project structure
2. Configure PostgreSQL and Neo4j
3. Implement database utilities
4. Create base models

### Phase 2: Core Agent
1. Build Pydantic AI agent
2. Implement RAG tools
3. Implement knowledge graph tools
4. Create prompts and configurations

### Phase 3: API Layer
1. Set up FastAPI application
2. Implement streaming endpoints
3. Add error handling
4. Create health checks

### Phase 4: Ingestion System
1. Build semantic chunker
2. Implement document processor
3. Create knowledge graph builder
4. Add cleanup utilities

### Phase 5: Testing & Documentation
1. Write comprehensive tests
2. Create detailed README
3. Generate API documentation
4. Add usage examples

## Configuration

### Environment Variables
```bash
# Database
DATABASE_URL=postgresql://user:password@localhost:5432/agentic_rag
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# LLM Configuration  
LLM_PROVIDER=openai  # openai, ollama, openrouter, gemini
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_CHOICE=gpt-4.1-mini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
INGESTION_LLM_CHOICE=gpt-4.1-nano

# Application
APP_ENV=development
LOG_LEVEL=INFO
APP_PORT=8058
```

### Database Schema
- **documents**: Store document metadata
- **chunks**: Store document chunks with embeddings
- **sessions**: Manage conversation sessions
- **messages**: Store conversation history

## Security Considerations
- Environment-based configuration
- No hardcoded credentials
- Input validation at all layers
- SQL injection prevention
- Rate limiting on API

## Performance Optimizations
- Connection pooling for databases
- Embedding caching
- Batch processing for ingestion
- Indexed vector searches
- Async operations throughout

## Monitoring & Logging
- Structured logging with context
- Performance metrics
- Error tracking
- Usage analytics

## Future Enhancements
- ✅ ~~Multi-model support~~ (Completed - Flexible provider system)
- Advanced reranking algorithms
- Real-time document updates
- GraphQL API option
- Web UI for exploration
- Additional LLM providers (Anthropic Claude direct, Cohere, etc.)
- Embedding provider diversity (Voyage, Cohere embeddings)
- Model performance optimization and caching