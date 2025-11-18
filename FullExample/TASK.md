# Hybrid RAG Development Checklist

> **📋 Use this as a template for building your own Hybrid RAG agent**
>
> This is a comprehensive checklist for implementing a Hybrid RAG system. Copy and adapt for your specific domain.

## Phase 1: Foundation & Planning

### 1.1 Define Your Domain
- [ ] Identify your primary entities (e.g., work_items, tickets, policies)
- [ ] Map entity relationships (e.g., Epic → Feature → Story)
- [ ] Define relationship types (CONTAINS, RELATES_TO, RESOLVES, etc.)
- [ ] Identify required metadata fields
- [ ] Determine temporal requirements (do you need history/timeline?)

### 1.2 Database Design
- [ ] Design PostgreSQL schema for your entities
- [ ] Add vector column for semantic search (pgvector)
- [ ] Add text search column for keyword search (tsvector)
- [ ] Create indexes (IVFFlat for vectors, GIN for text)
- [ ] Design SQL functions for search operations
- [ ] Plan for metadata storage (JSONB recommended)

### 1.3 Knowledge Graph Design
- [ ] Define Neo4j node types (map to your entities)
- [ ] Define relationship types between nodes
- [ ] Plan for temporal relationships (Graphiti supports this)
- [ ] Design episode structure (logical grouping of changes)
- [ ] Consider bi-temporal model (valid_at vs recorded_at)

### 1.4 Environment Setup
- [ ] Set up Python virtual environment
- [ ] Install dependencies (`requirements.txt`)
- [ ] Configure PostgreSQL with pgvector extension
- [ ] Set up Neo4j database
- [ ] Create `.env` file with all credentials
- [ ] Test database connections

## Phase 2: Database Implementation

### 2.1 PostgreSQL Setup
- [ ] Execute schema creation SQL
- [ ] Verify pgvector extension is enabled
- [ ] Test vector operations (cosine similarity)
- [ ] Create text search configuration
- [ ] Test full-text search
- [ ] Set up connection pooling (AsyncPG)

### 2.2 Neo4j Setup
- [ ] Install Neo4j (Desktop or Docker)
- [ ] Create database and set authentication
- [ ] Test connection from Python
- [ ] Install Graphiti and configure
- [ ] Verify graph operations work

### 2.3 SQL Functions
- [ ] Implement `match_[entities]` for semantic search
- [ ] Implement `hybrid_[entities]_search` for combined search
- [ ] Add specialized search functions (by date, type, etc.)
- [ ] Test all functions with sample data
- [ ] Optimize query performance

## Phase 3: Ingestion Pipeline

### 3.1 Data Source Integration
- [ ] Identify data sources (files, API, database)
- [ ] Design ingestion flow
- [ ] Implement data extraction
- [ ] Handle different formats (JSON, CSV, Markdown, etc.)
- [ ] Add error handling and logging

### 3.2 Chunking Strategy
- [ ] Implement semantic chunking (or use fixed size)
- [ ] Test chunk sizes (balance context vs specificity)
- [ ] Handle overlapping chunks if needed
- [ ] Preserve metadata through chunking
- [ ] Test with real data samples

### 3.3 Embedding Generation
- [ ] Choose embedding model (OpenAI, Ollama, etc.)
- [ ] Implement embedding generation
- [ ] Add caching for repeated queries
- [ ] Handle rate limits and retries
- [ ] Batch processing for efficiency

### 3.4 Knowledge Graph Construction
- [ ] Extract entities from your data
- [ ] Identify relationships between entities
- [ ] Create episodes for logical grouping
- [ ] Add temporal information
- [ ] Build graph incrementally
- [ ] Handle updates and deduplication

### 3.5 Testing Ingestion
- [ ] Test with small sample data
- [ ] Verify vector embeddings are correct
- [ ] Verify text search vectors are populated
- [ ] Check knowledge graph structure
- [ ] Test incremental updates
- [ ] Measure ingestion performance

## Phase 4: Agent Development

### 4.1 Agent Core
- [ ] Set up Pydantic AI agent
- [ ] Configure LLM provider (OpenAI, Ollama, etc.)
- [ ] Define agent dependencies (database clients, etc.)
- [ ] Create basic agent structure
- [ ] Test agent initialization

### 4.2 Pydantic Models
- [ ] Define models for your entities
- [ ] Add models for search results
- [ ] Create models for tool responses
- [ ] Add validation rules
- [ ] Test model serialization

### 4.3 System Prompt
- [ ] Adapt `SYSTEM_PROMPT` template for your domain
- [ ] Add domain-specific expertise
- [ ] Define tool selection strategy
- [ ] Add response guidelines
- [ ] Include example interactions
- [ ] Test prompt effectiveness

### 4.4 Tool Implementation
- [ ] Implement semantic search tool
- [ ] Implement keyword search tool
- [ ] Implement knowledge graph search tool
- [ ] Implement hybrid search tool
- [ ] Add entity retrieval tools (get by ID, etc.)
- [ ] Add specialized tools (timeline, relationships, etc.)
- [ ] Document all tools with clear descriptions
- [ ] Test each tool independently

### 4.5 Database Utilities
- [ ] Implement connection management (AsyncPG)
- [ ] Add query helper functions
- [ ] Implement result parsing
- [ ] Add error handling
- [ ] Test connection pooling

### 4.6 Graph Utilities
- [ ] Implement Graphiti client setup
- [ ] Add graph query functions
- [ ] Implement relationship traversal
- [ ] Add temporal query support
- [ ] Handle graph updates

## Phase 5: API Development

### 5.1 FastAPI Setup
- [ ] Create FastAPI app
- [ ] Add CORS configuration
- [ ] Set up request/response models
- [ ] Add health check endpoint
- [ ] Test basic API functionality

### 5.2 Chat Endpoints
- [ ] Implement `/chat` endpoint (simple response)
- [ ] Implement `/chat/stream` endpoint (SSE streaming)
- [ ] Add request validation
- [ ] Add response formatting
- [ ] Test both endpoints

### 5.3 Tool Usage Tracking
- [ ] Implement tool call extraction
- [ ] Add tool usage to responses
- [ ] Log tool performance
- [ ] Add debugging information

### 5.4 Error Handling
- [ ] Add global exception handler
- [ ] Handle database errors gracefully
- [ ] Handle LLM errors (rate limits, etc.)
- [ ] Return user-friendly error messages
- [ ] Log errors for debugging

## Phase 6: CLI Development

### 6.1 Interactive CLI
- [ ] Implement basic CLI interface
- [ ] Add conversation history
- [ ] Add commands (/help, /clear, /exit)
- [ ] Show tool usage
- [ ] Add colored output for readability

### 6.2 CLI Features
- [ ] Support multi-line input
- [ ] Show streaming responses
- [ ] Add command history
- [ ] Implement special commands
- [ ] Test user experience

## Phase 7: Testing

### 7.1 Unit Tests
- [ ] Test Pydantic models
- [ ] Test database utilities
- [ ] Test graph utilities
- [ ] Test embedding generation
- [ ] Test chunking logic
- [ ] Achieve >80% code coverage

### 7.2 Integration Tests
- [ ] Test ingestion pipeline end-to-end
- [ ] Test agent with TestModel
- [ ] Test API endpoints
- [ ] Test CLI functionality
- [ ] Test with real data samples

### 7.3 Agent Tests
- [ ] Test tool selection
- [ ] Test each tool independently
- [ ] Test tool combinations
- [ ] Test error handling
- [ ] Test with different LLM providers

### 7.4 Performance Tests
- [ ] Measure search query latency
- [ ] Test with large datasets
- [ ] Profile embedding generation
- [ ] Test concurrent requests
- [ ] Optimize slow operations

## Phase 8: Documentation

### 8.1 Code Documentation
- [ ] Add docstrings to all functions
- [ ] Document tool parameters
- [ ] Add inline comments for complex logic
- [ ] Document configuration options
- [ ] Add type hints everywhere

### 8.2 User Documentation
- [ ] Write README with quick start
- [ ] Document installation steps
- [ ] Add usage examples
- [ ] Document API endpoints
- [ ] Add troubleshooting guide
- [ ] Create adaptation guide for new domains

### 8.3 Architecture Documentation
- [ ] Document system architecture
- [ ] Explain tool selection strategy
- [ ] Document database schema
- [ ] Explain knowledge graph structure
- [ ] Add performance considerations

## Phase 9: Deployment Preparation

### 9.1 Configuration Management
- [ ] Use environment variables for all config
- [ ] Create `.env.example` file
- [ ] Document all configuration options
- [ ] Add validation for required settings
- [ ] Support multiple environments (dev, prod)

### 9.2 Security
- [ ] Never commit API keys or passwords
- [ ] Use secure password storage
- [ ] Add input validation
- [ ] Sanitize user inputs
- [ ] Rate limit API endpoints
- [ ] Add authentication if needed

### 9.3 Monitoring
- [ ] Add logging throughout
- [ ] Log tool usage statistics
- [ ] Monitor database performance
- [ ] Track LLM API usage
- [ ] Set up error alerting

### 9.4 Optimization
- [ ] Optimize slow queries
- [ ] Tune vector index parameters
- [ ] Implement caching where beneficial
- [ ] Batch operations when possible
- [ ] Use connection pooling

## Phase 10: Production Deployment

### 10.1 Infrastructure
- [ ] Choose hosting platform
- [ ] Set up PostgreSQL (managed or self-hosted)
- [ ] Set up Neo4j (managed or self-hosted)
- [ ] Configure networking and security
- [ ] Set up SSL/TLS if exposing API

### 10.2 Deployment
- [ ] Create deployment scripts
- [ ] Set up CI/CD pipeline
- [ ] Configure production environment variables
- [ ] Test deployment process
- [ ] Create rollback plan

### 10.3 Initial Data Load
- [ ] Prepare production data
- [ ] Run ingestion pipeline
- [ ] Verify data integrity
- [ ] Test searches on production data
- [ ] Validate knowledge graph

### 10.4 Launch
- [ ] Deploy to production
- [ ] Monitor for errors
- [ ] Test all endpoints
- [ ] Verify performance
- [ ] Document any issues

## Phase 11: Maintenance

### 11.1 Monitoring
- [ ] Set up health checks
- [ ] Monitor database performance
- [ ] Track API usage
- [ ] Monitor LLM costs
- [ ] Set up alerts for issues

### 11.2 Updates
- [ ] Plan for incremental data updates
- [ ] Implement data refresh strategy
- [ ] Handle schema migrations
- [ ] Update dependencies regularly
- [ ] Test updates in staging first

### 11.3 Improvements
- [ ] Gather user feedback
- [ ] Analyze tool usage patterns
- [ ] Optimize based on real usage
- [ ] Add new tools as needed
- [ ] Improve system prompt based on interactions

---

## Domain-Specific Checklists

### For Work Items (ADO/Jira)
- [ ] Integrate with ADO/Jira API
- [ ] Handle work item state changes
- [ ] Track parent-child relationships
- [ ] Implement sprint/iteration filtering
- [ ] Add priority and type filtering

### For SOC2 Compliance
- [ ] Map policies to frameworks
- [ ] Link controls to evidence
- [ ] Track audit periods
- [ ] Implement compliance scoring
- [ ] Generate audit reports

### For Customer Support
- [ ] Integrate with ticketing system
- [ ] Track resolution patterns
- [ ] Implement SLA monitoring
- [ ] Add sentiment analysis
- [ ] Generate support metrics

### For Your Domain
- [ ] [Add your domain-specific tasks here]
- [ ] [Map your unique requirements]
- [ ] [Define custom tools needed]
- [ ] [Plan domain-specific features]

---

## Quick Reference: Common Pitfalls

- ❌ **Wrong embedding dimensions** - Match your model (1536 for OpenAI, 768 for Ollama)
- ❌ **Missing indexes** - Vector and text search need proper indexes
- ❌ **Oversized chunks** - Too large = poor retrieval, too small = lost context
- ❌ **Vague system prompt** - Be specific about when to use each tool
- ❌ **No connection pooling** - AsyncPG handles this, but verify it's configured
- ❌ **Ignoring temporal data** - Knowledge graphs are powerful for time-based queries
- ❌ **Not testing with TestModel** - Always test agent logic before using real LLMs
- ❌ **Hardcoded values** - Use environment variables for all configuration

---

**Ready to build?** Start with Phase 1 and work through systematically. Each phase builds on the previous one.
