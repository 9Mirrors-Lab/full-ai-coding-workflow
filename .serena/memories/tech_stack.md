# Tech Stack

## Core Framework
- **Pydantic AI** (>=0.0.14) - Main AI agent framework
- **Pydantic** (>=2.0.0) - Data validation and settings
- **pydantic-settings** (>=2.0.0) - Configuration management

## LLM Providers
- **OpenAI** (>=1.0.0) - Primary LLM provider support
- Support for multiple providers (Anthropic, Gemini) via model-agnostic design

## Database & Storage
- **PostgreSQL** with **PGVector** - Vector similarity search
- **PostgreSQL** with **TSVector** - Full-text keyword search
- **asyncpg** (>=0.29.0) - Async PostgreSQL driver
- **Neo4j** with **Graphiti** - Knowledge graph (optional, may be private package)

## Web Framework
- **FastAPI** (>=0.115.0) - API framework
- **uvicorn** (>=0.30.0) - ASGI server

## HTTP & Networking
- **aiohttp** (>=3.9.0) - Async HTTP client

## Configuration
- **python-dotenv** (>=1.0.0) - Environment variable management

## Testing
- **pytest** (>=8.0.0) - Testing framework
- **pytest-asyncio** (>=0.23.0) - Async test support

## Python Version
- Python 3.9+ (project uses 3.9 in venv, system has 3.13.5 available)

## Development Environment
- Virtual environment (`venv/`) for dependency isolation
- Darwin (macOS) development system