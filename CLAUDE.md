# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 🎯 Repository Purpose

This repository is a **Pydantic AI Context Engineering Workshop** that teaches the full AI development workflow using the PRP (Product Requirements Prompt) framework. It contains:

1. **Working examples** of Pydantic AI agents (in `FullExample/` and `PRPs/examples/`)
2. **PRP templates and workflows** for building AI agents (`PRPs/templates/`)
3. **Specialized subagents** for agent development phases (`.claude/agents/`)
4. **Custom slash commands** for PRP-based development (`.claude/commands/`)
5. **AI Project Management agent** - Phase 1 complete (`ai_pm_agent/`), Phase 2 in progress (Hybrid RAG)

## 🚀 Essential Commands

### Virtual Environment
```bash
# Activate virtual environment (if exists)
source venv/bin/activate

# If venv doesn't exist, create it
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest tests/ -v --tb=short

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest tests/ -v --cov=ai_pm_agent --cov-report=term-missing
```

### Code Quality
```bash
# Lint code
ruff check . --fix

# Format code
black .

# Type checking (if mypy is configured)
mypy ai_pm_agent/
```

### Running the Agent
```bash
# Example: Run AI PM Agent (if implemented)
python -m ai_pm_agent
```

## 🏗️ Architecture & PRP Workflow

### The PRP Development Pattern

This repository follows a **structured PRP-based workflow** for building Pydantic AI agents:

**For NEW agents (greenfield)**:
1. Create `PRPs/INITIAL-Phase1.md` with requirements
2. Run `/generate-pydantic-ai-prp PRPs/INITIAL-Phase1.md` to generate comprehensive PRP
3. Run `/execute-pydantic-ai-prp PRPs/{generated-prp}.md` to implement

**For EXISTING codebase enhancements (multi-phase)**:
1. Investigate codebase patterns and architecture
2. Research and plan with back-and-forth discussions
3. Generate focused PRP: `/generate-pydantic-ai-prp "create PRP following findings"`
4. Execute PRP: `/execute-pydantic-ai-prp PRPs/{feature}.md`

**For Multi-Phase Development**:
Projects that evolve across phases (e.g., AI PM Agent):
1. **Phase 1:** `INITIAL-Phase1.md` → Generate PRP → Implement → Complete & Test
2. **Phase 2:** `INITIAL-Phase2.md` (references Phase 1 work) → Generate new PRP → Enhance

Example progression:
- Phase 1: Semantic search with pgvector (✅ complete, tested)
- Phase 2: Add Hybrid RAG (Neo4j/Graphiti + TSVector) building on Phase 1

### Agent Creation Workflow (via /execute-pydantic-ai-prp)

When executing a Pydantic AI PRP, the workflow uses **parallel subagent development**:

**Phase 1: Setup**
- Load PRP and create Archon project (if available) or TodoWrite tasks

**Phase 2: Parallel Component Development** (3 subagents run simultaneously)
- **pydantic-ai-prompt-engineer** → Creates `planning/prompts.md`
- **pydantic-ai-tool-integrator** → Creates `planning/tools.md`
- **pydantic-ai-dependency-manager** → Creates `planning/dependencies.md`

**Phase 3: Implementation**
- Main Claude Code implements agent using planning outputs
- Creates: `agent.py`, `tools.py`, `models.py`, `dependencies.py`, `settings.py`, `providers.py`

**Phase 4: Validation**
- **pydantic-ai-validator** creates comprehensive test suite
- Runs all validation gates from PRP

**Phase 5: Delivery**
- Documentation updates and final verification

### Directory Structure Patterns

**Pydantic AI Agent Structure** (see `ai_pm_agent/` and `FullExample/agent/`):
```
agent_name/
├── __init__.py
├── agent.py          # Main agent definition with @agent.tool decorators
├── tools.py          # Tool implementation functions
├── models.py         # Pydantic models for structured outputs
├── dependencies.py   # Dependencies class for RunContext
├── settings.py       # Environment configuration with python-dotenv
├── providers.py      # LLM model provider setup
└── prompts.py        # System prompt definitions
```

**Test Structure** (see `tests/`):
```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── pytest.ini               # Pytest configuration
├── test_models.py           # Pydantic model tests
└── VALIDATION_REPORT.md     # Validation results
```

**PRP Templates** (see `PRPs/templates/`):
```
PRPs/
├── INITIAL-Phase1.md             # Phase 1 requirements (completed)
├── INITIAL-Phase2.md             # Phase 2 requirements (Hybrid RAG)
├── ai-project-management-agent.md # Phase 1 generated PRP
├── templates/
│   └── prp_pydantic_ai_base.md  # Base PRP template
└── examples/                     # Reference implementations
    ├── main_agent_reference/     # **PRIMARY REFERENCE** - Best practices
    ├── basic_chat_agent/
    ├── tool_enabled_agent/
    ├── structured_output_agent/
    └── testing_examples/
```

**Planning Documents** (see `planning/`):
```
planning/
├── tools.md          # Phase 1 tool design decisions
├── prompts.md        # System prompt architecture
└── dependencies.md   # Dependency planning notes
```
**Why:** These provide context for understanding Phase 1 architecture when building Phase 2

## 🧠 Pydantic AI Development Principles

### Core Rules (from project CLAUDE.md)

1. **Always start with INITIAL-Phase{N}.md** for new agents or phases
2. **Follow main_agent_reference patterns** - This is the canonical example
3. **Use python-dotenv for configuration** - Never hardcode API keys
4. **Default to string output** - Only use `result_type` when validation needed
5. **Keep files under 500 lines** - Split into modules when approaching limit
6. **Test with TestModel first** - Before using real model providers
7. **Document phase lineage** - New phases should reference completed work

### Agent Architecture Standards

**Model Configuration** (follow `examples/main_agent_reference/`):
```python
# settings.py - Use python-dotenv
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

class Settings(BaseSettings):
    llm_provider: str = "openai"
    llm_api_key: str
    llm_model: str = "gpt-4"

# providers.py - Provider setup
from pydantic_ai.providers.openai import OpenAIProvider
def get_llm_model():
    settings = load_settings()
    provider = OpenAIProvider(api_key=settings.llm_api_key)
    return OpenAIModel(settings.llm_model, provider=provider)
```

**Tool Registration**:
```python
@agent.tool
async def tool_with_context(ctx: RunContext[AgentDependencies], param: str) -> str:
    """Tool with access to dependencies"""
    return await external_call(ctx.deps.api_key, param)

@agent.tool_plain
def simple_tool(param: str) -> str:
    """Tool without context dependencies"""
    return process(param)
```

**Testing Pattern**:
```python
from pydantic_ai.models.test import TestModel

async def test_agent_behavior():
    test_model = TestModel()
    with agent.override(model=test_model):
        result = await agent.run("test query")
        assert result.output
```

## 📋 Key Gotchas & Anti-Patterns

### DO:
- ✅ Use virtual environment (`venv/`) for all Python commands
- ✅ Read `examples/main_agent_reference/` for best practices
- ✅ Include `load_dotenv()` in settings.py (python-dotenv pattern)
- ✅ Use TestModel for development validation
- ✅ Keep dependency graphs simple
- ✅ Log all agent decisions for transparency

### DON'T:
- ❌ Skip agent testing with TestModel/FunctionModel
- ❌ Hardcode API keys or model strings
- ❌ Use `result_type` unless structured output is specifically needed
- ❌ Create files over 500 lines
- ❌ Mix async/sync patterns inconsistently
- ❌ Create complex dependency graphs

## 🔧 Slash Commands

This repository provides specialized PRP workflow commands:

- `/generate-pydantic-ai-prp PRPs/INITIAL-Phase{N}.md` - Research and create comprehensive PRP from initial requirements
- `/execute-pydantic-ai-prp PRPs/generated.md` - Implement agent using PRP with full workflow (subagents + validation)
- `/primer` - Analyze codebase for existing patterns

**Example Usage:**
```bash
# Phase 1 (new agent)
/generate-pydantic-ai-prp PRPs/INITIAL-Phase1.md

# Phase 2 (enhancement)
/generate-pydantic-ai-prp PRPs/INITIAL-Phase2.md
```

## 🧪 Testing Requirements

All new Pydantic AI agent features require:

1. **Unit tests** in `tests/` directory
2. **TestModel validation** before using real models
3. **Tool parameter validation** with Pydantic schemas
4. **Integration tests** for workflows
5. **Coverage report** showing >80% coverage

Test configuration is in `tests/pytest.ini` with markers: `asyncio`, `unit`, `integration`, `validation`

## 📚 Documentation References

**Internal Examples**:
- `PRPs/examples/main_agent_reference/` - **PRIMARY REFERENCE** for all agent development
- `FullExample/` - **Complete Hybrid RAG implementation** (Phase 2 reference architecture)
- `ai_pm_agent/` - AI PM Agent Phase 1 (complete, tested) | Phase 2 (in progress)

**External Documentation** (use Archon MCP for RAG queries):
- Pydantic AI: https://ai.pydantic.dev/
- Agent patterns: https://ai.pydantic.dev/agents/
- Tool integration: https://ai.pydantic.dev/tools/
- Testing: https://ai.pydantic.dev/testing/

## 🎓 Context Engineering Integration

This repository demonstrates the "Four Pillars of Context Engineering":

1. **RAG** - Use Archon MCP server for Pydantic AI documentation
2. **Task Management** - Archon integration or TodoWrite for tracking
3. **Memory** - PRP files maintain context across development phases
4. **Prompt Engineering** - PRP framework for structured AI development

When working in this codebase, leverage these pillars through slash commands and MCP tools.
