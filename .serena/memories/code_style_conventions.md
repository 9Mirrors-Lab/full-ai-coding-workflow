# Code Style and Conventions

## File Organization
- **Never create files > 500 lines** - Split into modules when approaching limit
- **Modular structure** by responsibility:
  - `agent.py` - Main agent definition and execution logic
  - `tools.py` - Tool functions used by the agent
  - `models.py` - Pydantic output models and dependency classes
  - `dependencies.py` - Context dependencies and external service integrations
  - `settings.py` - Configuration management with pydantic-settings

## Naming Conventions
- **snake_case** for functions, variables, and file names
- **PascalCase** for classes
- **UPPER_CASE** for constants
- Agent files: `agent.py`, `research_agent.py`, etc.
- Tool files: `tools.py`
- Model files: `models.py`

## Type Hints
- **Always use type hints** - Required for Pydantic AI agents
- Use `typing` module for complex types (Dict, List, Optional, Any)
- Use `dataclasses` for dependency classes
- Use Pydantic models for structured outputs

## Docstrings
- Use triple-quoted strings for all functions, classes, and modules
- Include Args, Returns, and Raises sections where applicable
- Document tool functions comprehensively (used by LLM)

## Imports
- Group imports: standard library, third-party, local
- Use absolute imports from project root
- Import from `pydantic_ai` appropriately

## Async Patterns
- **Consistent async/await** - All agent tools use async
- Use `@pytest.mark.asyncio` for async tests
- Use `asyncio.gather()` for parallel operations

## Error Handling
- Use try/except blocks with specific exceptions
- Log errors with appropriate log levels
- Never expose sensitive information in error messages
- Return structured error responses in tools

## Environment Variables
- **Always use python-dotenv** - Call `load_dotenv()` in settings
- Use `pydantic-settings` BaseSettings for configuration
- Never hardcode API keys or secrets
- Store in `.env` files (never commit)

## Agent Patterns
- Use `@agent.tool` decorator for context-aware tools with `RunContext[DepsType]`
- Use `@agent.tool_plain` decorator for simple tools without context
- Define dependencies as dataclasses
- Use `Agent.override()` for testing with TestModel/FunctionModel

## Code Quality Rules
- **Keep it simple** - Avoid overengineering
- **No backward compatibility** - Remove dead code
- **Avoid complex dependency graphs** - Keep dependencies simple and testable
- **Follow PRP workflow** - Don't skip validation steps