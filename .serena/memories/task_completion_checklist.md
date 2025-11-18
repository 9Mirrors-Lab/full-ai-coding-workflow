# Task Completion Checklist

When completing a task, ensure the following:

## Code Implementation
- [ ] Code follows modular structure (agent.py, tools.py, models.py)
- [ ] All files are under 500 lines
- [ ] Type hints are present on all functions
- [ ] Docstrings are comprehensive (especially for tools)
- [ ] Environment variables use python-dotenv pattern
- [ ] No hardcoded secrets or API keys
- [ ] Error handling is implemented properly

## Testing
- [ ] Unit tests created for all functions
- [ ] Integration tests for agent workflows
- [ ] Tests use TestModel/FunctionModel for agent validation
- [ ] Async tests properly marked with @pytest.mark.asyncio
- [ ] Test coverage meets requirements (if specified)

## Validation Gates
- [ ] Syntax and style checks pass (if tools installed)
- [ ] Type checking passes (if mypy configured)
- [ ] All tests pass
- [ ] Integration tests pass
- [ ] Human review completed (if required)

## Documentation
- [ ] Code is properly documented
- [ ] PRP requirements are met
- [ ] Examples follow established patterns
- [ ] README updated if needed

## Security
- [ ] No sensitive data in code
- [ ] Environment variables properly configured
- [ ] Input validation implemented
- [ ] Error messages don't expose sensitive info

## Agent-Specific
- [ ] Agent follows PRP workflow
- [ ] Tools properly decorated (@agent.tool or @agent.tool_plain)
- [ ] Dependencies properly defined as dataclasses
- [ ] System prompts are clear and focused
- [ ] Agent tested with TestModel before real models

## Final Steps
- [ ] Mark tasks complete in task management system
- [ ] Update project status if needed
- [ ] Commit changes with descriptive messages