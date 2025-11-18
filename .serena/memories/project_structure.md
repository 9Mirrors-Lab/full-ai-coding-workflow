# Project Structure

## Root Directory
```
full-ai-coding-workflow/
├── .claude/              # Claude Code configuration
│   ├── commands/         # Slash commands for automation
│   ├── agents/           # Subagent configurations
│   └── settings.local.json
├── .devcontainer/        # Dev container setup
├── .serena/              # Serena project metadata
├── FullExample/          # Complete RAG agent implementation
│   ├── agent/            # Agent code (agent.py, tools.py, models.py)
│   ├── ingestion/        # Document ingestion pipeline
│   ├── tests/            # Test suite
│   └── cli.py            # CLI entry point
├── PRPs/                 # PRP framework files
│   ├── examples/         # Reference implementations
│   │   ├── main_agent_reference/  # Main reference agent
│   │   ├── basic_chat_agent/
│   │   ├── rag_pipeline/
│   │   ├── structured_output_agent/
│   │   ├── testing_examples/
│   │   └── tool_enabled_agent/
│   ├── templates/        # PRP templates
│   └── INITIAL.md        # Starting point for new PRPs
├── PromptProgression/    # Prompt evolution examples
├── images/              # Documentation images
├── venv/                # Virtual environment
├── requirements.txt     # Python dependencies
├── CLAUDE.md            # Global rules
├── README.md            # Main documentation
└── copy_template.py     # Template deployment script
```

## Key Directories

### PRPs/examples/
Reference implementations demonstrating:
- Agent patterns and architectures
- Tool integration
- Dependency management
- Testing strategies
- Settings configuration

### FullExample/
Production-ready hybrid RAG agent with:
- Vector search (PGVector)
- Hybrid search (TSVector)
- Knowledge graph (Neo4j/Graphiti)
- Document ingestion pipeline
- Comprehensive test suite

### .claude/
Automation and configuration:
- Slash commands for PRP workflow
- Subagent definitions (prompt engineer, tool integrator, validator)
- Permission settings

## File Naming Patterns
- Agent files: `*_agent.py` or `agent.py`
- Tool files: `tools.py`
- Model files: `models.py`
- Settings: `settings.py`
- Tests: `test_*.py`
- Config: `pytest.ini`, `.env`