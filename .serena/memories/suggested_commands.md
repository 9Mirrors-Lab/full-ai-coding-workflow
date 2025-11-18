# Suggested Commands

## Virtual Environment
```bash
# Activate virtual environment
source venv/bin/activate

# Deactivate virtual environment
deactivate

# Install dependencies
pip install -r requirements.txt
```

## Testing
```bash
# Run all tests
pytest

# Run tests with verbose output
pytest -v

# Run specific test file
pytest tests/test_agent.py

# Run tests with coverage
pytest --cov=src/ --cov-report=term-missing

# Run async tests
pytest tests/ -v --asyncio-mode=auto
```

## Code Quality (if tools are installed)
```bash
# Format code (if black is installed)
black src/

# Check formatting
black src/ --check

# Lint code (if ruff is installed)
ruff check src/

# Type check (if mypy is installed)
mypy src/ --strict
```

## Development Workflow
```bash
# Generate PRP from INITIAL.md
/generate-pydantic-ai-prp PRPs/INITIAL.md

# Execute PRP to build agent
/execute-pydantic-ai-prp PRPs/generated_prp.md

# Run example agent (if implemented)
python FullExample/cli.py
```

## System Utilities (Darwin/macOS)
```bash
# List files
ls -la

# Change directory
cd /path/to/directory

# Search files
find . -name "*.py"

# Search content
grep -r "pattern" .

# Git operations
git status
git add .
git commit -m "message"
git push
```

## Python
```bash
# Check Python version
python3 --version

# Run Python script
python3 script.py

# Interactive Python
python3
```

## Project-Specific
```bash
# Copy template to new project
python copy_template.py /path/to/new-project

# Navigate to project root
cd /Volumes/Fulcrum/Develop/full-ai-coding-workflow
```