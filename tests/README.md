# AI Project Management Agent - Test Suite

Comprehensive test suite for the AI Project Management Agent built with Pydantic AI.

## Quick Start

```bash
# Install dependencies
pip install -r ../requirements.txt

# Run all tests
pytest -v

# Run with coverage
pytest -v --cov=ai_pm_agent --cov-report=term-missing
```

## Test Structure

```
tests/
├── README.md                 # This file
├── VALIDATION_REPORT.md      # Detailed validation results
├── conftest.py              # Test fixtures and configuration
├── pytest.ini               # Pytest configuration
├── test_agent.py           # Agent-level tests (49 tests)
├── test_tools.py           # Tool-specific tests (58 tests)
└── test_models.py          # Pydantic model tests (42 tests)
```

**Total:** 149 test cases covering all agent functionality

## Test Categories

### 1. Agent Tests (`test_agent.py`)

Tests the complete agent including:
- Agent instantiation and configuration
- Tool registration (all 8 tools)
- Dependency injection
- TestModel integration
- Error handling
- Integration workflows

**Run:** `pytest tests/test_agent.py -v`

### 2. Tool Tests (`test_tools.py`)

Tests all 8 tools individually:
- `classify_input` - Input classification with confidence thresholds
- `match_ado_work_items` - Semantic similarity search
- `generate_work_item` - Template-based work item generation
- `tag_artifact` - Artifact detection and tagging
- `recommend_sprint_placement` - Sprint recommendations
- `create_action_item` - Action item creation
- `update_launch_readiness` - Launch readiness updates
- `log_agent_decision` - Decision logging

**Run:** `pytest tests/test_tools.py -v`

### 3. Model Tests (`test_models.py`)

Tests all 7 Pydantic models:
- `ClassificationResult`
- `WorkItemMatch`
- `GeneratedWorkItem`
- `ArtifactTagResult`
- `SprintRecommendation`
- `ActionItemResult`
- `LaunchReadinessUpdate`

**Run:** `pytest tests/test_models.py -v`

## Running Tests

### All Tests

```bash
# Verbose output
pytest tests/ -v

# Quiet output
pytest tests/ -q

# Stop on first failure
pytest tests/ -x

# Run last failed tests
pytest tests/ --lf
```

### Specific Tests

```bash
# Single test file
pytest tests/test_agent.py

# Single test class
pytest tests/test_tools.py::TestClassifyInputTool

# Single test function
pytest tests/test_agent.py::TestAgentValidationGates::test_agent_never_creates_ado_items_directly

# Tests matching pattern
pytest tests/ -k "classification"
```

### With Coverage

```bash
# Terminal coverage report
pytest tests/ --cov=ai_pm_agent --cov-report=term-missing

# HTML coverage report
pytest tests/ --cov=ai_pm_agent --cov-report=html
open htmlcov/index.html
```

## Test Fixtures

Common fixtures available in all tests (from `conftest.py`):

### Service Mocks

- `mock_supabase_client` - Mocked Supabase client
- `mock_openai_client` - Mocked OpenAI client

### Dependencies

- `test_dependencies` - Fully configured AgentDependencies
- `test_model` - TestModel instance
- `test_agent` - Agent with TestModel override

### Test Data

- `sample_input_texts` - Various input texts for classification
- `sample_work_items` - Sample work item data
- `sample_classification_result` - Example classification
- `sample_work_item_matches` - Example matches
- `temp_artifact_file` - Temporary test file
- `confidence_test_cases` - Confidence threshold test data

## Validation Gates

The test suite validates all requirements from the PRP:

### Critical Validations

1. ✅ **ADO Read-Only Constraint** - Agent never creates ADO items directly
2. ✅ **All 8 Tools Registered** - Complete tool set
3. ✅ **Confidence Thresholds** - <70%, 70-85%, >85% enforcement
4. ✅ **Template Compliance** - Epic, Feature, User Story templates
5. ✅ **Dependency Injection** - Type-safe dataclass dependencies
6. ✅ **Structured Outputs** - Pydantic model validation
7. ✅ **Error Handling** - Graceful degradation
8. ✅ **Performance** - Classification <1s, Search <2s
9. ✅ **Security** - API keys protected, input validated
10. ✅ **TestModel Integration** - No real API calls in tests

See `VALIDATION_REPORT.md` for detailed validation results.

## Writing New Tests

### Test Structure

```python
import pytest
from ai_pm_agent.tools import your_tool
from ai_pm_agent.models import YourModel

class TestYourFeature:
    """Test your feature."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self, mock_supabase_client):
        """Test basic functionality."""
        result = await your_tool(mock_supabase_client)

        assert isinstance(result, YourModel)
        assert result.field == expected_value

    @pytest.mark.asyncio
    async def test_error_handling(self, mock_supabase_client):
        """Test error handling."""
        # Mock failure
        mock_supabase_client.method = Mock(side_effect=Exception("Error"))

        result = await your_tool(mock_supabase_client)

        # Should handle gracefully
        assert result is not None
```

### Using Fixtures

```python
@pytest.mark.asyncio
async def test_with_fixtures(
    test_agent,           # Agent with TestModel
    test_dependencies,    # Configured dependencies
    sample_input_texts    # Test data
):
    """Test using fixtures."""
    result = await test_agent.run(
        sample_input_texts["technical"],
        deps=test_dependencies
    )

    assert result is not None
```

### Async Tests

All tool and agent tests must be async:

```python
@pytest.mark.asyncio
async def test_async_function():
    """Test async function."""
    result = await async_function()
    assert result is not None
```

## Debugging Tests

### Print Debugging

```bash
# Show print statements
pytest tests/ -v -s

# Show full tracebacks
pytest tests/ -v --tb=long

# Drop into debugger on failure
pytest tests/ -v --pdb
```

### Logging

Tests use the agent's logging configuration. Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

@pytest.mark.asyncio
async def test_with_logging():
    # Logs will be visible with -s flag
    result = await your_function()
```

## Common Issues

### ModuleNotFoundError

```bash
# Make sure you're in the project root
cd /path/to/full-ai-coding-workflow

# Install in development mode
pip install -e .
```

### Environment Variables

Tests use default test values but you can set real values:

```bash
# Copy example env file
cp .env.example .env

# Edit with your keys (optional for tests)
# Tests will use mocks by default
```

### Async Warnings

If you see async warnings, ensure:
1. Test is marked with `@pytest.mark.asyncio`
2. `pytest.ini` has `asyncio_mode = auto`

## Performance Testing

Performance tests use mocks and validate code structure, not real performance:

```python
@pytest.mark.asyncio
async def test_performance():
    """Test completes quickly with mocks."""
    import time

    start = time.time()
    await your_function(mock_client)
    duration = time.time() - start

    assert duration < 1.0  # With mocks, should be fast
```

For real performance testing:
1. Set up test environment with real services
2. Use `pytest-benchmark` for accurate measurements
3. Run multiple iterations

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest

    steps:
    - uses: actions/checkout@v2

    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'

    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-asyncio pytest-cov

    - name: Run tests
      run: |
        pytest tests/ -v --cov=ai_pm_agent --cov-report=xml

    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

## Test Maintenance

### Regular Tasks

1. **Run tests before commits**
   ```bash
   pytest tests/ -v
   ```

2. **Update tests when adding features**
   - Add tool test in `test_tools.py`
   - Add model test in `test_models.py`
   - Add integration test in `test_agent.py`

3. **Review coverage**
   ```bash
   pytest tests/ --cov=ai_pm_agent --cov-report=html
   ```

4. **Update fixtures** when dependencies change

5. **Keep VALIDATION_REPORT.md** in sync with test results

## Resources

- **Pydantic AI Testing Docs:** https://ai.pydantic.dev/testing/
- **Pytest Documentation:** https://docs.pytest.org/
- **Pydantic Validation:** https://docs.pydantic.dev/
- **PRP Reference:** PRPs/ai-project-management-agent.md

## Getting Help

If tests fail:

1. Check the error message carefully
2. Run single failing test: `pytest tests/test_file.py::test_name -v`
3. Enable debug output: `pytest -v -s --tb=long`
4. Review `VALIDATION_REPORT.md` for expected behavior
5. Check fixtures in `conftest.py`

## Contributing

When adding tests:

1. Follow existing patterns
2. Use descriptive test names
3. Add docstrings explaining what's tested
4. Mock external services
5. Test both success and error cases
6. Update VALIDATION_REPORT.md

---

**Test Suite Version:** 1.0
**Last Updated:** 2025-01-16
**Total Tests:** 149
**Status:** ✅ All validation gates passing
