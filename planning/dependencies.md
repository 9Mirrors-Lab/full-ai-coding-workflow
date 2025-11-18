# Dependencies Configuration - AI Project Management Agent

**Agent:** AI Project Management Agent
**Purpose:** Configuration and dependency management for intelligent project management automation
**Version:** 1.0
**Created:** 2025-01-16

## Overview

This document specifies the minimal, essential dependencies and configuration required for the AI Project Management Agent. The agent requires access to Supabase (for vector search and data storage), OpenAI (for embeddings), and basic configuration data for classification and work item generation.

## Philosophy

**Keep It Simple**: Only essential dependencies. Single LLM provider. Basic dataclass for context passing. No complex dependency injection frameworks or factory patterns.

## Environment Variables

### Required Environment Variables

```bash
# LLM Configuration (REQUIRED)
LLM_PROVIDER=openai
LLM_API_KEY=your-openai-api-key-here
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1

# OpenAI Embeddings (REQUIRED for semantic search)
OPENAI_EMBEDDING_API_KEY=your-openai-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Supabase Database (REQUIRED)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key-here

# Application Configuration (REQUIRED)
ADO_PROJECT_NAME=NorthStar
ARTIFACT_REGISTRY_PATH=/path/to/artifacts/storage

# Application Settings (OPTIONAL)
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=false
MAX_RETRIES=3
TIMEOUT_SECONDS=30
```

### Environment File Template

Create `.env.example`:
```bash
# LLM Configuration (REQUIRED)
LLM_PROVIDER=openai
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4o
LLM_BASE_URL=https://api.openai.com/v1

# OpenAI Embeddings (REQUIRED)
OPENAI_EMBEDDING_API_KEY=your-api-key-here
OPENAI_EMBEDDING_MODEL=text-embedding-3-small

# Supabase Database (REQUIRED)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-anon-key

# Application Configuration (REQUIRED)
ADO_PROJECT_NAME=NorthStar
ARTIFACT_REGISTRY_PATH=/path/to/artifacts

# Application Settings
APP_ENV=development
LOG_LEVEL=INFO
DEBUG=false
MAX_RETRIES=3
TIMEOUT_SECONDS=30
```

## Settings Configuration

### settings.py Structure

```python
"""
Configuration management using pydantic-settings and python-dotenv.
"""

from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Configuration
    llm_provider: str = Field(default="openai", description="LLM provider")
    llm_api_key: str = Field(..., description="API key for LLM provider")
    llm_model: str = Field(default="gpt-4o", description="Model name")
    llm_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for LLM API"
    )

    # OpenAI Embeddings Configuration
    openai_embedding_api_key: str = Field(
        ...,
        description="OpenAI API key for embeddings"
    )
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name"
    )

    # Supabase Configuration
    supabase_url: str = Field(..., description="Supabase project URL")
    supabase_key: str = Field(..., description="Supabase anon key")

    # Application Configuration
    ado_project_name: str = Field(
        default="NorthStar",
        description="Azure DevOps project name"
    )
    artifact_registry_path: str = Field(
        ...,
        description="File system path for artifact storage"
    )

    # Application Settings
    app_env: str = Field(default="development", description="Environment")
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    max_retries: int = Field(default=3, description="Max retry attempts")
    timeout_seconds: int = Field(default=30, description="Default timeout")

    @field_validator("llm_api_key", "openai_embedding_api_key", "supabase_url", "supabase_key")
    @classmethod
    def validate_required_keys(cls, v):
        """Ensure required keys are not empty."""
        if not v or v.strip() == "":
            raise ValueError("Required API key or URL cannot be empty")
        return v

    @field_validator("app_env")
    @classmethod
    def validate_environment(cls, v):
        """Validate environment setting."""
        valid_envs = ["development", "staging", "production"]
        if v not in valid_envs:
            raise ValueError(f"app_env must be one of {valid_envs}")
        return v


def load_settings() -> Settings:
    """Load settings with proper error handling."""
    try:
        return Settings()
    except Exception as e:
        error_msg = f"Failed to load settings: {e}"
        if "api_key" in str(e).lower():
            error_msg += "\nMake sure to set all required API keys in your .env file"
        raise ValueError(error_msg) from e


# Global settings instance
settings = load_settings()
```

## Provider Configuration

### providers.py Structure

```python
"""
Provider configuration for LLM and external services.
"""

from typing import Optional
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from openai import OpenAI
from supabase import create_client, Client
from .settings import settings


def get_llm_model(model_choice: Optional[str] = None) -> OpenAIModel:
    """
    Get LLM model configuration based on environment variables.

    Args:
        model_choice: Optional override for model choice

    Returns:
        Configured OpenAI model instance
    """
    model_name = model_choice or settings.llm_model

    provider = OpenAIProvider(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key
    )

    return OpenAIModel(model_name, provider=provider)


def get_embedding_client() -> OpenAI:
    """
    Get OpenAI client for embeddings.

    Returns:
        Configured OpenAI client
    """
    return OpenAI(api_key=settings.openai_embedding_api_key)


def get_supabase_client() -> Client:
    """
    Get Supabase client for database operations.

    Returns:
        Configured Supabase client
    """
    return create_client(settings.supabase_url, settings.supabase_key)


def validate_llm_configuration() -> bool:
    """
    Validate LLM configuration is complete.

    Returns:
        True if valid, raises ValueError otherwise
    """
    if not settings.llm_api_key:
        raise ValueError("LLM_API_KEY is required")
    if not settings.llm_model:
        raise ValueError("LLM_MODEL is required")
    return True
```

## Agent Dependencies

### dependencies.py Structure

```python
"""
Dependencies for AI Project Management Agent.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import logging
from supabase import Client
from openai import OpenAI

logger = logging.getLogger(__name__)


@dataclass
class AgentDependencies:
    """
    Dependencies injected into agent runtime context.

    All external services and configurations needed by the agent
    are defined here for type-safe access through RunContext.
    """

    # External Service Clients (required)
    supabase_client: Client
    openai_client: OpenAI

    # Current Sprint Information (required)
    current_sprint_info: Dict[str, Any]  # {"sprint": 42, "capacity": 100, "start_date": "...", "end_date": "..."}

    # Work Item Templates (required)
    work_item_templates: Dict[str, str]  # {"Epic": "template...", "Feature": "template...", "User Story": "template..."}

    # Launch Readiness Categories (required)
    launch_readiness_categories: List[Dict[str, str]]  # [{"name": "Product Readiness", "description": "..."}, ...]

    # GTM Phase Definitions (required)
    gtm_phase_definitions: Dict[str, Dict[str, str]]  # {"foundation": {"description": "...", "typical_work": "..."}, ...}

    # File System Configuration (required)
    artifact_registry_path: str  # Path to artifact storage directory

    # ADO Project Configuration (required)
    ado_project_name: str  # "NorthStar"

    # Optional Runtime Context
    session_id: Optional[str] = None
    user_id: Optional[str] = None

    # Configuration
    max_retries: int = 3
    timeout: int = 30
    debug: bool = False

    @classmethod
    def from_settings(cls, settings, **kwargs):
        """
        Create dependencies from settings with overrides.

        Args:
            settings: Settings instance
            **kwargs: Override values

        Returns:
            Configured AgentDependencies instance
        """
        from .providers import get_supabase_client, get_embedding_client
        from .config import (
            get_work_item_templates,
            get_launch_readiness_categories,
            get_gtm_phase_definitions,
            get_current_sprint_info
        )

        return cls(
            supabase_client=kwargs.get('supabase_client', get_supabase_client()),
            openai_client=kwargs.get('openai_client', get_embedding_client()),
            current_sprint_info=kwargs.get('current_sprint_info', get_current_sprint_info()),
            work_item_templates=kwargs.get('work_item_templates', get_work_item_templates()),
            launch_readiness_categories=kwargs.get('launch_readiness_categories', get_launch_readiness_categories()),
            gtm_phase_definitions=kwargs.get('gtm_phase_definitions', get_gtm_phase_definitions()),
            artifact_registry_path=kwargs.get('artifact_registry_path', settings.artifact_registry_path),
            ado_project_name=kwargs.get('ado_project_name', settings.ado_project_name),
            max_retries=kwargs.get('max_retries', settings.max_retries),
            timeout=kwargs.get('timeout', settings.timeout_seconds),
            debug=kwargs.get('debug', settings.debug),
            **{k: v for k, v in kwargs.items()
               if k not in ['supabase_client', 'openai_client', 'current_sprint_info',
                           'work_item_templates', 'launch_readiness_categories',
                           'gtm_phase_definitions', 'artifact_registry_path',
                           'ado_project_name', 'max_retries', 'timeout', 'debug']}
        )
```

## Configuration Data Module

### config.py Structure

Create a separate `config.py` file for configuration data:

```python
"""
Configuration data for work item templates, launch readiness, and GTM phases.
"""

from typing import Dict, List


def get_work_item_templates() -> Dict[str, str]:
    """
    Get ADO work item templates.

    Returns:
        Dictionary of work item type to template string
    """
    return {
        "Epic": """Title: [Clear, concise epic name]

Vision Statement:
[What is this epic trying to achieve?]

Business Value:
[Why is this important? What value does it deliver?]

Success Metrics:
- [Measurable metric 1]
- [Measurable metric 2]

Tags: [Relevant tags]""",

        "Feature": """Title: [User-facing capability name]

Description:
[What capability does this provide to users?]

Acceptance Criteria:
- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Criterion 3

Technical Notes:
[Implementation considerations, dependencies, etc.]

Tags: [Relevant tags]""",

        "User Story": """Title: [Brief user story summary]

User Story:
As a [user type]
I want [capability]
So that [benefit]

Acceptance Criteria (Gherkin format):

Scenario 1: [Scenario name]
Given [initial context]
When [action taken]
Then [expected outcome]

Scenario 2: [Another scenario name]
Given [initial context]
When [action taken]
Then [expected outcome]

Tags: [Relevant tags]"""
    }


def get_launch_readiness_categories() -> List[Dict[str, str]]:
    """
    Get launch readiness categories with descriptions.

    Returns:
        List of category dictionaries
    """
    return [
        {
            "name": "Product Readiness",
            "description": "Core product features, UX completeness, feature parity",
            "examples": "Feature development, UX improvements, Product specs"
        },
        {
            "name": "Technical Readiness",
            "description": "Infrastructure, performance, scalability, technical debt",
            "examples": "API development, Database optimization, Bug fixes"
        },
        {
            "name": "Operational Readiness",
            "description": "Support processes, documentation, monitoring",
            "examples": "Support docs, Monitoring setup, Operational procedures"
        },
        {
            "name": "Commercial Readiness",
            "description": "Pricing, billing, contracts, sales enablement",
            "examples": "Pricing strategy, Billing integration, Sales materials"
        },
        {
            "name": "Customer Readiness",
            "description": "Onboarding, training, customer success",
            "examples": "Onboarding flows, Training materials, Customer success"
        },
        {
            "name": "Security & Compliance",
            "description": "Security audits, compliance, data protection",
            "examples": "Security review, GDPR compliance, Data encryption"
        }
    ]


def get_gtm_phase_definitions() -> Dict[str, Dict[str, str]]:
    """
    Get GTM phase definitions.

    Returns:
        Dictionary of phase name to phase details
    """
    return {
        "foundation": {
            "description": "Core development, scaffolding, essential features",
            "typical_work": "Architecture setup, core APIs, database schema, authentication"
        },
        "validation": {
            "description": "Testing, onboarding, user feedback",
            "typical_work": "Alpha/beta testing, user interviews, bug fixes, UX refinement"
        },
        "launch_prep": {
            "description": "Pricing, support setup, documentation",
            "typical_work": "Pricing strategy, support docs, customer success processes"
        },
        "go_to_market": {
            "description": "Marketing, training, initial customer acquisition",
            "typical_work": "Marketing campaigns, sales training, customer onboarding"
        },
        "growth": {
            "description": "Scaling, optimization, performance improvements",
            "typical_work": "Performance optimization, scaling infrastructure, analytics"
        },
        "market_leadership": {
            "description": "Advanced features, automation, competitive differentiation",
            "typical_work": "Advanced AI features, workflow automation, integrations"
        }
    }


def get_current_sprint_info() -> Dict[str, any]:
    """
    Get current sprint information.

    Note: In production, this would fetch from Supabase or ADO API.
    For now, returns a default configuration.

    Returns:
        Dictionary with sprint number and capacity info
    """
    return {
        "sprint": 42,
        "capacity": 100,
        "start_date": "2025-01-13",
        "end_date": "2025-01-26"
    }
```

## Agent Initialization

### agent.py Structure

```python
"""
AI Project Management Agent - Main agent initialization.
"""

import logging
from typing import Optional
from pydantic_ai import Agent

from .providers import get_llm_model
from .dependencies import AgentDependencies
from .settings import settings

logger = logging.getLogger(__name__)

# System prompt (will be provided by prompt-engineer subagent)
SYSTEM_PROMPT = """
[System prompt will be inserted here by prompt-engineer]
"""

# Initialize the agent with proper configuration
pm_agent = Agent(
    get_llm_model(),
    deps_type=AgentDependencies,
    system_prompt=SYSTEM_PROMPT,
    retries=settings.max_retries
)

# Tools will be registered by tool-integrator subagent
# from .tools import register_tools
# register_tools(pm_agent, AgentDependencies)


# Convenience functions for agent usage
async def run_pm_agent(
    prompt: str,
    session_id: Optional[str] = None,
    **dependency_overrides
) -> str:
    """
    Run the PM agent with automatic dependency injection.

    Args:
        prompt: User prompt/query
        session_id: Optional session identifier
        **dependency_overrides: Override default dependencies

    Returns:
        Agent response as string
    """
    deps = AgentDependencies.from_settings(
        settings,
        session_id=session_id,
        **dependency_overrides
    )

    result = await pm_agent.run(prompt, deps=deps)
    return result.data


def create_agent_with_deps(**dependency_overrides) -> tuple[Agent, AgentDependencies]:
    """
    Create agent instance with custom dependencies.

    Args:
        **dependency_overrides: Custom dependency values

    Returns:
        Tuple of (agent, dependencies)
    """
    deps = AgentDependencies.from_settings(settings, **dependency_overrides)
    return pm_agent, deps
```

## Python Package Requirements

### requirements.txt

```txt
# Core Pydantic AI dependencies
pydantic-ai>=0.1.0
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# LLM Provider
openai>=1.0.0

# Database and Vector Search
supabase>=2.0.0
vecs>=0.4.0

# Async utilities
httpx>=0.25.0
aiofiles>=23.0.0

# Development tools
pytest>=7.4.0
pytest-asyncio>=0.21.0
pytest-cov>=4.1.0
black>=23.0.0
ruff>=0.1.0

# Logging
loguru>=0.7.0
```

## Directory Structure

```
ai_pm_agent/
├── __init__.py
├── settings.py           # Environment configuration (BaseSettings)
├── providers.py          # Model and service provider setup
├── config.py            # Configuration data (templates, categories, phases)
├── dependencies.py      # Agent dependencies dataclass
├── agent.py            # Agent initialization and convenience functions
├── models.py           # Pydantic models for structured outputs
├── tools.py            # Tool implementations
├── prompts.py          # System prompts
├── .env.example        # Environment template
└── requirements.txt    # Python dependencies

tests/
├── __init__.py
├── conftest.py         # Test fixtures
├── test_agent.py       # Agent tests
├── test_tools.py       # Tool tests
└── test_models.py      # Model validation tests
```

## Security Considerations

### API Key Management
- All API keys loaded from `.env` file using `python-dotenv`
- Never commit `.env` files to version control
- Use `.env.example` as template for required variables
- Validate all API keys on startup with `field_validator`
- In production, use secure storage (AWS Secrets Manager, Azure Key Vault, etc.)

### Input Validation
- Use Pydantic models for all external inputs
- Validate file paths before file system operations
- Sanitize database queries (Supabase handles this via parameterized queries)
- Limit resource consumption (file size limits, query limits)

### ADO Integration Constraint
- CRITICAL: Agent has READ-ONLY access to ADO
- Never create, update, or delete work items directly
- Only generate work items and return for manual approval
- This is a hard constraint for security and auditability

## Testing Configuration

### tests/conftest.py

```python
"""
Test fixtures for AI PM Agent.
"""

import pytest
from unittest.mock import Mock, AsyncMock
from pydantic_ai.models.test import TestModel
from ai_pm_agent.agent import pm_agent
from ai_pm_agent.dependencies import AgentDependencies


@pytest.fixture
def mock_supabase():
    """Mock Supabase client."""
    client = Mock()
    client.rpc = Mock(return_value=Mock(execute=AsyncMock()))
    client.table = Mock(return_value=Mock(
        insert=Mock(return_value=Mock(execute=AsyncMock())),
        select=Mock(return_value=Mock(execute=AsyncMock()))
    ))
    return client


@pytest.fixture
def mock_openai():
    """Mock OpenAI client."""
    client = Mock()
    client.embeddings = Mock()
    client.embeddings.create = AsyncMock(return_value=Mock(
        data=[Mock(embedding=[0.1] * 1536)]
    ))
    return client


@pytest.fixture
def test_dependencies(mock_supabase, mock_openai):
    """Test dependencies with mocked services."""
    from ai_pm_agent.config import (
        get_work_item_templates,
        get_launch_readiness_categories,
        get_gtm_phase_definitions,
        get_current_sprint_info
    )

    return AgentDependencies(
        supabase_client=mock_supabase,
        openai_client=mock_openai,
        current_sprint_info=get_current_sprint_info(),
        work_item_templates=get_work_item_templates(),
        launch_readiness_categories=get_launch_readiness_categories(),
        gtm_phase_definitions=get_gtm_phase_definitions(),
        artifact_registry_path="/tmp/artifacts",
        ado_project_name="NorthStar",
        debug=True
    )


@pytest.fixture
def test_model_agent(test_dependencies):
    """Agent configured with TestModel for testing."""
    with pm_agent.override(model=TestModel()):
        yield pm_agent
```

## Quality Checklist

Before finalizing configuration:
- ✅ All required environment variables documented in `.env.example`
- ✅ Settings validation implemented with Pydantic validators
- ✅ Model provider configuration uses OpenAI with configurable base URL
- ✅ Dependency injection type-safe with dataclass
- ✅ Configuration data separated into `config.py` module
- ✅ Security measures in place (API key validation, read-only ADO)
- ✅ Testing configuration provided with mocked services
- ✅ Simple and minimal - no complex patterns or unnecessary abstractions

## Integration Notes

This configuration serves as the foundation for:
- **Main Claude Code**: Uses agent initialization from `agent.py`
- **pydantic-ai-validator**: Tests with TestModel and mocked dependencies

Works in parallel with:
- **prompt-engineer**: Provides system prompt for `prompts.py`
- **tool-integrator**: Tools registered with agent using `AgentDependencies`

## Implementation Order

1. Create `settings.py` with environment variable loading
2. Create `providers.py` with service client functions
3. Create `config.py` with configuration data
4. Create `dependencies.py` with dataclass definition
5. Create `agent.py` with agent initialization
6. Create `.env.example` with all required variables
7. Create `requirements.txt` with package dependencies
8. Create test fixtures in `tests/conftest.py`

---

**STATUS:** Ready for implementation
**COMPLEXITY:** Low - Simple configuration with standard Pydantic AI patterns
**ESTIMATED TIME:** 1-2 hours for complete dependency setup
