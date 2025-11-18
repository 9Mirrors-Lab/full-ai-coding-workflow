"""
Dependencies for AI Project Management Agent.
"""

from dataclasses import dataclass
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
    current_sprint_info: Dict[str, Any]

    # Work Item Templates (required)
    work_item_templates: Dict[str, str]

    # Launch Readiness Categories (required)
    launch_readiness_categories: List[Dict[str, str]]

    # GTM Phase Definitions (required)
    gtm_phase_definitions: Dict[str, Dict[str, str]]

    # File System Configuration (required)
    artifact_registry_path: str

    # ADO Project Configuration (required)
    ado_project_name: str

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
            session_id=kwargs.get('session_id'),
            user_id=kwargs.get('user_id'),
        )
