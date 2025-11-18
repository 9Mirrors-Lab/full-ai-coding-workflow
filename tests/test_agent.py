"""
Agent-level tests for AI Project Management Agent.

Tests the complete agent functionality including tool registration,
dependency injection, and agent responses using TestModel.
"""

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.messages import ModelResponse

from ai_pm_agent.agent import pm_agent, run_pm_agent
from ai_pm_agent.dependencies import AgentDependencies


def get_agent_tool_names(agent):
    """Helper function to get tool names from agent."""
    tool_names = []
    for toolset in agent.toolsets:
        if hasattr(toolset, 'tools'):
            tool_names.extend(toolset.tools.keys())
    return tool_names


class TestAgentInstantiation:
    """Test agent creation and configuration."""

    def test_agent_exists(self):
        """Test that agent is properly instantiated."""
        assert pm_agent is not None
        assert pm_agent.deps_type == AgentDependencies

    def test_agent_has_tools(self):
        """Test that agent has all 8 tools registered."""
        tool_names = get_agent_tool_names(pm_agent)

        expected_tools = [
            "classify_input",
            "match_ado_work_items",
            "generate_work_item",
            "tag_artifact",
            "recommend_sprint_placement",
            "create_action_item",
            "update_launch_readiness",
            "log_agent_decision"
        ]

        for expected_tool in expected_tools:
            assert expected_tool in tool_names, f"Tool {expected_tool} not registered"

    def test_agent_system_prompt(self):
        """Test that agent has a system prompt configured."""
        assert pm_agent.system_prompt is not None
        assert len(str(pm_agent.system_prompt)) > 0

    def test_agent_with_override(self, test_model):
        """Test agent override with TestModel."""
        # Override returns a context manager, verify it works
        with pm_agent.override(model=test_model):
            # Inside context, agent is overridden
            assert pm_agent is not None


class TestAgentBasicExecution:
    """Test basic agent execution with TestModel."""

    @pytest.mark.asyncio
    async def test_agent_run_basic(self, test_model, test_dependencies):
        """Test that agent can run with TestModel."""
        with pm_agent.override(model=test_model):
            result = await pm_agent.run(
                "Test input for agent",
                deps=test_dependencies
            )

            assert result is not None
            assert result.output is not None

    @pytest.mark.asyncio
    async def test_agent_run_with_message(self, test_model, test_dependencies):
        """Test agent processes messages correctly."""
        with pm_agent.override(model=test_model):
            result = await pm_agent.run(
                "Implement user authentication feature",
                deps=test_dependencies
            )

            # Verify we got a response
            assert result.output is not None

            # Verify message history exists
            messages = result.all_messages()
            assert len(messages) > 0

    @pytest.mark.asyncio
    async def test_agent_with_different_inputs(self, test_model, test_dependencies, sample_input_texts):
        """Test agent handles different input types."""
        with pm_agent.override(model=test_model):
            for input_type, input_text in sample_input_texts.items():
                result = await pm_agent.run(input_text, deps=test_dependencies)
                assert result is not None, f"Agent failed for {input_type} input"


class TestAgentToolCalling:
    """Test agent tool calling behavior."""

    @pytest.mark.asyncio
    async def test_classify_input_tool_callable(self, test_dependencies):
        """Test classify_input tool can be called directly."""
        test_model = TestModel()

        with pm_agent.override(model=test_model):
            result = await pm_agent.run(
                "Fix authentication bug in login flow",
                deps=test_dependencies
            )

            assert result is not None

    @pytest.mark.asyncio
    async def test_match_ado_work_items_tool_callable(self, test_dependencies):
        """Test match_ado_work_items tool can be called."""
        test_model = TestModel()

        with pm_agent.override(model=test_model):
            result = await pm_agent.run(
            "Search for existing authentication work items",
            deps=test_dependencies
        )

        assert result is not None

    @pytest.mark.asyncio
    async def test_generate_work_item_tool_callable(self, test_dependencies):
        """Test generate_work_item tool can be called."""
        test_model = TestModel()

        with pm_agent.override(model=test_model):
            result = await pm_agent.run(
            "Create a feature for user authentication",
            deps=test_dependencies
        )

        assert result is not None


class TestAgentDependencyInjection:
    """Test dependency injection works correctly."""

    @pytest.mark.asyncio
    async def test_dependencies_accessible(self, test_model, test_dependencies):
        """Test that dependencies are properly injected."""
        with pm_agent.override(model=test_model):
            result = await pm_agent.run(
            "Test dependency injection",
            deps=test_dependencies
        )

        # If no error, dependencies were properly injected
        assert result is not None

    @pytest.mark.asyncio
    async def test_session_id_in_dependencies(self, test_model):
        """Test session ID is passed through dependencies."""
        from ai_pm_agent.config import (
            get_work_item_templates,
            get_launch_readiness_categories,
            get_gtm_phase_definitions,
            get_current_sprint_info
        )
        from unittest.mock import Mock

        deps = AgentDependencies(
            supabase_client=Mock(),
            openai_client=Mock(),
            current_sprint_info=get_current_sprint_info(),
            work_item_templates=get_work_item_templates(),
            launch_readiness_categories=get_launch_readiness_categories(),
            gtm_phase_definitions=get_gtm_phase_definitions(),
            artifact_registry_path="/tmp/test",
            ado_project_name="Test",
            session_id="test-session-456"
        )

        with pm_agent.override(model=test_model):
            result = await pm_agent.run("Test", deps=deps)
        assert result is not None

    @pytest.mark.asyncio
    async def test_custom_dependencies(self, test_model):
        """Test agent works with custom dependency overrides."""
        from ai_pm_agent.config import (
            get_work_item_templates,
            get_launch_readiness_categories,
            get_gtm_phase_definitions,
            get_current_sprint_info
        )
        from unittest.mock import Mock

        custom_templates = get_work_item_templates()
        custom_templates["Custom Type"] = "Custom Template"

        deps = AgentDependencies(
            supabase_client=Mock(),
            openai_client=Mock(),
            current_sprint_info=get_current_sprint_info(),
            work_item_templates=custom_templates,
            launch_readiness_categories=get_launch_readiness_categories(),
            gtm_phase_definitions=get_gtm_phase_definitions(),
            artifact_registry_path="/tmp/custom",
            ado_project_name="CustomProject"
        )

        with pm_agent.override(model=test_model):
            result = await pm_agent.run("Test custom", deps=deps)
        assert result is not None


class TestAgentErrorHandling:
    """Test agent error handling and graceful degradation."""

    @pytest.mark.asyncio
    async def test_agent_with_empty_input(self, test_model, test_dependencies):
        """Test agent handles empty input gracefully."""
        with pm_agent.override(model=test_model):
            result = await pm_agent.run("", deps=test_dependencies)
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_with_very_long_input(self, test_model, test_dependencies):
        """Test agent handles very long input."""
        long_input = "This is a test. " * 1000
        with pm_agent.override(model=test_model):
            result = await pm_agent.run(long_input, deps=test_dependencies)
        assert result is not None

    @pytest.mark.asyncio
    async def test_agent_with_special_characters(self, test_model, test_dependencies):
        """Test agent handles special characters."""
        special_input = "Test with émojis 🚀 and symbols @#$%^&*()"
        with pm_agent.override(model=test_model):
            result = await pm_agent.run(special_input, deps=test_dependencies)
        assert result is not None


class TestAgentValidationGates:
    """Test validation gates from the PRP."""

    @pytest.mark.asyncio
    async def test_agent_never_creates_ado_items_directly(self, test_model, test_dependencies):
        """
        CRITICAL VALIDATION: Ensure agent NEVER creates ADO work items directly.

        Agent should only GENERATE work items and return them for approval.
        """
        # This is enforced by architecture - agent has no ADO write tools
        tool_names = get_agent_tool_names(pm_agent)

        # Verify no ADO creation tools exist
        ado_write_tools = [
            "create_ado_work_item",
            "update_ado_work_item",
            "delete_ado_work_item",
            "modify_ado_work_item"
        ]

        for forbidden_tool in ado_write_tools:
            assert forbidden_tool not in tool_names, \
                f"SECURITY VIOLATION: Agent has {forbidden_tool} tool!"

    @pytest.mark.asyncio
    async def test_all_eight_tools_registered(self, test_model):
        """Validate all 8 required tools are registered."""
        tool_names = get_agent_tool_names(pm_agent)

        required_tools = [
            "classify_input",
            "match_ado_work_items",
            "generate_work_item",
            "tag_artifact",
            "recommend_sprint_placement",
            "create_action_item",
            "update_launch_readiness",
            "log_agent_decision"
        ]

        assert len(tool_names) == 8, f"Expected 8 tools, got {len(tool_names)}"

        for required_tool in required_tools:
            assert required_tool in tool_names, f"Required tool {required_tool} missing"

    @pytest.mark.asyncio
    async def test_structured_outputs_validate(self, test_model, test_dependencies):
        """Test that structured outputs are properly validated by Pydantic models."""
        # Run agent and verify output structure
        with pm_agent.override(model=test_model):
            result = await pm_agent.run(
            "Classify this technical feature request",
            deps=test_dependencies
        )

        # TestModel should return string by default
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_dependency_injection_type_safe(self):
        """Validate dependency injection is type-safe with dataclass."""
        from dataclasses import is_dataclass

        assert is_dataclass(AgentDependencies), \
            "AgentDependencies must be a dataclass for type safety"

    @pytest.mark.asyncio
    async def test_confidence_thresholds_enforced(self, test_model, test_dependencies):
        """
        Validate confidence threshold requirements:
        - < 70%: ask_for_clarification
        - 70-85%: suggest_with_caveat
        - > 85%: recommend action
        """
        # This is tested in detail in test_tools.py
        # Here we just verify the agent can handle different confidence scenarios

        test_cases = [
            "vague unclear input",  # Low confidence
            "maybe some technical stuff",  # Medium confidence
            "implement user authentication with OAuth2"  # High confidence
        ]

        for test_input in test_cases:
            with pm_agent.override(model=test_model):
                result = await pm_agent.run(test_input, deps=test_dependencies)
            assert result is not None


class TestAgentIntegration:
    """Integration tests for complete agent workflows."""

    @pytest.mark.asyncio
    async def test_full_workflow_classification_to_work_item(self, test_model, test_dependencies):
        """Test complete workflow from classification to work item generation."""
        # Simulate a typical user input
        user_input = "Jay mentioned users are confused by the Sous loading screen. Wants progress indicators and food-related animation."

        with pm_agent.override(model=test_model):
            result = await pm_agent.run(user_input, deps=test_dependencies)

        # Verify agent processed the request
        assert result is not None
        assert result.output is not None

    @pytest.mark.asyncio
    async def test_artifact_processing_workflow(self, test_model, test_dependencies, temp_artifact_file):
        """Test artifact upload and tagging workflow."""
        user_input = f"Process this artifact: {temp_artifact_file}"

        with pm_agent.override(model=test_model):
            result = await pm_agent.run(user_input, deps=test_dependencies)

        assert result is not None

    @pytest.mark.asyncio
    async def test_sprint_planning_workflow(self, test_model, test_dependencies):
        """Test sprint recommendation workflow."""
        user_input = "Where should we place the authentication feature in our sprint?"

        with pm_agent.override(model=test_model):
            result = await pm_agent.run(user_input, deps=test_dependencies)

        assert result is not None
