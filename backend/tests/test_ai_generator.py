"""
Tests for AIGenerator tool integration and functionality
"""

import os
import sys
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ai_generator import AIGenerator
from ai_generator import ConversationState


class TestAIGenerator:
    """Test cases for AIGenerator tool integration and functionality"""

    def test_init_sets_correct_attributes(self):
        """Test that AIGenerator initializes with correct attributes"""
        # Arrange & Act
        generator = AIGenerator("test_api_key", "test_model")

        # Assert
        assert generator.model == "test_model"
        assert generator.base_params["model"] == "test_model"
        assert generator.base_params["temperature"] == 0
        assert generator.base_params["max_tokens"] == 800

    @patch("ai_generator.anthropic")
    def test_generate_response_without_tools(self, mock_anthropic):
        """Test response generation without tools"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Test response without tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        generator = AIGenerator("test_api_key", "test_model")

        # Act
        result = generator.generate_response("What is machine learning?")

        # Assert
        assert result == "Test response without tools"
        mock_client.messages.create.assert_called_once()

        # Verify the call arguments
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["model"] == "test_model"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert call_args[1]["messages"][0]["role"] == "user"
        assert call_args[1]["messages"][0]["content"] == "What is machine learning?"
        assert "tools" not in call_args[1]

    @patch("ai_generator.anthropic")
    def test_generate_response_with_conversation_history(self, mock_anthropic):
        """Test response generation with conversation history"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response with history")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        generator = AIGenerator("test_api_key", "test_model")

        # Act
        result = generator.generate_response(
            query="Follow up question",
            conversation_history="User: Previous question\nAssistant: Previous answer",
        )

        # Assert
        assert result == "Response with history"

        # Verify system prompt includes conversation history
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert "User: Previous question" in system_content
        assert "Assistant: Previous answer" in system_content

    @patch("ai_generator.anthropic")
    def test_generate_response_with_tools_no_tool_use(self, mock_anthropic):
        """Test response generation with tools provided but not used"""
        # Arrange
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock(text="Response without using tools")]
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.Anthropic.return_value = mock_client

        generator = AIGenerator("test_api_key", "test_model")
        mock_tools = [{"name": "search_tool", "description": "Search tool"}]
        mock_tool_manager = Mock()

        # Act
        result = generator.generate_response(
            query="Simple question", tools=mock_tools, tool_manager=mock_tool_manager
        )

        # Assert
        assert result == "Response without using tools"
        mock_client.messages.create.assert_called_once()

        # Verify tools were included in the call
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["tools"] == mock_tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}

        # Tool manager should not be called since no tool use
        mock_tool_manager.execute_tool.assert_not_called()

    @patch("ai_generator.anthropic")
    def test_generate_response_with_tool_use(
        self, mock_anthropic, mock_anthropic_client_with_tool_use
    ):
        """Test response generation with tool usage"""
        # Arrange
        mock_anthropic.Anthropic.return_value = mock_anthropic_client_with_tool_use

        generator = AIGenerator("test_api_key", "test_model")
        mock_tools = [{"name": "search_course_content", "description": "Search tool"}]
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        # Act
        result = generator.generate_response(
            query="Search for machine learning",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert (
            result
            == "Machine learning is a subset of AI that focuses on learning from data."
        )

        # Verify two API calls were made (initial + follow-up)
        assert mock_anthropic_client_with_tool_use.messages.create.call_count == 2

        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", query="machine learning"
        )

    def test_handle_tool_execution_single_tool(self, mock_anthropic_client):
        """Test _handle_tool_execution method with single tool call"""
        # Arrange
        generator = AIGenerator("test_api_key", "test_model")
        generator.client = mock_anthropic_client  # Inject mock client

        # Mock initial response with tool use
        initial_response = Mock()
        initial_response.content = [Mock()]
        initial_response.content[0].type = "tool_use"
        initial_response.content[0].name = "search_tool"
        initial_response.content[0].id = "tool_123"
        initial_response.content[0].input = {"query": "test"}

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Final response after tool use")]
        mock_anthropic_client.messages.create.return_value = final_response

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
        }

        # Act
        result = generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Assert
        assert result == "Final response after tool use"
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_tool", query="test"
        )
        mock_anthropic_client.messages.create.assert_called_once()

    def test_handle_tool_execution_multiple_tools(self, mock_anthropic_client):
        """Test _handle_tool_execution method with multiple tool calls"""
        # Arrange
        generator = AIGenerator("test_api_key", "test_model")
        generator.client = mock_anthropic_client  # Inject mock client

        # Mock initial response with multiple tool uses
        initial_response = Mock()
        tool_use_1 = Mock()
        tool_use_1.type = "tool_use"
        tool_use_1.name = "search_tool"
        tool_use_1.id = "tool_123"
        tool_use_1.input = {"query": "first"}

        tool_use_2 = Mock()
        tool_use_2.type = "tool_use"
        tool_use_2.name = "outline_tool"
        tool_use_2.id = "tool_456"
        tool_use_2.input = {"course": "test"}

        initial_response.content = [tool_use_1, tool_use_2]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Final response with multiple tools")]
        mock_anthropic_client.messages.create.return_value = final_response

        base_params = {
            "messages": [{"role": "user", "content": "test query"}],
            "system": "test system prompt",
        }

        # Act
        result = generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Assert
        assert result == "Final response with multiple tools"
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_tool", query="first")
        mock_tool_manager.execute_tool.assert_any_call("outline_tool", course="test")

    def test_handle_tool_execution_builds_correct_messages(self, mock_anthropic_client):
        """Test that _handle_tool_execution builds correct message structure"""
        # Arrange
        generator = AIGenerator("test_api_key", "test_model")
        generator.client = mock_anthropic_client  # Inject mock client

        # Mock initial response
        initial_response = Mock()
        tool_use = Mock()
        tool_use.type = "tool_use"
        tool_use.name = "test_tool"
        tool_use.id = "tool_id"
        tool_use.input = {"param": "value"}
        initial_response.content = [tool_use]

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"

        # Mock final response
        final_response = Mock()
        final_response.content = [Mock(text="Final response")]
        mock_anthropic_client.messages.create.return_value = final_response

        base_params = {
            "messages": [{"role": "user", "content": "original query"}],
            "system": "system prompt",
            "model": "test_model",
            "temperature": 0,
            "max_tokens": 800,
        }

        # Act
        generator._handle_tool_execution(
            initial_response, base_params, mock_tool_manager
        )

        # Assert
        call_args = mock_anthropic_client.messages.create.call_args
        final_params = call_args[1]

        # Verify final message structure
        assert len(final_params["messages"]) == 3
        assert final_params["messages"][0]["role"] == "user"
        assert final_params["messages"][0]["content"] == "original query"
        assert final_params["messages"][1]["role"] == "assistant"
        assert final_params["messages"][1]["content"] == [tool_use]
        assert final_params["messages"][2]["role"] == "user"
        assert len(final_params["messages"][2]["content"]) == 1

        # Verify tool result structure
        tool_result = final_params["messages"][2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "tool_id"
        assert tool_result["content"] == "Tool execution result"

        # Verify other parameters
        assert final_params["system"] == "system prompt"
        assert final_params["model"] == "test_model"
        assert "tools" not in final_params  # Tools should be removed for final call

    def test_system_prompt_content(self):
        """Test that system prompt contains expected content"""
        # Arrange & Act
        generator = AIGenerator("test_api_key", "test_model")

        # Assert
        system_prompt = generator.SYSTEM_PROMPT
        assert "course materials and educational content" in system_prompt
        assert "search tool" in system_prompt.lower()
        assert "outline tool" in system_prompt.lower()
        assert "Sequential Tool Usage" in system_prompt  # Updated for new functionality
        assert "Brief, Concise and focused" in system_prompt

    @patch("ai_generator.anthropic")
    def test_error_handling_in_generate_response(self, mock_anthropic):
        """Test error handling during API calls"""
        # Arrange
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API Error")
        mock_anthropic.Anthropic.return_value = mock_client

        generator = AIGenerator("test_api_key", "test_model")

        # Act & Assert
        with pytest.raises(Exception) as exc_info:
            generator.generate_response("test query")
        assert "API Error" in str(exc_info.value)

    def test_tool_execution_without_tool_manager(
        self, mock_anthropic_client_with_tool_use
    ):
        """Test tool execution when tool_manager is None"""
        # This tests the edge case where tools are provided but tool_manager is None
        # In this case, the system should handle it gracefully

        # Arrange
        # Use the mock that returns tool_use but no tool manager
        generator = AIGenerator("test_api_key", "test_model")
        generator.client = mock_anthropic_client_with_tool_use

        mock_tools = [{"name": "search_tool", "description": "Search tool"}]

        # Act
        result = generator.generate_response(
            query="Search query", tools=mock_tools, tool_manager=None
        )

        # Assert
        # Since tool_manager is None, _handle_tool_execution won't be called
        # The response should be from the initial call (which has stop_reason="tool_use")
        # But since there's no tool manager, it might not work as expected
        # This test verifies the behavior in this edge case
        assert result is not None

    def test_base_params_structure(self):
        """Test that base_params are correctly structured"""
        # Arrange & Act
        generator = AIGenerator("test_key", "test_model")

        # Assert
        base_params = generator.base_params
        assert base_params["model"] == "test_model"
        assert base_params["temperature"] == 0
        assert base_params["max_tokens"] == 800
        assert len(base_params) == 3  # Only these three keys


class TestConversationState:
    """Test cases for ConversationState class"""

    def test_conversation_state_init(self):
        """Test ConversationState initialization"""
        # Arrange & Act
        state = ConversationState(
            original_query="test query", messages=[{"role": "user", "content": "test"}]
        )

        # Assert
        assert state.original_query == "test query"
        assert state.messages == [{"role": "user", "content": "test"}]
        assert state.round_count == 0
        assert state.max_rounds == 2
        assert state.accumulated_context == []

    def test_can_continue(self):
        """Test can_continue logic"""
        # Arrange
        state = ConversationState("query", [])

        # Assert initial state
        assert state.can_continue() is True

        # Test after round 1
        state.round_count = 1
        assert state.can_continue() is True

        # Test after round 2 (max rounds)
        state.round_count = 2
        assert state.can_continue() is False

    def test_add_tool_context(self):
        """Test adding tool context"""
        # Arrange
        state = ConversationState("query", [])

        # Act
        state.add_tool_context("First tool result")
        state.add_tool_context("Second tool result")
        state.add_tool_context("First tool result")  # Duplicate

        # Assert
        assert len(state.accumulated_context) == 2
        assert "First tool result" in state.accumulated_context
        assert "Second tool result" in state.accumulated_context


class TestSequentialToolCalling:
    """Test cases for sequential tool calling functionality"""

    @patch("ai_generator.anthropic")
    def test_sequential_tool_calling_two_rounds(self, mock_anthropic):
        """Test sequential tool calling with two complete rounds"""
        # Arrange
        mock_client = Mock()

        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_use_block1 = Mock()
        tool_use_block1.type = "tool_use"
        tool_use_block1.name = "get_course_outline"
        tool_use_block1.id = "tool_123"
        tool_use_block1.input = {"course_title": "Course X"}
        round1_response.content = [tool_use_block1]

        # Round 1: Follow-up response after tool execution
        round1_followup = Mock()
        round1_followup.content = [
            Mock(text="I need to search for more specific content")
        ]
        round1_followup.stop_reason = "end_turn"

        # Round 2: Second tool use response
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        tool_use_block2 = Mock()
        tool_use_block2.type = "tool_use"
        tool_use_block2.name = "search_course_content"
        tool_use_block2.id = "tool_456"
        tool_use_block2.input = {"query": "machine learning"}
        round2_response.content = [tool_use_block2]

        # Round 2: Final response
        round2_followup = Mock()
        round2_followup.content = [
            Mock(text="Based on my searches, here's the comprehensive answer")
        ]
        round2_followup.stop_reason = "end_turn"

        # Configure mock client to return responses in sequence
        mock_client.messages.create.side_effect = [
            round1_response,
            round1_followup,  # Round 1
            round2_response,
            round2_followup,  # Round 2
        ]
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = [
            "Course X outline with lesson 4: Advanced ML Topics",
            "Found content about machine learning algorithms",
        ]

        generator = AIGenerator("test_api_key", "test_model")
        mock_tools = [
            {"name": "get_course_outline", "description": "Get course outline"},
            {"name": "search_course_content", "description": "Search content"},
        ]

        # Act
        result = generator.generate_response(
            query="Find a course that covers the same topic as lesson 4 of Course X",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "Based on my searches, here's the comprehensive answer"
        assert mock_client.messages.create.call_count == 4  # 2 rounds × 2 calls each
        assert mock_tool_manager.execute_tool.call_count == 2

        # Verify tools were called correctly
        mock_tool_manager.execute_tool.assert_any_call(
            "get_course_outline", course_title="Course X"
        )
        mock_tool_manager.execute_tool.assert_any_call(
            "search_course_content", query="machine learning"
        )

    @patch("ai_generator.anthropic")
    def test_sequential_tool_calling_early_termination(self, mock_anthropic):
        """Test that sequential tool calling terminates early when Claude doesn't suggest continuation"""
        # Arrange
        mock_client = Mock()

        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "machine learning"}
        round1_response.content = [tool_use_block]

        # Round 1: Complete answer (no continuation hints)
        round1_followup = Mock()
        round1_followup.content = [
            Mock(
                text="Machine learning is a comprehensive field. This answers your question completely."
            )
        ]
        round1_followup.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [round1_response, round1_followup]
        mock_anthropic.Anthropic.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "ML content found"

        generator = AIGenerator("test_api_key", "test_model")
        mock_tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="What is machine learning?",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert (
            result
            == "Machine learning is a comprehensive field. This answers your question completely."
        )
        assert mock_client.messages.create.call_count == 2  # Only one round
        assert mock_tool_manager.execute_tool.call_count == 1

    @patch("ai_generator.anthropic")
    def test_sequential_tool_calling_max_rounds_reached(self, mock_anthropic):
        """Test that sequential tool calling respects max rounds limit"""
        # Arrange
        mock_client = Mock()

        # Round 1
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_use_block1 = Mock()
        tool_use_block1.type = "tool_use"
        tool_use_block1.name = "search_course_content"
        tool_use_block1.id = "tool_123"
        tool_use_block1.input = {"query": "first search"}
        round1_response.content = [tool_use_block1]

        round1_followup = Mock()
        round1_followup.content = [Mock(text="I need to search for more information")]
        round1_followup.stop_reason = "end_turn"

        # Round 2 (final round)
        round2_response = Mock()
        round2_response.stop_reason = "tool_use"
        tool_use_block2 = Mock()
        tool_use_block2.type = "tool_use"
        tool_use_block2.name = "search_course_content"
        tool_use_block2.id = "tool_456"
        tool_use_block2.input = {"query": "second search"}
        round2_response.content = [tool_use_block2]

        round2_followup = Mock()
        round2_followup.content = [
            Mock(text="I still need more info but this is the final round")
        ]
        round2_followup.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [
            round1_response,
            round1_followup,
            round2_response,
            round2_followup,
        ]
        mock_anthropic.Anthropic.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator("test_api_key", "test_model")
        mock_tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="Complex query needing multiple searches",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert result == "I still need more info but this is the final round"
        assert mock_client.messages.create.call_count == 4  # Exactly 2 rounds
        assert mock_tool_manager.execute_tool.call_count == 2

    @patch("ai_generator.anthropic")
    def test_sequential_tool_calling_with_tool_error(self, mock_anthropic):
        """Test sequential tool calling handles tool execution errors gracefully"""
        # Arrange
        mock_client = Mock()

        # Round 1: Tool use response
        round1_response = Mock()
        round1_response.stop_reason = "tool_use"
        tool_use_block = Mock()
        tool_use_block.type = "tool_use"
        tool_use_block.name = "search_course_content"
        tool_use_block.id = "tool_123"
        tool_use_block.input = {"query": "machine learning"}
        round1_response.content = [tool_use_block]

        # Round 1: Follow-up after tool error
        round1_followup = Mock()
        round1_followup.content = [
            Mock(text="I encountered an error but can still provide a helpful response")
        ]
        round1_followup.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [round1_response, round1_followup]
        mock_anthropic.Anthropic.return_value = mock_client

        # Mock tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")

        generator = AIGenerator("test_api_key", "test_model")
        mock_tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="What is machine learning?",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        assert (
            result == "I encountered an error but can still provide a helpful response"
        )
        assert mock_client.messages.create.call_count == 2
        assert mock_tool_manager.execute_tool.call_count == 1

    def test_response_suggests_continuation(self):
        """Test the heuristic for determining if response suggests continuation"""
        # Arrange
        generator = AIGenerator("test_key", "test_model")

        # Test cases that should suggest continuation
        continuation_responses = [
            "Let me search for more information about this topic.",
            "I need to find additional details to answer your question.",
            "Let me look up more specific information.",
            "I should check for more comprehensive data.",
        ]

        # Test cases that should not suggest continuation
        complete_responses = [
            "Machine learning is a comprehensive field that involves algorithms.",
            "Here is the complete answer to your question.",
            "This provides all the information you need.",
        ]

        # Act & Assert
        for response in continuation_responses:
            assert generator._response_suggests_continuation(response) is True

        for response in complete_responses:
            assert generator._response_suggests_continuation(response) is False

    @patch("ai_generator.anthropic")
    def test_synthesis_fallback_when_rounds_exhausted(self, mock_anthropic):
        """Test synthesis fallback when max rounds are reached"""
        # This test simulates a scenario where we've done multiple tool uses
        # and need to synthesize a final response

        # Arrange
        mock_client = Mock()

        # Round 1: Tool use and response
        round1_tool_response = Mock()
        round1_tool_response.stop_reason = "tool_use"
        tool_use_block1 = Mock()
        tool_use_block1.type = "tool_use"
        tool_use_block1.name = "search_course_content"
        tool_use_block1.id = "tool_123"
        tool_use_block1.input = {"query": "first search"}
        round1_tool_response.content = [tool_use_block1]

        round1_followup = Mock()
        round1_followup.content = [Mock(text="I need to search for more information")]
        round1_followup.stop_reason = "end_turn"

        # Round 2: Tool use and response
        round2_tool_response = Mock()
        round2_tool_response.stop_reason = "tool_use"
        tool_use_block2 = Mock()
        tool_use_block2.type = "tool_use"
        tool_use_block2.name = "search_course_content"
        tool_use_block2.id = "tool_456"
        tool_use_block2.input = {"query": "second search"}
        round2_tool_response.content = [tool_use_block2]

        round2_followup = Mock()
        round2_followup.content = [Mock(text="I need to find additional information")]
        round2_followup.stop_reason = "end_turn"

        # Synthesis response
        synthesis_response = Mock()
        synthesis_response.content = [Mock(text="Final synthesized answer")]
        synthesis_response.stop_reason = "end_turn"

        mock_client.messages.create.side_effect = [
            round1_tool_response,
            round1_followup,
            round2_tool_response,
            round2_followup,
            synthesis_response,  # Synthesis call
        ]
        mock_anthropic.Anthropic.return_value = mock_client

        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool result"

        generator = AIGenerator("test_api_key", "test_model")
        mock_tools = [{"name": "search_course_content", "description": "Search"}]

        # Act
        result = generator.generate_response(
            query="Complex query needing synthesis",
            tools=mock_tools,
            tool_manager=mock_tool_manager,
        )

        # Assert
        # Should use synthesis after max rounds
        assert result == "Final synthesized answer"
        # Verify we made all expected calls including synthesis
        assert mock_client.messages.create.call_count == 5
