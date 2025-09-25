from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import anthropic


@dataclass
class ConversationState:
    """Manages state across multiple tool calling rounds"""

    original_query: str
    messages: List[Dict[str, Any]]
    round_count: int = 0
    max_rounds: int = 2
    system_prompt: str = ""
    accumulated_context: List[str] = None

    def __post_init__(self):
        if self.accumulated_context is None:
            self.accumulated_context = []

    def can_continue(self) -> bool:
        """Check if we can continue to next round"""
        return self.round_count < self.max_rounds

    def add_tool_context(self, context: str):
        """Add tool result context for future rounds"""
        if context and context not in self.accumulated_context:
            self.accumulated_context.append(context)


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Guidelines:
- **Content Search**: Use the search tool for questions about specific course content or detailed educational materials
- **Course Outline**: Use the outline tool for questions about course structure, lesson lists, or course overview
- **Sequential Tool Usage**: You can use tools multiple times (up to 2 rounds) to gather comprehensive information for complex queries
- **Multi-step Reasoning**: For complex questions, use initial tool results to inform follow-up tool searches
- Synthesize tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Sequential Tool Usage Examples:
- "Find a course that covers the same topic as lesson 4 of course X" → First get outline of course X to find lesson 4 title → Then search for courses covering that topic
- Comparing content across multiple courses → Search each course separately then synthesize
- Multi-part questions requiring different types of information

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course content questions**: Use search tool(s) as needed, then answer
- **Course outline/structure questions**: Use outline tool(s) as needed, then answer
- **Complex queries**: Break down into multiple tool uses if beneficial
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the outline"

For outline-related queries, always return:
- Course title
- Course link
- Complete lesson list with lesson numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

        # Pre-build base API parameters
        self.base_params = {"model": self.model, "temperature": 0, "max_tokens": 800}

    def generate_response(
        self,
        query: str,
        conversation_history: Optional[str] = None,
        tools: Optional[List] = None,
        tool_manager=None,
    ) -> str:
        """
        Generate AI response with optional sequential tool usage and conversation context.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Initialize conversation state
        initial_system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        state = ConversationState(
            original_query=query,
            messages=[{"role": "user", "content": query}],
            system_prompt=initial_system_content,
        )

        # Execute conversation rounds
        return self._execute_conversation_rounds(state, tools, tool_manager)

    def _execute_conversation_rounds(
        self, state: ConversationState, tools: Optional[List], tool_manager
    ) -> str:
        """
        Execute multiple conversation rounds with tool usage.

        Args:
            state: Conversation state to manage across rounds
            tools: Available tools for Claude
            tool_manager: Manager to execute tools

        Returns:
            Final response after all rounds
        """

        while state.can_continue():
            state.round_count += 1

            # Update system prompt with accumulated context for subsequent rounds
            if state.round_count > 1 and state.accumulated_context:
                context_summary = "\n".join(state.accumulated_context)
                enhanced_prompt = (
                    f"{state.system_prompt}\n\n"
                    f"Previous tool results from this query:\n{context_summary}\n\n"
                    f"This is round {state.round_count} of {state.max_rounds}. "
                    f"{'This is your final round of tool usage.' if state.round_count == state.max_rounds else 'You may use tools again if needed for follow-up searches.'}"
                )
            else:
                enhanced_prompt = state.system_prompt

            # Prepare API call parameters
            api_params = {
                **self.base_params,
                "messages": state.messages.copy(),
                "system": enhanced_prompt,
            }

            # Add tools if available
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            # Get response from Claude
            try:
                response = self.client.messages.create(**api_params)
            except Exception as e:
                # Handle API errors gracefully
                if state.accumulated_context:
                    return self._synthesize_final_response(state)
                else:
                    raise e

            # Check if Claude used tools
            if response.stop_reason == "tool_use" and tool_manager:
                # Execute tools and get the follow-up response for this round
                follow_up_response = self._handle_tool_execution_for_round(
                    response, state, api_params, tool_manager
                )

                # Check if we should continue to next round
                if not self._response_suggests_continuation(follow_up_response):
                    # Claude indicated completion - return this response
                    return follow_up_response
                elif not state.can_continue():
                    # Max rounds reached but Claude wants to continue - synthesize final response
                    return self._synthesize_final_response(state)

                # Add the final response from this round to continue to next round
                # (Tool use and results were already added in _handle_tool_execution_for_round)
                state.messages.append(
                    {"role": "assistant", "content": follow_up_response}
                )
            else:
                # No tool use - return this response
                return response.content[0].text

        # If we've exhausted all rounds, synthesize final response
        return self._synthesize_final_response(state)

    def _handle_tool_execution_for_round(
        self,
        initial_response,
        state: ConversationState,
        base_params: Dict[str, Any],
        tool_manager,
    ) -> str:
        """
        Handle tool execution within a conversation round.

        Args:
            initial_response: The response containing tool use requests
            state: Current conversation state
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Response text after tool execution
        """
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    # Add tool result to state context for future rounds
                    state.add_tool_context(f"{content_block.name}: {tool_result}")

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors
                    error_msg = f"Tool execution failed: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_msg,
                        }
                    )

        # Add tool use response and results to state messages
        state.messages.append(
            {"role": "assistant", "content": initial_response.content}
        )
        if tool_results:
            state.messages.append({"role": "user", "content": tool_results})

        # Prepare follow-up API call without tools for this round's final response
        follow_up_params = {
            **self.base_params,
            "messages": state.messages.copy(),
            "system": base_params["system"],
        }

        # Get follow-up response
        try:
            follow_up_response = self.client.messages.create(**follow_up_params)
            return follow_up_response.content[0].text
        except Exception as e:
            # If follow-up fails, try to synthesize from what we have
            if state.accumulated_context:
                return self._synthesize_final_response(state)
            else:
                raise e

    def _response_suggests_continuation(self, response: str) -> bool:
        """
        Heuristic to determine if Claude's response suggests more information is needed.
        """
        continuation_indicators = [
            "let me search for more",
            "i need to find",
            "let me look up",
            "i should check",
            "additional information",
            "more details needed",
            "need to search for more",
            "search for more specific",
        ]

        response_lower = response.lower()
        return any(indicator in response_lower for indicator in continuation_indicators)

    def _synthesize_final_response(self, state: ConversationState) -> str:
        """
        Create a synthesis response when rounds are exhausted or errors occur.
        """
        if not state.accumulated_context:
            return "I apologize, but I wasn't able to gather the information needed to answer your question."

        # Build synthesis prompt
        context_summary = "\n\n".join(state.accumulated_context)
        synthesis_prompt = f"""Based on the information I gathered:

{context_summary}

Please provide a comprehensive answer to: {state.original_query}"""

        # Prepare final synthesis call without tools
        final_params = {
            **self.base_params,
            "messages": [{"role": "user", "content": synthesis_prompt}],
            "system": "You are an AI assistant. Synthesize the provided information to answer the user's question comprehensively and accurately. Provide only the direct answer without mentioning the synthesis process.",
        }

        try:
            final_response = self.client.messages.create(**final_params)
            return final_response.content[0].text
        except Exception:
            # Fallback if synthesis fails
            return f"Based on my search, here's what I found:\n\n{context_summary}"

    def _handle_tool_execution(
        self, initial_response, base_params: Dict[str, Any], tool_manager
    ):
        """
        Legacy method for backwards compatibility.
        Handle execution of tool calls and get follow-up response (single round only).

        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools

        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()

        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})

        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                try:
                    tool_result = tool_manager.execute_tool(
                        content_block.name, **content_block.input
                    )

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": tool_result,
                        }
                    )
                except Exception as e:
                    # Handle tool execution errors
                    error_msg = f"Tool execution failed: {str(e)}"
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": content_block.id,
                            "content": error_msg,
                        }
                    )

        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})

        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"],
        }

        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
