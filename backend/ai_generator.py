import anthropic
from typing import List, Optional, Dict, Any


class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""

    MAX_TOOL_ROUNDS = 2

    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Search Tool Usage:
- Use the search tool **only** for questions about specific course content or detailed educational materials
- You may search **up to 2 times** per query if needed (e.g., to compare content across courses or to refine an initial search)
- If your first search does not fully answer the question, you may search again with a different query or course filter
- Synthesize search results into accurate, fact-based responses
- If search yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


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
        Generate AI response with optional tool usage and conversation context.

        Supports up to MAX_TOOL_ROUNDS sequential tool-call rounds. After all
        rounds are exhausted (or on tool error), a drain call without tools
        forces a final text response.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.SYSTEM_PROMPT
        )

        messages = [{"role": "user", "content": query}]

        for round in range(self.MAX_TOOL_ROUNDS):
            # Build API params with tools available so Claude can search again
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content,
            }
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)

            # If Claude didn't request a tool, return the text directly
            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text

            # Execute all tool calls in this round
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            error_occurred = False
            for content_block in response.content:
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
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": content_block.id,
                                "content": str(e),
                                "is_error": True,
                            }
                        )
                        error_occurred = True

            if tool_results:
                messages.append({"role": "user", "content": tool_results})

            if error_occurred:
                break

        # Drain call without tools to force a text response
        drain_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content,
        }
        final_response = self.client.messages.create(**drain_params)
        return final_response.content[0].text
