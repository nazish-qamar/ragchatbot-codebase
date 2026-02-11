"""Tests for AIGenerator (ai_generator.py)."""

from unittest.mock import MagicMock, patch, call
import pytest


# ---------------------------------------------------------------------------
# Helpers – build mock Anthropic response objects
# ---------------------------------------------------------------------------

def _text_block(text="Hello from Claude"):
    block = MagicMock()
    block.type = "text"
    block.text = text
    return block


def _tool_use_block(name="search_course_content", tool_input=None, tool_id="tool_1"):
    block = MagicMock()
    block.type = "tool_use"
    block.name = name
    block.input = tool_input or {"query": "RAG"}
    block.id = tool_id
    return block


def _make_response(content_blocks, stop_reason="end_turn"):
    resp = MagicMock()
    resp.content = content_blocks
    resp.stop_reason = stop_reason
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAIGeneratorDirectResponse:

    @patch("ai_generator.anthropic")
    def test_direct_response_no_tools(self, mock_anthropic_mod):
        """When stop_reason != 'tool_use', returns content[0].text directly."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_response(
            [_text_block("Direct answer")], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(query="What is AI?")
        assert result == "Direct answer"

    @patch("ai_generator.anthropic")
    def test_calls_api_with_tools_when_provided(self, mock_anthropic_mod):
        """Tools list is passed to the Anthropic API call."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_response(
            [_text_block()], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        tools = [{"name": "search_course_content", "description": "Search", "input_schema": {}}]
        gen.generate_response(query="q", tools=tools)

        api_call_kwargs = mock_client.messages.create.call_args
        assert "tools" in api_call_kwargs.kwargs or (
            api_call_kwargs[1] and "tools" in api_call_kwargs[1]
        )

    @patch("ai_generator.anthropic")
    def test_conversation_history_in_system_prompt(self, mock_anthropic_mod):
        """When history is provided, it's appended to the system prompt."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.return_value = _make_response(
            [_text_block()], stop_reason="end_turn"
        )

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", conversation_history="User: hi\nAssistant: hello")

        call_kwargs = mock_client.messages.create.call_args
        system_value = call_kwargs.kwargs.get("system") or call_kwargs[1].get("system")
        assert "Previous conversation" in system_value
        assert "User: hi" in system_value


class TestAIGeneratorToolExecution:

    @patch("ai_generator.anthropic")
    def test_tool_use_triggers_execution(self, mock_anthropic_mod):
        """When stop_reason == 'tool_use', calls tool_manager.execute_tool()."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        # First call returns tool_use, second call returns final text
        mock_client.messages.create.side_effect = [
            _make_response(
                [_tool_use_block("search_course_content", {"query": "RAG"})],
                stop_reason="tool_use",
            ),
            _make_response([_text_block("Final answer")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "search results text"

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(
            query="q",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_mgr,
        )

        tool_mgr.execute_tool.assert_called_once_with(
            "search_course_content", query="RAG"
        )
        assert result == "Final answer"

    @patch("ai_generator.anthropic")
    def test_intermediate_call_includes_tools(self, mock_anthropic_mod):
        """The follow-up API call after tool execution DOES include 'tools' (loop keeps tools available)."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response(
                [_tool_use_block()], stop_reason="tool_use"
            ),
            _make_response([_text_block("done")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "results"

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", tools=[{"name": "t"}], tool_manager=tool_mgr)

        # Second call (loop iteration 1) SHOULD contain 'tools'
        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        combined = {**second_call_kwargs.kwargs}
        assert "tools" in combined

    @patch("ai_generator.anthropic")
    def test_second_call_includes_tool_results(self, mock_anthropic_mod):
        """The follow-up messages include the tool result content."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response(
                [_tool_use_block(tool_id="abc123")], stop_reason="tool_use"
            ),
            _make_response([_text_block("final")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "tool output here"

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", tools=[{"name": "t"}], tool_manager=tool_mgr)

        second_call_kwargs = mock_client.messages.create.call_args_list[1]
        messages = second_call_kwargs.kwargs.get("messages") or second_call_kwargs[1].get("messages")
        # Last message should be the tool results
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        assert any(
            item["type"] == "tool_result" and item["content"] == "tool output here"
            for item in tool_result_msg["content"]
        )

    @patch("ai_generator.anthropic")
    def test_api_error_propagates(self, mock_anthropic_mod):
        """If client.messages.create raises, the exception propagates."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = RuntimeError("API down")

        gen = AIGenerator(api_key="fake", model="test-model")
        with pytest.raises(RuntimeError, match="API down"):
            gen.generate_response(query="q")


class TestAIGeneratorMultiRound:

    @patch("ai_generator.anthropic")
    def test_two_round_tool_calling(self, mock_anthropic_mod):
        """Two tool rounds produce 3 API calls and 2 execute_tool calls."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response(
                [_tool_use_block("search_course_content", {"query": "RAG"}, "t1")],
                stop_reason="tool_use",
            ),
            _make_response(
                [_tool_use_block("search_course_content", {"query": "MCP"}, "t2")],
                stop_reason="tool_use",
            ),
            _make_response([_text_block("Combined answer")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.side_effect = ["results 1", "results 2"]

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(
            query="Compare RAG and MCP",
            tools=[{"name": "search_course_content"}],
            tool_manager=tool_mgr,
        )

        assert mock_client.messages.create.call_count == 3
        assert tool_mgr.execute_tool.call_count == 2
        assert result == "Combined answer"

    @patch("ai_generator.anthropic")
    def test_tools_available_in_intermediate_rounds(self, mock_anthropic_mod):
        """Calls at index 0 and 1 have 'tools'; call at index 2 (drain) does NOT."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block(tool_id="t1")], stop_reason="tool_use"),
            _make_response([_tool_use_block(tool_id="t2")], stop_reason="tool_use"),
            _make_response([_text_block("done")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.side_effect = ["r1", "r2"]

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", tools=[{"name": "t"}], tool_manager=tool_mgr)

        calls = mock_client.messages.create.call_args_list
        # Calls 0 and 1 (loop iterations) should have tools
        assert "tools" in calls[0].kwargs
        assert "tools" in calls[1].kwargs
        # Call 2 (drain) should NOT have tools
        assert "tools" not in calls[2].kwargs

    @patch("ai_generator.anthropic")
    def test_max_rounds_drain_call_omits_tools(self, mock_anthropic_mod):
        """The drain call after exhausting all rounds has no 'tools' or 'tool_choice'."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block(tool_id="t1")], stop_reason="tool_use"),
            _make_response([_tool_use_block(tool_id="t2")], stop_reason="tool_use"),
            _make_response([_text_block("final")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.side_effect = ["r1", "r2"]

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", tools=[{"name": "t"}], tool_manager=tool_mgr)

        drain_call = mock_client.messages.create.call_args_list[2]
        assert "tools" not in drain_call.kwargs
        assert "tool_choice" not in drain_call.kwargs

    @patch("ai_generator.anthropic")
    def test_tool_error_breaks_loop_gracefully(self, mock_anthropic_mod):
        """Tool execution error produces is_error tool_result and doesn't raise."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response(
                [_tool_use_block(tool_id="t1")], stop_reason="tool_use"
            ),
            _make_response([_text_block("Recovered answer")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.side_effect = RuntimeError("search failed")

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(
            query="q",
            tools=[{"name": "t"}],
            tool_manager=tool_mgr,
        )

        assert result == "Recovered answer"
        # The drain call's messages should contain a tool_result with is_error
        drain_call = mock_client.messages.create.call_args_list[1]
        messages = drain_call.kwargs["messages"]
        tool_result_msg = messages[-1]
        assert tool_result_msg["role"] == "user"
        error_results = [
            item for item in tool_result_msg["content"]
            if item.get("is_error") is True
        ]
        assert len(error_results) == 1
        assert "search failed" in error_results[0]["content"]

    @patch("ai_generator.anthropic")
    def test_context_accumulates_across_rounds(self, mock_anthropic_mod):
        """After 2 tool rounds, the drain call's messages list has 5 entries."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block(tool_id="t1")], stop_reason="tool_use"),
            _make_response([_tool_use_block(tool_id="t2")], stop_reason="tool_use"),
            _make_response([_text_block("done")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.side_effect = ["r1", "r2"]

        gen = AIGenerator(api_key="fake", model="test-model")
        gen.generate_response(query="q", tools=[{"name": "t"}], tool_manager=tool_mgr)

        drain_call = mock_client.messages.create.call_args_list[2]
        messages = drain_call.kwargs["messages"]
        # user query, assistant tool_use #1, user tool_results #1,
        # assistant tool_use #2, user tool_results #2
        assert len(messages) == 5
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        assert messages[3]["role"] == "assistant"
        assert messages[4]["role"] == "user"

    @patch("ai_generator.anthropic")
    def test_single_round_no_unnecessary_drain(self, mock_anthropic_mod):
        """1 tool round + text response = exactly 2 API calls (no drain)."""
        from ai_generator import AIGenerator

        mock_client = MagicMock()
        mock_anthropic_mod.Anthropic.return_value = mock_client

        mock_client.messages.create.side_effect = [
            _make_response([_tool_use_block()], stop_reason="tool_use"),
            _make_response([_text_block("answer")], stop_reason="end_turn"),
        ]

        tool_mgr = MagicMock()
        tool_mgr.execute_tool.return_value = "results"

        gen = AIGenerator(api_key="fake", model="test-model")
        result = gen.generate_response(
            query="q", tools=[{"name": "t"}], tool_manager=tool_mgr,
        )

        assert result == "answer"
        assert mock_client.messages.create.call_count == 2
