"""Tests for CourseSearchTool and ToolManager (search_tools.py)."""

from unittest.mock import MagicMock
from vector_store import SearchResults


def make_search_results(documents=None, metadata=None, distances=None, error=None):
    return SearchResults(
        documents=documents or [],
        metadata=metadata or [],
        distances=distances or [],
        error=error,
    )


# ── CourseSearchTool.execute ─────────────────────────────────────────────

class TestCourseSearchToolExecute:

    def test_execute_returns_formatted_results(self, search_tool):
        """Successful search returns formatted string with course/lesson headers."""
        result = search_tool.execute(query="RAG")
        assert "[Intro to AI - Lesson 1]" in result
        assert "Chunk about RAG pipelines" in result
        assert "[Intro to AI - Lesson 2]" in result
        assert "Chunk about embeddings" in result

    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        """Passes course_name through to VectorStore.search()."""
        search_tool.execute(query="RAG", course_name="MCP")
        mock_vector_store.search.assert_called_once_with(
            query="RAG", course_name="MCP", lesson_number=None
        )

    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        """Passes lesson_number through to VectorStore.search()."""
        search_tool.execute(query="RAG", lesson_number=3)
        mock_vector_store.search.assert_called_once_with(
            query="RAG", course_name=None, lesson_number=3
        )

    def test_execute_empty_results(self, search_tool, mock_vector_store):
        """Returns 'No relevant content found' message when is_empty() is True."""
        mock_vector_store.search.return_value = make_search_results()
        result = search_tool.execute(query="nonexistent")
        assert "No relevant content found" in result

    def test_execute_empty_results_with_filters(self, search_tool, mock_vector_store):
        """Empty-results message includes filter info."""
        mock_vector_store.search.return_value = make_search_results()
        result = search_tool.execute(query="x", course_name="MCP", lesson_number=2)
        assert "MCP" in result
        assert "lesson 2" in result

    def test_execute_error_from_store(self, search_tool, mock_vector_store):
        """Returns error string when SearchResults.error is set."""
        mock_vector_store.search.return_value = make_search_results(
            error="No course found matching 'xyz'"
        )
        result = search_tool.execute(query="xyz")
        assert result == "No course found matching 'xyz'"

    def test_execute_populates_last_sources(self, search_tool):
        """After execution, last_sources contains source dicts with name/link."""
        search_tool.execute(query="RAG")
        sources = search_tool.last_sources
        assert len(sources) == 2
        assert sources[0]["name"] == "Intro to AI - Lesson 1"
        assert sources[0]["link"] == "https://example.com/lesson"


# ── ToolManager ──────────────────────────────────────────────────────────

class TestToolManager:

    def test_execute_delegates_to_search_tool(self, tool_manager, mock_vector_store):
        """ToolManager.execute_tool delegates to CourseSearchTool.execute()."""
        result = tool_manager.execute_tool("search_course_content", query="RAG")
        mock_vector_store.search.assert_called_once()
        assert "[Intro to AI" in result

    def test_get_last_sources(self, tool_manager):
        """Returns sources after search, empty after reset."""
        tool_manager.execute_tool("search_course_content", query="RAG")
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2

        tool_manager.reset_sources()
        assert tool_manager.get_last_sources() == []

    def test_get_tool_definitions(self, tool_manager):
        """Returns a list containing the search tool definition."""
        defs = tool_manager.get_tool_definitions()
        assert len(defs) == 1
        assert defs[0]["name"] == "search_course_content"

    def test_execute_unknown_tool(self, tool_manager):
        """Returns an error message for unknown tool names."""
        result = tool_manager.execute_tool("no_such_tool", query="x")
        assert "not found" in result
