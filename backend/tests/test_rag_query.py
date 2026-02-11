"""Tests for RAGSystem.query() and the FastAPI /api/query endpoint."""

from unittest.mock import MagicMock, patch, PropertyMock
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rag_system(mock_config):
    """Build a RAGSystem with all heavy dependencies mocked out."""
    with (
        patch("rag_system.DocumentProcessor"),
        patch("rag_system.VectorStore") as MockVS,
        patch("rag_system.AIGenerator") as MockAI,
    ):
        from rag_system import RAGSystem

        system = RAGSystem(mock_config)
        # Expose the mocks for assertions
        system._mock_ai = system.ai_generator
        system._mock_vs = system.vector_store
        return system


# ---------------------------------------------------------------------------
# RAGSystem.query tests
# ---------------------------------------------------------------------------

class TestRAGSystemQuery:

    def test_query_returns_response_and_sources(self, mock_config):
        """Full mock flow: query -> ai_generator -> tool_manager -> response + sources."""
        rag = _make_rag_system(mock_config)
        rag._mock_ai.generate_response.return_value = "Here is the answer"
        # Simulate that the tool_manager already has sources populated
        rag.tool_manager.get_last_sources = MagicMock(
            return_value=[{"name": "Course A", "link": None}]
        )
        rag.tool_manager.reset_sources = MagicMock()

        response, sources = rag.query("What is RAG?")
        assert response == "Here is the answer"
        assert len(sources) == 1
        assert sources[0]["name"] == "Course A"

    def test_query_creates_prompt_with_question(self, mock_config):
        """The prompt sent to AI includes the user's question."""
        rag = _make_rag_system(mock_config)
        rag._mock_ai.generate_response.return_value = "answer"
        rag.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag.tool_manager.reset_sources = MagicMock()

        rag.query("Explain embeddings")
        call_kwargs = rag._mock_ai.generate_response.call_args
        query_arg = call_kwargs.kwargs.get("query") or call_kwargs[1].get("query", call_kwargs[0][0])
        assert "Explain embeddings" in query_arg

    def test_query_passes_tools_to_generator(self, mock_config):
        """Tool definitions from ToolManager are passed to generate_response()."""
        rag = _make_rag_system(mock_config)
        rag._mock_ai.generate_response.return_value = "ans"
        rag.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag.tool_manager.reset_sources = MagicMock()

        rag.query("q")
        call_kwargs = rag._mock_ai.generate_response.call_args
        tools_arg = call_kwargs.kwargs.get("tools")
        assert tools_arg is not None
        assert isinstance(tools_arg, list)

    def test_query_resets_sources_after_retrieval(self, mock_config):
        """After query, reset_sources is called."""
        rag = _make_rag_system(mock_config)
        rag._mock_ai.generate_response.return_value = "ans"
        rag.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag.tool_manager.reset_sources = MagicMock()

        rag.query("q")
        rag.tool_manager.reset_sources.assert_called_once()

    def test_query_with_session_includes_history(self, mock_config):
        """When session_id provided, conversation history is passed."""
        rag = _make_rag_system(mock_config)
        rag._mock_ai.generate_response.return_value = "ans"
        rag.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag.tool_manager.reset_sources = MagicMock()

        # Pre-populate a session
        sid = rag.session_manager.create_session()
        rag.session_manager.add_exchange(sid, "Hi", "Hello!")

        rag.query("follow up", session_id=sid)
        call_kwargs = rag._mock_ai.generate_response.call_args
        history_arg = call_kwargs.kwargs.get("conversation_history")
        assert history_arg is not None
        assert "Hi" in history_arg

    def test_query_updates_session_after_response(self, mock_config):
        """After query, session has the exchange recorded."""
        rag = _make_rag_system(mock_config)
        rag._mock_ai.generate_response.return_value = "the answer"
        rag.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag.tool_manager.reset_sources = MagicMock()

        sid = rag.session_manager.create_session()
        rag.query("my question", session_id=sid)

        history = rag.session_manager.get_conversation_history(sid)
        assert "my question" in history
        assert "the answer" in history

    def test_query_without_session(self, mock_config):
        """Works correctly with session_id=None."""
        rag = _make_rag_system(mock_config)
        rag._mock_ai.generate_response.return_value = "works"
        rag.tool_manager.get_last_sources = MagicMock(return_value=[])
        rag.tool_manager.reset_sources = MagicMock()

        response, sources = rag.query("q", session_id=None)
        assert response == "works"


# ---------------------------------------------------------------------------
# FastAPI endpoint tests
# ---------------------------------------------------------------------------

class TestFastAPIQueryEndpoint:

    @pytest.fixture
    def test_client(self):
        """Create a FastAPI TestClient with the rag_system patched at module level."""
        import app as app_module
        from fastapi.testclient import TestClient

        mock_rag = MagicMock()
        mock_rag.query.return_value = ("Test answer", [{"name": "Src", "link": None}])
        mock_rag.session_manager.create_session.return_value = "session_1"

        # Patch the already-instantiated module-level rag_system
        with patch.object(app_module, "rag_system", mock_rag):
            client = TestClient(app_module.app, raise_server_exceptions=False)
            client._mock_rag = mock_rag
            yield client

    def test_api_endpoint_happy_path(self, test_client):
        """POST /api/query returns proper QueryResponse JSON."""
        resp = test_client.post("/api/query", json={"query": "What is AI?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data

    def test_api_endpoint_returns_500_on_error(self, test_client):
        """FastAPI endpoint converts exceptions to HTTP 500."""
        test_client._mock_rag.query.side_effect = RuntimeError("boom")
        resp = test_client.post(
            "/api/query", json={"query": "fail", "session_id": "session_1"}
        )
        assert resp.status_code == 500
