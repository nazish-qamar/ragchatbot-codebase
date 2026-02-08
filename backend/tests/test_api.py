from unittest.mock import MagicMock


class TestQueryEndpoint:
    """Tests for POST /api/query"""

    def test_query_creates_session_when_none_provided(self, client, mock_rag_system):
        response = client.post("/api/query", json={"query": "What is Python?"})

        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "This is a test answer."
        assert data["sources"] == ["Source 1", "Source 2"]
        assert data["session_id"] == "test_session_1"
        mock_rag_system.session_manager.create_session.assert_called_once()

    def test_query_uses_existing_session(self, client, mock_rag_system):
        response = client.post(
            "/api/query",
            json={"query": "Tell me more", "session_id": "existing_session"},
        )

        assert response.status_code == 200
        assert response.json()["session_id"] == "existing_session"
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with(
            "Tell me more", "existing_session"
        )

    def test_query_missing_query_field_returns_422(self, client):
        response = client.post("/api/query", json={})
        assert response.status_code == 422

    def test_query_invalid_json_returns_422(self, client):
        response = client.post(
            "/api/query", content="not json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

    def test_query_passes_query_to_rag_system(self, client, mock_rag_system):
        client.post("/api/query", json={"query": "How does RAG work?"})

        mock_rag_system.query.assert_called_once_with(
            "How does RAG work?", "test_session_1"
        )

    def test_query_rag_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.query.side_effect = Exception("AI service unavailable")

        response = client.post("/api/query", json={"query": "test"})

        assert response.status_code == 500
        assert "AI service unavailable" in response.json()["detail"]

    def test_query_returns_empty_sources(self, client, mock_rag_system):
        mock_rag_system.query.return_value = ("Direct answer.", [])

        response = client.post("/api/query", json={"query": "hi"})

        assert response.status_code == 200
        assert response.json()["sources"] == []

    def test_query_response_schema(self, client):
        response = client.post("/api/query", json={"query": "test"})
        data = response.json()

        assert set(data.keys()) == {"answer", "sources", "session_id"}
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["session_id"], str)


class TestCoursesEndpoint:
    """Tests for GET /api/courses"""

    def test_get_courses(self, client):
        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 3
        assert data["course_titles"] == ["Course A", "Course B", "Course C"]

    def test_get_courses_empty_catalog(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": [],
        }

        response = client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []

    def test_get_courses_error_returns_500(self, client, mock_rag_system):
        mock_rag_system.get_course_analytics.side_effect = Exception("DB error")

        response = client.get("/api/courses")

        assert response.status_code == 500
        assert "DB error" in response.json()["detail"]

    def test_get_courses_response_schema(self, client):
        response = client.get("/api/courses")
        data = response.json()

        assert set(data.keys()) == {"total_courses", "course_titles"}
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)


class TestUnknownRoutes:
    """Tests for undefined routes"""

    def test_root_returns_404_without_static_files(self, client):
        response = client.get("/")
        assert response.status_code == 404

    def test_unknown_api_path_returns_404(self, client):
        response = client.get("/api/nonexistent")
        assert response.status_code == 404

    def test_wrong_method_on_query_returns_405(self, client):
        response = client.get("/api/query")
        assert response.status_code == 405

    def test_wrong_method_on_courses_returns_405(self, client):
        response = client.post("/api/courses", json={})
        assert response.status_code == 405
