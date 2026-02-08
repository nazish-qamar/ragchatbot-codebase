import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from pydantic import BaseModel
from typing import List, Optional


def create_test_app(rag_system):
    """Create a FastAPI app that mirrors production routes without static file mounting.

    This avoids importing backend/app.py directly, which would fail in test
    environments due to the StaticFiles mount referencing ../frontend.
    """
    app = FastAPI(title="Course Materials RAG System - Test")

    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = rag_system.session_manager.create_session()
            answer, sources = rag_system.query(request.query, session_id)
            return QueryResponse(answer=answer, sources=sources, session_id=session_id)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"],
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def mock_rag_system():
    """Mock RAG system with sensible defaults for all endpoints."""
    rag = MagicMock()
    rag.query.return_value = ("This is a test answer.", ["Source 1", "Source 2"])
    rag.session_manager.create_session.return_value = "test_session_1"
    rag.get_course_analytics.return_value = {
        "total_courses": 3,
        "course_titles": ["Course A", "Course B", "Course C"],
    }
    return rag


@pytest.fixture
def app(mock_rag_system):
    """Test FastAPI application wired to the mock RAG system."""
    return create_test_app(mock_rag_system)


@pytest.fixture
def client(app):
    """Test client for making HTTP requests against the test app."""
    return TestClient(app)
