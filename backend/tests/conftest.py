"""
Pytest configuration and fixtures for RAG system testing
"""
import pytest
import json
import tempfile
import os
from unittest.mock import Mock, MagicMock, AsyncMock
from typing import Dict, List, Any

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fastapi import FastAPI
from fastapi.testclient import TestClient
from models import Course, Lesson, CourseChunk
from vector_store import SearchResults
from config import Config


@pytest.fixture
def sample_course():
    """Sample course data for testing"""
    return Course(
        title="Introduction to Machine Learning",
        course_link="https://example.com/ml-course",
        instructor="Dr. Jane Smith",
        lessons=[
            Lesson(
                lesson_number=1,
                title="What is Machine Learning?",
                lesson_link="https://example.com/ml-course/lesson-1"
            ),
            Lesson(
                lesson_number=2,
                title="Types of Machine Learning",
                lesson_link="https://example.com/ml-course/lesson-2"
            ),
            Lesson(
                lesson_number=3,
                title="Linear Regression",
                lesson_link="https://example.com/ml-course/lesson-3"
            )
        ]
    )


@pytest.fixture
def another_sample_course():
    """Another sample course for testing multiple courses"""
    return Course(
        title="Advanced Python Programming",
        course_link="https://example.com/python-course",
        instructor="Prof. John Doe",
        lessons=[
            Lesson(
                lesson_number=1,
                title="Object-Oriented Programming",
                lesson_link="https://example.com/python-course/lesson-1"
            ),
            Lesson(
                lesson_number=2,
                title="Decorators and Context Managers",
                lesson_link="https://example.com/python-course/lesson-2"
            )
        ]
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Supervised learning involves training models with labeled data to make predictions on new, unseen data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1
        ),
        CourseChunk(
            content="Linear regression is a fundamental supervised learning technique used for predicting continuous values.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def mock_vector_store():
    """Mock VectorStore for isolated testing"""
    mock_store = Mock()
    mock_store.max_results = 5

    # Configure common methods
    mock_store._resolve_course_name.return_value = "Introduction to Machine Learning"
    mock_store.get_lesson_link.return_value = "https://example.com/ml-course/lesson-1"

    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Mock Anthropic client for testing AI interactions"""
    mock_client = Mock()

    # Mock response structure
    mock_response = Mock()
    mock_response.content = [Mock(text="This is a test response.")]
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response

    return mock_client


@pytest.fixture
def mock_anthropic_client_with_tool_use():
    """Mock Anthropic client that simulates tool usage"""
    mock_client = Mock()

    # Mock initial tool use response
    mock_initial_response = Mock()
    mock_initial_response.stop_reason = "tool_use"

    # Mock tool use content block
    tool_use_block = Mock()
    tool_use_block.type = "tool_use"
    tool_use_block.name = "search_course_content"
    tool_use_block.id = "tool_123"
    tool_use_block.input = {"query": "machine learning"}

    mock_initial_response.content = [tool_use_block]

    # Mock final response after tool execution
    mock_final_response = Mock()
    mock_final_response.content = [Mock(text="Machine learning is a subset of AI that focuses on learning from data.")]
    mock_final_response.stop_reason = "end_turn"

    # Configure the client to return different responses on consecutive calls
    mock_client.messages.create.side_effect = [mock_initial_response, mock_final_response]

    return mock_client


@pytest.fixture
def search_results_with_data():
    """Sample SearchResults with mock data"""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Supervised learning uses labeled data for training."
        ],
        metadata=[
            {
                "course_title": "Introduction to Machine Learning",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Introduction to Machine Learning",
                "lesson_number": 1,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for testing no-match scenarios"""
    return SearchResults(
        documents=[],
        metadata=[],
        distances=[]
    )


@pytest.fixture
def search_results_with_error():
    """SearchResults with error for testing error scenarios"""
    return SearchResults.empty("Database connection failed")


@pytest.fixture
def test_config():
    """Test configuration with safe defaults"""
    config = Config()
    # Override sensitive settings for testing
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.CHROMA_PATH = "./test_chroma_db"
    config.MAX_RESULTS = 3
    config.MAX_HISTORY = 2
    return config


@pytest.fixture
def temp_directory():
    """Temporary directory for test file operations"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_tool_manager():
    """Mock ToolManager for testing tool interactions"""
    mock_manager = Mock()
    mock_manager.get_tool_definitions.return_value = [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        }
    ]
    mock_manager.execute_tool.return_value = "Mock search results"
    mock_manager.get_last_sources.return_value = ["Test Course - Lesson 1"]
    mock_manager.get_last_source_links.return_value = ["https://example.com/lesson-1"]
    mock_manager.reset_sources.return_value = None

    return mock_manager


@pytest.fixture
def mock_session_manager():
    """Mock SessionManager for testing session handling"""
    mock_manager = Mock()
    mock_manager.get_conversation_history.return_value = "User: Previous question\nAssistant: Previous answer"
    mock_manager.add_exchange.return_value = None
    mock_manager.create_session.return_value = "test-session-123"
    mock_manager.clear_session.return_value = None
    return mock_manager


@pytest.fixture
def mock_rag_system():
    """Mock RAGSystem for API testing"""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a test answer about machine learning.",
        ["Test Course - Lesson 1", "Test Course - Lesson 2"],
        ["https://example.com/lesson-1", "https://example.com/lesson-2"]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Introduction to Machine Learning", "Advanced Python Programming"]
    }

    # Add session manager to mock rag system
    mock_session_manager_instance = Mock()
    mock_session_manager_instance.create_session.return_value = "test-session-456"
    mock_session_manager_instance.clear_session.return_value = None
    mock_rag.session_manager = mock_session_manager_instance

    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """FastAPI test application with mocked dependencies"""
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    from typing import List, Optional

    # Create test app without static file mounting to avoid import issues
    app = FastAPI(title="Test Course Materials RAG System")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Define request/response models locally
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class QueryResponse(BaseModel):
        answer: str
        sources: List[str]
        source_links: List[Optional[str]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]

    class NewSessionResponse(BaseModel):
        session_id: str
        message: str

    class ClearSessionResponse(BaseModel):
        success: bool
        message: str

    # Define API endpoints inline
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()

            answer, sources, source_links = mock_rag_system.query(request.query, session_id)

            return QueryResponse(
                answer=answer,
                sources=sources,
                source_links=source_links,
                session_id=session_id
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/sessions/new", response_model=NewSessionResponse)
    async def create_new_session():
        try:
            session_id = mock_rag_system.session_manager.create_session()
            return NewSessionResponse(
                session_id=session_id,
                message="New session created successfully"
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/api/sessions/{session_id}/clear", response_model=ClearSessionResponse)
    async def clear_session(session_id: str):
        try:
            mock_rag_system.session_manager.clear_session(session_id)
            return ClearSessionResponse(
                success=True,
                message=f"Session {session_id} cleared successfully"
            )
        except Exception as e:
            from fastapi import HTTPException
            raise HTTPException(status_code=500, detail=str(e))

    return app


@pytest.fixture
def test_client(test_app):
    """FastAPI test client"""
    return TestClient(test_app)


@pytest.fixture
def api_test_data():
    """Common test data for API tests"""
    return {
        "valid_query": {
            "query": "What is machine learning?",
            "session_id": "test-session-123"
        },
        "query_without_session": {
            "query": "Explain supervised learning"
        },
        "invalid_query": {
            "invalid_field": "This should fail validation"
        },
        "expected_response": {
            "answer": "This is a test answer about machine learning.",
            "sources": ["Test Course - Lesson 1", "Test Course - Lesson 2"],
            "source_links": ["https://example.com/lesson-1", "https://example.com/lesson-2"],
            "session_id": "test-session-123"
        },
        "expected_courses": {
            "total_courses": 2,
            "course_titles": ["Introduction to Machine Learning", "Advanced Python Programming"]
        }
    }