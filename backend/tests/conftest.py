"""
Pytest configuration and fixtures for RAG system testing
"""

import os
import sys
import tempfile
from unittest.mock import Mock

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from models import Course
from models import CourseChunk
from models import Lesson
from vector_store import SearchResults


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
                lesson_link="https://example.com/ml-course/lesson-1",
            ),
            Lesson(
                lesson_number=2,
                title="Types of Machine Learning",
                lesson_link="https://example.com/ml-course/lesson-2",
            ),
            Lesson(
                lesson_number=3,
                title="Linear Regression",
                lesson_link="https://example.com/ml-course/lesson-3",
            ),
        ],
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
                lesson_link="https://example.com/python-course/lesson-1",
            ),
            Lesson(
                lesson_number=2,
                title="Decorators and Context Managers",
                lesson_link="https://example.com/python-course/lesson-2",
            ),
        ],
    )


@pytest.fixture
def sample_course_chunks(sample_course):
    """Sample course chunks for testing"""
    return [
        CourseChunk(
            content="Machine learning is a subset of artificial intelligence that focuses on algorithms that can learn from data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Supervised learning involves training models with labeled data to make predictions on new, unseen data.",
            course_title=sample_course.title,
            lesson_number=1,
            chunk_index=1,
        ),
        CourseChunk(
            content="Linear regression is a fundamental supervised learning technique used for predicting continuous values.",
            course_title=sample_course.title,
            lesson_number=3,
            chunk_index=2,
        ),
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
    mock_final_response.content = [
        Mock(
            text="Machine learning is a subset of AI that focuses on learning from data."
        )
    ]
    mock_final_response.stop_reason = "end_turn"

    # Configure the client to return different responses on consecutive calls
    mock_client.messages.create.side_effect = [
        mock_initial_response,
        mock_final_response,
    ]

    return mock_client


@pytest.fixture
def search_results_with_data():
    """Sample SearchResults with mock data"""
    return SearchResults(
        documents=[
            "Machine learning is a subset of artificial intelligence.",
            "Supervised learning uses labeled data for training.",
        ],
        metadata=[
            {
                "course_title": "Introduction to Machine Learning",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "Introduction to Machine Learning",
                "lesson_number": 1,
                "chunk_index": 1,
            },
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    """Empty SearchResults for testing no-match scenarios"""
    return SearchResults(documents=[], metadata=[], distances=[])


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
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
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
    mock_manager.get_conversation_history.return_value = (
        "User: Previous question\nAssistant: Previous answer"
    )
    mock_manager.add_exchange.return_value = None
    return mock_manager
