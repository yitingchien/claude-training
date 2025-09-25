"""
Tests for RAG system end-to-end content-query handling
"""

import os
import sys
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import Config
from rag_system import RAGSystem


class TestRAGSystem:
    """Test cases for RAG system end-to-end functionality"""

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_init_creates_all_components(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test that RAGSystem initializes all required components"""
        # Act
        rag_system = RAGSystem(test_config)

        # Assert
        mock_doc_proc.assert_called_once_with(
            test_config.CHUNK_SIZE, test_config.CHUNK_OVERLAP
        )
        mock_vector_store.assert_called_once_with(
            test_config.CHROMA_PATH,
            test_config.EMBEDDING_MODEL,
            test_config.MAX_RESULTS,
        )
        mock_ai_gen.assert_called_once_with(
            test_config.ANTHROPIC_API_KEY, test_config.ANTHROPIC_MODEL
        )
        mock_session_mgr.assert_called_once_with(test_config.MAX_HISTORY)

        # Verify tool manager setup
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.ToolManager")
    def test_query_without_session_id(
        self,
        mock_tool_manager_class,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test query processing without session ID"""
        # Arrange
        # Mock the tool manager instance
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        mock_tool_manager.get_last_sources.return_value = ["Test Course - Lesson 1"]
        mock_tool_manager.get_last_source_links.return_value = [
            "https://example.com/lesson1"
        ]
        mock_tool_manager.reset_sources.return_value = None
        mock_tool_manager_class.return_value = mock_tool_manager

        rag_system = RAGSystem(test_config)

        # Mock AI generator response
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = (
            "AI response about machine learning"
        )

        # Mock session manager
        mock_session_mgr_instance = mock_session_mgr.return_value
        mock_session_mgr_instance.get_conversation_history.return_value = None

        # Act
        response, sources, source_links = rag_system.query("What is machine learning?")

        # Assert
        assert response == "AI response about machine learning"
        assert sources == ["Test Course - Lesson 1"]
        assert source_links == ["https://example.com/lesson1"]

        # Verify AI generator was called with correct parameters
        mock_ai_gen_instance.generate_response.assert_called_once()
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert "What is machine learning?" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] is None
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

        # Session manager should not be called for history or exchange
        mock_session_mgr_instance.get_conversation_history.assert_not_called()
        mock_session_mgr_instance.add_exchange.assert_not_called()

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.ToolManager")
    def test_query_with_session_id(
        self,
        mock_tool_manager_class,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test query processing with session ID"""
        # Arrange
        # Mock the tool manager instance
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        mock_tool_manager.get_last_sources.return_value = ["Course A - Lesson 2"]
        mock_tool_manager.get_last_source_links.return_value = [
            "https://example.com/lesson2"
        ]
        mock_tool_manager.reset_sources.return_value = None
        mock_tool_manager_class.return_value = mock_tool_manager

        rag_system = RAGSystem(test_config)

        # Mock AI generator response
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Contextual AI response"

        # Mock session manager
        mock_session_mgr_instance = mock_session_mgr.return_value
        mock_session_mgr_instance.get_conversation_history.return_value = (
            "User: What is supervised learning?\n"
            "Assistant: Supervised learning uses labeled data for training."
        )

        # Act
        response, sources, source_links = rag_system.query(
            "Can you give me an example?", session_id="test_session"
        )

        # Assert
        assert response == "Contextual AI response"
        assert sources == ["Course A - Lesson 2"]
        assert source_links == ["https://example.com/lesson2"]

        # Verify session management
        mock_session_mgr_instance.get_conversation_history.assert_called_once_with(
            "test_session"
        )
        mock_session_mgr_instance.add_exchange.assert_called_once_with(
            "test_session", "Can you give me an example?", "Contextual AI response"
        )

        # Verify AI generator received conversation history
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert call_args[1]["conversation_history"] is not None
        assert "supervised learning" in call_args[1]["conversation_history"]

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.ToolManager")
    def test_query_prompt_construction(
        self,
        mock_tool_manager_class,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test that query prompt is properly constructed"""
        # Arrange
        # Mock the tool manager instance
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.get_last_source_links.return_value = []
        mock_tool_manager.reset_sources.return_value = None
        mock_tool_manager_class.return_value = mock_tool_manager

        rag_system = RAGSystem(test_config)
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Test response"

        # Act
        rag_system.query("What are neural networks?")

        # Assert
        call_args = mock_ai_gen_instance.generate_response.call_args
        query_arg = call_args[1]["query"]
        assert "Answer this question about course materials:" in query_arg
        assert "What are neural networks?" in query_arg

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.ToolManager")
    def test_query_tool_manager_integration(
        self,
        mock_tool_manager_class,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test that query integrates properly with tool manager"""
        # Arrange
        # Mock the tool manager instance
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        mock_tool_manager.get_last_sources.return_value = []
        mock_tool_manager.get_last_source_links.return_value = []
        mock_tool_manager.reset_sources.return_value = None
        mock_tool_manager_class.return_value = mock_tool_manager

        rag_system = RAGSystem(test_config)
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Tool-based response"

        # Act
        rag_system.query("Search for linear algebra concepts")

        # Assert
        call_args = mock_ai_gen_instance.generate_response.call_args

        # Verify tool definitions and tool manager are passed
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None

        # Verify tool manager methods are called
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.get_last_source_links.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.ToolManager")
    def test_query_sources_reset_after_retrieval(
        self,
        mock_tool_manager_class,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test that sources are reset after being retrieved"""
        # Arrange
        # Mock the tool manager instance
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        mock_tool_manager.get_last_sources.return_value = ["Source 1"]
        mock_tool_manager.get_last_source_links.return_value = ["Link 1"]
        mock_tool_manager.reset_sources.return_value = None
        mock_tool_manager_class.return_value = mock_tool_manager

        rag_system = RAGSystem(test_config)
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = "Response"

        # Act
        rag_system.query("Test query")

        # Assert
        # Sources should be retrieved before reset
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.get_last_source_links.assert_called_once()

        # Sources should be reset after retrieval
        mock_tool_manager.reset_sources.assert_called_once()

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_document_success(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
        sample_course,
        sample_course_chunks,
    ):
        """Test successful addition of a course document"""
        # Arrange
        rag_system = RAGSystem(test_config)

        # Mock document processor
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )

        # Mock vector store
        mock_vector_store_instance = mock_vector_store.return_value

        # Act
        course, chunk_count = rag_system.add_course_document("/path/to/course.pdf")

        # Assert
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

        # Verify document processing
        mock_doc_proc_instance.process_course_document.assert_called_once_with(
            "/path/to/course.pdf"
        )

        # Verify vector store operations
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(
            sample_course
        )
        mock_vector_store_instance.add_course_content.assert_called_once_with(
            sample_course_chunks
        )

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_document_failure(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test handling of failures during course document addition"""
        # Arrange
        rag_system = RAGSystem(test_config)

        # Mock document processor to raise exception
        mock_doc_proc_instance = mock_doc_proc.return_value
        mock_doc_proc_instance.process_course_document.side_effect = Exception(
            "Processing failed"
        )

        # Act
        course, chunk_count = rag_system.add_course_document("/path/to/invalid.pdf")

        # Assert
        assert course is None
        assert chunk_count == 0

    @patch("rag_system.os.path.isfile")
    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_add_course_folder_success(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        mock_listdir,
        mock_exists,
        mock_isfile,
        test_config,
        sample_course,
        sample_course_chunks,
    ):
        """Test successful addition of course folder"""
        # Arrange
        rag_system = RAGSystem(test_config)

        # Mock file system
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.pdf", "course2.txt", "not_a_doc.jpg"]

        # Mock isfile to return True for valid documents, False for jpg
        def mock_isfile_side_effect(path):
            return path.endswith((".pdf", ".txt", ".docx"))

        mock_isfile.side_effect = mock_isfile_side_effect

        # Mock document processor
        mock_doc_proc_instance = mock_doc_proc.return_value

        # Create a second course to simulate different documents
        from models import Course
        from models import CourseChunk
        from models import Lesson

        second_course = Course(
            title="Advanced Python Programming",
            course_link="https://example.com/python-course",
            instructor="Prof. John Doe",
            lessons=[
                Lesson(
                    lesson_number=1,
                    title="Object-Oriented Programming",
                    lesson_link="https://example.com/python-course/lesson-1",
                )
            ],
        )
        second_chunks = [
            CourseChunk(
                content="Object-oriented programming content",
                course_title="Advanced Python Programming",
                lesson_number=1,
                chunk_index=0,
            )
        ]

        # Mock to return different courses for different files
        mock_doc_proc_instance.process_course_document.side_effect = [
            (sample_course, sample_course_chunks),  # First call (course1.pdf)
            (second_course, second_chunks),  # Second call (course2.txt)
        ]

        # Mock vector store
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.get_existing_course_titles.return_value = []

        # Act
        total_courses, total_chunks = rag_system.add_course_folder("/path/to/courses")

        # Assert
        assert total_courses == 2  # Two valid document files
        assert total_chunks == len(sample_course_chunks) + len(
            second_chunks
        )  # 3 + 1 = 4 total chunks

        # Verify processing was called for valid files only
        assert mock_doc_proc_instance.process_course_document.call_count == 2

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_get_course_analytics(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test course analytics retrieval"""
        # Arrange
        rag_system = RAGSystem(test_config)

        # Mock vector store
        mock_vector_store_instance = mock_vector_store.return_value
        mock_vector_store_instance.get_course_count.return_value = 3
        mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
        ]

        # Act
        analytics = rag_system.get_course_analytics()

        # Assert
        assert analytics["total_courses"] == 3
        assert analytics["course_titles"] == ["Course A", "Course B", "Course C"]

        # Verify vector store methods were called
        mock_vector_store_instance.get_course_count.assert_called_once()
        mock_vector_store_instance.get_existing_course_titles.assert_called_once()

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    def test_tool_registration(
        self,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test that tools are properly registered with the tool manager"""
        # Act
        rag_system = RAGSystem(test_config)

        # Assert
        # Verify tools were registered (this tests the registration logic)
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

        # The actual tool definitions should be available
        tool_definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(tool_definitions) >= 2  # At least search and outline tools

    @patch("rag_system.SessionManager")
    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.ToolManager")
    def test_end_to_end_query_flow(
        self,
        mock_tool_manager_class,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_session_mgr,
        test_config,
    ):
        """Test complete end-to-end query processing flow"""
        # Arrange
        # Mock the tool manager instance
        mock_tool_manager = Mock()
        mock_tool_manager.get_tool_definitions.return_value = [{"name": "search_tool"}]
        mock_tool_manager.get_last_sources.return_value = [
            "ML Course - Lesson 1",
            "ML Course - Lesson 3",
        ]
        mock_tool_manager.get_last_source_links.return_value = [
            "https://example.com/ml/lesson1",
            "https://example.com/ml/lesson3",
        ]
        mock_tool_manager.reset_sources.return_value = None
        mock_tool_manager_class.return_value = mock_tool_manager

        rag_system = RAGSystem(test_config)

        # Mock all components for complete flow
        mock_ai_gen_instance = mock_ai_gen.return_value
        mock_ai_gen_instance.generate_response.return_value = (
            "Comprehensive answer about machine learning"
        )

        mock_session_mgr_instance = mock_session_mgr.return_value
        mock_session_mgr_instance.get_conversation_history.return_value = (
            "Previous context"
        )

        # Act
        response, sources, source_links = rag_system.query(
            "Explain machine learning algorithms with examples",
            session_id="session_123",
        )

        # Assert complete flow
        assert response == "Comprehensive answer about machine learning"
        assert len(sources) == 2
        assert len(source_links) == 2

        # Verify all major components were involved
        mock_session_mgr_instance.get_conversation_history.assert_called_once_with(
            "session_123"
        )
        mock_ai_gen_instance.generate_response.assert_called_once()
        mock_session_mgr_instance.add_exchange.assert_called_once()
        mock_tool_manager.get_last_sources.assert_called_once()
        mock_tool_manager.get_last_source_links.assert_called_once()
        mock_tool_manager.reset_sources.assert_called_once()

        # Verify AI generator received all necessary parameters
        call_args = mock_ai_gen_instance.generate_response.call_args
        assert "course materials" in call_args[1]["query"]
        assert call_args[1]["conversation_history"] == "Previous context"
        assert call_args[1]["tools"] is not None
        assert call_args[1]["tool_manager"] is not None
