"""
Tests for CourseSearchTool.execute() method evaluation
"""

import os
import sys
from unittest.mock import Mock
from unittest.mock import patch

import pytest

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from search_tools import CourseSearchTool
from vector_store import SearchResults


class TestCourseSearchTool:
    """Test cases for CourseSearchTool execution and behavior"""

    def test_execute_with_query_only_success(
        self, mock_vector_store, search_results_with_data
    ):
        """Test successful execution with query parameter only"""
        # Arrange
        mock_vector_store.search.return_value = search_results_with_data
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="machine learning")

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="machine learning", course_name=None, lesson_number=None
        )
        assert "Introduction to Machine Learning" in result
        assert "Machine learning is a subset of artificial intelligence" in result
        assert "Supervised learning uses labeled data" in result

        # Check that sources were tracked
        assert len(tool.last_sources) == 2
        assert "Introduction to Machine Learning - Lesson 1" in tool.last_sources

    def test_execute_with_course_name_filter(
        self, mock_vector_store, search_results_with_data
    ):
        """Test execution with course name filter"""
        # Arrange
        mock_vector_store.search.return_value = search_results_with_data
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(
            query="supervised learning", course_name="Introduction to Machine Learning"
        )

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="supervised learning",
            course_name="Introduction to Machine Learning",
            lesson_number=None,
        )
        assert "Introduction to Machine Learning" in result
        assert result.count("[Introduction to Machine Learning - Lesson 1]") == 2

    def test_execute_with_lesson_number_filter(
        self, mock_vector_store, search_results_with_data
    ):
        """Test execution with lesson number filter"""
        # Arrange
        mock_vector_store.search.return_value = search_results_with_data
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="machine learning basics", lesson_number=1)

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="machine learning basics", course_name=None, lesson_number=1
        )
        assert "Lesson 1" in result

    def test_execute_with_all_filters(
        self, mock_vector_store, search_results_with_data
    ):
        """Test execution with both course name and lesson number filters"""
        # Arrange
        mock_vector_store.search.return_value = search_results_with_data
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(
            query="linear regression",
            course_name="Introduction to Machine Learning",
            lesson_number=3,
        )

        # Assert
        mock_vector_store.search.assert_called_once_with(
            query="linear regression",
            course_name="Introduction to Machine Learning",
            lesson_number=3,
        )
        assert "Introduction to Machine Learning - Lesson 1" in result

    def test_execute_with_empty_results(self, mock_vector_store, empty_search_results):
        """Test execution when no results are found"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="nonexistent topic")

        # Assert
        assert result == "No relevant content found."
        assert len(tool.last_sources) == 0

    def test_execute_with_empty_results_and_course_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test execution with empty results when course filter is applied"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="nonexistent topic", course_name="Some Course")

        # Assert
        assert result == "No relevant content found in course 'Some Course'."

    def test_execute_with_empty_results_and_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test execution with empty results when lesson filter is applied"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="nonexistent topic", lesson_number=5)

        # Assert
        assert result == "No relevant content found in lesson 5."

    def test_execute_with_empty_results_and_both_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test execution with empty results when both filters are applied"""
        # Arrange
        mock_vector_store.search.return_value = empty_search_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(
            query="nonexistent topic", course_name="Some Course", lesson_number=3
        )

        # Assert
        assert (
            result == "No relevant content found in course 'Some Course' in lesson 3."
        )

    def test_execute_with_search_error(
        self, mock_vector_store, search_results_with_error
    ):
        """Test execution when vector store returns an error"""
        # Arrange
        mock_vector_store.search.return_value = search_results_with_error
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="any query")

        # Assert
        assert result == "Database connection failed"

    def test_result_formatting_with_mixed_lesson_numbers(self, mock_vector_store):
        """Test result formatting when chunks come from different lessons"""
        # Arrange
        mixed_results = SearchResults(
            documents=[
                "Machine learning basics from lesson 1",
                "Advanced concepts from lesson 3",
                "Course overview without lesson number",
            ],
            metadata=[
                {
                    "course_title": "Introduction to ML",
                    "lesson_number": 1,
                    "chunk_index": 0,
                },
                {
                    "course_title": "Introduction to ML",
                    "lesson_number": 3,
                    "chunk_index": 1,
                },
                {
                    "course_title": "Introduction to ML",
                    # No lesson_number (None)
                    "chunk_index": 2,
                },
            ],
            distances=[0.1, 0.2, 0.3],
        )
        mock_vector_store.search.return_value = mixed_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="mixed content")

        # Assert
        assert "[Introduction to ML - Lesson 1]" in result
        assert "[Introduction to ML - Lesson 3]" in result
        assert "[Introduction to ML]" in result  # No lesson number case
        assert "Machine learning basics from lesson 1" in result
        assert "Advanced concepts from lesson 3" in result
        assert "Course overview without lesson number" in result

    def test_source_tracking_and_links(
        self, mock_vector_store, search_results_with_data
    ):
        """Test that sources and source links are properly tracked"""
        # Arrange
        mock_vector_store.search.return_value = search_results_with_data
        mock_vector_store.get_lesson_link.side_effect = [
            "https://example.com/ml/lesson-1",
            "https://example.com/ml/lesson-1",
        ]
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="test query")

        # Assert
        assert len(tool.last_sources) == 2
        assert len(tool.last_source_links) == 2
        assert tool.last_sources[0] == "Introduction to Machine Learning - Lesson 1"
        assert tool.last_sources[1] == "Introduction to Machine Learning - Lesson 1"
        assert tool.last_source_links[0] == "https://example.com/ml/lesson-1"
        assert tool.last_source_links[1] == "https://example.com/ml/lesson-1"

        # Verify get_lesson_link was called correctly
        assert mock_vector_store.get_lesson_link.call_count == 2

    def test_source_tracking_without_lesson_numbers(self, mock_vector_store):
        """Test source tracking when chunks don't have lesson numbers"""
        # Arrange
        results_without_lessons = SearchResults(
            documents=["General course content"],
            metadata=[
                {
                    "course_title": "Test Course",
                    # No lesson_number
                    "chunk_index": 0,
                }
            ],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = results_without_lessons
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="general content")

        # Assert
        assert len(tool.last_sources) == 1
        assert len(tool.last_source_links) == 1
        assert tool.last_sources[0] == "Test Course"
        assert tool.last_source_links[0] is None  # No lesson link for general content

        # get_lesson_link should not be called without lesson numbers
        mock_vector_store.get_lesson_link.assert_not_called()

    def test_get_tool_definition(self, mock_vector_store):
        """Test that tool definition is properly structured"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)

        # Act
        definition = tool.get_tool_definition()

        # Assert
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition

        schema = definition["input_schema"]
        assert schema["type"] == "object"
        assert "properties" in schema
        assert "required" in schema

        properties = schema["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties

        assert schema["required"] == ["query"]
        assert properties["query"]["type"] == "string"
        assert properties["course_name"]["type"] == "string"
        assert properties["lesson_number"]["type"] == "integer"

    def test_multiple_queries_reset_sources(
        self, mock_vector_store, search_results_with_data
    ):
        """Test that sources are properly tracked across multiple queries"""
        # Arrange
        tool = CourseSearchTool(mock_vector_store)
        mock_vector_store.search.return_value = search_results_with_data

        # Act - First query
        tool.execute(query="first query")
        first_sources = tool.last_sources.copy()

        # Act - Second query (should replace first sources)
        tool.execute(query="second query")
        second_sources = tool.last_sources.copy()

        # Assert
        assert first_sources == second_sources  # Same mock data
        assert len(tool.last_sources) == 2  # Sources from second query only

    def test_execute_handles_missing_metadata_gracefully(self, mock_vector_store):
        """Test execution handles missing or malformed metadata gracefully"""
        # Arrange
        malformed_results = SearchResults(
            documents=["Content with missing metadata"],
            metadata=[
                {
                    # Missing course_title
                    "chunk_index": 0
                }
            ],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = malformed_results
        tool = CourseSearchTool(mock_vector_store)

        # Act
        result = tool.execute(query="malformed metadata")

        # Assert
        assert "[unknown]" in result  # Should handle missing course_title gracefully
        assert "Content with missing metadata" in result
