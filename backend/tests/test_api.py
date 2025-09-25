"""
API endpoint tests for the RAG system FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock


@pytest.mark.api
class TestQueryEndpoint:
    """Test the /api/query endpoint"""

    def test_query_with_session_id(self, test_client, api_test_data):
        """Test successful query with session ID"""
        response = test_client.post("/api/query", json=api_test_data["valid_query"])

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "source_links" in data
        assert "session_id" in data
        assert data["session_id"] == api_test_data["valid_query"]["session_id"]
        assert isinstance(data["sources"], list)
        assert isinstance(data["source_links"], list)

    def test_query_without_session_id(self, test_client, api_test_data):
        """Test query without session ID - should create new session"""
        response = test_client.post("/api/query", json=api_test_data["query_without_session"])

        assert response.status_code == 200
        data = response.json()

        assert "answer" in data
        assert "sources" in data
        assert "source_links" in data
        assert "session_id" in data
        assert data["session_id"] == "test-session-456"  # From mock

    def test_query_invalid_request_body(self, test_client, api_test_data):
        """Test query with invalid request body"""
        response = test_client.post("/api/query", json=api_test_data["invalid_query"])

        assert response.status_code == 422  # Validation error

    def test_query_missing_query_field(self, test_client):
        """Test query with missing required query field"""
        response = test_client.post("/api/query", json={"session_id": "test-123"})

        assert response.status_code == 422

    def test_query_empty_query(self, test_client):
        """Test query with empty query string"""
        response = test_client.post("/api/query", json={"query": ""})

        assert response.status_code == 200  # Empty query should still work

    def test_query_with_rag_system_error(self, test_client, mock_rag_system):
        """Test query when RAG system raises an exception"""
        mock_rag_system.query.side_effect = Exception("Database connection failed")

        response = test_client.post("/api/query", json={"query": "test query"})

        assert response.status_code == 500
        assert "Database connection failed" in response.json()["detail"]


@pytest.mark.api
class TestCoursesEndpoint:
    """Test the /api/courses endpoint"""

    def test_get_courses_success(self, test_client, api_test_data):
        """Test successful retrieval of course statistics"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == api_test_data["expected_courses"]["total_courses"]
        assert data["course_titles"] == api_test_data["expected_courses"]["course_titles"]
        assert isinstance(data["course_titles"], list)

    def test_get_courses_with_analytics_error(self, test_client, mock_rag_system):
        """Test courses endpoint when analytics raises an exception"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics service unavailable")

        response = test_client.get("/api/courses")

        assert response.status_code == 500
        assert "Analytics service unavailable" in response.json()["detail"]


@pytest.mark.api
class TestSessionEndpoints:
    """Test session management endpoints"""

    def test_create_new_session_success(self, test_client):
        """Test successful creation of new session"""
        response = test_client.post("/api/sessions/new")

        assert response.status_code == 200
        data = response.json()

        assert "session_id" in data
        assert "message" in data
        assert data["session_id"] == "test-session-456"  # From mock
        assert "created successfully" in data["message"]

    def test_create_new_session_with_error(self, test_client, mock_rag_system):
        """Test session creation when session manager raises an exception"""
        mock_rag_system.session_manager.create_session.side_effect = Exception("Session creation failed")

        response = test_client.post("/api/sessions/new")

        assert response.status_code == 500
        assert "Session creation failed" in response.json()["detail"]

    def test_clear_session_success(self, test_client):
        """Test successful session clearing"""
        session_id = "test-session-123"
        response = test_client.delete(f"/api/sessions/{session_id}/clear")

        assert response.status_code == 200
        data = response.json()

        assert "success" in data
        assert "message" in data
        assert data["success"] is True
        assert session_id in data["message"]
        assert "cleared successfully" in data["message"]

    def test_clear_session_with_error(self, test_client, mock_rag_system):
        """Test session clearing when session manager raises an exception"""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session clear failed")

        response = test_client.delete("/api/sessions/test-session/clear")

        assert response.status_code == 500
        assert "Session clear failed" in response.json()["detail"]

    def test_clear_nonexistent_session(self, test_client):
        """Test clearing a session that doesn't exist (should still succeed)"""
        response = test_client.delete("/api/sessions/nonexistent-session/clear")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True


@pytest.mark.api
class TestAPIIntegration:
    """Integration tests for API endpoints working together"""

    def test_full_workflow(self, test_client, api_test_data):
        """Test a complete workflow: create session, query, get courses, clear session"""
        # Create new session
        session_response = test_client.post("/api/sessions/new")
        assert session_response.status_code == 200
        session_id = session_response.json()["session_id"]

        # Make query with session
        query_data = {
            "query": "What is machine learning?",
            "session_id": session_id
        }
        query_response = test_client.post("/api/query", json=query_data)
        assert query_response.status_code == 200

        # Get courses
        courses_response = test_client.get("/api/courses")
        assert courses_response.status_code == 200

        # Clear session
        clear_response = test_client.delete(f"/api/sessions/{session_id}/clear")
        assert clear_response.status_code == 200

    def test_query_response_structure(self, test_client):
        """Test that query responses have the correct structure"""
        response = test_client.post("/api/query", json={"query": "test"})

        assert response.status_code == 200
        data = response.json()

        # Verify required fields exist
        required_fields = ["answer", "sources", "source_links", "session_id"]
        for field in required_fields:
            assert field in data

        # Verify field types
        assert isinstance(data["answer"], str)
        assert isinstance(data["sources"], list)
        assert isinstance(data["source_links"], list)
        assert isinstance(data["session_id"], str)

        # Verify sources and source_links have same length
        assert len(data["sources"]) == len(data["source_links"])

    def test_courses_response_structure(self, test_client):
        """Test that courses responses have the correct structure"""
        response = test_client.get("/api/courses")

        assert response.status_code == 200
        data = response.json()

        # Verify required fields exist
        assert "total_courses" in data
        assert "course_titles" in data

        # Verify field types
        assert isinstance(data["total_courses"], int)
        assert isinstance(data["course_titles"], list)

        # Verify consistency
        assert len(data["course_titles"]) == data["total_courses"]


@pytest.mark.api
class TestAPIErrorHandling:
    """Test API error handling scenarios"""

    def test_invalid_json(self, test_client):
        """Test API behavior with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_missing_content_type(self, test_client):
        """Test API behavior without content type header"""
        response = test_client.post("/api/query", data='{"query": "test"}')

        # FastAPI should still handle this correctly
        assert response.status_code in [200, 422]  # Either works or validation error

    def test_unsupported_http_methods(self, test_client):
        """Test unsupported HTTP methods on endpoints"""
        # GET on query endpoint (should be POST)
        response = test_client.get("/api/query")
        assert response.status_code == 405

        # POST on courses endpoint (should be GET)
        response = test_client.post("/api/courses")
        assert response.status_code == 405

        # GET on session creation (should be POST)
        response = test_client.get("/api/sessions/new")
        assert response.status_code == 405

    def test_nonexistent_endpoints(self, test_client):
        """Test requests to nonexistent endpoints"""
        response = test_client.get("/api/nonexistent")
        assert response.status_code == 404

        response = test_client.post("/api/invalid")
        assert response.status_code == 404


@pytest.mark.api
@pytest.mark.slow
class TestAPIPerformance:
    """Performance-related API tests"""

    def test_concurrent_queries(self, test_client):
        """Test that multiple concurrent queries work correctly"""
        import concurrent.futures
        import threading

        def make_query(query_num):
            return test_client.post("/api/query", json={"query": f"test query {query_num}"})

        # Make 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_query, i) for i in range(5)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # All requests should succeed
        for response in results:
            assert response.status_code == 200

    def test_large_query_text(self, test_client):
        """Test API behavior with large query text"""
        large_query = "x" * 10000  # 10KB query
        response = test_client.post("/api/query", json={"query": large_query})

        assert response.status_code == 200
        data = response.json()
        assert "answer" in data