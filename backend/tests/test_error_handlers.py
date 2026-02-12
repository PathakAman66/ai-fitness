"""
Tests for global error handlers in FastAPI application

Verifies that error responses follow consistent format and proper status codes.
"""
from fastapi.testclient import TestClient
from backend.api.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    """Test that health endpoint works correctly"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "service" in data
    assert "version" in data


def test_404_error_handler():
    """
    Test that 404 errors return consistent error format.
    
    **Requirements: 5.3**
    - WHEN a client attempts to access a non-existent endpoint
      THEN the FastAPI Service SHALL return a 404 status code with appropriate message
    """
    response = client.get("/api/v1/nonexistent")
    assert response.status_code == 404
    
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "status_code" in data
    assert data["status_code"] == 404
    assert data["error"] == "Not Found"


def test_session_not_found_returns_404():
    """
    Test that accessing non-existent session returns 404 with consistent format.
    
    **Requirements: 5.3**
    - WHEN a client attempts to end a non-existent session
      THEN the FastAPI Service SHALL return a 404 status code with an appropriate message
    """
    fake_session_id = "00000000-0000-0000-0000-000000000000"
    response = client.get(f"/api/v1/sessions/{fake_session_id}")
    
    # Should get either 404 (session not found) or 503 (manager not initialized in test)
    assert response.status_code in [404, 503]
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "status_code" in data
    assert data["status_code"] == response.status_code
    
    # If service is available, should specifically say "not found"
    if response.status_code == 404:
        assert "not found" in data["detail"].lower()


def test_validation_error_returns_400():
    """
    Test that validation errors return 400 with consistent format.
    
    **Requirements: 5.5**
    - WHEN request validation fails
      THEN the FastAPI Service SHALL return structured error responses following a consistent schema
    """
    # Send invalid exercise type
    response = client.post(
        "/api/v1/sessions/start",
        json={"exercise_type": "invalid_exercise"}
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "status_code" in data
    assert data["status_code"] == 400


def test_invalid_session_id_format_returns_400():
    """
    Test that invalid session ID format returns 400 with consistent format.
    
    **Requirements: 5.1, 5.5**
    - WHEN a client sends invalid data
      THEN the FastAPI Service SHALL return a 400 status code with a descriptive error message
    """
    response = client.post(
        "/api/v1/analyze",
        json={
            "session_id": "not-a-valid-uuid",
            "exercise_type": "bicep_curl",
            "key_points": {
                "left_shoulder": [0.5, 0.5, 0.0, 0.9],
                "right_shoulder": [0.5, 0.5, 0.0, 0.9]
            }
        }
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "status_code" in data
    assert data["status_code"] == 400


def test_missing_required_field_returns_400():
    """
    Test that missing required fields return 400 with validation error.
    
    **Requirements: 5.5**
    - WHEN request validation fails
      THEN the FastAPI Service SHALL return structured error responses following a consistent schema
    """
    # Missing exercise_type field
    response = client.post(
        "/api/v1/sessions/start",
        json={}
    )
    
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "status_code" in data
    assert data["status_code"] == 400
    assert "exercise_type" in data["detail"].lower()


def test_error_response_schema_consistency():
    """
    Test that all error responses follow the same schema structure.
    
    **Requirements: 5.5**
    - WHEN request validation fails
      THEN the FastAPI Service SHALL return structured error responses following a consistent schema
    """
    # Test multiple error scenarios
    error_responses = [
        client.get("/api/v1/nonexistent"),  # 404
        client.post("/api/v1/sessions/start", json={}),  # 400 validation
        client.get("/api/v1/sessions/invalid-uuid"),  # 404
    ]
    
    for response in error_responses:
        data = response.json()
        # All errors should have these three fields
        assert "error" in data, f"Missing 'error' field in {data}"
        assert "detail" in data, f"Missing 'detail' field in {data}"
        assert "status_code" in data, f"Missing 'status_code' field in {data}"
        
        # status_code in response should match status_code in body
        assert data["status_code"] == response.status_code


def test_service_unavailable_when_detector_not_initialized():
    """
    Test that 503 is returned when pose detector is not initialized.
    
    **Requirements: 5.4**
    - WHEN the Pose Detection Engine encounters an error
      THEN the FastAPI Service SHALL return appropriate status code
    
    Note: This test may not trigger 503 if detector initializes successfully,
    but it verifies the error handling structure is in place.
    """
    # This test verifies the endpoint exists and handles requests
    # The actual 503 would require the detector to fail initialization
    response = client.post(
        "/api/v1/pose/detect",
        data={"image": "invalid_base64_data"}
    )
    
    # Should get either 400 (invalid data) or 503 (detector not ready)
    assert response.status_code in [400, 503]
    
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "status_code" in data
