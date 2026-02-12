"""
Unit tests for SessionManager service class
Tests session creation, retrieval, ending, and resetting functionality
"""
import time
from backend.api.session_manager import SessionManager, SessionData


class TestSessionManager:
    """Test suite for SessionManager"""
    
    def test_create_session_generates_unique_id(self):
        """Test that creating a session generates a unique UUID"""
        manager = SessionManager()
        
        session_id = manager.create_session("bicep_curl")
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID format: 8-4-4-4-12
        assert manager.session_exists(session_id)
    
    def test_create_multiple_sessions_have_unique_ids(self):
        """Test that multiple sessions get unique IDs"""
        manager = SessionManager()
        
        session_id1 = manager.create_session("bicep_curl")
        session_id2 = manager.create_session("bicep_curl")
        
        assert session_id1 != session_id2
        assert manager.session_exists(session_id1)
        assert manager.session_exists(session_id2)
    
    def test_create_session_with_user_id(self):
        """Test creating a session with optional user_id"""
        manager = SessionManager()
        
        session_id = manager.create_session("squat", user_id="user123")
        session_data = manager.get_session(session_id)
        
        assert session_data is not None
        assert session_data.user_id == "user123"
        assert session_data.exercise_type == "squat"
    
    def test_get_session_returns_session_data(self):
        """Test retrieving an existing session"""
        manager = SessionManager()
        
        session_id = manager.create_session("push_up")
        session_data = manager.get_session(session_id)
        
        assert session_data is not None
        assert isinstance(session_data, SessionData)
        assert session_data.session_id == session_id
        assert session_data.exercise_type == "push_up"
        assert session_data.status == "active"
    
    def test_get_session_returns_none_for_invalid_id(self):
        """Test that getting a non-existent session returns None"""
        manager = SessionManager()
        
        session_data = manager.get_session("invalid-uuid")
        
        assert session_data is None
    
    def test_session_has_analyzer_instance(self):
        """Test that created session has an associated analyzer"""
        manager = SessionManager()
        
        session_id = manager.create_session("bicep_curl")
        session_data = manager.get_session(session_id)
        
        assert session_data is not None
        assert session_data.analyzer is not None
        assert session_data.analyzer.exercise_type == "bicep_curl"
        assert session_data.analyzer.rep_count == 0
    
    def test_reset_session_clears_state(self):
        """Test that resetting a session clears rep count and calories"""
        manager = SessionManager()
        
        session_id = manager.create_session("bicep_curl")
        session_data = manager.get_session(session_id)
        
        # Simulate some activity
        session_data.analyzer.rep_count = 10
        session_data.analyzer.calories_burned = 5.0
        session_data.analyzer.current_stage = "up"
        
        # Reset the session
        result = manager.reset_session(session_id)
        
        assert result is True
        assert session_data.analyzer.rep_count == 0
        assert session_data.analyzer.calories_burned == 0
        assert session_data.analyzer.current_stage == "start"
    
    def test_reset_nonexistent_session_returns_false(self):
        """Test that resetting a non-existent session returns False"""
        manager = SessionManager()
        
        result = manager.reset_session("invalid-uuid")
        
        assert result is False
    
    def test_end_session_removes_from_active(self):
        """Test that ending a session removes it from active sessions"""
        manager = SessionManager()
        
        session_id = manager.create_session("squat")
        assert manager.session_exists(session_id)
        
        # End the session
        session_data = manager.end_session(session_id)
        
        assert session_data is not None
        assert session_data.status == "completed"
        assert not manager.session_exists(session_id)
    
    def test_end_nonexistent_session_returns_none(self):
        """Test that ending a non-existent session returns None"""
        manager = SessionManager()
        
        result = manager.end_session("invalid-uuid")
        
        assert result is None
    
    def test_concurrent_sessions_are_isolated(self):
        """Test that multiple concurrent sessions maintain separate state"""
        manager = SessionManager()
        
        # Create two sessions
        session_id1 = manager.create_session("bicep_curl")
        session_id2 = manager.create_session("squat")
        
        session1 = manager.get_session(session_id1)
        session2 = manager.get_session(session_id2)
        
        # Modify session 1
        session1.analyzer.rep_count = 5
        session1.analyzer.calories_burned = 2.5
        
        # Verify session 2 is unaffected
        assert session2.analyzer.rep_count == 0
        assert session2.analyzer.calories_burned == 0
        assert session1.exercise_type == "bicep_curl"
        assert session2.exercise_type == "squat"
    
    def test_get_all_sessions(self):
        """Test retrieving all active sessions"""
        manager = SessionManager()
        
        session_id1 = manager.create_session("bicep_curl")
        session_id2 = manager.create_session("squat")
        
        all_sessions = manager.get_all_sessions()
        
        assert len(all_sessions) == 2
        assert session_id1 in all_sessions
        assert session_id2 in all_sessions
    
    def test_session_timestamps(self):
        """Test that sessions have proper timestamps"""
        manager = SessionManager()
        
        session_id = manager.create_session("plank")
        session_data = manager.get_session(session_id)
        
        assert session_data.created_at is not None
        assert session_data.last_activity is not None
        
        # Verify last_activity updates on get
        time.sleep(0.01)  # Small delay
        original_activity = session_data.last_activity
        session_data = manager.get_session(session_id)
        
        assert session_data.last_activity >= original_activity
    
    def test_session_has_quality_evaluator(self):
        """Test that created session has an associated quality evaluator"""
        manager = SessionManager()
        
        session_id = manager.create_session("bicep_curl")
        session_data = manager.get_session(session_id)
        
        assert session_data is not None
        assert session_data.quality_evaluator is not None
        assert session_data.quality_evaluator.exercise_type == "bicep_curl"
        assert len(session_data.quality_evaluator.history) == 0
    
    def test_reset_session_clears_quality_evaluator_history(self):
        """Test that resetting a session clears quality evaluator history"""
        manager = SessionManager()
        
        session_id = manager.create_session("squat")
        session_data = manager.get_session(session_id)
        
        # Simulate some quality scores
        session_data.quality_evaluator.update_history(85.0)
        session_data.quality_evaluator.update_history(90.0)
        session_data.quality_evaluator.update_history(88.0)
        
        assert len(session_data.quality_evaluator.history) == 3
        assert session_data.quality_evaluator.get_historical_average() is not None
        
        # Reset the session
        result = manager.reset_session(session_id)
        
        assert result is True
        assert len(session_data.quality_evaluator.history) == 0
        assert session_data.quality_evaluator.get_historical_average() is None
