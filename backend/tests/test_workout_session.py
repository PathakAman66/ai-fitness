"""
Unit tests for WorkoutSession
Tests session management, persistence, and data loading functionality
"""
import pytest
import os
import tempfile
import shutil
from backend.api.workout_session import WorkoutSession


class TestWorkoutSession:
    """Test suite for WorkoutSession"""
    
    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    def test_initialization(self, temp_data_dir):
        """Test that WorkoutSession initializes correctly"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        assert session_manager is not None
        assert session_manager.sessions == []
        assert session_manager.current_session is None
        assert os.path.exists(temp_data_dir)
    
    def test_start_session_generates_uuid(self, temp_data_dir):
        """Test that starting a session generates a UUID"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("bicep_curl")
        
        assert session_id is not None
        assert isinstance(session_id, str)
        assert len(session_id) == 36  # UUID format
        assert session_manager.current_session is not None
        assert session_manager.current_session['session_id'] == session_id
    
    def test_start_session_with_custom_id(self, temp_data_dir):
        """Test starting a session with a custom session ID"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        custom_id = "custom-session-id-123"
        session_id = session_manager.start_session("squat", session_id=custom_id)
        
        assert session_id == custom_id
        assert session_manager.current_session['session_id'] == custom_id
    
    def test_start_session_with_user_id(self, temp_data_dir):
        """Test starting a session with a user ID"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("push_up", user_id="user123")
        
        assert session_manager.current_session['user_id'] == "user123"
        assert session_manager.current_session['exercise_type'] == "push_up"
    
    def test_update_session(self, temp_data_dir):
        """Test updating session with rep count and calories"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("bicep_curl")
        session_manager.update_session(rep_count=5, calories=2.5)
        
        assert session_manager.current_session['reps'] == 5
        assert session_manager.current_session['calories'] == 2.5
        assert session_manager.current_session['duration'] >= 0
    
    def test_end_session(self, temp_data_dir):
        """Test ending a session"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("squat")
        session_manager.update_session(rep_count=10, calories=10.0)
        
        completed_session = session_manager.end_session()
        
        assert completed_session is not None
        assert completed_session['status'] == 'completed'
        assert completed_session['end_time'] is not None
        assert completed_session['reps'] == 10
        assert session_manager.current_session is None
        assert len(session_manager.sessions) == 1
    
    def test_end_session_with_analysis_result(self, temp_data_dir):
        """Test ending a session with analysis result"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("bicep_curl")
        
        analysis_result = {
            'rep_count': 15,
            'calories': 7.5
        }
        
        completed_session = session_manager.end_session(analysis_result)
        
        assert completed_session['reps'] == 15
        assert completed_session['calories'] == 7.5
    
    def test_end_session_without_active_session(self, temp_data_dir):
        """Test that ending without an active session returns None"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        result = session_manager.end_session()
        
        assert result is None
    
    def test_get_session_by_id_current(self, temp_data_dir):
        """Test retrieving the current active session by ID"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("plank")
        
        retrieved_session = session_manager.get_session_by_id(session_id)
        
        assert retrieved_session is not None
        assert retrieved_session['session_id'] == session_id
        assert retrieved_session['exercise_type'] == "plank"
    
    def test_get_session_by_id_completed(self, temp_data_dir):
        """Test retrieving a completed session by ID"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("squat")
        session_manager.end_session()
        
        retrieved_session = session_manager.get_session_by_id(session_id)
        
        assert retrieved_session is not None
        assert retrieved_session['session_id'] == session_id
        assert retrieved_session['status'] == 'completed'
    
    def test_get_session_by_id_not_found(self, temp_data_dir):
        """Test that getting a non-existent session returns None"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        result = session_manager.get_session_by_id("non-existent-id")
        
        assert result is None
    
    def test_save_and_load_sessions(self, temp_data_dir):
        """Test that sessions are saved and can be loaded"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        # Create and end a session
        session_id = session_manager.start_session("bicep_curl")
        session_manager.update_session(rep_count=20, calories=10.0)
        session_manager.end_session()
        
        # Create a new session manager and load sessions
        new_session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        assert len(new_session_manager.sessions) == 1
        assert new_session_manager.sessions[0]['session_id'] == session_id
        assert new_session_manager.sessions[0]['reps'] == 20
    
    def test_get_all_sessions(self, temp_data_dir):
        """Test retrieving all sessions"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        # Create multiple sessions
        session_manager.start_session("bicep_curl")
        session_manager.end_session()
        
        session_manager.start_session("squat")
        session_manager.end_session()
        
        all_sessions = session_manager.get_all_sessions()
        
        assert len(all_sessions) == 2
    
    def test_get_all_sessions_with_user_filter(self, temp_data_dir):
        """Test filtering sessions by user_id"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        # Create sessions for different users
        session_manager.start_session("bicep_curl", user_id="user1")
        session_manager.end_session()
        
        session_manager.start_session("squat", user_id="user2")
        session_manager.end_session()
        
        session_manager.start_session("plank", user_id="user1")
        session_manager.end_session()
        
        user1_sessions = session_manager.get_all_sessions(user_id="user1")
        
        assert len(user1_sessions) == 2
        assert all(s['user_id'] == "user1" for s in user1_sessions)
    
    def test_get_all_sessions_with_pagination(self, temp_data_dir):
        """Test pagination of session retrieval"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        # Create multiple sessions
        for i in range(5):
            session_manager.start_session("bicep_curl")
            session_manager.end_session()
        
        # Get first 2 sessions
        page1 = session_manager.get_all_sessions(limit=2, offset=0)
        assert len(page1) == 2
        
        # Get next 2 sessions
        page2 = session_manager.get_all_sessions(limit=2, offset=2)
        assert len(page2) == 2
        
        # Verify they're different
        assert page1[0]['session_id'] != page2[0]['session_id']
    
    def test_reset_current_session(self, temp_data_dir):
        """Test resetting the current session"""
        session_manager = WorkoutSession(data_dir=temp_data_dir)
        
        session_id = session_manager.start_session("bicep_curl")
        session_manager.update_session(rep_count=10, calories=5.0)
        
        session_manager.reset_current_session()
        
        assert session_manager.current_session['reps'] == 0
        assert session_manager.current_session['calories'] == 0
        # Session ID and start time should remain unchanged
        assert session_manager.current_session['session_id'] == session_id
