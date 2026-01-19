"""
Test edge cases and error scenarios for workout history module
Tests Requirements: 1.3, 1.5, 2.5
"""
import json
import os
import tempfile
import shutil
from pathlib import Path
import pytest
import sys

# Add frontend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))

from workout_history import (
    WorkoutHistoryLoader,
    WorkoutHistoryFilter,
    WorkoutHistoryAggregator,
    WorkoutHistoryFormatter
)


class TestEmptyDataDirectory:
    """Test handling of empty data directory - Requirement 1.3"""
    
    def test_nonexistent_directory(self):
        """Test loading from a directory that doesn't exist"""
        loader = WorkoutHistoryLoader(data_dir="/nonexistent/path/to/data")
        sessions = loader.load_all_sessions()
        
        # Should return empty list, not crash
        assert sessions == []
        assert isinstance(sessions, list)
    
    def test_empty_directory(self):
        """Test loading from an empty directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should return empty list
            assert sessions == []
            assert isinstance(sessions, list)
    
    def test_directory_with_no_json_files(self):
        """Test loading from directory with non-JSON files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some non-JSON files
            Path(tmpdir, "readme.txt").write_text("This is not JSON")
            Path(tmpdir, "data.csv").write_text("col1,col2\nval1,val2")
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should return empty list
            assert sessions == []


class TestCorruptedJSONFiles:
    """Test handling of corrupted JSON files - Requirement 1.5"""
    
    def test_invalid_json_syntax(self):
        """Test loading file with invalid JSON syntax"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a file with invalid JSON
            corrupted_file = Path(tmpdir, "corrupted.json")
            corrupted_file.write_text("{invalid json syntax")
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should skip corrupted file and return empty list
            assert sessions == []
    
    def test_empty_json_file(self):
        """Test loading an empty JSON file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an empty file
            empty_file = Path(tmpdir, "empty.json")
            empty_file.write_text("")
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should skip empty file and return empty list
            assert sessions == []
    
    def test_mixed_valid_and_corrupted_files(self):
        """Test loading mix of valid and corrupted files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a valid session file
            valid_session = {
                "session_id": "test-123",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
            Path(tmpdir, "valid.json").write_text(json.dumps(valid_session))
            
            # Create corrupted files
            Path(tmpdir, "corrupted1.json").write_text("{invalid")
            Path(tmpdir, "corrupted2.json").write_text("")
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should load only the valid session
            assert len(sessions) == 1
            assert sessions[0]["session_id"] == "test-123"


class TestMissingRequiredFields:
    """Test handling of sessions with missing required fields - Requirement 1.5"""
    
    def test_missing_session_id(self):
        """Test session missing session_id field"""
        with tempfile.TemporaryDirectory() as tmpdir:
            incomplete_session = {
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
            Path(tmpdir, "incomplete.json").write_text(json.dumps(incomplete_session))
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should skip session with missing required field
            assert sessions == []
    
    def test_missing_exercise_field(self):
        """Test session missing exercise field"""
        with tempfile.TemporaryDirectory() as tmpdir:
            incomplete_session = {
                "session_id": "test-123",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
            Path(tmpdir, "incomplete.json").write_text(json.dumps(incomplete_session))
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should skip session with missing required field
            assert sessions == []
    
    def test_missing_multiple_fields(self):
        """Test session missing multiple required fields"""
        with tempfile.TemporaryDirectory() as tmpdir:
            incomplete_session = {
                "session_id": "test-123",
                "exercise": "squat"
            }
            Path(tmpdir, "incomplete.json").write_text(json.dumps(incomplete_session))
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should skip session with missing required fields
            assert sessions == []
    
    def test_invalid_field_types(self):
        """Test session with invalid field types"""
        with tempfile.TemporaryDirectory() as tmpdir:
            invalid_session = {
                "session_id": 123,  # Should be string
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": "ten",  # Should be number
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
            Path(tmpdir, "invalid.json").write_text(json.dumps(invalid_session))
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should skip session with invalid field types
            assert sessions == []
    
    def test_mixed_valid_and_invalid_sessions(self):
        """Test loading mix of valid and invalid sessions"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Valid session
            valid_session = {
                "session_id": "valid-123",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
            Path(tmpdir, "valid.json").write_text(json.dumps(valid_session))
            
            # Invalid sessions
            invalid_session1 = {
                "exercise": "squat",  # Missing session_id
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
            Path(tmpdir, "invalid1.json").write_text(json.dumps(invalid_session1))
            
            invalid_session2 = {
                "session_id": "invalid-123",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": "ten",  # Invalid type
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
            Path(tmpdir, "invalid2.json").write_text(json.dumps(invalid_session2))
            
            loader = WorkoutHistoryLoader(data_dir=tmpdir)
            sessions = loader.load_all_sessions()
            
            # Should load only the valid session
            assert len(sessions) == 1
            assert sessions[0]["session_id"] == "valid-123"


class TestNoMatchingSessionsForFilter:
    """Test handling when no sessions match the filter - Requirement 2.5"""
    
    def test_filter_with_no_matches(self):
        """Test filtering by exercise type that doesn't exist"""
        sessions = [
            {
                "session_id": "1",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            },
            {
                "session_id": "2",
                "exercise": "bicep_curl",
                "start_time": "2024-01-15T11:00:00",
                "reps": 15,
                "duration": 45.0,
                "calories": 30.0,
                "status": "completed"
            }
        ]
        
        filter_obj = WorkoutHistoryFilter()
        filtered = filter_obj.filter_by_exercise(sessions, "push_up")
        
        # Should return empty list
        assert filtered == []
        assert isinstance(filtered, list)
    
    def test_filter_empty_session_list(self):
        """Test filtering an empty session list"""
        sessions = []
        
        filter_obj = WorkoutHistoryFilter()
        filtered = filter_obj.filter_by_exercise(sessions, "squat")
        
        # Should return empty list
        assert filtered == []
    
    def test_aggregation_with_empty_filtered_results(self):
        """Test aggregation when filter returns no results"""
        sessions = [
            {
                "session_id": "1",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            }
        ]
        
        filter_obj = WorkoutHistoryFilter()
        aggregator = WorkoutHistoryAggregator()
        
        # Filter to get no results
        filtered = filter_obj.filter_by_exercise(sessions, "push_up")
        
        # Aggregate the empty results
        total_workouts = aggregator.calculate_total_workouts(filtered)
        total_reps = aggregator.calculate_total_reps(filtered)
        total_calories = aggregator.calculate_total_calories(filtered)
        total_duration = aggregator.calculate_total_duration(filtered)
        
        # All should be zero
        assert total_workouts == 0
        assert total_reps == 0
        assert total_calories == 0.0
        assert total_duration == 0.0


class TestGracefulErrorHandling:
    """Test graceful error handling throughout the system"""
    
    def test_negative_values_in_aggregation(self):
        """Test that negative values are handled gracefully"""
        sessions = [
            {
                "session_id": "1",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": -10,  # Negative reps
                "duration": -60.0,  # Negative duration
                "calories": -50.0,  # Negative calories
                "status": "completed"
            }
        ]
        
        aggregator = WorkoutHistoryAggregator()
        
        # Should treat negative values as 0
        total_reps = aggregator.calculate_total_reps(sessions)
        total_calories = aggregator.calculate_total_calories(sessions)
        total_duration = aggregator.calculate_total_duration(sessions)
        
        assert total_reps == 0
        assert total_calories == 0.0
        assert total_duration == 0.0
    
    def test_invalid_date_formatting(self):
        """Test formatting of invalid date strings"""
        formatter = WorkoutHistoryFormatter()
        
        # Test with invalid date string
        result = formatter.format_date("not-a-date")
        
        # Should return the original string, not crash
        assert result == "not-a-date"
    
    def test_invalid_duration_formatting(self):
        """Test formatting of invalid duration values"""
        formatter = WorkoutHistoryFormatter()
        
        # Test with negative duration
        result = formatter.format_duration(-100)
        assert result == "0s"
        
        # Test with None (should handle gracefully)
        result = formatter.format_duration(None)
        assert result == "0s"
    
    def test_unknown_exercise_type_formatting(self):
        """Test formatting of unknown exercise types"""
        formatter = WorkoutHistoryFormatter()
        
        # Test with unknown exercise type
        name = formatter.format_exercise_name("unknown_exercise")
        icon = formatter.get_exercise_icon("unknown_exercise")
        
        # Should return formatted name and default icon
        assert name == "Unknown Exercise"
        assert icon == "🏋️"  # Default icon
    
    def test_case_insensitive_filtering(self):
        """Test that filtering is case-insensitive"""
        sessions = [
            {
                "session_id": "1",
                "exercise": "Squat",  # Mixed case
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            },
            {
                "session_id": "2",
                "exercise": "BICEP_CURL",  # Upper case
                "start_time": "2024-01-15T11:00:00",
                "reps": 15,
                "duration": 45.0,
                "calories": 30.0,
                "status": "completed"
            }
        ]
        
        filter_obj = WorkoutHistoryFilter()
        
        # Filter with lowercase
        filtered_squat = filter_obj.filter_by_exercise(sessions, "squat")
        filtered_bicep = filter_obj.filter_by_exercise(sessions, "bicep_curl")
        
        # Should match case-insensitively
        assert len(filtered_squat) == 1
        assert len(filtered_bicep) == 1
    
    def test_all_exercises_filter(self):
        """Test that 'all' filter returns all sessions"""
        sessions = [
            {
                "session_id": "1",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            },
            {
                "session_id": "2",
                "exercise": "bicep_curl",
                "start_time": "2024-01-15T11:00:00",
                "reps": 15,
                "duration": 45.0,
                "calories": 30.0,
                "status": "completed"
            }
        ]
        
        filter_obj = WorkoutHistoryFilter()
        
        # Test with "all"
        filtered_all = filter_obj.filter_by_exercise(sessions, "all")
        assert len(filtered_all) == 2
        
        # Test with empty string
        filtered_empty = filter_obj.filter_by_exercise(sessions, "")
        assert len(filtered_empty) == 2
    
    def test_sorting_with_missing_timestamps(self):
        """Test sorting when some sessions have missing timestamps"""
        sessions = [
            {
                "session_id": "1",
                "exercise": "squat",
                "start_time": "2024-01-15T10:00:00",
                "reps": 10,
                "duration": 60.0,
                "calories": 50.0,
                "status": "completed"
            },
            {
                "session_id": "2",
                "exercise": "bicep_curl",
                "start_time": "",  # Empty timestamp
                "reps": 15,
                "duration": 45.0,
                "calories": 30.0,
                "status": "completed"
            }
        ]
        
        filter_obj = WorkoutHistoryFilter()
        
        # Should not crash when sorting
        sorted_sessions = filter_obj.sort_by_date(sessions)
        
        # Should return a list
        assert isinstance(sorted_sessions, list)
        assert len(sorted_sessions) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
