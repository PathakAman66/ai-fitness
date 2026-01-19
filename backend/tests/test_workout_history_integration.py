"""
Integration tests for workout history with real data scenarios
Tests Requirements: 1.3, 1.5, 2.5
"""
import json
import os
import tempfile
import shutil
from pathlib import Path
import sys

# Add frontend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))

from workout_history import (
    WorkoutHistoryLoader,
    WorkoutHistoryFilter,
    WorkoutHistoryAggregator,
    WorkoutHistoryFormatter,
    WorkoutHistoryUI
)


def test_real_data_directory():
    """Test loading from the actual data directory"""
    loader = WorkoutHistoryLoader(data_dir="backend/data/reports")
    sessions = loader.load_all_sessions()
    
    print(f"\n✓ Loaded {len(sessions)} sessions from real data directory")
    
    # Should load sessions without crashing
    assert isinstance(sessions, list)
    
    # If sessions exist, verify they have required fields
    for session in sessions:
        assert 'session_id' in session
        assert 'exercise' in session
        assert 'start_time' in session
        assert 'reps' in session
        assert 'duration' in session
        assert 'calories' in session
        assert 'status' in session
        print(f"  - Session {session['session_id']}: {session['exercise']}")


def test_filtering_real_data():
    """Test filtering with real workout data"""
    loader = WorkoutHistoryLoader(data_dir="backend/data/reports")
    sessions = loader.load_all_sessions()
    
    if not sessions:
        print("\n⚠ No sessions found, skipping filter test")
        return
    
    filter_obj = WorkoutHistoryFilter()
    
    # Get unique exercise types
    exercise_types = filter_obj.get_unique_exercise_types(sessions)
    print(f"\n✓ Found exercise types: {exercise_types}")
    
    # Test filtering by each type
    for exercise_type in exercise_types:
        filtered = filter_obj.filter_by_exercise(sessions, exercise_type)
        print(f"  - {exercise_type}: {len(filtered)} sessions")
        assert len(filtered) > 0
        
        # Verify all filtered sessions match the type
        for session in filtered:
            session_exercise = session.get('exercise', '').lower()
            assert session_exercise == exercise_type.lower()
    
    # Test filtering by non-existent type (Requirement 2.5)
    filtered_none = filter_obj.filter_by_exercise(sessions, "non_existent_exercise")
    print(f"  - non_existent_exercise: {len(filtered_none)} sessions (expected 0)")
    assert len(filtered_none) == 0


def test_aggregation_real_data():
    """Test aggregation with real workout data"""
    loader = WorkoutHistoryLoader(data_dir="backend/data/reports")
    sessions = loader.load_all_sessions()
    
    if not sessions:
        print("\n⚠ No sessions found, skipping aggregation test")
        return
    
    aggregator = WorkoutHistoryAggregator()
    
    total_workouts = aggregator.calculate_total_workouts(sessions)
    total_reps = aggregator.calculate_total_reps(sessions)
    total_calories = aggregator.calculate_total_calories(sessions)
    total_duration = aggregator.calculate_total_duration(sessions)
    
    print(f"\n✓ Aggregation results:")
    print(f"  - Total workouts: {total_workouts}")
    print(f"  - Total reps: {total_reps}")
    print(f"  - Total calories: {total_calories:.1f}")
    print(f"  - Total duration: {total_duration:.1f}s")
    
    # Verify aggregation makes sense
    assert total_workouts == len(sessions)
    assert total_reps >= 0
    assert total_calories >= 0
    assert total_duration >= 0


def test_formatting_real_data():
    """Test formatting with real workout data"""
    loader = WorkoutHistoryLoader(data_dir="backend/data/reports")
    sessions = loader.load_all_sessions()
    
    if not sessions:
        print("\n⚠ No sessions found, skipping formatting test")
        return
    
    formatter = WorkoutHistoryFormatter()
    
    print(f"\n✓ Formatting test:")
    
    # Test formatting on first session
    session = sessions[0]
    
    formatted_date = formatter.format_date(session['start_time'])
    formatted_duration = formatter.format_duration(session['duration'])
    formatted_name = formatter.format_exercise_name(session['exercise'])
    exercise_icon = formatter.get_exercise_icon(session['exercise'])
    
    print(f"  - Date: {session['start_time']} → {formatted_date}")
    print(f"  - Duration: {session['duration']}s → {formatted_duration}")
    print(f"  - Exercise: {session['exercise']} → {exercise_icon} {formatted_name}")
    
    # Verify formatting doesn't crash
    assert isinstance(formatted_date, str)
    assert isinstance(formatted_duration, str)
    assert isinstance(formatted_name, str)
    assert isinstance(exercise_icon, str)


def test_corrupted_data_scenario():
    """Test scenario with mix of valid and corrupted data (Requirement 1.5)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy a real session file
        real_data_dir = "backend/data/reports"
        if os.path.exists(real_data_dir):
            real_files = [f for f in os.listdir(real_data_dir) if f.endswith('.json')]
            if real_files:
                # Copy first real file
                shutil.copy(
                    os.path.join(real_data_dir, real_files[0]),
                    os.path.join(tmpdir, "valid_session.json")
                )
        
        # Add corrupted files
        Path(tmpdir, "corrupted1.json").write_text("{invalid json")
        Path(tmpdir, "corrupted2.json").write_text("")
        
        # Add invalid session (missing required fields)
        invalid_session = {
            "exercise": "squat",
            "reps": 10
        }
        Path(tmpdir, "invalid.json").write_text(json.dumps(invalid_session))
        
        # Load sessions
        loader = WorkoutHistoryLoader(data_dir=tmpdir)
        sessions = loader.load_all_sessions()
        
        print(f"\n✓ Corrupted data scenario:")
        print(f"  - Created 3 corrupted/invalid files + 1 valid file")
        print(f"  - Loaded {len(sessions)} valid session(s)")
        
        # Should load only the valid session
        assert len(sessions) <= 1  # 1 if real file was copied, 0 otherwise
        
        # If a session was loaded, verify it's valid
        for session in sessions:
            assert 'session_id' in session
            assert 'exercise' in session


def test_empty_directory_scenario():
    """Test scenario with empty data directory (Requirement 1.3)"""
    with tempfile.TemporaryDirectory() as tmpdir:
        loader = WorkoutHistoryLoader(data_dir=tmpdir)
        sessions = loader.load_all_sessions()
        
        print(f"\n✓ Empty directory scenario:")
        print(f"  - Loaded {len(sessions)} sessions (expected 0)")
        
        assert sessions == []
        
        # Test aggregation with empty sessions
        aggregator = WorkoutHistoryAggregator()
        total_workouts = aggregator.calculate_total_workouts(sessions)
        total_reps = aggregator.calculate_total_reps(sessions)
        
        print(f"  - Total workouts: {total_workouts}")
        print(f"  - Total reps: {total_reps}")
        
        assert total_workouts == 0
        assert total_reps == 0


def test_filter_no_matches_scenario():
    """Test scenario where filter returns no matches (Requirement 2.5)"""
    loader = WorkoutHistoryLoader(data_dir="backend/data/reports")
    sessions = loader.load_all_sessions()
    
    if not sessions:
        print("\n⚠ No sessions found, skipping no-match filter test")
        return
    
    filter_obj = WorkoutHistoryFilter()
    aggregator = WorkoutHistoryAggregator()
    
    # Filter by non-existent exercise type
    filtered = filter_obj.filter_by_exercise(sessions, "yoga")
    
    print(f"\n✓ No-match filter scenario:")
    print(f"  - Total sessions: {len(sessions)}")
    print(f"  - Filtered sessions (yoga): {len(filtered)}")
    
    assert len(filtered) == 0
    
    # Test aggregation with empty filtered results
    total_workouts = aggregator.calculate_total_workouts(filtered)
    total_reps = aggregator.calculate_total_reps(filtered)
    total_calories = aggregator.calculate_total_calories(filtered)
    
    print(f"  - Aggregation on empty results:")
    print(f"    - Workouts: {total_workouts}")
    print(f"    - Reps: {total_reps}")
    print(f"    - Calories: {total_calories}")
    
    assert total_workouts == 0
    assert total_reps == 0
    assert total_calories == 0.0


def test_sorting_consistency():
    """Test that sorting maintains consistency"""
    loader = WorkoutHistoryLoader(data_dir="backend/data/reports")
    sessions = loader.load_all_sessions()
    
    if len(sessions) < 2:
        print("\n⚠ Not enough sessions for sorting test")
        return
    
    filter_obj = WorkoutHistoryFilter()
    
    # Sort in reverse chronological order
    sorted_sessions = filter_obj.sort_by_date(sessions, reverse=True)
    
    print(f"\n✓ Sorting test:")
    print(f"  - Total sessions: {len(sorted_sessions)}")
    
    # Verify sorting order (newest first)
    for i in range(len(sorted_sessions) - 1):
        current_time = sorted_sessions[i].get('start_time', '')
        next_time = sorted_sessions[i + 1].get('start_time', '')
        
        if current_time and next_time:
            # Current should be >= next (reverse chronological)
            assert current_time >= next_time, f"Sorting error: {current_time} should be >= {next_time}"
    
    print(f"  - First session: {sorted_sessions[0].get('start_time', 'N/A')}")
    print(f"  - Last session: {sorted_sessions[-1].get('start_time', 'N/A')}")


if __name__ == "__main__":
    print("=" * 60)
    print("WORKOUT HISTORY INTEGRATION TESTS")
    print("=" * 60)
    
    try:
        test_real_data_directory()
        test_filtering_real_data()
        test_aggregation_real_data()
        test_formatting_real_data()
        test_corrupted_data_scenario()
        test_empty_directory_scenario()
        test_filter_no_matches_scenario()
        test_sorting_consistency()
        
        print("\n" + "=" * 60)
        print("✅ ALL INTEGRATION TESTS PASSED")
        print("=" * 60)
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        raise
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        raise
