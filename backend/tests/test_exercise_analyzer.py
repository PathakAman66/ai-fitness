"""
Unit tests for EnhancedExerciseAnalyzer
Tests exercise analysis and rep counting functionality
"""
from backend.api.exercise_analyzer import EnhancedExerciseAnalyzer


class TestEnhancedExerciseAnalyzer:
    """Test suite for EnhancedExerciseAnalyzer"""
    
    def test_initialization(self):
        """Test that analyzer initializes with correct defaults"""
        analyzer = EnhancedExerciseAnalyzer("bicep_curl")
        
        assert analyzer.exercise_type == "bicep_curl"
        assert analyzer.rep_count == 0
        assert analyzer.current_stage == "start"
        assert analyzer.calories_burned == 0.0
        assert analyzer.rep_history == []
    
    def test_calculate_angle(self):
        """Test angle calculation between three points"""
        analyzer = EnhancedExerciseAnalyzer()
        
        # Test 90 degree angle
        a = (0, 1, 0, 1)
        b = (0, 0, 0, 1)
        c = (1, 0, 0, 1)
        
        angle = analyzer.calculate_angle(a, b, c)
        
        assert 85 <= angle <= 95  # Allow small floating point error
    
    def test_calculate_distance(self):
        """Test distance calculation between two points"""
        analyzer = EnhancedExerciseAnalyzer()
        
        point1 = (0, 0, 0, 1)
        point2 = (3, 4, 0, 1)
        
        distance = analyzer.calculate_distance(point1, point2)
        
        assert distance == 5.0  # 3-4-5 triangle
    
    def test_analyze_form_with_no_key_points(self):
        """Test that analyze_form handles None key_points gracefully"""
        analyzer = EnhancedExerciseAnalyzer("bicep_curl")
        
        result = analyzer.analyze_form(None)
        
        assert result is not None
        assert 'errors' in result
        assert 'No person detected' in result['errors']
        assert result['rep_count'] == 0
    
    def test_analyze_form_with_unsupported_exercise(self):
        """Test that unsupported exercise type returns error"""
        analyzer = EnhancedExerciseAnalyzer("invalid_exercise")
        
        key_points = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.5, 0.5, 0, 1),
            'right_wrist': (0.5, 0.7, 0, 1)
        }
        
        result = analyzer.analyze_form(key_points)
        
        assert 'errors' in result
        assert 'Exercise not supported' in result['errors']
    
    def test_bicep_curl_stage_transitions(self):
        """Test bicep curl stage transitions based on elbow angle"""
        analyzer = EnhancedExerciseAnalyzer("bicep_curl")
        
        # Start position (straight arm, angle > 160)
        key_points_start = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.5, 0.5, 0, 1),
            'right_wrist': (0.5, 0.7, 0, 1)
        }
        
        result = analyzer.analyze_form(key_points_start)
        assert analyzer.current_stage == "start"
        
        # Curl up (angle < 80)
        key_points_up = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.5, 0.5, 0, 1),
            'right_wrist': (0.52, 0.35, 0, 1)  # Wrist closer to shoulder
        }
        
        result = analyzer.analyze_form(key_points_up)
        assert analyzer.current_stage == "up"
        
        # Return to start (angle > 160) - should count a rep
        result = analyzer.analyze_form(key_points_start)
        assert analyzer.current_stage == "down"
        assert analyzer.rep_count == 1
    
    def test_squat_returns_expected_fields(self):
        """Test that squat analysis returns expected fields"""
        analyzer = EnhancedExerciseAnalyzer("squat")
        
        key_points = {
            'right_hip': (0.5, 0.4, 0, 1),
            'right_knee': (0.5, 0.7, 0, 1),
            'right_ankle': (0.5, 0.95, 0, 1),
            'right_shoulder': (0.5, 0.2, 0, 1)
        }
        
        result = analyzer.analyze_form(key_points)
        
        # Verify expected fields are present
        assert 'angles' in result
        assert 'knee' in result['angles']
        assert 'hip' in result['angles']
        assert 'rep_count' in result
        assert 'stage' in result
        assert 'feedback' in result
        assert 'errors' in result
        assert 'warnings' in result
    
    def test_plank_returns_duration(self):
        """Test that plank analysis returns duration instead of reps"""
        analyzer = EnhancedExerciseAnalyzer("plank")
        
        key_points = {
            'right_shoulder': (0.5, 0.4, 0, 1),
            'right_hip': (0.5, 0.5, 0, 1),
            'right_ankle': (0.5, 0.6, 0, 1)
        }
        
        result = analyzer.analyze_form(key_points)
        
        assert 'duration' in result
        assert result['duration'] >= 0
        assert result['stage'] == 'hold'
    
    def test_reset_clears_state(self):
        """Test that reset method clears analyzer state"""
        analyzer = EnhancedExerciseAnalyzer("bicep_curl")
        
        # Set some state
        analyzer.rep_count = 10
        analyzer.current_stage = "up"
        analyzer.calories_burned = 5.0
        analyzer.rep_history = [1.0, 2.0, 3.0]
        
        # Reset
        analyzer.reset()
        
        assert analyzer.rep_count == 0
        assert analyzer.current_stage == "start"
        assert analyzer.calories_burned == 0.0
        assert analyzer.rep_history == []
