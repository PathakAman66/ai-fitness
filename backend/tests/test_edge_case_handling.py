"""
Unit tests for edge case handling in pose quality evaluation
Tests no pose detected, missing metrics, and invalid configuration scenarios
"""
import logging
from backend.api.pose_quality_evaluator import PoseQualityEvaluator
from backend.api.exercise_config import get_exercise_config


class TestEdgeCaseHandling:
    """Test suite for edge case handling"""
    
    def test_no_pose_detected_empty_key_points(self):
        """Test that empty key_points returns empty metrics"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Empty key points (no pose detected)
        form_metrics = evaluator.calculate_form_metrics({}, {})
        
        assert form_metrics == {}
    
    def test_no_pose_detected_neutral_feedback(self):
        """Test that no pose detected scenario returns neutral feedback"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Empty metrics should result in neutral score
        quality_score = evaluator.calculate_quality_score({}, {})
        
        assert quality_score == 50.0
    
    def test_missing_key_points_bicep_curl(self):
        """Test that missing key points for bicep curl returns neutral scores"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Missing all key points (all zeros)
        key_points = {
            'right_shoulder': (0, 0, 0, 0),
            'right_elbow': (0, 0, 0, 0),
            'right_wrist': (0, 0, 0, 0)
        }
        
        form_metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert form_metrics == {
            "elbow_angle_score": 50.0,
            "elbow_stability_score": 50.0
        }
    
    def test_missing_key_points_squat(self):
        """Test that missing key points for squat returns neutral scores"""
        evaluator = PoseQualityEvaluator("squat")
        
        # Missing all key points (all zeros)
        key_points = {
            'right_hip': (0, 0, 0, 0),
            'right_knee': (0, 0, 0, 0),
            'right_ankle': (0, 0, 0, 0),
            'right_shoulder': (0, 0, 0, 0)
        }
        
        form_metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert form_metrics == {
            "depth_score": 50.0,
            "knee_alignment_score": 50.0,
            "back_position_score": 50.0
        }
    
    def test_missing_key_points_pushup(self):
        """Test that missing key points for push-up returns neutral scores"""
        evaluator = PoseQualityEvaluator("push_up")
        
        # Missing all key points (all zeros)
        key_points = {
            'right_shoulder': (0, 0, 0, 0),
            'right_elbow': (0, 0, 0, 0),
            'right_wrist': (0, 0, 0, 0),
            'right_hip': (0, 0, 0, 0),
            'right_ankle': (0, 0, 0, 0)
        }
        
        form_metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert form_metrics == {
            "body_alignment_score": 50.0,
            "elbow_angle_score": 50.0
        }
    
    def test_partially_missing_key_points_bicep_curl(self):
        """Test that partially missing key points returns neutral scores"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Only shoulder is valid, others are missing
        key_points = {
            'right_shoulder': (0.5, 0.5, 0.5, 0.9),
            'right_elbow': (0, 0, 0, 0),
            'right_wrist': (0, 0, 0, 0)
        }
        
        form_metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert form_metrics == {
            "elbow_angle_score": 50.0,
            "elbow_stability_score": 50.0
        }
    
    def test_unknown_exercise_type_returns_empty_metrics(self):
        """Test that unknown exercise type returns empty metrics"""
        evaluator = PoseQualityEvaluator("unknown_exercise")
        
        key_points = {
            'right_shoulder': (0.5, 0.5, 0.5, 0.9),
            'right_elbow': (0.4, 0.6, 0.5, 0.9)
        }
        
        form_metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert form_metrics == {}
    
    def test_unknown_exercise_type_uses_default_config(self):
        """Test that unknown exercise type uses default configuration"""
        evaluator = PoseQualityEvaluator("unknown_exercise")
        
        assert evaluator.config.exercise_type == "default"
        assert evaluator.config.thresholds.poor_threshold == 60.0
        assert evaluator.config.thresholds.good_threshold == 85.0
    
    def test_invalid_configuration_fallback(self):
        """Test that invalid configuration falls back to defaults"""
        # This tests the get_exercise_config function directly
        # We can't easily create an invalid config in EXERCISE_CONFIGS
        # but we can test the validation logic
        
        config = get_exercise_config("unknown_type")
        
        assert config.exercise_type == "default"
        assert config.validate() is True
    
    def test_missing_form_metrics_default_to_neutral(self):
        """Test that missing form metrics default to 50.0 in quality score calculation"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Only provide one metric, the other should default to 50.0
        form_metrics = {
            "elbow_angle_score": 80.0
            # elbow_stability_score is missing
        }
        
        quality_score = evaluator.calculate_quality_score(form_metrics, {})
        
        # Should be between 50 and 80 (weighted average with one at 80, one at 50)
        assert 50.0 <= quality_score <= 80.0
    
    def test_empty_metrics_with_no_config_weights(self):
        """Test that empty metrics with no configured weights returns neutral score"""
        evaluator = PoseQualityEvaluator("default")
        
        # Default config has no metric weights
        quality_score = evaluator.calculate_quality_score({}, {})
        
        assert quality_score == 50.0
    
    def test_feedback_message_with_empty_metrics(self):
        """Test that feedback message generation works with empty metrics"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Empty metrics should still generate a message
        message = evaluator.generate_feedback_message(
            quality_score=50.0,
            category='average',
            form_metrics={},
            analysis_result={}
        )
        
        assert len(message) > 0
        assert isinstance(message, str)
    
    def test_configuration_validation_logs_error(self, caplog):
        """Test that invalid configuration logs an error"""
        with caplog.at_level(logging.WARNING):
            config = get_exercise_config("nonexistent_exercise")
        
        assert "Unknown exercise type" in caplog.text
        assert config.exercise_type == "default"
    
    def test_missing_key_points_logs_warning(self, caplog):
        """Test that missing key points logs a warning"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        with caplog.at_level(logging.WARNING):
            key_points = {
                'right_shoulder': (0, 0, 0, 0),
                'right_elbow': (0, 0, 0, 0),
                'right_wrist': (0, 0, 0, 0)
            }
            evaluator.calculate_form_metrics({}, key_points)
        
        assert "Missing key points" in caplog.text
    
    def test_empty_key_points_logs_warning(self, caplog):
        """Test that empty key points logs a warning"""
        evaluator = PoseQualityEvaluator("squat")
        
        with caplog.at_level(logging.WARNING):
            evaluator.calculate_form_metrics({}, {})
        
        assert "No key points provided" in caplog.text
    
    def test_quality_score_with_all_missing_metrics(self):
        """Test quality score calculation when all metrics are missing"""
        evaluator = PoseQualityEvaluator("squat")
        
        # Provide empty metrics dict
        quality_score = evaluator.calculate_quality_score({}, {})
        
        # Should default to 50.0 (neutral)
        assert quality_score == 50.0
    
    def test_feedback_category_with_neutral_score(self):
        """Test feedback category with neutral score from missing data"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        category = evaluator.get_feedback_category(50.0)
        
        # 50.0 is below 60.0 threshold, so should be 'poor'
        assert category == 'poor'
    
    def test_valid_key_points_after_missing(self):
        """Test that valid key points work correctly after handling missing ones"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # First call with missing key points
        missing_key_points = {
            'right_shoulder': (0, 0, 0, 0),
            'right_elbow': (0, 0, 0, 0),
            'right_wrist': (0, 0, 0, 0)
        }
        metrics1 = evaluator.calculate_form_metrics({}, missing_key_points)
        assert metrics1 == {
            "elbow_angle_score": 50.0,
            "elbow_stability_score": 50.0
        }
        
        # Second call with valid key points
        valid_key_points = {
            'right_shoulder': (0.5, 0.5, 0.5, 0.9),
            'right_elbow': (0.4, 0.6, 0.5, 0.9),
            'right_wrist': (0.3, 0.7, 0.5, 0.9)
        }
        metrics2 = evaluator.calculate_form_metrics({}, valid_key_points)
        
        # Should calculate actual metrics, not neutral scores
        assert 'elbow_angle_score' in metrics2
        assert 'elbow_stability_score' in metrics2
        # Scores should be calculated, not necessarily 50.0
        assert isinstance(metrics2['elbow_angle_score'], float)
        assert isinstance(metrics2['elbow_stability_score'], float)
