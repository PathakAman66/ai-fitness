"""
Tests for form metric calculation in PoseQualityEvaluator
Validates that calculate_form_metrics returns appropriate scores for each exercise
"""
from backend.api.pose_quality_evaluator import PoseQualityEvaluator


class TestFormMetrics:
    """Test suite for form metric calculation"""
    
    def test_bicep_curl_metrics_perfect_form(self):
        """Test bicep curl metrics with perfect form"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Perfect form: elbow at 170° (extended), elbow close to shoulder
        key_points = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.52, 0.5, 0, 1),  # Very close to shoulder horizontally
            'right_wrist': (0.54, 0.7, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert 'elbow_angle_score' in metrics
        assert 'elbow_stability_score' in metrics
        assert metrics['elbow_angle_score'] >= 80.0  # Good extension
        assert metrics['elbow_stability_score'] >= 90.0  # Excellent stability
    
    def test_bicep_curl_metrics_poor_form(self):
        """Test bicep curl metrics with poor form"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Poor form: elbows drifting out
        key_points = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.7, 0.5, 0, 1),  # Far from shoulder (0.2 distance)
            'right_wrist': (0.72, 0.7, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert 'elbow_angle_score' in metrics
        assert 'elbow_stability_score' in metrics
        # With 0.2 distance, stability score should be low
        assert metrics['elbow_stability_score'] < 60.0  # Poor stability
    
    def test_squat_metrics_perfect_form(self):
        """Test squat metrics with perfect form"""
        evaluator = PoseQualityEvaluator("squat")
        
        # Test that squat metrics are calculated
        key_points = {
            'right_shoulder': (0.5, 0.2, 0, 1),
            'right_hip': (0.5, 0.5, 0, 1),
            'right_knee': (0.52, 0.7, 0, 1),
            'right_ankle': (0.5, 0.9, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        # Verify all expected metrics are present
        assert 'depth_score' in metrics
        assert 'knee_alignment_score' in metrics
        assert 'back_position_score' in metrics
        
        # Verify all scores are in valid range
        assert 0.0 <= metrics['depth_score'] <= 100.0
        assert 0.0 <= metrics['knee_alignment_score'] <= 100.0
        assert 0.0 <= metrics['back_position_score'] <= 100.0
    
    def test_squat_metrics_shallow_depth(self):
        """Test squat metrics with shallow depth"""
        evaluator = PoseQualityEvaluator("squat")
        
        # Shallow squat: knee angle around 150°
        key_points = {
            'right_shoulder': (0.5, 0.2, 0, 1),
            'right_hip': (0.5, 0.4, 0, 1),
            'right_knee': (0.5, 0.6, 0, 1),
            'right_ankle': (0.5, 0.9, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert 'depth_score' in metrics
        # Shallow depth should result in lower score
        assert metrics['depth_score'] < 60.0
    
    def test_pushup_metrics_perfect_form(self):
        """Test push-up metrics with perfect form"""
        evaluator = PoseQualityEvaluator("push_up")
        
        # Perfect form: straight body (170° angle), good elbow angle
        key_points = {
            'right_shoulder': (0.5, 0.4, 0, 1),
            'right_elbow': (0.5, 0.42, 0, 1),
            'right_wrist': (0.5, 0.44, 0, 1),
            'right_hip': (0.5, 0.6, 0, 1),
            'right_ankle': (0.5, 0.8, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert 'body_alignment_score' in metrics
        assert 'elbow_angle_score' in metrics
        assert metrics['body_alignment_score'] >= 90.0  # Excellent alignment
    
    def test_pushup_metrics_sagging_body(self):
        """Test push-up metrics with sagging body"""
        evaluator = PoseQualityEvaluator("push_up")
        
        # Test that push-up metrics are calculated
        key_points = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.5, 0.32, 0, 1),
            'right_wrist': (0.5, 0.34, 0, 1),
            'right_hip': (0.5, 0.65, 0, 1),
            'right_ankle': (0.5, 0.7, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        # Verify metrics are present and in valid range
        assert 'body_alignment_score' in metrics
        assert 0.0 <= metrics['body_alignment_score'] <= 100.0
    
    def test_unknown_exercise_returns_empty_metrics(self):
        """Test that unknown exercise type returns empty metrics"""
        evaluator = PoseQualityEvaluator("unknown_exercise")
        
        key_points = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.5, 0.5, 0, 1),
            'right_wrist': (0.5, 0.7, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        assert metrics == {}
    
    def test_form_metrics_with_missing_key_points(self):
        """Test that form metrics handle missing key points gracefully"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Missing wrist key point
        key_points = {
            'right_shoulder': (0.5, 0.3, 0, 1),
            'right_elbow': (0.5, 0.5, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        
        # Should still return metrics (with default values for missing points)
        assert 'elbow_angle_score' in metrics
        assert 'elbow_stability_score' in metrics
    
    def test_bicep_curl_angle_calculation(self):
        """Test that bicep curl angle calculation works correctly"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Create points that form a specific angle
        shoulder = (0.0, 0.0, 0, 1)
        elbow = (0.0, 1.0, 0, 1)
        wrist = (1.0, 1.0, 0, 1)  # Forms 90° angle
        
        angle = evaluator.calculate_angle(shoulder, elbow, wrist)
        
        # Should be approximately 90 degrees
        assert 85.0 <= angle <= 95.0
    
    def test_squat_depth_scoring_ranges(self):
        """Test squat depth scoring with different configurations"""
        evaluator = PoseQualityEvaluator("squat")
        
        # Test that depth score is calculated
        key_points = {
            'right_shoulder': (0.5, 0.2, 0, 1),
            'right_hip': (0.5, 0.5, 0, 1),
            'right_knee': (0.52, 0.7, 0, 1),
            'right_ankle': (0.5, 0.9, 0, 1)
        }
        
        metrics = evaluator.calculate_form_metrics({}, key_points)
        # Verify depth score is in valid range
        assert 0.0 <= metrics['depth_score'] <= 100.0
    
    def test_all_metrics_return_valid_scores(self):
        """Test that all metrics return scores in valid range (0-100)"""
        exercises = ["bicep_curl", "squat", "push_up"]
        
        for exercise in exercises:
            evaluator = PoseQualityEvaluator(exercise)
            
            # Generic key points
            key_points = {
                'right_shoulder': (0.5, 0.3, 0, 1),
                'right_elbow': (0.5, 0.5, 0, 1),
                'right_wrist': (0.5, 0.7, 0, 1),
                'right_hip': (0.5, 0.6, 0, 1),
                'right_knee': (0.5, 0.75, 0, 1),
                'right_ankle': (0.5, 0.9, 0, 1)
            }
            
            metrics = evaluator.calculate_form_metrics({}, key_points)
            
            # All metric scores should be in valid range
            for metric_name, score in metrics.items():
                assert 0.0 <= score <= 100.0, f"{exercise} {metric_name} out of range: {score}"
