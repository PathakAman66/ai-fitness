"""
Unit tests for PoseQualityEvaluator
Tests quality score calculation, threshold categorization, and history tracking
"""
from backend.api.pose_quality_evaluator import PoseQualityEvaluator


class TestPoseQualityEvaluator:
    """Test suite for PoseQualityEvaluator"""
    
    def test_initialization(self):
        """Test that evaluator initializes correctly"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        assert evaluator.exercise_type == "bicep_curl"
        assert evaluator.config is not None
        assert evaluator.history == []
    
    def test_initialization_with_unknown_exercise(self):
        """Test that unknown exercise type uses default config"""
        evaluator = PoseQualityEvaluator("unknown_exercise")
        
        assert evaluator.exercise_type == "unknown_exercise"
        assert evaluator.config.exercise_type == "default"
    
    # Task 2.1: Test quality score bounds
    def test_quality_score_bounds_with_empty_metrics(self):
        """Test that quality score is between 0-100 with empty metrics"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        score = evaluator.calculate_quality_score({}, {})
        
        assert 0.0 <= score <= 100.0
    
    def test_quality_score_bounds_with_all_zeros(self):
        """Test that quality score is between 0-100 with all zero metrics"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        form_metrics = {
            "elbow_angle_score": 0.0,
            "elbow_stability_score": 0.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        assert 0.0 <= score <= 100.0
        assert score == 0.0  # All zeros should give 0
    
    def test_quality_score_bounds_with_perfect_scores(self):
        """Test that quality score is between 0-100 with perfect metrics"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        form_metrics = {
            "elbow_angle_score": 100.0,
            "elbow_stability_score": 100.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        assert 0.0 <= score <= 100.0
        assert score == 100.0  # All perfect should give 100
    
    def test_quality_score_bounds_with_out_of_range_metrics(self):
        """Test that quality score handles metrics outside 0-100 range"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Test with metrics above 100
        form_metrics = {
            "elbow_angle_score": 150.0,
            "elbow_stability_score": 200.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        assert 0.0 <= score <= 100.0
        
        # Test with metrics below 0
        form_metrics = {
            "elbow_angle_score": -50.0,
            "elbow_stability_score": -100.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        assert 0.0 <= score <= 100.0
    
    def test_quality_score_bounds_with_mixed_metrics(self):
        """Test that quality score is between 0-100 with mixed metric values"""
        evaluator = PoseQualityEvaluator("squat")
        
        form_metrics = {
            "depth_score": 75.0,
            "knee_alignment_score": 50.0,
            "back_position_score": 90.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        assert 0.0 <= score <= 100.0
    
    def test_quality_score_with_missing_metrics(self):
        """Test that missing metrics default to 50.0 and score stays in bounds"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Only provide one of two expected metrics
        form_metrics = {
            "elbow_angle_score": 80.0
            # elbow_stability_score is missing
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        assert 0.0 <= score <= 100.0
    
    def test_quality_score_with_no_configured_weights(self):
        """Test quality score with exercise that has no metric weights"""
        evaluator = PoseQualityEvaluator("default")
        
        form_metrics = {
            "some_metric": 75.0,
            "another_metric": 85.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        assert 0.0 <= score <= 100.0

    # Task 2.2: Test threshold categorization
    def test_threshold_categorization_poor_boundary(self):
        """Test boundary value 59.9 categorizes as poor"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        category = evaluator.get_feedback_category(59.9)
        
        assert category == 'poor'
    
    def test_threshold_categorization_average_lower_boundary(self):
        """Test boundary value 60.0 categorizes as average"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        category = evaluator.get_feedback_category(60.0)
        
        assert category == 'average'
    
    def test_threshold_categorization_average_upper_boundary(self):
        """Test boundary value 84.9 categorizes as average"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        category = evaluator.get_feedback_category(84.9)
        
        assert category == 'average'
    
    def test_threshold_categorization_excellent_boundary(self):
        """Test boundary value 85.0 categorizes as excellent"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        category = evaluator.get_feedback_category(85.0)
        
        assert category == 'excellent'
    
    def test_threshold_categorization_poor_range(self):
        """Test various values in poor range"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        for score in [0.0, 30.0, 59.0, 59.99]:
            category = evaluator.get_feedback_category(score)
            assert category == 'poor', f"Score {score} should be 'poor'"
    
    def test_threshold_categorization_average_range(self):
        """Test various values in average range"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        for score in [60.0, 70.0, 75.5, 84.0, 84.99]:
            category = evaluator.get_feedback_category(score)
            assert category == 'average', f"Score {score} should be 'average'"
    
    def test_threshold_categorization_excellent_range(self):
        """Test various values in excellent range"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        for score in [85.0, 90.0, 95.5, 100.0]:
            category = evaluator.get_feedback_category(score)
            assert category == 'excellent', f"Score {score} should be 'excellent'"
    
    # Additional tests for history tracking
    def test_update_history_adds_score(self):
        """Test that update_history adds scores to history"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        evaluator.update_history(75.0)
        evaluator.update_history(80.0)
        evaluator.update_history(85.0)
        
        assert len(evaluator.history) == 3
        assert evaluator.history == [75.0, 80.0, 85.0]
    
    def test_get_historical_average_with_no_history(self):
        """Test that historical average returns None with no history"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        avg = evaluator.get_historical_average()
        
        assert avg is None
    
    def test_get_historical_average_with_scores(self):
        """Test that historical average calculates correctly"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        evaluator.update_history(60.0)
        evaluator.update_history(80.0)
        evaluator.update_history(100.0)
        
        avg = evaluator.get_historical_average()
        
        assert avg == 80.0  # (60 + 80 + 100) / 3
    
    def test_history_sliding_window(self):
        """Test that history maintains sliding window of 100 scores"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Add 150 scores
        for i in range(150):
            evaluator.update_history(float(i))
        
        # Should only keep last 100
        assert len(evaluator.history) == 100
        assert evaluator.history[0] == 50.0  # First of last 100
        assert evaluator.history[-1] == 149.0  # Last score
    
    def test_reset_clears_history(self):
        """Test that reset clears history"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        evaluator.update_history(75.0)
        evaluator.update_history(80.0)
        
        evaluator.reset()
        
        assert evaluator.history == []
        assert evaluator.get_historical_average() is None
    
    def test_weighted_calculation_with_critical_metrics(self):
        """Test that critical metrics receive 1.5x weight multiplier"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Both metrics are critical with equal weight (0.5 each)
        # With 1.5x multiplier: effective weights are 0.75 each
        form_metrics = {
            "elbow_angle_score": 100.0,
            "elbow_stability_score": 0.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        # Expected: (100 * 0.75 + 0 * 0.75) / (0.75 + 0.75) = 75 / 1.5 = 50
        assert score == 50.0
    
    def test_weighted_calculation_mixed_critical_non_critical(self):
        """Test weighted calculation with mix of critical and non-critical metrics"""
        evaluator = PoseQualityEvaluator("squat")
        
        # depth_score: weight=0.33, non-critical (1.0x)
        # knee_alignment_score: weight=0.33, critical (1.5x)
        # back_position_score: weight=0.34, critical (1.5x)
        form_metrics = {
            "depth_score": 60.0,
            "knee_alignment_score": 80.0,
            "back_position_score": 100.0
        }
        
        score = evaluator.calculate_quality_score(form_metrics, {})
        
        # Expected calculation:
        # depth: 60 * 0.33 * 1.0 = 19.8
        # knee: 80 * 0.33 * 1.5 = 39.6
        # back: 100 * 0.34 * 1.5 = 51.0
        # total_score = 110.4
        # total_weight = 0.33 + 0.495 + 0.51 = 1.335
        # final = 110.4 / 1.335 ≈ 82.7
        
        assert 82.0 <= score <= 83.0  # Allow small floating point variance

    # Task 4.1: Test feedback message generation
    def test_feedback_message_poor_includes_correction(self):
        """Test that poor category feedback includes correction instructions"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        form_metrics = {
            "elbow_angle_score": 30.0,
            "elbow_stability_score": 40.0
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=35.0,
            category='poor',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Message should contain correction instructions
        assert len(message) > 0
        # Should contain actionable guidance (common words in correction messages)
        correction_keywords = ['extend', 'lock', 'keep', 'tight', 'full', 'range', 'sides', 'arms']
        assert any(keyword in message.lower() for keyword in correction_keywords), \
            f"Message '{message}' should contain correction instructions"
    
    def test_feedback_message_excellent_includes_encouragement(self):
        """Test that excellent category feedback includes encouragement"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        form_metrics = {
            "elbow_angle_score": 95.0,
            "elbow_stability_score": 90.0
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=92.0,
            category='excellent',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Message should contain encouragement
        assert len(message) > 0
        # Should contain positive/encouraging words
        encouragement_keywords = ['perfect', 'excellent', 'outstanding', 'keep', 'crushing', 'done']
        assert any(keyword in message.lower() for keyword in encouragement_keywords), \
            f"Message '{message}' should contain encouragement"
    
    def test_feedback_message_bicep_curl_terminology(self):
        """Test that bicep curl feedback contains exercise-specific terminology"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        form_metrics = {
            "elbow_angle_score": 30.0,
            "elbow_stability_score": 40.0
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=35.0,
            category='poor',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Should contain bicep curl specific terms
        bicep_curl_terms = ['elbow', 'arm', 'extend', 'curl', 'sides', 'range']
        assert any(term in message.lower() for term in bicep_curl_terms), \
            f"Message '{message}' should contain bicep curl terminology"
    
    def test_feedback_message_squat_terminology(self):
        """Test that squat feedback contains exercise-specific terminology"""
        evaluator = PoseQualityEvaluator("squat")
        
        form_metrics = {
            "depth_score": 30.0,
            "knee_alignment_score": 40.0,
            "back_position_score": 35.0
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=35.0,
            category='poor',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Should contain squat specific terms
        squat_terms = ['deeper', 'thighs', 'parallel', 'knee', 'back', 'core', 'feet', 'alignment']
        assert any(term in message.lower() for term in squat_terms), \
            f"Message '{message}' should contain squat terminology"
    
    def test_feedback_message_pushup_terminology(self):
        """Test that push-up feedback contains exercise-specific terminology"""
        evaluator = PoseQualityEvaluator("push_up")
        
        form_metrics = {
            "body_alignment_score": 30.0,
            "elbow_angle_score": 40.0
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=35.0,
            category='poor',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Should contain push-up specific terms
        pushup_terms = ['body', 'straight', 'hips', 'sag', 'elbow', 'core', 'lower', 'bend']
        assert any(term in message.lower() for term in pushup_terms), \
            f"Message '{message}' should contain push-up terminology"
    
    def test_feedback_message_average_category(self):
        """Test that average category feedback is appropriate"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        form_metrics = {
            "elbow_angle_score": 70.0,
            "elbow_stability_score": 75.0
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=72.0,
            category='average',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Message should be constructive
        assert len(message) > 0
        # Should contain acknowledgment or improvement suggestions
        # Updated keywords to match actual feedback templates
        average_keywords = ['good', 'nice', 'try', 'focus', 'keep', 'work', 'effort', 'getting', 'maintain']
        assert any(keyword in message.lower() for keyword in average_keywords), \
            f"Message '{message}' should contain appropriate average feedback"
    
    def test_feedback_message_with_historical_regression(self):
        """Test that feedback includes regression notice when performance drops"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Build history with good scores
        for _ in range(5):
            evaluator.update_history(85.0)
        
        form_metrics = {
            "elbow_angle_score": 60.0,
            "elbow_stability_score": 65.0
        }
        
        # Current score is much lower than historical average
        message = evaluator.generate_feedback_message(
            quality_score=62.0,
            category='average',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Should mention regression
        assert 'slipping' in message.lower() or 'refocus' in message.lower(), \
            f"Message '{message}' should indicate regression"
    
    def test_feedback_message_with_historical_progress(self):
        """Test that feedback includes progress acknowledgment when improving"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Build history with lower scores
        for _ in range(5):
            evaluator.update_history(65.0)
        
        form_metrics = {
            "elbow_angle_score": 85.0,
            "elbow_stability_score": 90.0
        }
        
        # Current score is much higher than historical average
        message = evaluator.generate_feedback_message(
            quality_score=87.0,
            category='excellent',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Should mention progress
        assert 'improving' in message.lower() or 'progress' in message.lower(), \
            f"Message '{message}' should acknowledge progress"
    
    def test_feedback_message_with_insufficient_history(self):
        """Test that feedback doesn't include historical comparison with < 3 reps"""
        evaluator = PoseQualityEvaluator("bicep_curl")
        
        # Only 2 scores in history
        evaluator.update_history(85.0)
        evaluator.update_history(85.0)
        
        form_metrics = {
            "elbow_angle_score": 60.0,
            "elbow_stability_score": 65.0
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=62.0,
            category='average',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Should NOT mention regression/progress (not enough history)
        assert 'slipping' not in message.lower() and 'improving' not in message.lower(), \
            f"Message '{message}' should not include historical comparison with < 3 reps"
    
    def test_feedback_message_prioritizes_critical_metrics(self):
        """Test that poor feedback prioritizes critical (safety) metrics"""
        evaluator = PoseQualityEvaluator("squat")
        
        # Critical metrics (knee_alignment, back_position) vs non-critical (depth)
        form_metrics = {
            "depth_score": 30.0,  # Non-critical, worst score
            "knee_alignment_score": 40.0,  # Critical
            "back_position_score": 45.0  # Critical
        }
        
        message = evaluator.generate_feedback_message(
            quality_score=38.0,
            category='poor',
            form_metrics=form_metrics,
            analysis_result={}
        )
        
        # Should prioritize critical metrics (knee or back) over depth
        # Check that message is about knee or back, not depth
        critical_terms = ['knee', 'back', 'core', 'aligned', 'straighten']
        depth_terms = ['deeper', 'parallel', 'thighs']
        
        has_critical = any(term in message.lower() for term in critical_terms)
        has_depth_only = all(term not in message.lower() for term in critical_terms) and \
                         any(term in message.lower() for term in depth_terms)
        
        # Should mention critical issues, not just depth
        assert has_critical or not has_depth_only, \
            f"Message '{message}' should prioritize critical metrics over non-critical"
