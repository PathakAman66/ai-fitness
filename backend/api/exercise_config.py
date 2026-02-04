"""
Exercise configuration infrastructure for pose quality evaluation.

This module defines the configuration structure for exercise-specific
threshold-based feedback, including metric weights and feedback templates.
"""

from dataclasses import dataclass
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)


# -----------------------------
# Core Data Models
# -----------------------------

@dataclass
class ThresholdConfig:
    """Configuration for pose quality thresholds."""
    poor_threshold: float = 60.0
    good_threshold: float = 85.0

    def validate(self) -> bool:
        if self.poor_threshold >= self.good_threshold:
            logger.error(
                f"Invalid threshold configuration: poor_threshold ({self.poor_threshold}) "
                f"must be less than good_threshold ({self.good_threshold})"
            )
            return False
        return True


@dataclass
class MetricWeight:
    """Weight configuration for a form metric."""
    name: str
    weight: float
    is_critical: bool


@dataclass
class ExerciseConfig:
    """Complete configuration for an exercise type."""
    exercise_type: str
    thresholds: ThresholdConfig
    metric_weights: List[MetricWeight]
    feedback_templates: Dict[str, List[str]]

    def validate(self) -> bool:
        # Validate thresholds
        if not self.thresholds.validate():
            return False

        # Normalize exercise type
        self.exercise_type = self.exercise_type.lower().strip()

        # Validate feedback categories
        required_categories = {"poor", "average", "excellent"}
        if not required_categories.issubset(self.feedback_templates.keys()):
            logger.error(
                f"Missing feedback templates for {self.exercise_type}. "
                f"Required: {required_categories}, Found: {set(self.feedback_templates.keys())}"
            )
            return False

        # Ensure feedback lists are not empty
        for k, v in self.feedback_templates.items():
            if not v:
                logger.error(
                    f"Feedback list for '{k}' in {self.exercise_type} is empty."
                )
                return False

        # Validate metric weights
        total_weight = sum(m.weight for m in self.metric_weights)

        if self.metric_weights:
            if not (0.95 <= total_weight <= 1.05):
                logger.error(
                    f"Metric weights for {self.exercise_type} must sum to 1.0, "
                    f"got {total_weight}"
                )
                return False

            for m in self.metric_weights:
                if not (0.0 <= m.weight <= 1.0):
                    logger.error(
                        f"Invalid weight for {m.name}: {m.weight}. Must be between 0 and 1."
                    )
                    return False

        return True


# -----------------------------
# Feedback Templates
# -----------------------------

BICEP_CURL_FEEDBACK = {
    "poor": [
        "Lock those elbows in! Keep them tight to your sides.",
        "Slow down and control the movement. Quality over speed!",
        "Extend your arms fully at the bottom - get that full range!",
    ],
    "average": [
        "Good effort! Try keeping your elbows even closer to your body.",
        "Nice work! Focus on that full extension at the bottom.",
        "You're getting there! Maintain a steady, controlled tempo.",
    ],
    "excellent": [
        "Perfect form! Keep it up!",
        "Excellent control! That's how it's done!",
        "Outstanding technique! You're crushing it!",
    ]
}

SQUAT_FEEDBACK = {
    "poor": [
        "Go deeper! Aim to get your thighs parallel to the ground.",
        "Keep your knees aligned with your feet - don't let them cave in!",
        "Straighten that back! Engage your core for stability.",
    ],
    "average": [
        "Good depth! Try going just a bit lower for maximum benefit.",
        "Nice form! Keep focusing on that knee alignment.",
        "Solid work! Remember to push through your heels.",
    ],
    "excellent": [
        "Perfect squat! That's textbook form!",
        "Incredible depth and control! Keep it going!",
        "Flawless technique! You're a squat master!",
    ]
}

PUSHUP_FEEDBACK = {
    "poor": [
        "Keep your body straight! Don't let those hips sag.",
        "Lower yourself more - aim for a 90-degree elbow bend.",
        "Engage your core! Your body should be a straight line.",
    ],
    "average": [
        "Good form! Try to keep your body even straighter.",
        "Nice work! Focus on that full range of motion.",
        "You're doing well! Keep that core tight throughout.",
    ],
    "excellent": [
        "Perfect push-up! Textbook form!",
        "Excellent alignment and control! Keep it up!",
        "Outstanding! That's how push-ups are done!",
    ]
}


# -----------------------------
# Exercise Configurations
# -----------------------------

EXERCISE_CONFIGS = {
    "bicep_curl": ExerciseConfig(
        exercise_type="bicep_curl",
        thresholds=ThresholdConfig(),
        metric_weights=[
            MetricWeight("elbow_angle_score", 0.5, True),
            MetricWeight("elbow_stability_score", 0.5, True),
        ],
        feedback_templates=BICEP_CURL_FEEDBACK
    ),
    "squat": ExerciseConfig(
        exercise_type="squat",
        thresholds=ThresholdConfig(),
        metric_weights=[
            MetricWeight("depth_score", 0.33, False),
            MetricWeight("knee_alignment_score", 0.33, True),
            MetricWeight("back_position_score", 0.34, True),
        ],
        feedback_templates=SQUAT_FEEDBACK
    ),
    "push_up": ExerciseConfig(
        exercise_type="push_up",
        thresholds=ThresholdConfig(),
        metric_weights=[
            MetricWeight("body_alignment_score", 0.5, True),
            MetricWeight("elbow_angle_score", 0.5, False),
        ],
        feedback_templates=PUSHUP_FEEDBACK
    ),
}


DEFAULT_CONFIG = ExerciseConfig(
    exercise_type="default",
    thresholds=ThresholdConfig(),
    metric_weights=[],
    feedback_templates={
        "poor": ["Focus on your form!"],
        "average": ["Good work! Keep improving!"],
        "excellent": ["Perfect! Keep it up!"]
    }
)


# -----------------------------
# Public Access API
# -----------------------------

def get_exercise_config(exercise_type: str) -> ExerciseConfig:
    if not exercise_type:
        logger.warning("Empty exercise type provided, using default config.")
        return DEFAULT_CONFIG

    exercise_type = exercise_type.lower().strip()
    config = EXERCISE_CONFIGS.get(exercise_type)

    if config is None:
        logger.warning(
            f"Unknown exercise type '{exercise_type}', using default configuration. "
            f"Available: {list(EXERCISE_CONFIGS.keys())}"
        )
        return DEFAULT_CONFIG

    if not config.validate():
        logger.error(
            f"Invalid configuration for '{exercise_type}', falling back to default."
        )
        return DEFAULT_CONFIG

    return config
