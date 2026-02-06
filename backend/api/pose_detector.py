"""
Enhanced Pose Detector for API use
Production-ready version without print statements.
Uses structured logging and clean error handling.
"""

import mediapipe as mp
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Tuple, Any


logger = logging.getLogger(__name__)


class EnhancedPoseDetector:
    """
    Detects human pose landmarks from images using MediaPipe.
    Designed for backend/API usage.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize MediaPipe pose detector.

        Raises:
            ValueError: If confidence values are invalid
            RuntimeError: If MediaPipe initialization fails
        """

        if not (0.0 <= min_detection_confidence <= 1.0):
            raise ValueError(
                f"min_detection_confidence must be between 0.0 and 1.0, got {min_detection_confidence}"
            )

        if not (0.0 <= min_tracking_confidence <= 1.0):
            raise ValueError(
                f"min_tracking_confidence must be between 0.0 and 1.0, got {min_tracking_confidence}"
            )

        try:
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils

            self.pose = self.mp_pose.Pose(
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

            self.initialized = True
            logger.info("Pose detector initialized successfully")

        except Exception as e:
            self.initialized = False
            logger.exception("Failed to initialize pose detector")
            raise RuntimeError("MediaPipe Pose initialization failed") from e

    def detect_pose(
        self,
        image: np.ndarray,
        draw_landmarks: bool = False,
    ) -> Tuple[Optional[Any], np.ndarray]:
        """
        Detect pose landmarks in an image.

        Returns:
            (landmarks, processed_image)
        """

        if not self.initialized:
            raise RuntimeError("Pose detector not initialized")

        if image is None or image.size == 0:
            raise ValueError("Invalid image provided")

        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Image must be a 3-channel BGR image")

        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False

            results = self.pose.process(image_rgb)

            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            landmarks = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image_bgr,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(
                            color=(0, 255, 0),
                            thickness=2,
                            circle_radius=3,
                        ),
                        self.mp_drawing.DrawingSpec(
                            color=(255, 0, 0),
                            thickness=2,
                        ),
                    )

            return landmarks, image_bgr

        except cv2.error as e:
            logger.exception("OpenCV error during pose detection")
            raise RuntimeError("OpenCV processing failed") from e

        except Exception as e:
            logger.exception("Unexpected error during pose detection")
            raise RuntimeError("Pose detection failed") from e

    def extract_key_points(
        self,
        landmarks: Any,
    ) -> Optional[Dict[str, Tuple[float, float, float, float]]]:
        """
        Extract selected key landmarks into dictionary.
        """

        if landmarks is None:
            return None

        landmark_indices = {
            "nose": 0,
            "left_eye": 1,
            "right_eye": 2,
            "left_ear": 7,
            "right_ear": 8,
            "left_shoulder": 11,
            "right_shoulder": 12,
            "left_elbow": 13,
            "right_elbow": 14,
            "left_wrist": 15,
            "right_wrist": 16,
            "left_hip": 23,
            "right_hip": 24,
            "left_knee": 25,
            "right_knee": 26,
            "left_ankle": 27,
            "right_ankle": 28,
            "left_heel": 29,
            "right_heel": 30,
            "left_foot_index": 31,
            "right_foot_index": 32,
        }

        key_points = {}

        for name, idx in landmark_indices.items():
            if idx < len(landmarks):
                lm = landmarks[idx]
                key_points[name] = (lm.x, lm.y, lm.z, lm.visibility)

        return key_points

    def close(self):
        """Release MediaPipe resources."""
        if self.pose:
            self.pose.close()
