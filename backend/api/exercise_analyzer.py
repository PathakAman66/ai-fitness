"""
Enhanced Pose Detector for API use
Production-safe, optimized & thread-safe
"""

import cv2
import mediapipe as mp
import numpy as np
import logging
import threading
from typing import Optional, Dict, Tuple, List

logger = logging.getLogger("pose")


class EnhancedPoseDetector:

    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        if not 0 <= min_detection_confidence <= 1:
            raise ValueError("min_detection_confidence must be 0-1")
        if not 0 <= min_tracking_confidence <= 1:
            raise ValueError("min_tracking_confidence must be 0-1")

        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        self.lock = threading.Lock()
        self.prev_landmarks: Optional[List[Tuple[float, float, float, float]]] = None
        self.alpha = 0.6  # EMA smoothing

        logger.info("EnhancedPoseDetector initialized")

    # -----------------------------
    # Internal helpers
    # -----------------------------

    def _resize(self, image):
        h, w = image.shape[:2]
        if w > 640:
            scale = 640 / w
            image = cv2.resize(image, None, fx=scale, fy=scale)
        return image

    def _smooth(self, landmarks):
        current = [
            (lm.x, lm.y, lm.z, lm.visibility if lm.visibility is not None else 1.0)
            for lm in landmarks
        ]

        if self.prev_landmarks is None or len(self.prev_landmarks) != len(current):
            self.prev_landmarks = current
            return current

        smoothed = []
        for p, c in zip(self.prev_landmarks, current):
            smoothed.append((
                self.alpha * c[0] + (1 - self.alpha) * p[0],
                self.alpha * c[1] + (1 - self.alpha) * p[1],
                self.alpha * c[2] + (1 - self.alpha) * p[2],
                self.alpha * c[3] + (1 - self.alpha) * p[3],
            ))

        self.prev_landmarks = smoothed
        return smoothed

    # -----------------------------
    # Main API
    # -----------------------------

    def detect_pose(self, image: np.ndarray, draw_landmarks=False):
        if image is None or image.size == 0:
            logger.error("Empty image")
            return None, image

        if len(image.shape) != 3 or image.shape[2] != 3:
            logger.error("Invalid image format")
            return None, image

        image = self._resize(image)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        with self.lock:
            results = self.pose.process(rgb)

            landmarks = None
            if results.pose_landmarks:
                landmarks = self._smooth(results.pose_landmarks.landmark)

                if draw_landmarks:
                    self.mp_drawing.draw_landmarks(
                        image,
                        results.pose_landmarks,
                        self.mp_pose.POSE_CONNECTIONS
                    )

        return landmarks, image

    # -----------------------------
    # Landmark extraction
    # -----------------------------

    def extract_key_points(self, landmarks) -> Optional[Dict[str, Tuple]]:
        if landmarks is None:
            return None

        idx = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28
        }

        keypoints = {}
        for name, i in idx.items():
            x, y, z, v = landmarks[i]
            keypoints[name] = (x, y, z, v)

        return keypoints

    # -----------------------------
    # Cleanup
    # -----------------------------

    def close(self):
        with self.lock:
            self.pose.close()
            logger.info("Pose detector closed")
