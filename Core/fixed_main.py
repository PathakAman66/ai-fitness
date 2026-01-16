"""
Fixed AI Fitness Trainer - Standalone version
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import os
import json
from datetime import datetime
from report_manager import WorkoutReportManager


# Configuration
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720
MIN_DETECTION_CONFIDENCE = 0.7
MIN_TRACKING_CONFIDENCE = 0.5

class SimplePoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE
        )

    def detect_pose(self, image):
        """Detect human pose in the image"""
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            landmarks = None
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                self.mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
            
            return landmarks, image_bgr
        except Exception as e:
            print(f"Pose detection error: {e}")
            return None, image

    def extract_key_points(self, landmarks):
        """Extract key points for exercise analysis"""
        if not landmarks:
            return None
            
        key_points = {}
        landmark_indices = {
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                key_points[name] = (landmark.x, landmark.y)
                
        return key_points

class SimpleExerciseAnalyzer:
    def __init__(self, exercise_type="bicep_curl"):
        self.exercise_type = exercise_type
        self.rep_count = 0
        self.current_stage = "start"
        self.start_time = time.time()
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        try:
            a = np.array(a)
            b = np.array(b)
            c = np.array(c)
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1, 1)
            
            return np.degrees(np.arccos(cosine_angle))
        except:
            return 0

    def analyze_bicep_curl(self, key_points):
        """Analyze bicep curl form"""
        feedback = []
        errors = []
        
        # Use right arm by default
        shoulder = key_points.get('right_shoulder', (0, 0))
        elbow = key_points.get('right_elbow', (0, 0))
        wrist = key_points.get('right_wrist', (0, 0))
        
        # Calculate elbow angle
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Simple rep counting
        if self.current_stage == "start" and elbow_angle < 60:
            self.current_stage = "up"
        elif self.current_stage == "up" and elbow_angle > 140:
            self.current_stage = "down"
            self.rep_count += 1
            feedback.append(f"Rep {self.rep_count} completed!")
        elif self.current_stage == "down" and elbow_angle < 60:
            self.current_stage = "up"
            
        # Form checks
        if abs(elbow[0] - shoulder[0]) > 0.1:
            errors.append("Keep elbow close to body")
            
        return {
            'angles': {'elbow': elbow_angle},
            'rep_count': self.rep_count,
            'feedback': feedback,
            'errors': errors,
            'stage': self.current_stage
        }

    def analyze_form(self, key_points):
        """Main analysis method"""
        if not key_points:
            return {'errors': ['No pose detected'], 'rep_count': self.rep_count}
            
        if self.exercise_type == "bicep_curl":
            return self.analyze_bicep_curl(key_points)
        else:
            return {'errors': ['Exercise type not supported'], 'rep_count': self.rep_count}
    def get_duration(self):
        return int(time.time() - self.start_time)


class SimpleFitnessTrainer:
    def __init__(self):
        self.pose_detector = SimplePoseDetector()
        self.exercise_analyzer = SimpleExerciseAnalyzer()
        self.is_running = False
        self.camera = None

    def initialize_camera(self):
        """Initialize camera"""
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        return self.camera.isOpened()

    def add_overlay(self, frame, analysis_result):
        """Add analysis overlay to frame"""
        # Rep count
        cv2.putText(frame, f"Reps: {analysis_result.get('rep_count', 0)}", 
                   (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Stage
        cv2.putText(frame, f"Stage: {analysis_result.get('stage', 'start')}", 
                   (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Angles
        angles = analysis_result.get('angles', {})
        y_offset = 110
        for joint, angle in angles.items():
            cv2.putText(frame, f"{joint}: {angle:.1f}°", 
                       (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
            
        # Errors
        errors = analysis_result.get('errors', [])
        if errors:
            cv2.putText(frame, "Form Issues:", 
                       (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            y_offset += 30
            for error in errors[:2]:
                cv2.putText(frame, f"- {error}", 
                           (70, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                y_offset += 25
                
        return frame

    def run(self):
        """Main application loop"""
        if not self.initialize_camera():
            print("❌ Could not initialize camera")
            return
            
        print("✅ Camera initialized")
        print("🎯 Starting AI Fitness Trainer...")
        print("📝 Press 'q' to quit, 'r' to reset rep counter")
        
        self.is_running = True
        self.exercise_analyzer = SimpleExerciseAnalyzer("bicep_curl")
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
                
            # Detect pose and analyze
            landmarks, processed_frame = self.pose_detector.detect_pose(frame)
            
            if landmarks:
                key_points = self.pose_detector.extract_key_points(landmarks)
                analysis_result = self.exercise_analyzer.analyze_form(key_points)
                processed_frame = self.add_overlay(processed_frame, analysis_result)
                
                # Print feedback to console
                if analysis_result.get('feedback'):
                    for msg in analysis_result['feedback']:
                        print(f"🎉 {msg}")
                        
            # Display frame
            cv2.imshow('AI Fitness Trainer - Press Q to quit, R to reset', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.exercise_analyzer.rep_count = 0
                self.exercise_analyzer.current_stage = "start"
                print("🔄 Rep counter reset")
                
        self.cleanup()

    def cleanup(self):
        """ Cleanup resources and save workout report """
        self.is_running = False

    # Save workout session
        try:
            WorkoutReportManager.save(self.exercise_analyzer)
        except Exception as e:
            print(f"⚠ Workout report not saved: {e}")

        if self.camera:
            self.camera.release()

        cv2.destroyAllWindows()
        print("👋 Application closed")


def main():
    """Main function"""
    print("=" * 50)
    print("🏋️ AI Fitness Trainer - Simple Version")
    print("=" * 50)
    print("This version uses minimal dependencies")
    print("and focuses on bicep curl detection.")
    print()
    
    trainer = SimpleFitnessTrainer()
    
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
    except Exception as e:
        print(f"💥 Error: {e}")
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()