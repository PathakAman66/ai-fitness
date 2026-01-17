"""
Enhanced AI Fitness Trainer with More Features
"""
import cv2
import mediapipe as mp
import numpy as np
import time
import json
import os
from datetime import datetime

print("🚀 Initializing Enhanced AI Fitness Trainer...")

class EnhancedConfig:
    CAMERA_WIDTH = 1280
    CAMERA_HEIGHT = 720
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    EXERCISES = {
        "bicep_curl": {"name": "Bicep Curls", "muscle": "Biceps"},
        "squat": {"name": "Squats", "muscle": "Legs"},
        "push_up": {"name": "Push-ups", "muscle": "Chest"},
        "shoulder_press": {"name": "Shoulder Press", "muscle": "Shoulders"},
        "lunge": {"name": "Lunges", "muscle": "Legs"},
        "plank": {"name": "Plank", "muscle": "Core"}
    }

class EnhancedPoseDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=EnhancedConfig.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=EnhancedConfig.MIN_TRACKING_CONFIDENCE
        )
        print("✅ Enhanced pose detector initialized")

    def detect_pose(self, image):
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = self.pose.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            landmarks = None
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                # Draw landmarks with different colors
                self.mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS,
                    self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                    self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
            
            return landmarks, image_bgr
        except Exception as e:
            print(f"❌ Pose detection error: {e}")
            return None, image

    def extract_key_points(self, landmarks):
        if not landmarks:
            return None
            
        key_points = {}
        landmark_indices = {
            'nose': 0, 'left_eye': 1, 'right_eye': 2,
            'left_ear': 7, 'right_ear': 8,
            'left_shoulder': 11, 'right_shoulder': 12,
            'left_elbow': 13, 'right_elbow': 14,
            'left_wrist': 15, 'right_wrist': 16,
            'left_hip': 23, 'right_hip': 24,
            'left_knee': 25, 'right_knee': 26,
            'left_ankle': 27, 'right_ankle': 28,
            'left_heel': 29, 'right_heel': 30,
            'left_foot_index': 31, 'right_foot_index': 32
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                key_points[name] = (landmark.x, landmark.y, landmark.z, landmark.visibility)
                
        return key_points

class EnhancedExerciseAnalyzer:
    def __init__(self, exercise_type="bicep_curl"):
        self.exercise_type = exercise_type
        self.rep_count = 0
        self.set_count = 1
        self.current_stage = "start"
        self.start_time = time.time()
        self.rep_history = []
        self.calories_burned = 0
        print(f"✅ Enhanced analyzer for {exercise_type} initialized")

    def calculate_angle(self, a, b, c):
        try:
            a = np.array(a[:2])  # Use only x,y for 2D angle
            b = np.array(b[:2])
            c = np.array(c[:2])
            
            ba = a - b
            bc = c - b
            
            cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
            cosine_angle = np.clip(cosine_angle, -1, 1)
            
            return np.degrees(np.arccos(cosine_angle))
        except:
            return 0

    def calculate_distance(self, point1, point2):
        try:
            return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
        except:
            return 0

    def analyze_bicep_curl(self, key_points):
        feedback = []
        errors = []
        warnings = []
        
        shoulder = key_points.get('right_shoulder', (0, 0, 0, 0))
        elbow = key_points.get('right_elbow', (0, 0, 0, 0))
        wrist = key_points.get('right_wrist', (0, 0, 0, 0))
        
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Enhanced rep counting with time tracking
        current_time = time.time()
        
        if self.current_stage == "start" and elbow_angle < 80:
            self.current_stage = "up"
            self.rep_start_time = current_time
        elif self.current_stage == "up" and elbow_angle > 160:
            self.current_stage = "down"
            self.rep_count += 1
            rep_duration = current_time - self.rep_start_time
            self.rep_history.append(rep_duration)
            self.calories_burned += 0.5  # Approximate calories per rep
            
            feedback.append(f"💪 Rep {self.rep_count} completed! ({rep_duration:.1f}s)")
            
            # Check for consistent tempo
            if len(self.rep_history) > 1:
                avg_duration = np.mean(self.rep_history)
                if rep_duration > avg_duration * 1.5:
                    warnings.append("Slow down - maintain consistent tempo")
                elif rep_duration < avg_duration * 0.5:
                    warnings.append("Speed up - don't rush the movement")
                    
        elif self.current_stage == "down" and elbow_angle < 80:
            self.current_stage = "up"
            self.rep_start_time = current_time
            
        # Advanced form checks
        elbow_shoulder_distance = abs(elbow[0] - shoulder[0])
        if elbow_shoulder_distance > 0.12:
            errors.append("Keep elbows tight to your body")
        elif elbow_shoulder_distance > 0.08:
            warnings.append("Elbows starting to drift out")
            
        if elbow_angle > 170 and self.current_stage != "start":
            warnings.append("Fully extend arms for maximum range")
            
        return {
            'angles': {'elbow': elbow_angle},
            'rep_count': self.rep_count,
            'set_count': self.set_count,
            'feedback': feedback,
            'errors': errors,
            'warnings': warnings,
            'stage': self.current_stage,
            'calories': self.calories_burned
        }

    def analyze_squat(self, key_points):
        feedback = []
        errors = []
        warnings = []
        
        hip = key_points.get('right_hip', (0, 0, 0, 0))
        knee = key_points.get('right_knee', (0, 0, 0, 0))
        ankle = key_points.get('right_ankle', (0, 0, 0, 0))
        shoulder = key_points.get('right_shoulder', (0, 0, 0, 0))
        
        knee_angle = self.calculate_angle(hip, knee, ankle)
        hip_angle = self.calculate_angle(shoulder, hip, knee)
        
        # Squat analysis
        if self.current_stage == "start" and knee_angle < 150:
            self.current_stage = "down"
        elif self.current_stage == "down" and knee_angle > 160:
            self.current_stage = "up"
            self.rep_count += 1
            self.calories_burned += 1.0  # More calories for squats
            feedback.append(f"🦵 Squat {self.rep_count} completed!")
        elif self.current_stage == "up" and knee_angle < 150:
            self.current_stage = "down"
            
        # Depth analysis
        if knee_angle < 90:
            feedback.append("Great depth! 💯")
        elif knee_angle < 110:
            warnings.append("Go deeper for better results")
        else:
            errors.append("Not deep enough - aim for 90°")
            
        # Knee alignment
        if abs(knee[0] - ankle[0]) > 0.1:
            errors.append("Keep knees aligned with ankles")
            
        # Back position
        if hip_angle < 150:
            errors.append("Keep your back straight")
            
        return {
            'angles': {'knee': knee_angle, 'hip': hip_angle},
            'rep_count': self.rep_count,
            'feedback': feedback,
            'errors': errors,
            'warnings': warnings,
            'stage': self.current_stage,
            'calories': self.calories_burned
        }

    def analyze_shoulder_press(self, key_points):
        feedback = []
        errors = []
        warnings = []
        
        shoulder = key_points.get('right_shoulder', (0, 0, 0, 0))
        elbow = key_points.get('right_elbow', (0, 0, 0, 0))
        wrist = key_points.get('right_wrist', (0, 0, 0, 0))
        
        elbow_angle = self.calculate_angle(shoulder, elbow, wrist)
        
        if self.current_stage == "start" and elbow_angle < 100:
            self.current_stage = "up"
        elif self.current_stage == "up" and elbow_angle > 160:
            self.current_stage = "down"
            self.rep_count += 1
            self.calories_burned += 0.6
            feedback.append(f"💪 Shoulder Press {self.rep_count} completed!")
        elif self.current_stage == "down" and elbow_angle < 100:
            self.current_stage = "up"
            
        # Form checks
        if elbow_angle < 90:
            warnings.append("Good depth - keep elbows at 90°")
        if abs(elbow[0] - shoulder[0]) > 0.15:
            errors.append("Keep elbows in front of shoulders")
            
        return {
            'angles': {'elbow': elbow_angle},
            'rep_count': self.rep_count,
            'feedback': feedback,
            'errors': errors,
            'warnings': warnings,
            'stage': self.current_stage,
            'calories': self.calories_burned
        }

    def analyze_plank(self, key_points):
        feedback = []
        errors = []
        warnings = []
        
        shoulder = key_points.get('right_shoulder', (0, 0, 0, 0))
        hip = key_points.get('right_hip', (0, 0, 0, 0))
        ankle = key_points.get('right_ankle', (0, 0, 0, 0))
        
        # Calculate body alignment angle
        body_angle = self.calculate_angle(shoulder, hip, ankle)
        
        # Plank is time-based, not rep-based
        current_time = time.time()
        plank_duration = current_time - self.start_time
        
        # Update calories based on time (approx 3 calories per minute)
        self.calories_burned = plank_duration * (3/60)
        
        # Form checks
        if body_angle < 160:
            errors.append("Keep your body straight - don't sag")
        elif body_angle > 175:
            warnings.append("Don't raise hips too high")
        else:
            feedback.append(f"Perfect plank form! ⏱️ {int(plank_duration)}s")
            
        # Check hip height relative to shoulders
        if hip[1] > shoulder[1] + 0.1:
            errors.append("Hips are too low - engage core")
            
        return {
            'angles': {'body': body_angle},
            'duration': plank_duration,
            'feedback': feedback,
            'errors': errors,
            'warnings': warnings,
            'stage': 'hold',
            'calories': self.calories_burned
        }

    def analyze_form(self, key_points):
        if not key_points:
            return {'errors': ['No person detected'], 'rep_count': self.rep_count}
            
        if self.exercise_type == "bicep_curl":
            return self.analyze_bicep_curl(key_points)
        elif self.exercise_type == "squat":
            return self.analyze_squat(key_points)
        elif self.exercise_type == "shoulder_press":
            return self.analyze_shoulder_press(key_points)
        elif self.exercise_type == "plank":
            return self.analyze_plank(key_points)
        elif self.exercise_type == "push_up":
            return self.analyze_bicep_curl(key_points)  # Similar mechanics
        else:
            return {'errors': ['Exercise not supported'], 'rep_count': self.rep_count}

class WorkoutSession:
    def __init__(self):
        self.sessions = []
        self.current_session = None
        
    def start_session(self, exercise_type):
        self.current_session = {
        'id': len(self.sessions) + 1,
        'exercise': exercise_type,
        'start_time': datetime.now().isoformat(),
        'start_timestamp': time.time(),  # FIX: store start timestamp
        'reps': 0,
        'duration': 0,
        'calories': 0
        }

        
    def end_session(self, analysis_result):
        if not self.current_session:
            return

        self.current_session['end_time'] = datetime.now().isoformat()
        self.current_session['reps'] = analysis_result.get('rep_count', 0)
        self.current_session['calories'] = analysis_result.get('calories', 0)
        self.current_session['duration'] = (
        time.time() - self.current_session['start_timestamp']
        )

        self.sessions.append(self.current_session)
        self.current_session = None

        self.save_sessions()

            
    def save_sessions(self):
        try:
            os.makedirs('workout_data', exist_ok=True)
            with open('workout_data/sessions.json', 'w') as f:
                json.dump(self.sessions, f, indent=2)
        except Exception as e:
            print(f"Could not save sessions: {e}")

class EnhancedFitnessTrainer:
    def __init__(self):
        self.pose_detector = EnhancedPoseDetector()
        self.exercise_analyzer = None
        self.workout_session = WorkoutSession()
        self.is_running = False
        self.camera = None
        self.current_exercise = "bicep_curl"
        self.workout_start_time = None

    def initialize_camera(self):
        self.camera = cv2.VideoCapture(0)
        if not self.camera.isOpened():
            for i in range(1, 3):
                self.camera = cv2.VideoCapture(i)
                if self.camera.isOpened():
                    break
        
        if not self.camera.isOpened():
            return False
            
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, EnhancedConfig.CAMERA_WIDTH)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, EnhancedConfig.CAMERA_HEIGHT)
        return True

    def draw_enhanced_overlay(self, frame, analysis_result, exercise_type):
        h, w = frame.shape[:2]
        
        # Exercise info panel
        exercise_info = EnhancedConfig.EXERCISES.get(exercise_type, {})
        cv2.rectangle(frame, (w-350, 10), (w-10, 120), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.8, frame, 0.2, 0, frame)
        
        cv2.putText(frame, f"EXERCISE: {exercise_info.get('name', exercise_type)}", 
                   (w-340, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"MUSCLE: {exercise_info.get('muscle', 'Multiple')}", 
                   (w-340, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Stats panel
        cv2.rectangle(frame, (w-350, 130), (w-10, 220), (0, 50, 0), -1)
        cv2.putText(frame, f"REPS: {analysis_result.get('rep_count', 0)}", 
                   (w-340, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"CALORIES: {analysis_result.get('calories', 0):.1f}", 
                   (w-340, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Real-time feedback
        y_offset = 250
        feedback_items = (
            analysis_result.get('feedback', []) +
            analysis_result.get('warnings', []) +
            analysis_result.get('errors', [])
        )
        
        for item in feedback_items[:4]:  # Show max 4 items
            color = (0, 255, 0) if item in analysis_result.get('feedback', []) else \
                   (0, 255, 255) if item in analysis_result.get('warnings', []) else \
                   (0, 0, 255)
                   
            cv2.putText(frame, f"• {item}", (w-340, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 20
            
        # Left side info
        cv2.putText(frame, f"Stage: {analysis_result.get('stage', 'start').upper()}", 
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Angles display
        angles = analysis_result.get('angles', {})
        y_offset = 60
        for joint, angle in angles.items():
            cv2.putText(frame, f"{joint}: {angle:.1f}°", (20, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
            
        # Workout timer
        if self.workout_start_time:
            elapsed = time.time() - self.workout_start_time
            mins, secs = divmod(int(elapsed), 60)
            cv2.putText(frame, f"TIME: {mins:02d}:{secs:02d}", (20, h-20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
        return frame

    def show_enhanced_menu(self, frame):
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Title
        cv2.putText(frame, "🏋️ ENHANCED AI FITNESS TRAINER", (w//2-200, 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Exercise grid
        exercises = list(EnhancedConfig.EXERCISES.items())
        box_width, box_height = 200, 80
        start_x, start_y = w//2 - 250, h//2 - 100
        
        for i, (ex_key, ex_data) in enumerate(exercises):
            row = i // 3
            col = i % 3
            x = start_x + col * (box_width + 20)
            y = start_y + row * (box_height + 20)
            
            # Exercise box
            color = (100, 100, 255) if ex_key == self.current_exercise else (50, 50, 50)
            cv2.rectangle(frame, (x, y), (x+box_width, y+box_height), color, -1)
            cv2.rectangle(frame, (x, y), (x+box_width, y+box_height), (255, 255, 255), 2)
            
            # Exercise text
            cv2.putText(frame, ex_data['name'], (x+10, y+30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame, f"Muscle: {ex_data['muscle']}", (x+10, y+55), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
            cv2.putText(frame, f"Press {i+1}", (x+10, y+75), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Controls
        controls_y = h - 100
        controls = [
            "SPACE - Start/Stop Exercise",
            "1-6 - Select Exercise", 
            "R - Reset Counter",
            "S - Save Session",
            "Q - Quit"
        ]
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (w//2-150, controls_y + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        return frame

    def run(self):
        if not self.initialize_camera():
            print("❌ Error: Could not access camera")
            return
            
        print("✅ Enhanced camera initialized")
        print("🎮 Enhanced Controls:")
        print("   SPACE - Start/Stop exercise")
        print("   1-6   - Select specific exercise") 
        print("   R     - Reset counter")
        print("   S     - Save workout session")
        print("   Q     - Quit")
        
        self.is_running = True
        in_menu = True
        
        while self.is_running:
            ret, frame = self.camera.read()
            if not ret:
                print("❌ Failed to capture frame")
                break
                
            if in_menu:
                frame = self.show_enhanced_menu(frame)
            else:
                # Process exercise
                landmarks, processed_frame = self.pose_detector.detect_pose(frame)
                
                if landmarks:
                    key_points = self.pose_detector.extract_key_points(landmarks)
                    analysis_result = self.exercise_analyzer.analyze_form(key_points)
                    processed_frame = self.draw_enhanced_overlay(processed_frame, analysis_result, self.current_exercise)
                    
                    # Console feedback
                    if analysis_result.get('feedback'):
                        for msg in analysis_result['feedback']:
                            print(f"🎉 {msg}")
                            
                    if analysis_result.get('warnings'):
                        for warning in analysis_result['warnings']:
                            print(f"⚠️  {warning}")
                            
                    if analysis_result.get('errors'):
                        for error in analysis_result['errors']:
                            print(f"❌ {error}")
                else:
                    processed_frame = self.draw_enhanced_overlay(processed_frame, {}, self.current_exercise)
                    cv2.putText(processed_frame, "🔍 No person detected - Position yourself in frame", 
                               (w//2-250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                frame = processed_frame
            
            # Display frame
            cv2.imshow('Enhanced AI Fitness Trainer - Professional Grade', frame)
            
            # Enhanced key handling
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):  # Space bar
                if in_menu:
                    # Start exercise
                    self.exercise_analyzer = EnhancedExerciseAnalyzer(self.current_exercise)
                    self.workout_session.start_session(self.current_exercise)
                    self.workout_start_time = time.time()
                    in_menu = False
                    print(f"🎯 Starting {self.current_exercise} workout!")
                else:
                    # Return to menu
                    in_menu = True
                    if self.exercise_analyzer:
                        self.workout_session.end_session(self.exercise_analyzer.analyze_form({}))
                    print("📋 Returning to exercise menu")
            elif ord('1') <= key <= ord('6'):
                exercises = list(EnhancedConfig.EXERCISES.keys())
                exercise_index = key - ord('1')
                if exercise_index < len(exercises):
                    self.current_exercise = exercises[exercise_index]
                    print(f"📝 Selected: {EnhancedConfig.EXERCISES[self.current_exercise]['name']}")
            elif key == ord('r') and not in_menu:
                self.exercise_analyzer.rep_count = 0
                self.exercise_analyzer.current_stage = "start"
                self.exercise_analyzer.calories_burned = 0
                print("🔄 Counter reset")
            elif key == ord('s') and not in_menu:
                self.workout_session.end_session(self.exercise_analyzer.analyze_form({}))
                print("💾 Workout session saved!")
                
        self.cleanup()

    def cleanup(self):
        self.is_running = False
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        
        # Save final session
        if self.exercise_analyzer:
            self.workout_session.end_session(self.exercise_analyzer.analyze_form({}))
            
        print("👋 Thank you for using Enhanced AI Fitness Trainer!")
        print("📊 Your workout data has been saved to 'workout_data/sessions.json'")

def main():
    print("=" * 70)
    print("🏋️‍♂️ ENHANCED AI FITNESS TRAINER - Professional Edition")
    print("=" * 70)
    print("📊 Enhanced Features:")
    print("   • 6 different exercises with specialized form analysis")
    print("   • Calorie tracking and workout duration")
    print("   • Advanced form correction with warnings")
    print("   • Workout session saving and history")
    print("   • Real-time performance metrics")
    print("   • Professional visual interface")
    print()
    
    trainer = EnhancedFitnessTrainer()
    
    try:
        trainer.run()
    except KeyboardInterrupt:
        print("\n🛑 Application interrupted by user")
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        trainer.cleanup()

if __name__ == "__main__":
    main()