"""
Enhanced Exercise Analyzer for API use
Refactored from app.py - removed print statements and added API-specific features
"""
import numpy as np
import time
import logging

logger = logging.getLogger(__name__)

class EnhancedExerciseAnalyzer:
    def __init__(self, exercise_type="bicep_curl"):
        self.exercise_type = exercise_type
        self.rep_count = 0
        self.set_count = 1
        self.current_stage = "start"
        self.start_time = time.time()
        self.rep_history = []
        self.calories_burned = 0.0
        self.rep_start_time = None
        
        logger.info(f"✅ Enhanced analyzer for {exercise_type} initialized")

    def reset(self):
        """Reset the analyzer state to initial values"""
        self.rep_count = 0
        self.current_stage = "start"
        self.start_time = time.time()
        self.rep_history = []
        self.calories_burned = 0.0
        self.rep_start_time = None
        logger.info(f"Analyzer for {self.exercise_type} reset")

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
        except Exception as e:
            logger.warning(f"Error calculating angle: {e}")
            return 0

    def calculate_distance(self, point1, point2):
        try:
            return np.sqrt((point2[0]-point1[0])**2 + (point2[1]-point1[1])**2)
        except Exception:
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
            
            rep_duration = 0
            if self.rep_start_time:
                rep_duration = current_time - self.rep_start_time
                self.rep_history.append(rep_duration)
            
            self.calories_burned += 0.5  # Approximate calories per rep
            
            feedback.append(f"💪 Rep {self.rep_count} completed!")
            if rep_duration > 0:
                feedback[-1] += f" ({rep_duration:.1f}s)"
            
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
            # Push-up mechanics are similar to bicep curl regarding elbow angle for simplicity here,
            # but ideally should be different. Reusing bicep curl logic as placeholder 
            # or could verify if app.py had push_up specific logic.
            # App.py used analyze_bicep_curl for push_up too: `return self.analyze_bicep_curl(key_points)  # Similar mechanics`
            return self.analyze_bicep_curl(key_points)
        elif self.exercise_type == "lunge":
             # App.py didn't have explicit lunge logic in the snippet above, let's Default to squat or empty?
             # Actually I didn't see lunge logic in app.py's analyze_form.
             # It defaults to error/unsupported in app.py if not in list.
             # Wait, app.py had:
             # elif self.exercise_type == "push_up": return self.analyze_bicep_curl(key_points)
             # else: return ...
             
             # I will stick to what app.py had.
             return {'errors': ['Exercise not supported'], 'rep_count': self.rep_count}
             
        else:
            return {'errors': ['Exercise not supported'], 'rep_count': self.rep_count}
