"""
Web Interface for AI Fitness Trainer
"""
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from datetime import datetime
import time

st.set_page_config(
    page_title="AI Fitness Trainer",
    page_icon="🏋️",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .navbar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .navbar-brand {
        font-size: 1.8rem;
        font-weight: bold;
        color: white;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .navbar-menu {
        display: flex;
        gap: 1rem;
        align-items: center;
    }
    .navbar-link {
        color: white;
        text-decoration: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        transition: background 0.3s;
        font-weight: 500;
    }
    .navbar-link:hover {
        background: rgba(255, 255, 255, 0.2);
    }
    .navbar-badge {
        background: rgba(255, 255, 255, 0.3);
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        color: white;
        font-size: 0.9rem;
    }
    .main-header {
        font-size: 3rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .feedback-positive {
        color: #00D26A;
        font-weight: bold;
    }
    .feedback-warning {
        color: #FFB02E;
        font-weight: bold;
    }
    .feedback-error {
        color: #FF4B4B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("""
<div class="navbar">
    <div class="navbar-brand">
        <span>🏋️</span>
        <span>AI Fitness Trainer</span>
    </div>
    <div class="navbar-menu">
        <a href="#" class="navbar-link">🏠 Home</a>
        <a href="#workout" class="navbar-link">💪 Workout</a>
        <a href="#analytics" class="navbar-link">📊 Analytics</a>
        <a href="#settings" class="navbar-link">⚙️ Settings</a>
        <span class="navbar-badge">✨ AI Powered</span>
    </div>
</div>
""", unsafe_allow_html=True)

class SimpleTrainer:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.5)
        self.rep_count = 0
        self.current_stage = "start"
        
    def calculate_angle(self, a, b, c):
        a, b, c = np.array(a), np.array(b), np.array(c)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1, 1)
        return np.degrees(np.arccos(cosine_angle))
    
    def analyze_bicep_curl(self, landmarks):
        if not landmarks: return {'rep_count': self.rep_count, 'errors': ['No pose detected']}
        
        # Get key points
        key_points = {}
        indices = {11: 'shoulder', 13: 'elbow', 15: 'wrist'}
        for idx, name in indices.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                key_points[name] = (landmark.x, landmark.y)
        
        if len(key_points) != 3:
            return {'rep_count': self.rep_count, 'errors': ['Not all points visible']}
            
        shoulder = key_points['shoulder']
        elbow = key_points['elbow']
        wrist = key_points['wrist']
        
        angle = self.calculate_angle(shoulder, elbow, wrist)
        
        # Rep counting
        if self.current_stage == "start" and angle < 80:
            self.current_stage = "up"
        elif self.current_stage == "up" and angle > 160:
            self.current_stage = "down"
            self.rep_count += 1
        elif self.current_stage == "down" and angle < 80:
            self.current_stage = "up"
            
        errors = []
        if abs(elbow[0] - shoulder[0]) > 0.15:
            errors.append("Keep elbow close to body")
            
        return {
            'rep_count': self.rep_count,
            'angle': angle,
            'stage': self.current_stage,
            'errors': errors
        }

def main():
    st.markdown('<h1 class="main-header">🏋️ AI Fitness Trainer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("Workout Settings")
        exercise = st.selectbox("Select Exercise", ["Bicep Curls", "Squats", "Push-ups"])
        target_reps = st.slider("Target Reps", 5, 20, 10)
        
        st.header("Camera Settings")
        start_camera = st.button("Start Camera")
        stop_camera = st.button("Stop Camera")
        
        st.header("Session Info")
        if 'rep_count' not in st.session_state:
            st.session_state.rep_count = 0
        if 'session_start' not in st.session_state:
            st.session_state.session_start = datetime.now()
            
        st.metric("Current Reps", st.session_state.rep_count)
        st.metric("Target Reps", target_reps)
        
    # Main area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Live Camera Feed")
        
        if start_camera:
            # Initialize camera
            cap = cv2.VideoCapture(0)
            trainer = SimpleTrainer()
            
            # Create placeholder for video
            video_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            while cap.isOpened() and not stop_camera:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = trainer.pose.process(image_rgb)
                
                if results.pose_landmarks:
                    # Draw landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)
                    
                    # Analyze exercise
                    analysis = trainer.analyze_bicep_curl(results.pose_landmarks.landmark)
                    st.session_state.rep_count = analysis['rep_count']
                    
                    # Add overlay
                    cv2.putText(frame, f"Reps: {analysis['rep_count']}", (50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Stage: {analysis['stage']}", (50, 80), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"Angle: {analysis.get('angle', 0):.1f}°", (50, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display frame
                video_placeholder.image(frame, channels="BGR", use_column_width=True)
                
                # Display feedback
                with feedback_placeholder.container():
                    if analysis.get('errors'):
                        for error in analysis['errors']:
                            st.error(f"⚠️ {error}")
                    if analysis['rep_count'] > 0:
                        st.success(f"🎉 {analysis['rep_count']} reps completed!")
                        
                time.sleep(0.1)
                
            cap.release()
            
    with col2:
        st.header("Workout Analytics")
        
        # Progress
        progress = min(st.session_state.rep_count / target_reps, 1.0)
        st.progress(progress)
        st.metric("Completion", f"{progress*100:.1f}%")
        
        # Form tips
        st.subheader("Form Tips")
        if exercise == "Bicep Curls":
            st.info("""
            💡 **Bicep Curl Tips:**
            • Keep elbows close to your body
            • Fully extend arms at the bottom
            • Control the movement both ways
            • Don't swing your torso
            """)
        elif exercise == "Squats":
            st.info("""
            💡 **Squat Tips:**
            • Keep knees aligned with feet
            • Maintain straight back
            • Go parallel to the floor
            • Push through your heels
            """)
        
        # Session summary
        st.subheader("Session Summary")
        duration = datetime.now() - st.session_state.session_start
        st.metric("Workout Duration", f"{duration.seconds // 60}:{duration.seconds % 60:02d}")

if __name__ == "__main__":
    main()