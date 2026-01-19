"""
Web Interface for AI Fitness Trainer
Refactored to use FastAPI backend
"""
import streamlit as st
import cv2
import numpy as np
import requests
import base64
from datetime import datetime
from typing import Optional, Dict, Any
import time

# Page Configuration
st.set_page_config(
    page_title="AI Fitness Trainer",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Enhanced Responsive CSS (Restored from your original) ---
st.markdown("""
<style>
    /* Global Styles */
    * { box-sizing: border-box; }
    .main { padding: 2rem 1rem; }
    .block-container { padding-top: 2rem; max-width: 1400px; margin: 0 auto; }
    
    /* Typography */
    .main-header {
        font-size: clamp(1.5rem, 4vw, 3.5rem);
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 0 0 1rem 0;
        padding: 0.5rem 0;
        line-height: 1.2;
    }
    
    /* Workout Selection Cards */
    .workout-card {
        background: #f8f9fa;
        padding: 40px;
        border-radius: 20px;
        border: 2px solid #eee;
        text-align: center;
        transition: all 0.3s ease;
    }
    .workout-card:hover {
        transform: translateY(-5px);
        border-color: #764ba2;
        box-shadow: 0 10px 20px rgba(0,0,0,0.05);
    }
    
    /* Metrics & Info Cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .info-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# FastAPI Backend Configuration
API_BASE_URL = "http://localhost:8000"

# Exercise type mapping (Streamlit display name -> API enum)
EXERCISE_TYPE_MAP = {
    "Bicep Curls": "bicep_curl",
    "Squats": "squat",
    "Push-ups": "push_up"
}


class APIClient:
    """Client for communicating with FastAPI backend"""
    
    @staticmethod
    def encode_frame(frame: np.ndarray) -> str:
        """Encode frame to base64 string"""
        _, buffer = cv2.imencode('.jpg', frame)
        return base64.b64encode(buffer).decode('utf-8')
    
    @staticmethod
    def decode_frame(base64_str: str) -> np.ndarray:
        """Decode base64 string to frame"""
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def check_health() -> bool:
        """Check if backend is available"""
        try:
            response = requests.get(f"{API_BASE_URL}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    @staticmethod
    def start_session(exercise_type: str) -> Optional[str]:
        """Start a new workout session"""
        try:
            api_exercise_type = EXERCISE_TYPE_MAP.get(exercise_type, exercise_type)
            response = requests.post(
                f"{API_BASE_URL}/api/v1/sessions/start",
                json={"exercise_type": api_exercise_type}
            )
            if response.status_code == 200:
                return response.json()["session_id"]
            return None
        except Exception as e:
            st.error(f"Failed to start session: {str(e)}")
            return None
    
    @staticmethod
    def end_session(session_id: str) -> Optional[Dict[str, Any]]:
        """End a workout session"""
        try:
            response = requests.post(
                f"{API_BASE_URL}/api/v1/sessions/{session_id}/end"
            )
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Failed to end session: {str(e)}")
            return None
    
    @staticmethod
    def detect_pose(frame: np.ndarray, draw_landmarks: bool = True) -> Optional[Dict[str, Any]]:
        """Detect pose in frame"""
        try:
            # Encode frame
            encoded_frame = APIClient.encode_frame(frame)
            
            # Send to API
            response = requests.post(
                f"{API_BASE_URL}/api/v1/pose/detect",
                data={
                    "image": encoded_frame,
                    "draw_landmarks": str(draw_landmarks).lower()
                }
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Pose detection failed: {str(e)}")
            return None
    
    @staticmethod
    def analyze_exercise(session_id: str, exercise_type: str, key_points: Dict) -> Optional[Dict[str, Any]]:
        """Analyze exercise form"""
        try:
            api_exercise_type = EXERCISE_TYPE_MAP.get(exercise_type, exercise_type)
            response = requests.post(
                f"{API_BASE_URL}/api/v1/analyze",
                json={
                    "session_id": session_id,
                    "exercise_type": api_exercise_type,
                    "key_points": key_points
                }
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            return None

def main():
    st.markdown('<h1 class="main-header">🏋️ AI Fitness Trainer</h1>', unsafe_allow_html=True)
    
    # Check backend health
    if not APIClient.check_health():
        st.error("⚠️ Backend API is not available. Please start the FastAPI server.")
        st.info("Run: `uvicorn backend.api.main:app --reload`")
        return

    # --- UI COMPONENT: WORKOUT SELECTION MENU (PRE-SESSION) ---
    if 'active_exercise' not in st.session_state:
        st.markdown("<h3 style='text-align: center;'>Pick your workout to start the session</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        
        workouts = [("💪 Bicep Curls", "Bicep Curls"), ("🦵 Squats", "Squats"), ("🏃 Push-ups", "Push-ups")]
        for i, (label, key) in enumerate(workouts):
            with [col1, col2, col3][i]:
                st.markdown('<div class="workout-card">', unsafe_allow_html=True)
                if st.button(label, key=f"btn_{key}", use_container_width=True):
                    # Start session via API
                    session_id = APIClient.start_session(key)
                    if session_id:
                        st.session_state.active_exercise = key
                        st.session_state.session_id = session_id
                        st.session_state.session_start = datetime.now()
                        st.session_state.rep_count = 0
                        st.rerun()
                    else:
                        st.error("Failed to start session")
                st.markdown('</div>', unsafe_allow_html=True)

    # --- MAIN SESSION DASHBOARD ---
    else:
        exercise = st.session_state.active_exercise
        session_id = st.session_state.session_id
        
        with st.sidebar:
            st.markdown(f"### ⚙️ Mode: {exercise}")
            st.caption(f"Session: {session_id[:8]}...")
            
            if st.button("🔄 Back to Menu"):
                # End session via API
                APIClient.end_session(session_id)
                del st.session_state.active_exercise
                del st.session_state.session_id
                if 'rep_count' in st.session_state:
                    del st.session_state.rep_count
                st.rerun()
            
            st.markdown("---")
            target_reps = st.slider("Target Reps", 5, 20, 10)
            
            st.markdown("### 📹 Camera Controls")
            col_a, col_b = st.columns(2)
            start_camera = col_a.button("▶️ Start", use_container_width=True)
            stop_camera = col_b.button("⏹️ Stop", use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Session Stats")
            current_reps = st.session_state.get('rep_count', 0)
            calories = st.session_state.get('calories', 0.0)
            col1, col2 = st.columns(2)
            col1.metric("Current", current_reps)
            col2.metric("Target", target_reps)
            st.metric("Calories", f"{calories:.1f}")

        # Main Layout
        col_video, col_analytics = st.columns([2.5, 1.5], gap="medium")

        with col_video:
            st.markdown("### 📹 Live Camera Feed")
            video_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            if start_camera:
                cap = cv2.VideoCapture(0)
                
                while cap.isOpened() and not stop_camera:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Detect pose via API
                    pose_result = APIClient.detect_pose(frame, draw_landmarks=True)
                    
                    if pose_result and pose_result.get('detected'):
                        key_points = pose_result.get('key_points')
                        
                        # Get annotated frame if available
                        if pose_result.get('annotated_image'):
                            frame = APIClient.decode_frame(pose_result['annotated_image'])
                        
                        # Analyze exercise via API
                        analysis = APIClient.analyze_exercise(session_id, exercise, key_points)
                        
                        if analysis:
                            # Update session state
                            st.session_state.rep_count = analysis.get('rep_count', 0)
                            st.session_state.calories = analysis.get('calories', 0.0)
                            
                            # Add rep count overlay
                            cv2.putText(
                                frame, 
                                f"Reps: {analysis['rep_count']}", 
                                (50, 50), 
                                cv2.FONT_HERSHEY_SIMPLEX, 
                                1, 
                                (0, 255, 0), 
                                2
                            )
                            
                            # Display feedback
                            with feedback_placeholder.container():
                                # Show errors
                                if analysis.get('errors'):
                                    for error in analysis['errors']:
                                        st.error(f"❌ {error}")
                                
                                # Show warnings
                                if analysis.get('warnings'):
                                    for warning in analysis['warnings']:
                                        st.warning(f"⚠️ {warning}")
                                
                                # Show positive feedback
                                if analysis.get('feedback'):
                                    for fb in analysis['feedback']:
                                        st.success(f"✅ {fb}")
                    else:
                        # No pose detected
                        cv2.putText(
                            frame, 
                            "No pose detected", 
                            (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1, 
                            (0, 0, 255), 
                            2
                        )
                    
                    # Display frame
                    video_placeholder.image(frame, channels="BGR", use_column_width=True)
                    
                    # Small delay to prevent overwhelming the API
                    time.sleep(0.05)
                
                cap.release()

        with col_analytics:
            st.markdown("### 📈 Workout Analytics")
            reps = st.session_state.get('rep_count', 0)
            calories = st.session_state.get('calories', 0.0)
            progress = min(reps / target_reps, 1.0)
            st.progress(progress)
            st.metric("Completion", f"{progress*100:.1f}%")
            
            if progress >= 1.0:
                st.balloons()
                st.success("🎉 Target reached! Great job!")

            # --- RESTORED FORM GUIDE ---
            st.markdown("### 💡 Form Guide")
            if exercise == "Bicep Curls":
                st.info("""
                        **Bicep Curl Tips:**
                        
                        ✓ Keep elbows close to body  
                        ✓ Fully extend arms at bottom  
                        ✓ Control the movement  
                        ✓ Avoid swinging torso
                    """)
            elif exercise == "Squats":
                st.info("""
                        **Squat Tips:**
                       
                        ✓ Knees aligned with feet  
                        ✓ Maintain straight back  
                        ✓ Go parallel to floor  
                        ✓ Push through heels
                    """)

            elif exercise == "Push-ups":
                st.info("""
                        **Push-up Tips:**
                        
                        ✓ Keep body straight  
                        ✓ Elbows at 45° angle  
                        ✓ Full range of motion  
                        ✓ Engage core muscles
                    """)
                st.markdown("")

        

            # --- RESTORED SUMMARY ---
            st.markdown("### ⏱️ Session Summary")
            duration = datetime.now() - st.session_state.session_start
            st.metric("Duration", f"{duration.seconds // 60}m {duration.seconds % 60}s")

if __name__ == "__main__":
    main()