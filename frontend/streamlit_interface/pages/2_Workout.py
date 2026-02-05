"""
Workout page for the AI Fitness Trainer Streamlit application.

This page allows users to select an exercise type and start a workout session.
It handles exercise selection, session initialization, and error handling for API failures.
"""

import streamlit as st
import sys
from pathlib import Path
import cv2
import numpy as np
from datetime import datetime, timedelta
import time
import base64


# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from styles.custom_css import inject_custom_css, apply_page_config
from styles.theme import COLORS, TYPOGRAPHY, SPACING, BORDER_RADIUS, SHADOWS
from utils.icons import (
    inject_material_icons_cdn,
    render_exercise_card_icon,
    get_icon_name,
)
from utils.state_manager import StateManager
from components.navigation import Navigation
from services.api_client import APIClient, get_api_client
from utils.error_handler import ErrorHandler


def check_backend_health() -> bool:
    """
    Check if the backend API is available.
    
    Returns:
        True if backend is healthy, False otherwise
    """
    return APIClient.check_health()


def render_backend_error():
    """Render error message when backend is unavailable."""
    st.markdown(
        f"""
        <div style="
            background-color: {COLORS['error']}15;
            border-left: 4px solid {COLORS['error']};
            border-radius: {BORDER_RADIUS['md']};
            padding: {SPACING['lg']};
            margin: {SPACING['xl']} 0;
            text-align: center;
        ">
            <div style="
                display: flex;
                justify-content: center;
                margin-bottom: {SPACING['md']};
            ">
                <span class="material-icons" style="
                    font-size: 48px;
                    color: {COLORS['error']};
                ">error_outline</span>
            </div>
            <h3 style="
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_xl']};
                font-weight: {TYPOGRAPHY['font_weight_semibold']};
                color: {COLORS['text_primary']};
                margin-bottom: {SPACING['sm']};
            ">
                Backend Service Unavailable
            </h3>
            <p style="
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_base']};
                color: {COLORS['text_secondary']};
                line-height: {TYPOGRAPHY['line_height_relaxed']};
                margin-bottom: {SPACING['md']};
            ">
                The AI Fitness Trainer backend service is not running or cannot be reached.
                Please ensure the FastAPI backend is started before using the workout feature.
            </p>
            <p style="
                font-family: {TYPOGRAPHY['font_family_mono']};
                font-size: {TYPOGRAPHY['font_size_sm']};
                color: {COLORS['text_tertiary']};
                background-color: {COLORS['background_secondary']};
                padding: {SPACING['sm']};
                border-radius: {BORDER_RADIUS['sm']};
                display: inline-block;
            ">
                Expected backend at: http://localhost:8000
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Add retry button
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("🔄 Retry Connection", use_container_width=True):
            # Clear health check cache and rerun
            APIClient._health_check_cache = None
            APIClient._health_check_timestamp = None
            st.rerun()


def start_workout_session(exercise_type: str) -> bool:
    """
    Start a new workout session via API.
    
    Args:
        exercise_type: Type of exercise to start
        
    Returns:
        True if session started successfully, False otherwise
    """
    api_client = get_api_client()
    
    # Map internal exercise type to display name for API
    exercise_display_map = {
        "bicep_curl": "Bicep Curls",
        "squat": "Squats",
        "push_up": "Push-ups",
    }
    
    display_name = exercise_display_map.get(exercise_type, exercise_type)
    
    # Start session via API
    session_id = api_client.start_session(display_name)
    
    if session_id:
        # Use StateManager to initialize workout session state
        StateManager.start_workout_session(
            exercise_type=exercise_type,
            session_id=session_id,
            session_start=datetime.now()
        )
        return True
    else:
        return False


def render_session_start_error(exercise_name: str):
    """
    Render error message when session fails to start.
    
    Args:
        exercise_name: Name of the exercise that failed to start
    """
    ErrorHandler.render_session_error("start")


def render_page_header():
    """Render the workout page header."""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: {SPACING['2xl']} {SPACING['lg']} {SPACING['xl']};
            margin-bottom: {SPACING['xl']};
        ">
            <h1 style="
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_4xl']};
                font-weight: {TYPOGRAPHY['font_weight_bold']};
                color: var(--text-color);
                line-height: {TYPOGRAPHY['line_height_tight']};
                margin-bottom: {SPACING['sm']};
                letter-spacing: {TYPOGRAPHY['letter_spacing_tight']};
            ">
                Start Your Workout
            </h1>
            <p style="
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_lg']};
                font-weight: {TYPOGRAPHY['font_weight_normal']};
                color: var(--text-secondary);
                line-height: {TYPOGRAPHY['line_height_relaxed']};
                max-width: 600px;
                margin: 0 auto;
            ">
                Select an exercise to begin your AI-powered workout session with real-time form analysis
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_all_exercises():
    """
    Get comprehensive list of all available exercises organized by category.
    Only includes exercises supported by the backend API.
    
    Returns:
        dict: Dictionary of exercises organized by category
    """
    return {
        "Upper Body": [
            {
                "type": "bicep_curl",
                "display_name": "Bicep Curls",
                "description": "Build arm strength with proper curl form",
                "icon": "fitness_center",
                "difficulty": "Beginner",
                "duration": "10-15 min",
                "calories": "50-80",
                "benefits": ["Strengthens biceps", "Improves arm definition", "Enhances grip strength"],
                "category": "Upper Body",
            },
            {
                "type": "push_up",
                "display_name": "Push-ups",
                "description": "Develop upper body strength and endurance",
                "icon": "fitness_center",
                "difficulty": "Beginner",
                "duration": "10-15 min",
                "calories": "60-100",
                "benefits": ["Strengthens chest", "Builds shoulders", "Engages core"],
                "category": "Upper Body",
            },
            {
                "type": "shoulder_press",
                "display_name": "Shoulder Press",
                "description": "Build strong shoulders with overhead pressing",
                "icon": "fitness_center",
                "difficulty": "Intermediate",
                "duration": "12-18 min",
                "calories": "70-110",
                "benefits": ["Builds deltoids", "Improves posture", "Increases upper body power"],
                "category": "Upper Body",
            },
        ],
        "Lower Body": [
            {
                "type": "squat",
                "display_name": "Squats",
                "description": "Strengthen legs and core with perfect form",
                "icon": "accessibility_new",
                "difficulty": "Beginner",
                "duration": "12-18 min",
                "calories": "80-120",
                "benefits": ["Builds leg muscles", "Strengthens core", "Improves mobility"],
                "category": "Lower Body",
            },
            {
                "type": "lunge",
                "display_name": "Lunges",
                "description": "Build leg strength and balance",
                "icon": "directions_walk",
                "difficulty": "Beginner",
                "duration": "10-15 min",
                "calories": "70-100",
                "benefits": ["Strengthens quads", "Improves balance", "Builds glutes"],
                "category": "Lower Body",
            },
        ],
        "Core": [
            {
                "type": "plank",
                "display_name": "Planks",
                "description": "Build core stability and endurance",
                "icon": "self_improvement",
                "difficulty": "Beginner",
                "duration": "8-12 min",
                "calories": "40-70",
                "benefits": ["Strengthens core", "Improves posture", "Builds endurance"],
                "category": "Core",
            },
        ],
    }


def get_category_style(category):
    """Get theme-aware glassmorphism styles for each category."""
    styles = {
        "Upper Body": (
            "linear-gradient(145deg, rgba(37, 99, 235, 0.4) 0%, rgba(30, 58, 138, 0.4) 100%)", # Blue bg
            "rgba(59, 130, 246, 0.3)", # Blue border
            "0 0 20px rgba(37, 99, 235, 0.2)" # Blue glow
        ),
        "Lower Body": (
            "linear-gradient(145deg, rgba(139, 92, 246, 0.4) 0%, rgba(88, 28, 135, 0.4) 100%)", # Purple bg
            "rgba(167, 139, 250, 0.3)", # Purple border
            "0 0 20px rgba(139, 92, 246, 0.2)" # Purple glow
        ),
        "Core": (
            "linear-gradient(145deg, rgba(249, 115, 22, 0.4) 0%, rgba(154, 52, 18, 0.4) 100%)", # Orange bg
            "rgba(251, 146, 60, 0.3)", # Orange border
            "0 0 20px rgba(249, 115, 22, 0.2)" # Orange glow
        ),
        "Cardio": (
            "linear-gradient(145deg, rgba(239, 68, 68, 0.4) 0%, rgba(153, 27, 27, 0.4) 100%)", # Red bg
            "rgba(248, 113, 113, 0.3)", # Red border
            "0 0 20px rgba(239, 68, 68, 0.2)" # Red glow
        ),
    }
    # Default to Slate/Gray
    return styles.get(category, (
        "linear-gradient(145deg, rgba(100, 116, 139, 0.4) 0%, rgba(51, 65, 85, 0.4) 100%)",
        "rgba(148, 163, 184, 0.3)",
        "0 0 20px rgba(148, 163, 184, 0.2)"
    ))


def get_difficulty_color(difficulty):
    """Get color for difficulty badge."""
    colors = {
        "Beginner": "#10b981",  # Green
        "Intermediate": "#f59e0b",  # Amber
        "Advanced": "#ef4444",  # Red
    }
    return colors.get(difficulty, "#64748b")


def render_exercise_selection():
    """Render enhanced exercise selection with categories and modern UI."""
    
    # Get all exercises
    all_exercises = get_all_exercises()
    

    
    # Category tabs
    categories = ["All"] + list(all_exercises.keys())
    selected_category = st.selectbox(
        "Filter by Category",
        categories,
        key="category_selector"
    )
    
    # Filter exercises based on category
    if selected_category == "All":
        exercises_to_show = []
        for category_exercises in all_exercises.values():
            exercises_to_show.extend(category_exercises)
    else:
        exercises_to_show = all_exercises[selected_category]
    
    # Render exercise grid
    st.markdown("---")
    
    # Create responsive grid (3 columns)
    cols_per_row = 3
    for i in range(0, len(exercises_to_show), cols_per_row):
        cols = st.columns(cols_per_row)
        
        for col_idx, exercise in enumerate(exercises_to_show[i:i + cols_per_row]):
            with cols[col_idx]:
                render_enhanced_exercise_card(exercise)
                continue
                
                # Get category color
                category_for_exercise = None
                for cat, exercises in all_exercises.items():
                    if exercise in exercises:
                        category_for_exercise = cat
                        break
                
                bg_color, accent_color = get_category_color(category_for_exercise) if category_for_exercise else ("rgba(100, 116, 139, 0.1)", "#64748b")
                difficulty_color = get_difficulty_color(exercise["difficulty"])
                
                # Card container
                st.markdown(
                    f"""
                    <div style="
                        background: {bg_color};
                        border-radius: 16px;
                        padding: 24px;
                        border: 2px solid #e5e7eb;
                        margin-bottom: 16px;
                        text-align: center;
                    ">
                        <div style="
                            background-color: {difficulty_color};
                            color: white;
                            padding: 4px 12px;
                            border-radius: 12px;
                            font-size: 11px;
                            font-weight: 600;
                            text-transform: uppercase;
                            display: inline-block;
                            margin-bottom: 16px;
                        ">
                            {exercise['difficulty']}
                        </div>
                        
                        <div style="margin: 16px 0;">
                            <div style="
                                width: 80px;
                                height: 80px;
                                background: {accent_color};
                                border-radius: 16px;
                                display: inline-flex;
                                align-items: center;
                                justify-content: center;
                                margin: 0 auto;
                            ">
                                <span class="material-icons" style="font-size: 48px; color: white;">
                                    {exercise['icon']}
                                </span>
                            </div>
                        </div>
                        
                        <h3 style="
                            font-size: 20px;
                            font-weight: 700;
                            color: #111827;
                            margin: 12px 0;
                        ">
                            {exercise['display_name']}
                        </h3>
                        
                        <p style="
                            font-size: 14px;
                            color: #6b7280;
                            margin: 12px 0;
                            min-height: 40px;
                        ">
                            {exercise['description']}
                        </p>
                        
                        <div style="
                            display: flex;
                            justify-content: space-around;
                            margin: 16px 0;
                            padding: 12px 0;
                            border-top: 1px solid #e5e7eb;
                            border-bottom: 1px solid #e5e7eb;
                        ">
                            <div>
                                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Duration</div>
                                <div style="font-size: 14px; font-weight: 600; color: #111827;">{exercise['duration']}</div>
                            </div>
                            <div>
                                <div style="font-size: 11px; color: #9ca3af; text-transform: uppercase;">Calories</div>
                                <div style="font-size: 14px; font-weight: 600; color: #111827;">{exercise['calories']}</div>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                
                # Start button
                if st.button(
                    f"▶️ Start Workout",
                    key=f"start_{exercise['type']}",
                    use_container_width=True,
                ):
                    with st.spinner(f"Starting {exercise['display_name']} session..."):
                        success = start_workout_session(exercise["type"])
                        
                        if success:
                            st.success(
                                f"✅ {exercise['display_name']} session started successfully!",
                                icon="🎉"
                            )
                            time.sleep(1)
                            st.rerun()
                        else:
                            render_session_start_error(exercise['display_name'])


def inject_hover_styles():
    """Inject CSS for hover effects on exercise cards and dark mode support."""
    st.markdown(
        f"""
        <style>
        /* Dark mode CSS variables */
        :root {{
            --text-color: #111827;
            --text-secondary: #6b7280;
            --card-bg: #f9fafb;
            --border-color: #e5e7eb;
        }}
        
        @media (prefers-color-scheme: dark) {{
            :root {{
                --text-color: #f9fafb;
                --text-secondary: #9ca3af;
                --card-bg: #1f2937;
                --border-color: #374151;
            }}
        }}
        
        [data-testid="stAppViewContainer"][data-theme="dark"] {{
            --text-color: #f9fafb;
            --text-secondary: #9ca3af;
            --card-bg: #1f2937;
            --border-color: #374151;
        }}
        
        /* Enhanced Exercise Card Hover Effects */
        .exercise-card-enhanced {{
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }}
        
        .exercise-card-enhanced:hover {{
            transform: translateY(-8px) scale(1.02);
            box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 
                        0 10px 10px -5px rgba(0, 0, 0, 0.04),
                        0 0 0 3px rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.3);
        }}
        
        /* Category Radio Buttons Styling */
        div[data-testid="stRadio"] > div {{
            display: flex;
            gap: 0.5rem;
            flex-wrap: wrap;
        }}
        
        div[data-testid="stRadio"] label {{
            background: linear-gradient(135deg, {COLORS['card_background']}, {COLORS['background_secondary']});
            border: 2px solid {COLORS['border']};
            border-radius: {BORDER_RADIUS['lg']};
            padding: {SPACING['sm']} {SPACING['lg']};
            cursor: pointer;
            transition: all 0.2s ease;
            font-weight: {TYPOGRAPHY['font_weight_medium']};
        }}
        
        div[data-testid="stRadio"] label:hover {{
            border-color: {COLORS['primary']};
            background: linear-gradient(135deg, {COLORS['primary']}15, {COLORS['primary']}05);
            transform: translateY(-2px);
        }}
        
        div[data-testid="stRadio"] input:checked + div {{
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['primary']}dd);
            border-color: {COLORS['primary']};
            color: white;
        }}
        
        /* Button Styling */
        .stButton > button {{
            background: linear-gradient(135deg, {COLORS['primary']}, {COLORS['secondary']});
            color: white;
            border: none;
            border-radius: {BORDER_RADIUS['lg']};
            padding: {SPACING['md']} {SPACING['xl']};
            font-weight: {TYPOGRAPHY['font_weight_semibold']};
            transition: all 0.2s ease;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
        }}
        
        .stButton > button:active {{
            transform: translateY(0);
        }}
        
        /* Force all headings and text to use proper colors */
        h1, h2, h3, h4, h5, h6 {{
            color: var(--text-color) !important;
        }}
        
        p {{
            color: var(--text-secondary) !important;
        }}
        
        [data-testid="stAppViewContainer"][data-theme="dark"] h1,
        [data-testid="stAppViewContainer"][data-theme="dark"] h2,
        [data-testid="stAppViewContainer"][data-theme="dark"] h3 {{
            color: #f9fafb !important;
        }}
        
        [data-testid="stAppViewContainer"][data-theme="dark"] p {{
            color: #9ca3af !important;
        }}
        
        /* Remove white space at top */
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        
        [data-testid="stToolbar"] {{
            background-color: transparent !important;
        }}
        
        [data-testid="stToolbar"] > div:not(:first-child) {{
            display: none !important;
        }}
        
        .main .block-container {{
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }}
        
        section[data-testid="stSidebar"] > div:first-child {{
            padding-top: 1rem !important;
        }}
        
        /* Fix button text color to white - FORCE IT */
        .stButton > button {{
            color: #ffffff !important;
        }}
        
        .stButton > button:hover {{
            color: #ffffff !important;
        }}
        
        .stButton > button p {{
            color: #ffffff !important;
        }}
        
        .stButton > button span {{
            color: #ffffff !important;
        }}
        
        .stButton > button div {{
            color: #ffffff !important;
        }}
        
        /* Fix button text in dark mode */
        [data-testid="stAppViewContainer"][data-theme="dark"] .stButton > button {{
            color: #ffffff !important;
        }}
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stButton > button:hover {{
            color: #ffffff !important;
        }}
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stButton > button p {{
            color: #ffffff !important;
        }}
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stButton > button span {{
            color: #ffffff !important;
        }}
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stButton > button div {{
            color: #ffffff !important;
        }}
        
        /* Hover effects */
        .exercise-card:hover {{
            transform: translateY(-5px);
            box-shadow: {SHADOWS['lg']};
            border-color: {COLORS['primary']};
        }}
        
        /* Responsive adjustments */
        @media (max-width: 768px) {{
            .exercise-card {{
                margin-bottom: {SPACING['lg']};
            }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def end_workout_session() -> bool:
    """
    End the current workout session via API.
    
    Returns:
        True if session ended successfully, False otherwise
    """
    session_id = StateManager.get("session_id")
    if not session_id:
        return False
    
    api_client = get_api_client()
    result = api_client.end_session(session_id)
    
    if result:
        # Use StateManager to clean up workout session state
        StateManager.end_workout_session()
        return True
    else:
        return False


def render_session_metrics():
    """Render session metrics in metric cards."""
    # Calculate duration
    duration_seconds = 0
    session_start = StateManager.get("session_start")
    if session_start:
        duration = datetime.now() - session_start
        duration_seconds = int(duration.total_seconds())
    
    # Format duration as MM:SS
    minutes = duration_seconds // 60
    seconds = duration_seconds % 60
    duration_str = f"{minutes:02d}:{seconds:02d}"
    
    # Get metrics from state
    rep_count = StateManager.get("rep_count", 0)
    calories = StateManager.get("calories", 0.0)
    quality_score = StateManager.get("quality_score", 0.0)
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f"""
            <div style="
                background-color: {COLORS['card_background']};
                border-radius: {BORDER_RADIUS['lg']};
                padding: {SPACING['lg']};
                box-shadow: {SHADOWS['sm']};
                border: 2px solid {COLORS['border']};
                text-align: center;
            ">
                <div style="
                    display: flex;
                    justify-content: center;
                    margin-bottom: {SPACING['sm']};
                ">
                    <span class="material-icons" style="
                        font-size: 32px;
                        color: {COLORS['primary']};
                    ">fitness_center</span>
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_3xl']};
                    font-weight: {TYPOGRAPHY['font_weight_bold']};
                    color: {COLORS['text_primary']};
                    margin-bottom: {SPACING['xs']};
                ">
                    {rep_count}
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_sm']};
                    color: {COLORS['text_secondary']};
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">
                    Reps
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col2:
        st.markdown(
            f"""
            <div style="
                background-color: {COLORS['card_background']};
                border-radius: {BORDER_RADIUS['lg']};
                padding: {SPACING['lg']};
                box-shadow: {SHADOWS['sm']};
                border: 2px solid {COLORS['border']};
                text-align: center;
            ">
                <div style="
                    display: flex;
                    justify-content: center;
                    margin-bottom: {SPACING['sm']};
                ">
                    <span class="material-icons" style="
                        font-size: 32px;
                        color: {COLORS['secondary']};
                    ">timer</span>
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_3xl']};
                    font-weight: {TYPOGRAPHY['font_weight_bold']};
                    color: {COLORS['text_primary']};
                    margin-bottom: {SPACING['xs']};
                ">
                    {duration_str}
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_sm']};
                    color: {COLORS['text_secondary']};
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">
                    Duration
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col3:
        st.markdown(
            f"""
            <div style="
                background-color: {COLORS['card_background']};
                border-radius: {BORDER_RADIUS['lg']};
                padding: {SPACING['lg']};
                box-shadow: {SHADOWS['sm']};
                border: 2px solid {COLORS['border']};
                text-align: center;
            ">
                <div style="
                    display: flex;
                    justify-content: center;
                    margin-bottom: {SPACING['sm']};
                ">
                    <span class="material-icons" style="
                        font-size: 32px;
                        color: {COLORS['accent']};
                    ">local_fire_department</span>
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_3xl']};
                    font-weight: {TYPOGRAPHY['font_weight_bold']};
                    color: {COLORS['text_primary']};
                    margin-bottom: {SPACING['xs']};
                ">
                    {calories:.1f}
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_sm']};
                    color: {COLORS['text_secondary']};
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">
                    Calories
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    
    with col4:
        # Determine quality color based on score
        if quality_score >= 80:
            quality_color = COLORS['success']
        elif quality_score >= 60:
            quality_color = COLORS['warning']
        else:
            quality_color = COLORS['error']
        
        st.markdown(
            f"""
            <div style="
                background-color: {COLORS['card_background']};
                border-radius: {BORDER_RADIUS['lg']};
                padding: {SPACING['lg']};
                box-shadow: {SHADOWS['sm']};
                border: 2px solid {COLORS['border']};
                text-align: center;
            ">
                <div style="
                    display: flex;
                    justify-content: center;
                    margin-bottom: {SPACING['sm']};
                ">
                    <span class="material-icons" style="
                        font-size: 32px;
                        color: {quality_color};
                    ">grade</span>
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_3xl']};
                    font-weight: {TYPOGRAPHY['font_weight_bold']};
                    color: {COLORS['text_primary']};
                    margin-bottom: {SPACING['xs']};
                ">
                    {quality_score:.0f}%
                </div>
                <div style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_sm']};
                    color: {COLORS['text_secondary']};
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                ">
                    Quality
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_feedback_messages():
    """Render real-time feedback messages."""
    current_feedback = StateManager.get("current_feedback", [])
    
    if not current_feedback:
        # Show default message when no feedback
        st.markdown(
            f"""
            <div style="
                background-color: {COLORS['info']}15;
                border-left: 4px solid {COLORS['info']};
                border-radius: {BORDER_RADIUS['md']};
                padding: {SPACING['md']};
                margin: {SPACING['lg']} 0;
            ">
                <div style="
                    display: flex;
                    align-items: center;
                    gap: {SPACING['sm']};
                ">
                    <span class="material-icons" style="
                        font-size: 24px;
                        color: {COLORS['info']};
                    ">info</span>
                    <span style="
                        font-family: {TYPOGRAPHY['font_family_primary']};
                        font-size: {TYPOGRAPHY['font_size_base']};
                        color: var(--text-color);
                    ">
                        Position yourself in frame and start exercising to receive feedback
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        # Display feedback messages
        for feedback in current_feedback:
            # Determine feedback type and color
            feedback_lower = feedback.lower()
            if "good" in feedback_lower or "great" in feedback_lower or "excellent" in feedback_lower:
                feedback_color = COLORS['success']
                icon = "check_circle"
            elif "warning" in feedback_lower or "careful" in feedback_lower or "watch" in feedback_lower:
                feedback_color = COLORS['warning']
                icon = "warning"
            else:
                feedback_color = COLORS['error']
                icon = "error"
            
            st.markdown(
                f"""
                <div style="
                    background-color: {feedback_color}15;
                    border-left: 4px solid {feedback_color};
                    border-radius: {BORDER_RADIUS['md']};
                    padding: {SPACING['md']};
                    margin: {SPACING['sm']} 0;
                ">
                    <div style="
                        display: flex;
                        align-items: center;
                        gap: {SPACING['sm']};
                    ">
                        <span class="material-icons" style="
                            font-size: 24px;
                            color: {feedback_color};
                        ">{icon}</span>
                        <span style="
                            font-family: {TYPOGRAPHY['font_family_primary']};
                            font-size: {TYPOGRAPHY['font_size_base']};
                            color: var(--text-color);
                        ">
                            {feedback}
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_camera_feed_with_feedback():
    """Render camera feed component with pose detection and feedback inside the loop."""
    st.markdown(
        f"""
        <div style="
            text-align: center;
            margin: {SPACING['xl']} 0;
        ">
            <h3 style="
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_2xl']};
                font-weight: {TYPOGRAPHY['font_weight_semibold']};
                color: {COLORS['text_primary']};
                margin-bottom: {SPACING['md']};
            ">
                Live Camera Feed
            </h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Create placeholder for camera feed
    camera_placeholder = st.empty()
    
    # Create placeholder for feedback (will be updated inside camera loop)
    st.markdown(
        f"""
        <h3 style="
            font-family: {TYPOGRAPHY['font_family_primary']};
            font-size: {TYPOGRAPHY['font_size_xl']};
            font-weight: {TYPOGRAPHY['font_weight_semibold']};
            color: {COLORS['text_primary']};
            margin: {SPACING['lg']} 0 {SPACING['md']} 0;
        ">
            Real-Time Feedback
        </h3>
        """,
        unsafe_allow_html=True,
    )
    feedback_placeholder = st.empty()
    
    # Camera control buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        camera_active = StateManager.get("camera_active", False)
        if st.button("📹 Start Camera", use_container_width=True, disabled=camera_active):
            StateManager.set("camera_active", True)
            st.rerun()
    
    with col2:
        camera_active = StateManager.get("camera_active", False)
        if st.button("⏸️ Stop Camera", use_container_width=True, disabled=not camera_active):
            StateManager.set("camera_active", False)
            st.rerun()
    
    # If camera is active, show live feed
    if StateManager.get("camera_active", False):
        try:
            # Open camera
            cap = cv2.VideoCapture(0)
            
            if not cap.isOpened():
                ErrorHandler.render_camera_error("Camera device not found or already in use")
                StateManager.set("camera_active", False)
                return
            
            # Get API client
            api_client = get_api_client()
            
            # Process frames
            frame_count = 0
            max_frames = 300  # Process for ~10 seconds at 30fps
            
            while StateManager.get("camera_active", False) and frame_count < max_frames:
                ret, frame = cap.read()
                
                if not ret:
                    ErrorHandler.render_camera_error("Failed to read frame from camera")
                    break
                
                # Detect pose every frame
                pose_result = api_client.detect_pose(frame, draw_landmarks=True)
                
                if pose_result and pose_result.get('detected'):
                    # Get annotated image
                    if 'annotated_image' in pose_result:
                        annotated_frame = api_client.decode_frame(pose_result['annotated_image'])
                    else:
                        annotated_frame = frame
                    
                    # Analyze exercise
                    key_points = pose_result.get('key_points', {})
                    
                    # Map exercise type to display name
                    exercise_display_map = {
                        "bicep_curl": "Bicep Curls",
                        "squat": "Squats",
                        "push_up": "Push-ups",
                    }
                    active_exercise = StateManager.get("active_exercise")
                    display_name = exercise_display_map.get(active_exercise, active_exercise)
                    
                    session_id = StateManager.get("session_id")
                    analysis_result = api_client.analyze_exercise(
                        session_id,
                        display_name,
                        key_points
                    )
                    
                    if analysis_result:
                        # Extract all feedback types from API response
                        feedback_list = []
                        
                        # Add errors
                        if analysis_result.get('errors'):
                            feedback_list.extend(analysis_result.get('errors', []))
                        
                        # Add warnings
                        if analysis_result.get('warnings'):
                            feedback_list.extend(analysis_result.get('warnings', []))
                        
                        # Add positive feedback
                        if analysis_result.get('feedback'):
                            feedback_list.extend(analysis_result.get('feedback', []))
                        
                        # Add real-time feedback message if available
                        if analysis_result.get('real_time_feedback'):
                            feedback_list.append(analysis_result.get('real_time_feedback'))
                        
                        # Update session state with analysis results using StateManager
                        StateManager.update_workout_metrics(
                            rep_count=analysis_result.get('rep_count'),
                            calories=analysis_result.get('calories'),
                            quality_score=analysis_result.get('quality_score'),
                            feedback=feedback_list
                        )
                        
                        # Update feedback display dynamically INSIDE the loop
                        with feedback_placeholder.container():
                            render_feedback_messages()
                    else:
                        # API call failed - show warning but continue
                        if frame_count % 30 == 0:  # Show warning every second
                            st.warning("⚠️ Temporary connection issue with backend. Retrying...", icon="⚠️")
                    
                    # Display annotated frame
                    camera_placeholder.image(annotated_frame, channels="BGR", use_container_width=True)
                elif pose_result is None:
                    # API call failed completely
                    if frame_count % 30 == 0:  # Show error every second
                        st.error("❌ Lost connection to backend. Please check backend status.", icon="🚨")
                    # Still show the frame
                    camera_placeholder.image(frame, channels="BGR", use_container_width=True)
                else:
                    # No pose detected - show overlay message
                    overlay_frame = frame.copy()
                    h, w = overlay_frame.shape[:2]
                    cv2.putText(
                        overlay_frame,
                        "Step into frame",
                        (w // 2 - 150, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.5,
                        (0, 0, 255),
                        3
                    )
                    camera_placeholder.image(overlay_frame, channels="BGR", use_container_width=True)
                
                frame_count += 1
                time.sleep(0.033)  # ~30 fps
            
            # Release camera
            cap.release()
            
            if frame_count >= max_frames:
                st.info("ℹ️ Camera feed stopped after timeout. Click 'Start Camera' to continue.", icon="ℹ️")
                StateManager.set("camera_active", False)
                
        except cv2.error as e:
            ErrorHandler.render_camera_error(f"OpenCV error: {str(e)}")
            StateManager.set("camera_active", False)
        except Exception as e:
            ErrorHandler.render_camera_error(f"Unexpected error: {str(e)}")
            StateManager.set("camera_active", False)
    else:
        # Show placeholder when camera is not active
        camera_placeholder.markdown(
            f"""
            <div style="
                background-color: {COLORS['background_secondary']};
                border: 2px dashed {COLORS['border']};
                border-radius: {BORDER_RADIUS['lg']};
                padding: {SPACING['3xl']};
                text-align: center;
                min-height: 400px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
            ">
                <span class="material-icons" style="
                    font-size: 72px;
                    color: {COLORS['text_tertiary']};
                    margin-bottom: {SPACING['lg']};
                ">videocam_off</span>
                <p style="
                    font-family: {TYPOGRAPHY['font_family_primary']};
                    font-size: {TYPOGRAPHY['font_size_lg']};
                    color: {COLORS['text_secondary']};
                ">
                    Camera is not active. Click "Start Camera" to begin.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Show initial feedback message when camera is off
        with feedback_placeholder.container():
            render_feedback_messages()


def render_active_session():
    """Render the active workout session view."""
    # Get exercise display name
    exercise_display_map = {
        "bicep_curl": "Bicep Curls",
        "squat": "Squats",
        "push_up": "Push-ups",
    }
    active_exercise = StateManager.get("active_exercise")
    exercise_name = exercise_display_map.get(
        active_exercise,
        active_exercise.replace('_', ' ').title() if active_exercise else "Unknown"
    )
    
    session_id = StateManager.get("session_id", "")
    
    # Page header
    st.markdown(
        f"""
        <div style="
            text-align: center;
            padding: {SPACING['xl']} {SPACING['lg']} {SPACING['md']};
            margin-bottom: {SPACING['lg']};
        ">
            <h1 style="
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_3xl']};
                font-weight: {TYPOGRAPHY['font_weight_bold']};
                color: {COLORS['text_primary']};
                line-height: {TYPOGRAPHY['line_height_tight']};
                margin-bottom: {SPACING['xs']};
            ">
                {exercise_name} Session
            </h1>
            <p style="
                font-family: {TYPOGRAPHY['font_family_primary']};
                font-size: {TYPOGRAPHY['font_size_base']};
                color: {COLORS['text_secondary']};
            ">
                Session ID: <code style="
                    font-family: {TYPOGRAPHY['font_family_mono']};
                    background-color: {COLORS['background_secondary']};
                    padding: 2px 6px;
                    border-radius: {BORDER_RADIUS['sm']};
                ">{session_id}</code>
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Session metrics
    render_session_metrics()
    
    st.markdown(f"<div style='margin: {SPACING['xl']} 0;'></div>", unsafe_allow_html=True)
    
    # Camera feed (includes feedback inside the loop)
    render_camera_feed_with_feedback()
    
    st.markdown(f"<div style='margin: {SPACING['2xl']} 0;'></div>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔙 Back to Exercise Selection", use_container_width=True):
            # Stop camera if active
            StateManager.set("camera_active", False)
            
            # End session
            with st.spinner("Ending session..."):
                try:
                    success = end_workout_session()
                    
                    if success:
                        st.success("✅ Session ended successfully!", icon="🎉")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("⚠️ Session ended but backend communication failed. Data may not be saved.", icon="⚠️")
                        # Clear state anyway using StateManager
                        StateManager.end_workout_session()
                        time.sleep(1)
                        st.rerun()
                except Exception as e:
                    ErrorHandler.handle_exception(
                        e,
                        "ending workout session",
                        show_to_user=True,
                        fallback_message="Failed to end session properly. Returning to exercise selection."
                    )
                    # Clear state anyway
                    StateManager.end_workout_session()
                    time.sleep(1)
                    st.rerun()
    
    with col3:
        if st.button("🏁 End Workout", use_container_width=True, type="primary"):
            # Stop camera if active
            StateManager.set("camera_active", False)
            
            # Get current metrics before ending
            rep_count = StateManager.get("rep_count", 0)
            calories = StateManager.get("calories", 0.0)
            
            # End session
            with st.spinner("Saving workout session..."):
                try:
                    success = end_workout_session()
                    
                    if success:
                        st.success(
                            f"🎉 Great workout! You completed {rep_count} reps "
                            f"and burned {calories:.1f} calories!",
                            icon="🏆"
                        )
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.warning(
                            f"⚠️ Workout completed ({rep_count} reps, {calories:.1f} calories) "
                            "but failed to save to backend. Data may not be persisted.",
                            icon="⚠️"
                        )
                        time.sleep(2)
                        st.rerun()
                except Exception as e:
                    ErrorHandler.handle_exception(
                        e,
                        "saving workout session",
                        show_to_user=True,
                        fallback_message="Failed to save workout session. Your progress may not be recorded."
                    )
                    time.sleep(1)
                    st.rerun()


def main():
    """Main function to render the Workout page."""
    # Apply page configuration
    apply_page_config(
        page_title="Workout - AI Fitness Trainer",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Inject custom CSS and Material Icons
    inject_custom_css()
    inject_material_icons_cdn()
    inject_hover_styles()

    # Add HD fitness background - Unique "App Code" aesthetic
    # Using high-quality gym equipment close-up (Weights)
    bg_image_url = "https://images.unsplash.com/photo-1526506118085-60ce8714f8c5?q=80&w=2670&auto=format&fit=crop"
    
    st.markdown(
        f"""
        <style>
        /* Force Main Background Override - Robust Selectors */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.85)), 
                url('{bg_image_url}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        
        /* Make all headers and toolbars transparent */
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background-color: transparent !important;
        }}
        
        .main {{
            background: transparent !important;
        }}
        
        /* Ensure text visibility */
        h1, h2, h3, p, span, div, label {{
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
        

    
    # Initialize session state
    StateManager.initialize_all()
    
    # Set current page
    Navigation.set_current_page("workout")
    
    # Render navigation
    Navigation.render_sidebar_nav()
    
    # Check backend health
    if not check_backend_health():
        render_backend_error()
        return
    
    # Check if there's an active session using StateManager
    if StateManager.is_workout_active():
        # Render active session view
        render_active_session()
    else:
        # Render page header
        render_page_header()
        

        
        # Render exercise selection view
        render_exercise_selection()
        






def render_enhanced_exercise_card(exercise):
    """Render a modern exercise card structure with glassmorphism."""
    
    # Get theme-aware semi-transparent colors
    bg_style, border_color, glow_color = get_category_style(exercise.get('category'))
    
    st.markdown(
        f"""
<div style="background: {bg_style}; backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px); border-radius: 16px; padding: 24px; margin-bottom: 24px; box-shadow: 0 4px 6px rgba(0,0,0,0.1), {glow_color}; border: 1px solid {border_color}; color: #f8fafc; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden;" class="exercise-card-enhanced">
<div style="font-size: 48px; margin-bottom: 16px; text-align: center; filter: drop-shadow(0 0 10px rgba(255,255,255,0.3));">
{get_icon_for_exercise(exercise['type'])}
</div>
<h3 style="margin: 0 0 8px 0; font-size: 22px; font-weight: 700; text-align: center; color: #f8fafc !important; text-shadow: 0 2px 4px rgba(0,0,0,0.3);">{exercise['display_name']}</h3>
<div style="text-align: center; margin-bottom: 20px;">
<span style="background: rgba(255,255,255,0.1); border: 1px solid rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 20px; font-size: 12px; font-weight: 600; letter-spacing: 0.05em; text-transform: uppercase; color: #e2e8f0;">{exercise['difficulty']}</span>
</div>
<p style="font-size: 15px; line-height: 1.6; opacity: 0.9; margin-bottom: 24px; text-align: center; min-height: 48px; color: #cbd5e1 !important;">
{exercise['description']}
</p>
<div style="display: flex; justify-content: space-between; font-size: 14px; margin-bottom: 0; background: rgba(0,0,0,0.2); padding: 12px 16px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05);">
<div style="display: flex; align-items: center; gap: 6px; color: #e2e8f0;">
<span style="font-size: 16px;">⏱️</span> {exercise['duration']}
</div>
<div style="display: flex; align-items: center; gap: 6px; color: #e2e8f0;">
<span style="font-size: 16px;">🔥</span> {exercise['calories']}
</div>
</div>
</div>
""",
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns([1, 1])
    with col1:
         with st.expander("Details"):
             st.markdown(f"**Best for:** {exercise.get('difficulty')}")
             for benefit in exercise.get('benefits', []):
                 st.caption(f"• {benefit}")
                 
    with col2:
         # Start button
         if st.button(
             f"Start",
             key=f"start_enhanced_{exercise['type']}",
             use_container_width=True,
             type="primary"
         ):
             with st.spinner(f"Starting {exercise['display_name']} session..."):
                 success = start_workout_session(exercise["type"])
                 if success:
                     st.rerun()
                 else:
                     render_session_start_error(exercise['display_name'])


def get_icon_for_exercise(exercise_type):
    """Return an emoji icon for the exercise type."""
    icons = {
        "bicep_curl": "💪",
        "push_up": "🙇",
        "shoulder_press": "🏋️",
        "squat": "🦵",
        "lunge": "🚶",
        "plank": "🪜"
    }
    return icons.get(exercise_type, "🏃")


if __name__ == "__main__":
    main()
