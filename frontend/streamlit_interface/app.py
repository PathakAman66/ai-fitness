"""
Main entry point for the AI Fitness Trainer Streamlit application.

This module serves as the primary entry point for the multi-page Streamlit
application. It configures page settings, injects custom CSS and theme,
initializes session state, and performs backend health checks on startup.

Usage:
    streamlit run frontend/streamlit_interface/app.py
"""

import streamlit as st
import sys
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from styles.custom_css import inject_custom_css, apply_page_config
from styles.theme import COLORS, TYPOGRAPHY, SPACING
from utils.icons import inject_material_icons_cdn
from utils.state_manager import StateManager
from services.api_client import APIClient


def configure_page():
    """
    Configure Streamlit page settings.
    
    Sets up the page title, icon, layout, and initial sidebar state
    for the entire application.
    """
    apply_page_config(
        page_title="AI Fitness Trainer",
        page_icon="💪",
        layout="wide",
        initial_sidebar_state="expanded"
    )


def inject_styles():
    """
    Inject custom CSS and theme styling.
    
    Applies professional styling including typography, colors, spacing,
    and responsive design across all pages.
    """
    # Inject dark mode CSS variables and sidebar fixes
    st.markdown(
        """
        <style>
        :root {
            --text-color: #111827;
            --text-secondary: #6b7280;
            --card-bg: #f9fafb;
            --border-color: #e5e7eb;
            --success-bg: #f0fdf4;
        }
        
        /* Dark mode support */
        @media (prefers-color-scheme: dark) {
            :root {
                --text-color: #f9fafb;
                --text-secondary: #9ca3af;
                --card-bg: #1f2937;
                --border-color: #374151;
                --success-bg: #064e3b;
            }
        }
        
        /* Force dark mode for Streamlit dark theme */
        [data-testid="stAppViewContainer"][data-theme="dark"] {
            --text-color: #f9fafb;
            --text-secondary: #9ca3af;
            --card-bg: #1f2937;
            --border-color: #374151;
            --success-bg: #064e3b;
        }
        
        /* Remove top white space - Fix header/toolbar */
        [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        [data-testid="stToolbar"] {
            background-color: transparent !important;
        }
        
        /* Hide the toolbar buttons but keep the hamburger menu */
        [data-testid="stToolbar"] > div:not(:first-child) {
            display: none !important;
        }
        
        [data-theme="dark"] [data-testid="stHeader"] {
            background-color: transparent !important;
        }
        
        [data-theme="dark"] [data-testid="stToolbar"] {
            background-color: transparent !important;
        }
        
        /* Fix sidebar menu in dark mode */
        [data-testid="stSidebar"] {
            background-color: transparent !important;
        }
        
        /* Fix sidebar navigation links in dark mode */
        [data-testid="stSidebarNav"] {
            background-color: transparent !important;
        }
        
        [data-testid="stSidebarNav"] ul {
            background-color: transparent !important;
        }
        
        [data-testid="stSidebarNav"] li {
            background-color: transparent !important;
        }
        
        [data-testid="stSidebarNav"] a {
            background-color: transparent !important;
        }
        
        /* Ensure text is visible in dark mode */
        [data-theme="dark"] [data-testid="stSidebarNav"] a {
            color: #f9fafb !important;
        }
        
        [data-theme="dark"] [data-testid="stSidebarNav"] a:hover {
            background-color: rgba(255, 255, 255, 0.1) !important;
        }
        
        /* Remove top padding on main content */
        .main .block-container {
            padding-top: 1rem !important;
            padding-bottom: 1rem !important;
        }
        
        /* Remove top margin/padding from main container */
        .main {
            padding-top: 0 !important;
        }
        
        /* Fix sidebar padding */
        section[data-testid="stSidebar"] > div:first-child {
            padding-top: 1rem !important;
        }
        
        /* Fix button text visibility - WHITE TEXT ONLY */
        .stButton > button {
            color: #ffffff !important;
        }
        
        .stButton > button:hover {
            color: #ffffff !important;
        }
        
        /* Fix button text in dark mode - WHITE TEXT ONLY */
        [data-testid="stAppViewContainer"][data-theme="dark"] .stButton > button {
            color: #ffffff !important;
        }
        
        [data-testid="stAppViewContainer"][data-theme="dark"] .stButton > button:hover {
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    inject_custom_css()
    inject_material_icons_cdn()


def initialize_state():
    """
    Initialize session state with default values.
    
    Ensures all required state variables exist with appropriate defaults
    before any page is rendered.
    """
    StateManager.initialize_all()


def check_backend_health():
    """
    Perform backend health check on startup.
    
    Checks if the FastAPI backend is available and displays a warning
    if it's not accessible. This helps users identify connectivity issues
    early in the application lifecycle.
    """
    # Check backend health (using cache to avoid repeated checks)
    is_healthy = APIClient.check_health(use_cache=True)
    
    if not is_healthy:
        st.warning(
            "⚠️ **Backend API is not available**\n\n"
            "The FastAPI backend server is not responding. "
            "Some features may not work correctly.\n\n"
            "**To start the backend:**\n"
            "```bash\n"
            "uvicorn backend.api.main:app --reload\n"
            "```",
            icon="⚠️"
        )


def render_welcome_page():
    """
    Render the main welcome page.
    
    This page is displayed when users access the root URL of the application.
    It provides a welcome message and directs users to navigate using the sidebar.
    """
    # Centered header with icon and title at the top
    st.markdown(
        """
        <div style='display: flex; align-items: center; justify-content: center; gap: 0.75rem; margin-bottom: 1.5rem; padding-top: 0.5rem;'>
            <span class="material-icons" style='font-size: 2rem; color: #2563eb;'>fitness_center</span>
            <h1 style='font-size: 1.75rem; font-weight: 700; margin: 0; color: var(--text-color);'>AI Fitness Trainer</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Get Started section
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem;'>
            <span class="material-icons" style='color: #2563eb;'>rocket_launch</span>
            <h2 style='margin: 0; color: var(--text-color);'>Get Started</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.info(
        "Use the navigation menu in the sidebar to explore the application and start your fitness journey!"
    )
    
    # Navigation cards with icons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            """
            <div style='text-align: center; padding: 1.5rem; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color);'>
                <span class="material-icons" style='font-size: 3rem; color: #2563eb; margin-bottom: 0.5rem;'>home</span>
                <div style='font-weight: 600; color: var(--text-color);'>Home</div>
                <div style='font-size: 0.875rem; color: var(--text-secondary);'>Start here</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style='text-align: center; padding: 1.5rem; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color);'>
                <span class="material-icons" style='font-size: 3rem; color: #10b981; margin-bottom: 0.5rem;'>fitness_center</span>
                <div style='font-weight: 600; color: var(--text-color);'>Workout</div>
                <div style='font-size: 0.875rem; color: var(--text-secondary);'>Train now</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div style='text-align: center; padding: 1.5rem; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color);'>
                <span class="material-icons" style='font-size: 3rem; color: #8b5cf6; margin-bottom: 0.5rem;'>history</span>
                <div style='font-weight: 600; color: var(--text-color);'>History</div>
                <div style='font-size: 0.875rem; color: var(--text-secondary);'>View past</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col4:
        st.markdown(
            """
            <div style='text-align: center; padding: 1.5rem; background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border-color);'>
                <span class="material-icons" style='font-size: 3rem; color: #f59e0b; margin-bottom: 0.5rem;'>bar_chart</span>
                <div style='font-weight: 600; color: var(--text-color);'>Stats</div>
                <div style='font-size: 0.875rem; color: var(--text-secondary);'>Track progress</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Features section
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem; margin-top: 2rem;'>
            <span class="material-icons" style='color: #2563eb;'>auto_awesome</span>
            <h2 style='margin: 0; color: var(--text-color);'>Features</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(
            """
            <div style='margin-bottom: 1.5rem;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span class="material-icons" style='color: #2563eb;'>videocam</span>
                    <h3 style='margin: 0; color: var(--text-color);'>Real-Time Analysis</h3>
                </div>
                <p style='color: var(--text-secondary); margin-left: 2rem;'>
                    Get instant feedback on your form using advanced pose detection technology
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style='margin-bottom: 1.5rem;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span class="material-icons" style='color: #10b981;'>trending_up</span>
                    <h3 style='margin: 0; color: var(--text-color);'>Track Progress</h3>
                </div>
                <p style='color: var(--text-secondary); margin-left: 2rem;'>
                    Monitor your workout history and performance trends over time
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style='margin-bottom: 1.5rem;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span class="material-icons" style='color: #f59e0b;'>emoji_events</span>
                    <h3 style='margin: 0; color: var(--text-color);'>Achieve Goals</h3>
                </div>
                <p style='color: var(--text-secondary); margin-left: 2rem;'>
                    Set targets, track achievements, and celebrate your fitness milestones
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            """
            <div style='margin-bottom: 1.5rem;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span class="material-icons" style='color: #8b5cf6;'>psychology</span>
                    <h3 style='margin: 0; color: var(--text-color);'>Smart Feedback</h3>
                </div>
                <p style='color: var(--text-secondary); margin-left: 2rem;'>
                    Receive personalized recommendations to improve your exercise technique
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # Supported Exercises
    st.markdown(
        """
        <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.5rem; margin-top: 2rem;'>
            <span class="material-icons" style='color: #2563eb;'>sports_gymnastics</span>
            <h2 style='margin: 0; color: var(--text-color);'>Supported Exercises</h2>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            """
            <div style='padding: 1rem; background: var(--success-bg); border-radius: 8px; border-left: 4px solid #10b981;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span class="material-icons" style='color: #10b981;'>fitness_center</span>
                    <strong style='color: var(--text-color);'>Bicep Curls</strong>
                </div>
                <p style='font-size: 0.875rem; color: var(--text-secondary); margin: 0;'>
                    Build arm strength with proper form
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            """
            <div style='padding: 1rem; background: var(--success-bg); border-radius: 8px; border-left: 4px solid #10b981;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span class="material-icons" style='color: #10b981;'>accessibility</span>
                    <strong style='color: var(--text-color);'>Squats</strong>
                </div>
                <p style='font-size: 0.875rem; color: var(--text-secondary); margin: 0;'>
                    Strengthen legs and core
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col3:
        st.markdown(
            """
            <div style='padding: 1rem; background: var(--success-bg); border-radius: 8px; border-left: 4px solid #10b981;'>
                <div style='display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.5rem;'>
                    <span class="material-icons" style='color: #10b981;'>self_improvement</span>
                    <strong style='color: var(--text-color);'>Push-ups</strong>
                </div>
                <p style='font-size: 0.875rem; color: var(--text-secondary); margin: 0;'>
                    Develop upper body strength
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


def main():
    """
    Main application entry point.
    
    This function orchestrates the application startup sequence:
    1. Configure page settings
    2. Inject custom styles
    3. Initialize session state
    4. Check backend health
    5. Render welcome page
    """
    # Configure page settings
    configure_page()
    
    # Inject Landing Page Background
    inject_landing_background()
    
    # Inject custom CSS and theme
    inject_styles()


def inject_landing_background():
    """Inject specific background image for the Landing page."""
    # Unique HD Gym Aesthetic Image (Dark Moody Gym)
    bg_image_url = "https://images.unsplash.com/photo-1599058945522-28d584b6f0ff?q=80&w=2669&auto=format&fit=crop"
    
    st.markdown(
        f"""
        <style>
        /* Force background on the main container */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.9)), 
                        url('{bg_image_url}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        
        /* Make header transparent */
        [data-testid="stHeader"] {{
            background-color: transparent !important;
        }}
        
        .main {{
            background: transparent !important;
        }}
        
        /* content styling for specific landing page elements to ensure contrast */
        div[data-testid="stMarkdownContainer"] h1, 
        div[data-testid="stMarkdownContainer"] h2, 
        div[data-testid="stMarkdownContainer"] h3 {{
            text-shadow: 0 2px 4px rgba(0,0,0,0.8);
        }}
        
        /* Card transparency adjustments */
        div[style*="background: var(--card-bg)"] {{
            background: rgba(31, 41, 55, 0.7) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
        }}
        
        /* Feature section enhancements */
        div[style*="background: var(--success-bg)"] {{
            background: rgba(6, 78, 59, 0.6) !important; 
            backdrop-filter: blur(5px);
            border-left: 4px solid #10b981 !important;
            border: 1px solid rgba(16, 185, 129, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Initialize session state
    initialize_state()
    
    # Check backend health
    check_backend_health()
    

    # Render welcome page
    render_welcome_page()



if __name__ == "__main__":
    main()
