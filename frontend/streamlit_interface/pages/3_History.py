"""
History page for the AI Fitness Trainer Streamlit application.

This page displays past workout sessions with filtering capabilities,
formatted data display, and summary statistics.
"""

import streamlit as st
import textwrap
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from styles.custom_css import inject_custom_css, apply_page_config
from styles.theme import COLORS, TYPOGRAPHY, SPACING, BORDER_RADIUS, SHADOWS
from utils.icons import (
    inject_material_icons_cdn,
    get_icon_name,
    render_icon,
)
from utils.state_manager import StateManager
from components.navigation import Navigation
from components.auth_header import render_auth_header
from services.workout_loader import WorkoutHistoryLoader
from services.workout_filter import WorkoutHistoryFilter
from services.workout_formatter import WorkoutHistoryFormatter
from services.workout_aggregator import WorkoutHistoryAggregator
from utils.error_handler import ErrorHandler


def inject_history_background():
    """Inject specific background image for the History page."""
    # High-quality dark gym aesthetic image (Dumbbells/Weights)
    bg_image_url = "https://images.unsplash.com/photo-1534438327276-14e5300c3a48?q=80&w=2070&auto=format&fit=crop"
    
    st.markdown(
        f"""
        <style>
        /* Force background on the main container - Targeting multiple possible selectors */
        .stApp, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
            background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.8)), 
                        url('{bg_image_url}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        
        /* Make header and main content container transparent */
        [data-testid="stHeader"], [data-testid="stToolbar"] {{
            background-color: transparent !important;
        }}
        
        .main {{
            background: transparent !important;
        }}
        
        /* Ensure readable text on dark background */
        h1, h2, h3, p, span, div, label {{
            color: #ffffff !important;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.8);
        }}
        
        /* Specific adjustments for custom cards */
        .history-card {{
            background: rgba(30, 41, 59, 0.7) !important;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2) !important;
            box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        }}
        
        /* Ensure inputs and selects are visible */
        .stSelectbox div[data-baseweb="select"] > div {{
            background-color: rgba(30, 41, 59, 0.9) !important;
            color: white !important;
            border-color: rgba(255, 255, 255, 0.2);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )




def load_workout_sessions():
    """Load workout sessions from the backend data directory."""
    # Use cached sessions if available
    loaded_sessions = StateManager.get("loaded_sessions")
    loader_state = StateManager.get("loader_state")
    
    if loaded_sessions is not None and loader_state is not None:
        return loaded_sessions, loader_state
    
    # Load sessions using WorkoutHistoryLoader
    loader = WorkoutHistoryLoader()
    
    try:
        sessions = loader.load_all_sessions()
        
        # Store loader state for error reporting
        loader_state = {
            'corrupted_count': loader.get_corrupted_file_count(),
            'has_errors': loader.has_load_errors(),
            'errors': loader.load_errors
        }
        
        # Cache sessions and loader state in session state
        StateManager.set("loaded_sessions", sessions)
        StateManager.set("loader_state", loader_state)
        
        return sessions, loader_state
    except Exception as e:
        ErrorHandler.handle_exception(
            e,
            "loading workout history",
            show_to_user=True,
            fallback_message="Failed to load workout history. Please try refreshing the page."
        )
        return [], {'corrupted_count': 0, 'has_errors': True, 'errors': {}}


def render_page_header():
    """Render the history page header."""
    st.title("📜 Workout History")
    st.write("Review your past workout sessions and track your progress over time")
    st.divider()


def render_filter_controls(sessions):
    """
    Render filter controls dropdown for exercise type selection.
    
    Args:
        sessions: List of all workout sessions
        
    Returns:
        Selected filter value
    """
    # Get unique exercise types
    exercise_types = WorkoutHistoryFilter.get_unique_exercise_types(sessions)
    
    # Create filter options
    filter_options = ["All Exercises"] + [
        WorkoutHistoryFormatter.format_exercise_name(ex) for ex in exercise_types
    ]
    
    # Map display names back to internal types
    display_to_internal = {
        WorkoutHistoryFormatter.format_exercise_name(ex): ex 
        for ex in exercise_types
    }
    display_to_internal["All Exercises"] = "all"
    
    # Render filter section
    st.subheader("🔍 Filter Workouts")
    st.write("")
    
    # Create columns for filter controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        # Exercise type filter dropdown
        selected_display = st.selectbox(
            "Exercise Type",
            options=filter_options,
            index=0,
            key="exercise_filter_dropdown",
            label_visibility="collapsed"
        )
        
        # Convert display name back to internal type
        selected_filter = display_to_internal[selected_display]
        
        # Update session state
        StateManager.set("selected_exercise_filter", selected_filter)
        
        return selected_filter
    
    with col2:
        # Refresh button
        if st.button("🔄 Refresh", use_container_width=True):
            StateManager.clear_history_cache()
            st.rerun()
    
    with col3:
        # Clear filter button
        if st.button("✖️ Clear Filter", use_container_width=True):
            StateManager.set("selected_exercise_filter", "all")
            st.rerun()


def render_summary_statistics(sessions):
    """
    Render summary statistics cards showing totals.
    
    Args:
        sessions: List of workout sessions to aggregate
    """
    # Calculate aggregate statistics
    total_workouts = WorkoutHistoryAggregator.calculate_total_workouts(sessions)
    total_reps = WorkoutHistoryAggregator.calculate_total_reps(sessions)
    total_calories = WorkoutHistoryAggregator.calculate_total_calories(sessions)
    total_duration = WorkoutHistoryAggregator.calculate_total_duration(sessions)
    
    # Format duration
    duration_formatted = WorkoutHistoryFormatter.format_duration(total_duration)
    
    # Render statistics cards
    st.subheader("📊 Summary Statistics")
    st.write("")
    
    # Create columns for metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        {
            "icon": "fitness_center",
            "value": str(total_workouts),
            "label": "Total Workouts",
            "color": COLORS['primary'],
        },
        {
            "icon": "repeat",
            "value": str(total_reps),
            "label": "Total Reps",
            "color": COLORS['secondary'],
        },
        {
            "icon": "local_fire_department",
            "value": f"{total_calories:.1f}",
            "label": "Total Calories",
            "color": COLORS['accent'],
        },
        {
            "icon": "schedule",
            "value": duration_formatted,
            "label": "Total Duration",
            "color": COLORS['info'],
        },
    ]
    
    for col, metric in zip([col1, col2, col3, col4], metrics):
        with col:
            st.metric(
                label=metric['label'],
                value=metric['value'],
                delta=None
            )


from components.footer import render_footer

def render_session_card(session):
    """
    Render a single workout session card using custom HTML/CSS.
    
    Args:
        session: Workout session dictionary
    """
    # Format session data
    raw_exercise_name = session.get('exercise', 'Unknown')
    exercise_name = WorkoutHistoryFormatter.format_exercise_name(raw_exercise_name)
    
    # Get icon for the exercise
    icon_name = get_icon_name("exercise", raw_exercise_name)
    
    # Format date (e.g., "Today", "Yesterday", or "Oct 12, 2023")
    # For now ensuring we have a readable string, assuming format_date handles it well
    date_formatted = WorkoutHistoryFormatter.format_date(
        session.get('start_time', '')
    )
    
    duration_formatted = WorkoutHistoryFormatter.format_duration(
        session.get('duration', 0)
    )
    reps = session.get('reps', 0)
    calories = session.get('calories', 0.0)
    status = session.get('status', 'unknown').lower()
    
    # Determine status class and icon
    status_class = "status-incomplete"
    status_icon = "⚠️"
    if status == 'completed':
        status_class = "status-completed"
        status_icon = "✅"
    elif status == 'active':
        status_class = "status-active"
        status_icon = "🏃"

    # Get additional session info
    start_time = session.get('start_time', '')
    quality_score = session.get('quality_score', 0)
    
    # Format start time for display
    time_display = ""
    if start_time:
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
            time_display = dt.strftime("%I:%M %p")
        except:
            time_display = ""
    
    # Custom HTML Card
    card_html = f"""
<div class="history-card animate-slide-up">
<div class="history-content">
<!-- Icon Section -->
<div class="history-icon-wrapper">
<div class="history-icon-bg"></div>
<span class="material-icons icon-glow-primary">{icon_name}</span>
</div>
<!-- Details Section -->
<div class="history-details">
<div class="history-header">
<div>
<div class="history-title">{exercise_name}</div>
<div class="history-date">
<span class="material-icons" style="font-size: 14px;">calendar_today</span>
{date_formatted}
{f'<span style="margin-left: 1rem; opacity: 0.7;">🕐 {time_display}</span>' if time_display else ''}
</div>
</div>
<span class="status-badge {status_class}">
{status_icon} {status.upper()}
</span>
</div>
<div class="history-stats">
<div class="stat-item">
<div class="stat-value">{reps}</div>
<div class="stat-label">Reps</div>
</div>
<div class="stat-item">
<div class="stat-value">{duration_formatted}</div>
<div class="stat-label">Duration</div>
</div>
<div class="stat-item">
<div class="stat-value">{calories:.0f}</div>
<div class="stat-label">Calories</div>
</div>
{f'<div class="stat-item"><div class="stat-value">{quality_score}%</div><div class="stat-label">Quality</div></div>' if quality_score > 0 else ''}
</div>
</div>
</div>
</div>
"""
    st.markdown(card_html, unsafe_allow_html=True)


def render_session_list(sessions):
    """
    Render list of workout sessions with formatted data.
    
    Args:
        sessions: List of workout sessions to display
    """
    st.subheader(f"📝 Recent Sessions ({len(sessions)})")
    st.write("")
    
    # Render each session card
    for session in sessions:
        render_session_card(session)


def render_empty_state(filter_applied=False):
    """
    Render empty state when no sessions exist or filter returns no results.
    
    Args:
        filter_applied: Whether a filter is currently applied
    """
    if filter_applied:
        st.info("🔍 No workout sessions match your current filter. Try selecting a different exercise type or clear the filter.")
    else:
        st.info("💪 You haven't completed any workout sessions yet. Start your first workout to begin tracking your progress!")
    
    # Add action button
    if not filter_applied:
        if st.button("🏋️ Start Your First Workout", use_container_width=True):
            StateManager.set_current_page("workout")
            st.switch_page("pages/2_Workout.py")


def main():
    """Main function to render the History page."""
    # Apply page configuration
    apply_page_config(
        page_title="History - AI Fitness Trainer",
        page_icon="📜",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Render global auth header
    render_auth_header()
    
    # Inject custom CSS and Material Icons
    inject_custom_css()
    inject_material_icons_cdn()
    
    # Inject page-specific background
    inject_history_background()
    
    # Initialize session state
    StateManager.initialize_all()
    
    # Set current page
    Navigation.set_current_page("history")
    
    # Render navigation
    Navigation.render_sidebar_nav()
    
    # Render page header
    render_page_header()
    
    # Load workout sessions
    all_sessions, loader_state = load_workout_sessions()
    
    # Show warning for corrupted files if any
    if loader_state.get('corrupted_count', 0) > 0:
        ErrorHandler.render_corrupted_data_warning(loader_state['corrupted_count'])
    
    # Check if any sessions exist
    if not all_sessions:
        # Display empty state
        render_empty_state(filter_applied=False)
    else:
        # Render filter controls
        selected_filter = render_filter_controls(all_sessions)
        
        # Apply filter to sessions
        filtered_sessions = WorkoutHistoryFilter.filter_by_exercise(
            all_sessions, selected_filter
        )
        
        # Sort sessions in reverse chronological order (newest first)
        sorted_sessions = WorkoutHistoryFilter.sort_by_date(
            filtered_sessions, reverse=True
        )
        
        # Check if filtered results are empty
        if not sorted_sessions:
            # Display empty state for filtered results
            render_empty_state(filter_applied=True)
        else:
            # Render summary statistics
            render_summary_statistics(sorted_sessions)
            
            # Add divider
            st.divider()
            
            # Render session list
            render_session_list(sorted_sessions)
    
    # Render Footer at the bottom
    render_footer()


if __name__ == "__main__":
    main()
