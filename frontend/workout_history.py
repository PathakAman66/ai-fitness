"""
Workout History Module for AI Fitness Trainer
Handles loading, filtering, and displaying workout session history
"""
import json
import os
from typing import List, Dict, Optional
from pathlib import Path
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkoutHistoryLoader:
    """Loads workout session data from the backend data directory"""
    
    def __init__(self, data_dir: str = "backend/data/reports"):
        """
        Initialize the WorkoutHistoryLoader
        
        Args:
            data_dir: Path to the directory containing workout session JSON files
        """
        self.data_dir = data_dir
    
    def load_all_sessions(self) -> List[Dict]:
        """
        Load all workout session JSON files from the data directory
        
        Returns:
            List of workout session dictionaries
        """
        sessions = []
        
        # Check if data directory exists
        if not os.path.exists(self.data_dir):
            logger.warning(f"Data directory does not exist: {self.data_dir}")
            return sessions
        
        # Check if it's a directory
        if not os.path.isdir(self.data_dir):
            logger.error(f"Path is not a directory: {self.data_dir}")
            return sessions
        
        # Iterate through all files in the directory
        try:
            for filename in os.listdir(self.data_dir):
                # Only process JSON files
                if filename.endswith('.json'):
                    filepath = os.path.join(self.data_dir, filename)
                    session = self.parse_session_file(filepath)
                    
                    # Only add valid sessions
                    if session is not None:
                        sessions.append(session)
        except PermissionError:
            logger.error(f"Permission denied accessing directory: {self.data_dir}")
        except Exception as e:
            logger.error(f"Error reading directory {self.data_dir}: {str(e)}")
        
        logger.info(f"Loaded {len(sessions)} workout sessions from {self.data_dir}")
        return sessions
    
    def parse_session_file(self, filepath: str) -> Optional[Dict]:
        """
        Parse a single JSON file and return session data
        
        Args:
            filepath: Path to the JSON file
            
        Returns:
            Session dictionary if valid, None if invalid or corrupted
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                session_data = json.load(f)
            
            # Validate the session data
            if self.validate_session_data(session_data):
                return session_data
            else:
                logger.warning(f"Invalid session data in file: {filepath}")
                return None
                
        except json.JSONDecodeError as e:
            logger.warning(f"Corrupted JSON file {filepath}: {str(e)}")
            return None
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return None
        except PermissionError:
            logger.warning(f"Permission denied reading file: {filepath}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing file {filepath}: {str(e)}")
            return None
    
    def validate_session_data(self, session: Dict) -> bool:
        """
        Validate that a session contains required fields
        
        Args:
            session: Session dictionary to validate
            
        Returns:
            True if session is valid, False otherwise
        """
        # Define required fields based on the data model
        required_fields = [
            'session_id',
            'exercise',
            'start_time',
            'reps',
            'duration',
            'calories',
            'status'
        ]
        
        # Check if all required fields are present
        for field in required_fields:
            if field not in session:
                logger.warning(f"Missing required field: {field}")
                return False
        
        # Additional validation: check data types
        try:
            # session_id should be a string
            if not isinstance(session['session_id'], str):
                logger.warning(f"Invalid session_id type: {type(session['session_id'])}")
                return False
            
            # exercise should be a string
            if not isinstance(session['exercise'], str):
                logger.warning(f"Invalid exercise type: {type(session['exercise'])}")
                return False
            
            # start_time should be a string
            if not isinstance(session['start_time'], str):
                logger.warning(f"Invalid start_time type: {type(session['start_time'])}")
                return False
            
            # reps should be a number (int or float)
            if not isinstance(session['reps'], (int, float)):
                logger.warning(f"Invalid reps type: {type(session['reps'])}")
                return False
            
            # duration should be a number
            if not isinstance(session['duration'], (int, float)):
                logger.warning(f"Invalid duration type: {type(session['duration'])}")
                return False
            
            # calories should be a number
            if not isinstance(session['calories'], (int, float)):
                logger.warning(f"Invalid calories type: {type(session['calories'])}")
                return False
            
            # status should be a string
            if not isinstance(session['status'], str):
                logger.warning(f"Invalid status type: {type(session['status'])}")
                return False
            
        except Exception as e:
            logger.warning(f"Error validating session data: {str(e)}")
            return False
        
        return True


class WorkoutHistoryFilter:
    """Filters and sorts workout session data"""
    
    @staticmethod
    def filter_by_exercise(sessions: List[Dict], exercise_type: str) -> List[Dict]:
        """
        Filter sessions by exercise type
        
        Args:
            sessions: List of workout session dictionaries
            exercise_type: Exercise type to filter by (e.g., "bicep_curl", "squat")
                          Use "all" or empty string to return all sessions
            
        Returns:
            List of sessions matching the exercise type
        """
        # If exercise_type is "all" or empty, return all sessions
        if not exercise_type or exercise_type.lower() == "all":
            return sessions
        
        # Filter sessions by exercise type
        filtered_sessions = []
        for session in sessions:
            # Check both 'exercise' and 'exercise_type' fields for compatibility
            session_exercise = session.get('exercise', session.get('exercise_type', ''))
            
            if session_exercise.lower() == exercise_type.lower():
                filtered_sessions.append(session)
        
        return filtered_sessions
    
    @staticmethod
    def sort_by_date(sessions: List[Dict], reverse: bool = True) -> List[Dict]:
        """
        Sort sessions chronologically by start time
        
        Args:
            sessions: List of workout session dictionaries
            reverse: If True, sort in reverse chronological order (newest first)
                    If False, sort in chronological order (oldest first)
            
        Returns:
            Sorted list of sessions
        """
        # Create a copy to avoid modifying the original list
        sorted_sessions = sessions.copy()
        
        # Sort by start_time (ISO format strings sort correctly lexicographically)
        # If start_time is not available, try start_timestamp
        def get_sort_key(session: Dict) -> str:
            # Try to get start_time first (ISO format)
            start_time = session.get('start_time', '')
            
            # If start_time is not available, try start_timestamp
            if not start_time and 'start_timestamp' in session:
                # Convert timestamp to string for sorting
                start_time = str(session['start_timestamp'])
            
            return start_time
        
        sorted_sessions.sort(key=get_sort_key, reverse=reverse)
        
        return sorted_sessions
    
    @staticmethod
    def get_unique_exercise_types(sessions: List[Dict]) -> List[str]:
        """
        Extract unique exercise types from sessions
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Sorted list of unique exercise types
        """
        exercise_types = set()
        
        for session in sessions:
            # Check both 'exercise' and 'exercise_type' fields for compatibility
            exercise = session.get('exercise', session.get('exercise_type', ''))
            
            if exercise:  # Only add non-empty exercise types
                exercise_types.add(exercise)
        
        # Return sorted list for consistent ordering
        return sorted(list(exercise_types))


class WorkoutHistoryAggregator:
    """Calculates summary statistics from workout session data"""
    
    @staticmethod
    def calculate_total_workouts(sessions: List[Dict]) -> int:
        """
        Calculate the total number of workout sessions
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Total number of workout sessions
        """
        return len(sessions)
    
    @staticmethod
    def calculate_total_reps(sessions: List[Dict]) -> int:
        """
        Calculate the total number of reps across all sessions
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Total number of reps performed
        """
        total_reps = 0
        
        for session in sessions:
            # Get reps from session, default to 0 if not present
            reps = session.get('reps', 0)
            
            # Handle negative values by treating them as 0
            if isinstance(reps, (int, float)) and reps > 0:
                total_reps += int(reps)
        
        return total_reps
    
    @staticmethod
    def calculate_total_calories(sessions: List[Dict]) -> float:
        """
        Calculate the total calories burned across all sessions
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Total calories burned
        """
        total_calories = 0.0
        
        for session in sessions:
            # Get calories from session, default to 0 if not present
            calories = session.get('calories', 0)
            
            # Handle negative values by treating them as 0
            if isinstance(calories, (int, float)) and calories > 0:
                total_calories += float(calories)
        
        return total_calories
    
    @staticmethod
    def calculate_total_duration(sessions: List[Dict]) -> float:
        """
        Calculate the total workout time across all sessions
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Total duration in seconds
        """
        total_duration = 0.0
        
        for session in sessions:
            # Get duration from session, default to 0 if not present
            duration = session.get('duration', 0)
            
            # Handle negative values by treating them as 0
            if isinstance(duration, (int, float)) and duration > 0:
                total_duration += float(duration)
        
        return total_duration



class WorkoutHistoryFormatter:
    """Formats workout data for display in the UI"""
    
    @staticmethod
    def format_date(timestamp: str) -> str:
        """
        Convert ISO timestamp to human-readable format
        
        Args:
            timestamp: ISO format timestamp string (e.g., "2024-01-15T14:30:00")
            
        Returns:
            Human-readable date string (e.g., "Jan 15, 2024 2:30 PM")
        """
        try:
            # Parse the ISO format timestamp
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Format as human-readable string
            # Example: "Jan 15, 2024 2:30 PM"
            return dt.strftime("%b %d, %Y %I:%M %p")
        except (ValueError, AttributeError) as e:
            logger.warning(f"Error formatting date {timestamp}: {str(e)}")
            # Return the original timestamp if parsing fails
            return timestamp
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """
        Convert seconds to "Xm Ys" format
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration string (e.g., "5m 30s")
        """
        try:
            # Handle negative values by treating them as 0
            if seconds < 0:
                seconds = 0
            
            # Convert to integer seconds
            total_seconds = int(seconds)
            
            # Calculate minutes and remaining seconds
            minutes = total_seconds // 60
            remaining_seconds = total_seconds % 60
            
            # Format the string
            if minutes > 0:
                return f"{minutes}m {remaining_seconds}s"
            else:
                return f"{remaining_seconds}s"
        except (ValueError, TypeError) as e:
            logger.warning(f"Error formatting duration {seconds}: {str(e)}")
            return "0s"
    
    @staticmethod
    def format_exercise_name(exercise_type: str) -> str:
        """
        Convert exercise type to display name
        
        Args:
            exercise_type: Exercise type identifier (e.g., "bicep_curl", "squat")
            
        Returns:
            Human-readable exercise name (e.g., "Bicep Curl", "Squat")
        """
        # Mapping of exercise types to display names
        exercise_name_map = {
            'bicep_curl': 'Bicep Curl',
            'squat': 'Squat',
            'push_up': 'Push Up',
            'pushup': 'Push Up',
            'pull_up': 'Pull Up',
            'pullup': 'Pull Up',
            'plank': 'Plank',
            'lunge': 'Lunge',
            'jumping_jack': 'Jumping Jack',
            'burpee': 'Burpee',
            'sit_up': 'Sit Up',
            'situp': 'Sit Up',
        }
        
        # Convert to lowercase for lookup
        exercise_lower = exercise_type.lower()
        
        # Return mapped name if available, otherwise format the raw type
        if exercise_lower in exercise_name_map:
            return exercise_name_map[exercise_lower]
        else:
            # Convert underscores to spaces and capitalize each word
            return exercise_type.replace('_', ' ').title()
    
    @staticmethod
    def get_exercise_icon(exercise_type: str) -> str:
        """
        Get emoji icon for exercise type
        
        Args:
            exercise_type: Exercise type identifier (e.g., "bicep_curl", "squat")
            
        Returns:
            Emoji icon string for the exercise type
        """
        # Mapping of exercise types to emoji icons
        exercise_icon_map = {
            'bicep_curl': '💪',
            'squat': '🦵',
            'push_up': '🤸',
            'pushup': '🤸',
            'pull_up': '🏋️',
            'pullup': '🏋️',
            'plank': '🧘',
            'lunge': '🏃',
            'jumping_jack': '🤾',
            'burpee': '🔥',
            'sit_up': '🧘',
            'situp': '🧘',
        }
        
        # Convert to lowercase for lookup
        exercise_lower = exercise_type.lower()
        
        # Return mapped icon if available, otherwise return a default icon
        return exercise_icon_map.get(exercise_lower, '🏋️')



class WorkoutHistoryUI:
    """Renders the workout history interface using Streamlit components"""
    
    def __init__(self):
        """Initialize the WorkoutHistoryUI"""
        self.loader = WorkoutHistoryLoader()
        self.filter = WorkoutHistoryFilter()
        self.aggregator = WorkoutHistoryAggregator()
        self.formatter = WorkoutHistoryFormatter()
    
    def render_navigation(self):
        """
        Render navigation controls for returning to main menu
        """
        import streamlit as st
        
        # Header with consistent styling
        st.markdown(
            '<h2 style="font-size: clamp(1.5rem, 3vw, 2.5rem); font-weight: 700; '
            'background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
            '-webkit-background-clip: text; -webkit-text-fill-color: transparent; '
            'margin-bottom: 1.5rem;">📊 Workout History</h2>',
            unsafe_allow_html=True
        )
        
        # Back to menu button
        if st.button("🔙 Back to Main Menu", use_container_width=True):
            # Clear the history view flag
            if 'view_history' in st.session_state:
                del st.session_state.view_history
            st.rerun()
    
    def render_filter_controls(self, exercise_types: List[str]) -> str:
        """
        Render exercise type filter dropdown
        
        Args:
            exercise_types: List of unique exercise types available
            
        Returns:
            Selected exercise type (or "All Exercises")
        """
        import streamlit as st
        
        # Create filter options with "All Exercises" as default
        filter_options = ["All Exercises"] + [
            self.formatter.format_exercise_name(ex) for ex in exercise_types
        ]
        
        # Render selectbox with styled label
        st.markdown(
            '<p style="font-weight: 600; margin-bottom: 0.5rem; color: #667eea;">🔍 Filter by Exercise Type</p>',
            unsafe_allow_html=True
        )
        selected = st.selectbox(
            "Filter by Exercise Type",
            options=filter_options,
            index=0,
            key="exercise_filter",
            label_visibility="collapsed"
        )
        
        # Convert back to internal format if not "All Exercises"
        if selected == "All Exercises":
            return "all"
        else:
            # Find the original exercise type
            for ex in exercise_types:
                if self.formatter.format_exercise_name(ex) == selected:
                    return ex
            return "all"
    
    def render_summary_stats(self, sessions: List[Dict]):
        """
        Render summary statistics cards
        
        Args:
            sessions: List of workout sessions to calculate stats from
        """
        import streamlit as st
        
        # Calculate statistics
        total_workouts = self.aggregator.calculate_total_workouts(sessions)
        total_reps = self.aggregator.calculate_total_reps(sessions)
        total_calories = self.aggregator.calculate_total_calories(sessions)
        total_duration = self.aggregator.calculate_total_duration(sessions)
        
        # Format duration
        formatted_duration = self.formatter.format_duration(total_duration)
        
        # Render metrics in columns with enhanced styling
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: clamp(1.5rem, 3vw, 2rem); font-weight: 700;">{total_workouts}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: clamp(0.8rem, 2vw, 1rem); opacity: 0.9;">Total Workouts</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: clamp(1.5rem, 3vw, 2rem); font-weight: 700;">{total_reps}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: clamp(0.8rem, 2vw, 1rem); opacity: 0.9;">Total Reps</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: clamp(1.5rem, 3vw, 2rem); font-weight: 700;">{total_calories:.1f}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: clamp(0.8rem, 2vw, 1rem); opacity: 0.9;">Total Calories</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col4:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3 style="margin: 0; font-size: clamp(1.5rem, 3vw, 2rem); font-weight: 700;">{formatted_duration}</h3>
                    <p style="margin: 0.5rem 0 0 0; font-size: clamp(0.8rem, 2vw, 1rem); opacity: 0.9;">Total Time</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def render_session_list(self, sessions: List[Dict]):
        """
        Render list of workout sessions
        
        Args:
            sessions: List of workout sessions to display
        """
        import streamlit as st
        
        # Sort sessions in reverse chronological order
        sorted_sessions = self.filter.sort_by_date(sessions, reverse=True)
        
        # Render each session with enhanced styling
        for session in sorted_sessions:
            # Get session details
            exercise = session.get('exercise', 'Unknown')
            exercise_icon = self.formatter.get_exercise_icon(exercise)
            exercise_name = self.formatter.format_exercise_name(exercise)
            
            start_time = session.get('start_time', '')
            formatted_date = self.formatter.format_date(start_time)
            
            reps = session.get('reps', 0)
            duration = session.get('duration', 0)
            formatted_duration = self.formatter.format_duration(duration)
            calories = session.get('calories', 0)
            
            # Create a card for each session using info-card style with responsive design
            st.markdown(
                f"""
                <div class="info-card" style="transition: all 0.3s ease;">
                    <h4 style="margin: 0 0 0.75rem 0; font-size: clamp(1rem, 2.5vw, 1.25rem); 
                               color: #667eea; font-weight: 600;">
                        {exercise_icon} {exercise_name}
                    </h4>
                    <p style="margin: 0.25rem 0; font-size: clamp(0.85rem, 2vw, 0.95rem); color: #555;">
                        <strong>📅 Date:</strong> {formatted_date}
                    </p>
                    <p style="margin: 0.25rem 0; font-size: clamp(0.85rem, 2vw, 0.95rem); color: #555;">
                        <strong>🔢 Reps:</strong> {reps} | 
                        <strong>⏱️ Duration:</strong> {formatted_duration} | 
                        <strong>🔥 Calories:</strong> {calories:.1f}
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def render_empty_state(self, filter_applied: bool = False):
        """
        Render message when no sessions are available
        
        Args:
            filter_applied: Whether a filter is currently applied
        """
        import streamlit as st
        
        if filter_applied:
            st.markdown(
                """
                <div class="info-card" style="text-align: center; padding: 2rem; background: #fff3cd; border-left-color: #ffc107;">
                    <h3 style="margin: 0 0 0.5rem 0; color: #856404;">🔍 No Results Found</h3>
                    <p style="margin: 0; color: #856404; font-size: clamp(0.9rem, 2vw, 1rem);">
                        No workout sessions found for the selected exercise type.<br>
                        Try selecting a different filter.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                """
                <div class="info-card" style="text-align: center; padding: 2rem; background: #d1ecf1; border-left-color: #17a2b8;">
                    <h3 style="margin: 0 0 0.5rem 0; color: #0c5460;">📭 No Workout History Yet</h3>
                    <p style="margin: 0; color: #0c5460; font-size: clamp(0.9rem, 2vw, 1rem);">
                        Complete your first workout to see it here!<br>
                        Start tracking your fitness journey today.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
    
    def render_history_section(self):
        """
        Main entry point for rendering the workout history UI
        This method orchestrates all other rendering methods
        """
        import streamlit as st
        
        # Inject responsive CSS for workout history
        st.markdown("""
        <style>
            /* Responsive adjustments for workout history */
            @media (max-width: 768px) {
                .metric-card {
                    padding: 0.75rem;
                    margin-bottom: 0.75rem;
                }
                .info-card {
                    padding: 0.75rem;
                    margin: 0.75rem 0;
                }
                .stColumns {
                    flex-direction: column;
                }
            }
            
            /* Hover effects for session cards */
            .info-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                border-left-color: #764ba2;
            }
            
            /* Smooth transitions */
            .info-card, .metric-card {
                transition: all 0.3s ease;
            }
            
            /* Ensure proper spacing on small screens */
            @media (max-width: 480px) {
                .main-header {
                    font-size: 1.5rem !important;
                }
                .metric-card h3 {
                    font-size: 1.5rem !important;
                }
                .metric-card p {
                    font-size: 0.8rem !important;
                }
            }
        </style>
        """, unsafe_allow_html=True)
        
        # Render navigation
        self.render_navigation()
        
        # Load all sessions
        sessions = self.loader.load_all_sessions()
        
        # Check if there are any sessions
        if not sessions:
            self.render_empty_state(filter_applied=False)
            return
        
        # Get unique exercise types
        exercise_types = self.filter.get_unique_exercise_types(sessions)
        
        # Render filter controls with spacing
        st.markdown('<div style="margin: 1.5rem 0;">', unsafe_allow_html=True)
        selected_exercise = self.render_filter_controls(exercise_types)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Filter sessions based on selection
        if selected_exercise != "all":
            filtered_sessions = self.filter.filter_by_exercise(sessions, selected_exercise)
        else:
            filtered_sessions = sessions
        
        # Check if filtered results are empty
        if not filtered_sessions:
            self.render_empty_state(filter_applied=True)
            return
        
        # Render summary statistics with section header
        st.markdown(
            '<h3 style="font-size: clamp(1.2rem, 2.5vw, 1.5rem); font-weight: 600; '
            'color: #667eea; margin: 2rem 0 1rem 0;">📈 Summary Statistics</h3>',
            unsafe_allow_html=True
        )
        self.render_summary_stats(filtered_sessions)
        
        # Render session list with section header
        st.markdown(
            '<h3 style="font-size: clamp(1.2rem, 2.5vw, 1.5rem); font-weight: 600; '
            'color: #667eea; margin: 2rem 0 1rem 0;">📋 Workout Sessions</h3>',
            unsafe_allow_html=True
        )
        self.render_session_list(filtered_sessions)
