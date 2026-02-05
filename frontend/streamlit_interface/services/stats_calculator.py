"""
Statistics Calculator Service
Calculates advanced statistics and trends from workout session data
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StatsCalculator:
    """Calculates advanced statistics and trends from workout session data"""
    
    @staticmethod
    def calculate_workout_frequency(sessions: List[Dict], interval: str = "auto") -> Dict[str, int]:
        """
        Calculate workouts per time interval (daily/weekly/monthly)
        
        Args:
            sessions: List of workout session dictionaries
            interval: Time interval for grouping ("daily", "weekly", "monthly", or "auto")
                     "auto" will select the appropriate interval based on data density
            
        Returns:
            Dictionary mapping time period labels to workout counts
        """
        if not sessions:
            return {}
        
        # Parse timestamps and sort sessions by date
        dated_sessions = []
        for session in sessions:
            try:
                timestamp = session.get('start_time', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                dated_sessions.append(dt)
            except (ValueError, AttributeError) as e:
                logger.warning(f"Error parsing timestamp {timestamp}: {str(e)}")
                continue
        
        if not dated_sessions:
            return {}
        
        # Sort dates
        dated_sessions.sort()
        
        # Determine interval if set to auto
        if interval == "auto":
            date_range = (dated_sessions[-1] - dated_sessions[0]).days
            if date_range < 14:
                interval = "daily"
            elif date_range <= 90:
                interval = "weekly"
            else:
                interval = "monthly"
        
        # Group sessions by interval
        frequency = defaultdict(int)
        
        for dt in dated_sessions:
            if interval == "daily":
                key = dt.strftime("%Y-%m-%d")
            elif interval == "weekly":
                # Get the Monday of the week
                monday = dt - timedelta(days=dt.weekday())
                key = monday.strftime("%Y-%m-%d")
            elif interval == "monthly":
                key = dt.strftime("%Y-%m")
            else:
                # Default to daily if invalid interval
                key = dt.strftime("%Y-%m-%d")
            
            frequency[key] += 1
        
        # Convert defaultdict to regular dict and sort by key
        return dict(sorted(frequency.items()))
    
    @staticmethod
    def calculate_exercise_distribution(sessions: List[Dict]) -> Dict[str, int]:
        """
        Calculate count of each exercise type
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Dictionary mapping exercise types to counts
        """
        distribution = defaultdict(int)
        
        for session in sessions:
            exercise_type = session.get('exercise', 'unknown')
            distribution[exercise_type] += 1
        
        # Convert defaultdict to regular dict and sort by count (descending)
        return dict(sorted(distribution.items(), key=lambda x: x[1], reverse=True))
    
    @staticmethod
    def calculate_performance_trends(sessions: List[Dict]) -> List[Dict]:
        """
        Calculate average quality scores over time
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            List of dictionaries with date and average quality score
            Format: [{"date": "2024-01-15", "quality_score": 85.5, "count": 3}, ...]
        """
        if not sessions:
            return []
        
        # Group sessions by date and collect quality scores
        daily_scores = defaultdict(list)
        
        for session in sessions:
            try:
                # Parse timestamp
                timestamp = session.get('start_time', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                date_key = dt.strftime("%Y-%m-%d")
                
                # Get quality score if available
                quality_score = session.get('quality_score')
                if quality_score is not None and isinstance(quality_score, (int, float)):
                    daily_scores[date_key].append(float(quality_score))
            except (ValueError, AttributeError) as e:
                logger.warning(f"Error processing session for trends: {str(e)}")
                continue
        
        # Calculate average quality score for each date
        trends = []
        for date_key in sorted(daily_scores.keys()):
            scores = daily_scores[date_key]
            avg_score = sum(scores) / len(scores) if scores else 0
            trends.append({
                "date": date_key,
                "quality_score": round(avg_score, 2),
                "count": len(scores)
            })
        
        return trends
    
    @staticmethod
    def identify_personal_bests(sessions: List[Dict]) -> Dict[str, Any]:
        """
        Identify highest reps, longest duration, best quality scores
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Dictionary containing personal best records:
            {
                "max_reps": {"value": int, "session_id": str, "exercise": str, "date": str},
                "longest_duration": {"value": float, "session_id": str, "exercise": str, "date": str},
                "best_quality": {"value": float, "session_id": str, "exercise": str, "date": str},
                "most_calories": {"value": float, "session_id": str, "exercise": str, "date": str}
            }
        """
        if not sessions:
            return {
                "max_reps": None,
                "longest_duration": None,
                "best_quality": None,
                "most_calories": None
            }
        
        # Initialize tracking variables
        max_reps_session = None
        max_reps = -1
        
        longest_duration_session = None
        longest_duration = -1
        
        best_quality_session = None
        best_quality = -1
        
        most_calories_session = None
        most_calories = -1
        
        # Iterate through sessions to find bests
        for session in sessions:
            # Check reps
            reps = session.get('reps', 0)
            if isinstance(reps, (int, float)) and reps > max_reps:
                max_reps = reps
                max_reps_session = session
            
            # Check duration
            duration = session.get('duration', 0)
            if isinstance(duration, (int, float)) and duration > longest_duration:
                longest_duration = duration
                longest_duration_session = session
            
            # Check quality score
            quality_score = session.get('quality_score')
            if quality_score is not None and isinstance(quality_score, (int, float)) and quality_score > best_quality:
                best_quality = quality_score
                best_quality_session = session
            
            # Check calories
            calories = session.get('calories', 0)
            if isinstance(calories, (int, float)) and calories > most_calories:
                most_calories = calories
                most_calories_session = session
        
        # Helper function to format personal best record
        def format_record(session: Optional[Dict], value: float, metric_name: str) -> Optional[Dict]:
            if session is None or value < 0:
                return None
            
            try:
                timestamp = session.get('start_time', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                date_str = dt.strftime("%Y-%m-%d")
            except (ValueError, AttributeError):
                date_str = "Unknown"
            
            return {
                "value": value,
                "session_id": session.get('session_id', 'unknown'),
                "exercise": session.get('exercise', 'unknown'),
                "date": date_str
            }
        
        # Build results
        return {
            "max_reps": format_record(max_reps_session, max_reps, "reps"),
            "longest_duration": format_record(longest_duration_session, longest_duration, "duration"),
            "best_quality": format_record(best_quality_session, best_quality, "quality_score"),
            "most_calories": format_record(most_calories_session, most_calories, "calories")
        }

    @staticmethod
    def get_recent_workouts(sessions: List[Dict], limit: int = 5) -> List[Dict]:
        """
        Get the most recent N workouts sorted by date descending.
        
        Args:
            sessions: List of workout session dictionaries
            limit: Number of sessions to return
            
        Returns:
            List of session dictionaries sorted by time (newest first)
        """
        if not sessions:
            return []
            
        # Sort sessions by start_time descending
        sorted_sessions = sorted(
            sessions, 
            key=lambda x: x.get('start_time', ''), 
            reverse=True
        )
        
        return sorted_sessions[:limit]

    @staticmethod
    def calculate_current_streak(sessions: List[Dict]) -> int:
        """
        Calculate the current streak of consecutive days with workouts.
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Number of consecutive days (including today)
        """
        if not sessions:
            return 0
            
        # Get set of unique dates with workouts
        workout_dates = set()
        for session in sessions:
            try:
                timestamp = session.get('start_time', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                workout_dates.add(dt.date())
            except (ValueError, AttributeError):
                continue
        
        if not workout_dates:
            return 0
            
        # Check streak counting backwards from today
        streak = 0
        today = datetime.now().date()
        current_check = today
        
        # If no workout today, check if there was one yesterday to keep streak alive
        if today not in workout_dates:
            if (today - timedelta(days=1)) in workout_dates:
                 current_check = today - timedelta(days=1)
            else:
                return 0 # No workout today or yesterday = 0 streak
        
        while current_check in workout_dates:
            streak += 1
            current_check -= timedelta(days=1)
            
        return streak

    @staticmethod
    def calculate_weekly_stats(sessions: List[Dict]) -> Dict[str, Any]:
        """
        Calculate statistics for the current week (starting Monday).
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            Dictionary containing weekly stats
        """
        now = datetime.now()
        start_of_week = now - timedelta(days=now.weekday())
        start_of_week = start_of_week.replace(hour=0, minute=0, second=0, microsecond=0)
        
        weekly_sessions = []
        for session in sessions:
            try:
                timestamp = session.get('start_time', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                # Remove timezone info for comparison consistency
                dt = dt.replace(tzinfo=None) 
                if dt >= start_of_week:
                    weekly_sessions.append(session)
            except (ValueError, AttributeError):
                continue
                
        return {
            "count": len(weekly_sessions),
            "calories": sum(s.get('calories', 0) for s in weekly_sessions),
            "duration": sum(s.get('duration', 0) for s in weekly_sessions),
            "reps": sum(s.get('reps', 0) for s in weekly_sessions)
        }

    @staticmethod
    def calculate_calorie_trends(sessions: List[Dict]) -> List[Dict]:
        """
        Calculate average calories burned over time.
        
        Args:
            sessions: List of workout session dictionaries
            
        Returns:
            List of dictionaries with date and total calories
        """
        if not sessions:
            return []
        
        # Group sessions by date and sum calories
        daily_calories = defaultdict(float)
        
        for session in sessions:
            try:
                timestamp = session.get('start_time', '')
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                date_key = dt.strftime("%Y-%m-%d")
                
                calories = session.get('calories', 0)
                if isinstance(calories, (int, float)):
                    daily_calories[date_key] += calories
            except (ValueError, AttributeError):
                continue
        
        # Format for chart
        trends = []
        for date_key in sorted(daily_calories.keys()):
            trends.append({
                "date": date_key,
                "calories": round(daily_calories[date_key], 1)
            })
        
        return trends
