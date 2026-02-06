"""
API Client for communicating with FastAPI backend
Enhanced with error handling, retry logic, timeout configuration, and logging
"""
import base64
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

import cv2
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI Backend Configuration
API_BASE_URL = "http://localhost:8000"

# Exercise type mapping (Streamlit display name -> API enum)
EXERCISE_TYPE_MAP = {
    "Bicep Curls": "bicep_curl",
    "Squats": "squat",
    "Push-ups": "push_up"
}

# Timeout configuration (in seconds)
DEFAULT_TIMEOUT = 5
HEALTH_CHECK_TIMEOUT = 2
POSE_DETECT_TIMEOUT = 10

# Retry configuration
MAX_RETRIES = 3
BACKOFF_FACTOR = 0.5  # Wait 0.5s, 1s, 2s between retries

# Health check caching configuration
HEALTH_CHECK_CACHE_DURATION = 30  # Cache health check for 30 seconds


class APIClient:
    """Client for communicating with FastAPI backend with enhanced error handling"""
    
    # Class-level cache for health check
    _health_check_cache: Optional[Dict[str, Any]] = None
    _health_check_timestamp: Optional[datetime] = None
    
    def __init__(self):
        """Initialize API client with retry configuration"""
        self.session = self._create_session_with_retries()
    
    @staticmethod
    def _create_session_with_retries() -> requests.Session:
        """Create a requests session with retry logic"""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=MAX_RETRIES,
            backoff_factor=BACKOFF_FACTOR,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    @staticmethod
    def encode_frame(frame: np.ndarray) -> str:
        """Encode frame to base64 string"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            encoded = base64.b64encode(buffer).decode('utf-8')
            logger.debug("Frame encoded successfully")
            return encoded
        except Exception as e:
            logger.error(f"Failed to encode frame: {str(e)}")
            raise
    
    @staticmethod
    def decode_frame(base64_str: str) -> np.ndarray:
        """Decode base64 string to frame"""
        try:
            img_bytes = base64.b64decode(base64_str)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            logger.debug("Frame decoded successfully")
            return frame
        except Exception as e:
            logger.error(f"Failed to decode frame: {str(e)}")
            raise
    
    @classmethod
    def _is_health_check_cached(cls) -> bool:
        """Check if health check result is still valid in cache"""
        if cls._health_check_cache is None or cls._health_check_timestamp is None:
            return False
        
        cache_age = datetime.now() - cls._health_check_timestamp
        return cache_age < timedelta(seconds=HEALTH_CHECK_CACHE_DURATION)
    
    @classmethod
    def _cache_health_check(cls, is_healthy: bool) -> None:
        """Cache health check result"""
        cls._health_check_cache = {"healthy": is_healthy}
        cls._health_check_timestamp = datetime.now()
        logger.debug(f"Health check cached: {is_healthy}")
    
    @classmethod
    def check_health(cls, use_cache: bool = True) -> bool:
        """
        Check if backend is available
        
        Args:
            use_cache: If True, use cached result if available
            
        Returns:
            True if backend is healthy, False otherwise
        """
        # Return cached result if available and valid
        if use_cache and cls._is_health_check_cached():
            logger.debug("Using cached health check result")
            return cls._health_check_cache["healthy"]
        
        try:
            logger.info(f"Checking backend health at {API_BASE_URL}/health")
            response = requests.get(
                f"{API_BASE_URL}/health",
                timeout=HEALTH_CHECK_TIMEOUT
            )
            is_healthy = response.status_code == 200
            
            # Cache the result
            cls._cache_health_check(is_healthy)
            
            if is_healthy:
                logger.info("Backend is healthy")
            else:
                logger.warning(f"Backend returned status code: {response.status_code}")
            
            return is_healthy
            
        except requests.exceptions.Timeout:
            logger.error("Health check timed out")
            cls._cache_health_check(False)
            return False
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to backend")
            cls._cache_health_check(False)
            return False
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            cls._cache_health_check(False)
            return False
    
    def login(self, username, password) -> Dict[str, Any]:
        """Authenticate user"""
        try:
            response = self.session.post(f"{API_BASE_URL}/api/v1/auth/login", json={
                "username": username,
                "password": password
            }, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Login failed: {str(e)}")
            return {"success": False, "message": "Connection error or invalid credentials"}

    def signup(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Register new user"""
        try:
            response = self.session.post(f"{API_BASE_URL}/api/v1/auth/signup", json=user_data, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Signup failed: {str(e)}")
            return {"success": False, "message": "Connection error or user already exists"}
            
    def get_exercises(self) -> List[Dict[str, Any]]:
        """
        Get a list of available exercises from the backend.
        
        Returns:
            A list of dictionaries, each representing an exercise, or an empty list on failure.
        """
        try:
            logger.info(f"Fetching available exercises from {API_BASE_URL}/api/v1/exercises")
            response = self.session.get(
                f"{API_BASE_URL}/api/v1/exercises",
                timeout=DEFAULT_TIMEOUT
            )
            response.raise_for_status()
            exercises = response.json()
            logger.debug(f"Successfully fetched {len(exercises)} exercises.")
            return exercises
        except requests.exceptions.Timeout:
            logger.error("Get exercises request timed out")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error fetching exercises: {e.response.status_code} - {e.response.text}")
            return []
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed to fetch exercises: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error fetching exercises: {str(e)}")
            return []
    
    def start_session(self, exercise_type: str) -> Optional[str]:
        """
        Start a new workout session
        
        Args:
            exercise_type: Type of exercise (display name)
            
        Returns:
            Session ID if successful, None otherwise
        """
        try:
            api_exercise_type = EXERCISE_TYPE_MAP.get(exercise_type, exercise_type)
            logger.info(f"Starting session for exercise: {api_exercise_type}")
            
            response = self.session.post(
                f"{API_BASE_URL}/api/v1/sessions/start",
                json={"exercise_type": api_exercise_type},
                timeout=DEFAULT_TIMEOUT
            )
            
            response.raise_for_status()
            session_id = response.json()["session_id"]
            
            logger.info(f"Session started successfully: {session_id}")
            return session_id
            
        except requests.exceptions.Timeout:
            logger.error("Start session request timed out")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error starting session: {e.response.status_code} - {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed to start session: {str(e)}")
            return None
        except KeyError:
            logger.error("Invalid response format: missing session_id")
            return None
        except Exception as e:
            logger.error(f"Unexpected error starting session: {str(e)}")
            return None
    
    def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End a workout session
        
        Args:
            session_id: ID of the session to end
            
        Returns:
            Session summary if successful, None otherwise
        """
        try:
            logger.info(f"Ending session: {session_id}")
            
            response = self.session.post(
                f"{API_BASE_URL}/api/v1/sessions/{session_id}/end",
                timeout=DEFAULT_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.info(f"Session ended successfully: {session_id}")
            logger.debug(f"Session summary: {result}")
            return result
            
        except requests.exceptions.Timeout:
            logger.error("End session request timed out")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error ending session: {e.response.status_code} - {e.response.text}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed to end session: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error ending session: {str(e)}")
            return None
    
    def detect_pose(self, frame: np.ndarray, draw_landmarks: bool = True) -> Optional[Dict[str, Any]]:
        """
        Detect pose in frame
        
        Args:
            frame: Image frame as numpy array
            draw_landmarks: Whether to draw pose landmarks on the frame
            
        Returns:
            Pose detection result if successful, None otherwise
        """
        try:
            # Encode frame
            encoded_frame = self.encode_frame(frame)
            
            logger.debug("Sending pose detection request")
            
            # Send to API
            response = self.session.post(
                f"{API_BASE_URL}/api/v1/pose/detect",
                data={
                    "image": encoded_frame,
                    "draw_landmarks": str(draw_landmarks).lower()
                },
                timeout=POSE_DETECT_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            if result.get('detected'):
                logger.debug("Pose detected successfully")
            else:
                logger.debug("No pose detected in frame")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("Pose detection request timed out")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in pose detection: {e.response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for pose detection: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in pose detection: {str(e)}")
            return None
    
    def analyze_exercise(self, session_id: str, exercise_type: str, key_points: Dict) -> Optional[Dict[str, Any]]:
        """
        Analyze exercise form
        
        Args:
            session_id: Current session ID
            exercise_type: Type of exercise (display name)
            key_points: Detected pose key points
            
        Returns:
            Analysis result if successful, None otherwise
        """
        try:
            api_exercise_type = EXERCISE_TYPE_MAP.get(exercise_type, exercise_type)
            logger.debug(f"Analyzing exercise: {api_exercise_type}")
            
            response = self.session.post(
                f"{API_BASE_URL}/api/v1/analyze",
                json={
                    "session_id": session_id,
                    "exercise_type": api_exercise_type,
                    "key_points": key_points
                },
                timeout=DEFAULT_TIMEOUT
            )
            
            response.raise_for_status()
            result = response.json()
            
            logger.debug(f"Analysis complete - Reps: {result.get('rep_count', 0)}, "
                        f"Quality: {result.get('quality_score', 0)}")
            
            return result
            
        except requests.exceptions.Timeout:
            logger.error("Exercise analysis request timed out")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error in exercise analysis: {e.response.status_code}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for exercise analysis: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in exercise analysis: {str(e)}")
            return None


# Create a singleton instance for convenience
_api_client_instance = None


def get_api_client() -> APIClient:
    """Get or create the singleton API client instance"""
    global _api_client_instance
    if _api_client_instance is None:
        _api_client_instance = APIClient()
    return _api_client_instance
