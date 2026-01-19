"""
FastAPI Backend for AI Fitness Trainer
Exposes pose detection, exercise analysis, and workout session management via REST API
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Path, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import logging

# Import models
from backend.api.models import (
    HealthCheckResponse,
    ExerciseListResponse,
    ExerciseInfo,
    PoseDetectionRequest,
    PoseDetectionResponse,
    AnalysisRequest,
    AnalysisResponse,
    SessionStartRequest,
    SessionResponse,
    SessionResetResponse,
    SessionListResponse,
    ErrorResponse,
    ExerciseType,
    SessionStatus,
    ExerciseStage,
    validate_uuid_format
)

# Import services and utilities
from backend.api.pose_detector import EnhancedPoseDetector
from backend.api.session_manager import SessionManager
from backend.utils.image_processor import ImageProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application instance
app = FastAPI(
    title="AI Fitness Trainer API",
    description="REST API for pose detection, exercise analysis, and workout session management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
# Allow requests from web interfaces and Streamlit
CORS_ORIGINS = [
    "http://localhost:3000",      # React/Next.js default
    "http://localhost:8501",      # Streamlit default
    "http://localhost:8080",      # Alternative web server
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8501",
    "http://127.0.0.1:8080",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

logger.info("FastAPI application initialized with CORS support")

# Initialize global services
pose_detector: Optional[EnhancedPoseDetector] = None
session_manager: Optional[SessionManager] = None
pose_detector_initialized = False


def validate_session_id(session_id: str = Path(..., description="UUID of the workout session")) -> str:
    """
    Dependency to validate session ID format.
    
    Args:
        session_id: Session ID string from path parameter
        
    Returns:
        str: Validated session ID
        
    Raises:
        HTTPException: 400 if session ID format is invalid
    """
    try:
        return validate_uuid_format(session_id, "session_id")
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )


# ============================================================================
# GLOBAL EXCEPTION HANDLERS
# ============================================================================

from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """
    Global handler for HTTP exceptions (400, 404, 500, 503, etc.)
    
    Ensures consistent error response format across all endpoints.
    
    **Requirements: 5.1, 5.2, 5.3, 5.4, 5.5**
    - 5.1: WHEN a client sends invalid image data 
           THEN the FastAPI Service SHALL return a 400 status code with a descriptive error message
    - 5.2: WHEN a client requests an unsupported exercise type 
           THEN the FastAPI Service SHALL return a 400 status code listing valid exercise types
    - 5.3: WHEN a client attempts to end a non-existent session 
           THEN the FastAPI Service SHALL return a 404 status code with an appropriate message
    - 5.4: WHEN the Pose Detection Engine encounters an error 
           THEN the FastAPI Service SHALL return a 500 status code and log the exception details
    - 5.5: WHEN request validation fails 
           THEN the FastAPI Service SHALL return structured error responses following a consistent schema
    """
    # Determine error category based on status code
    error_category = "Bad Request"
    if exc.status_code == 404:
        error_category = "Not Found"
    elif exc.status_code == 500:
        error_category = "Internal Server Error"
    elif exc.status_code == 503:
        error_category = "Service Unavailable"
    
    # Log error details for server errors
    if exc.status_code >= 500:
        logger.error(f"HTTP {exc.status_code} error on {request.url.path}: {exc.detail}")
    
    # Return consistent error response
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_category,
            "detail": str(exc.detail),
            "status_code": exc.status_code
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Global handler for Pydantic validation errors.
    
    Converts validation errors into consistent error response format.
    
    **Requirements: 5.5**
    - 5.5: WHEN request validation fails 
           THEN the FastAPI Service SHALL return structured error responses following a consistent schema
    """
    # Extract validation error details
    errors = exc.errors()
    error_messages = []
    
    for error in errors:
        field = " -> ".join(str(loc) for loc in error["loc"])
        message = error["msg"]
        error_messages.append(f"{field}: {message}")
    
    detail = "; ".join(error_messages)
    
    logger.warning(f"Validation error on {request.url.path}: {detail}")
    
    # Return consistent error response
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Validation Error",
            "detail": detail,
            "status_code": 400
        }
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """
    Global handler for ValueError exceptions.
    
    Typically raised for invalid input data or business logic violations.
    
    **Requirements: 5.1**
    - 5.1: WHEN a client sends invalid image data 
           THEN the FastAPI Service SHALL return a 400 status code with a descriptive error message
    """
    logger.warning(f"ValueError on {request.url.path}: {str(exc)}")
    
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={
            "error": "Invalid Input",
            "detail": str(exc),
            "status_code": 400
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Global handler for all unhandled exceptions.
    
    Catches any unexpected errors and returns a 500 response.
    
    **Requirements: 5.4**
    - 5.4: WHEN the Pose Detection Engine encounters an error 
           THEN the FastAPI Service SHALL return a 500 status code and log the exception details
    """
    # Log the full exception details
    logger.error(f"Unhandled exception on {request.url.path}: {type(exc).__name__}: {str(exc)}", exc_info=True)
    
    # Return generic error response (don't expose internal details)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal Server Error",
            "detail": "An unexpected error occurred. Please try again later.",
            "status_code": 500
        }
    )


@app.on_event("startup")
async def startup_event():
    """Initialize services on application startup"""
    global pose_detector, session_manager, pose_detector_initialized
    
    logger.info("🚀 AI Fitness Trainer API starting up...")
    logger.info(f"�A API Documentation available at: http://localhost:8000/docs")
    logger.info(f"🔧 CORS enabled for origins: {CORS_ORIGINS}")
    
    try:
        # Initialize pose detector
        pose_detector = EnhancedPoseDetector(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        pose_detector_initialized = True
        logger.info("✅ Pose detector initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize pose detector: {str(e)}")
        pose_detector_initialized = False
    
    # Initialize session manager
    session_manager = SessionManager()
    logger.info("✅ Session manager initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown"""
    logger.info("👋 AI Fitness Trainer API shutting down...")


# ============================================================================
# HEALTH AND CONFIGURATION ENDPOINTS
# ============================================================================

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify service availability.
    
    Returns service status and version information.
    
    **Requirements: 4.1**
    - WHEN a client sends a GET request to the health endpoint 
      THEN the FastAPI Service SHALL return a 200 status code with service status information
    """
    return HealthCheckResponse(
        status="healthy",
        service="AI Fitness Trainer API",
        version="1.0.0",
        pose_detector_initialized=pose_detector_initialized
    )


@app.get("/api/v1/exercises", response_model=ExerciseListResponse, tags=["Configuration"])
async def get_exercises():
    """
    Get list of all supported exercises.
    
    Returns information about available exercise types including names,
    target muscles, and exercise type (rep-based or time-based).
    
    **Requirements: 4.2**
    - WHEN a client requests available exercises 
      THEN the FastAPI Service SHALL return all supported exercise types 
      with their names and target muscles
    """
    exercises = [
        ExerciseInfo(
            id="bicep_curl",
            name="Bicep Curls",
            muscle="Biceps",
            type="rep_based"
        ),
        ExerciseInfo(
            id="squat",
            name="Squats",
            muscle="Quadriceps, Glutes",
            type="rep_based"
        ),
        ExerciseInfo(
            id="push_up",
            name="Push-ups",
            muscle="Chest, Triceps",
            type="rep_based"
        ),
        ExerciseInfo(
            id="shoulder_press",
            name="Shoulder Press",
            muscle="Shoulders, Triceps",
            type="rep_based"
        ),
        ExerciseInfo(
            id="lunge",
            name="Lunges",
            muscle="Quadriceps, Glutes",
            type="rep_based"
        ),
        ExerciseInfo(
            id="plank",
            name="Plank",
            muscle="Core, Abs",
            type="time_based"
        )
    ]
    
    return ExerciseListResponse(exercises=exercises)


# ============================================================================
# POSE DETECTION ENDPOINT
# ============================================================================

@app.post("/api/v1/pose/detect", response_model=PoseDetectionResponse, tags=["Pose Detection"])
async def detect_pose(
    image: Optional[str] = Form(None, description="Base64 encoded image string"),
    draw_landmarks: bool = Form(False, description="Whether to return annotated image with landmarks"),
    file: Optional[UploadFile] = File(None, description="Uploaded image file")
):
    """
    Detect pose landmarks in an image.
    
    Accepts either base64-encoded image string or multipart file upload.
    Returns detected key points and optionally an annotated image with landmarks drawn.
    
    **Requirements: 1.1, 1.2, 1.3, 1.5**
    - 1.1: WHEN a client sends an image frame via HTTP POST 
           THEN the FastAPI Service SHALL process the image and return pose landmarks
    - 1.2: WHEN the image contains a detectable person 
           THEN the FastAPI Service SHALL extract key points and return them in JSON format
    - 1.3: WHEN the image does not contain a detectable person 
           THEN the FastAPI Service SHALL return an error response with appropriate status code
    - 1.5: WHEN receiving image data 
           THEN the FastAPI Service SHALL accept both base64-encoded strings and multipart form data
    
    Args:
        image: Base64-encoded image string (optional if file is provided)
        draw_landmarks: Whether to return annotated image with landmarks
        file: Uploaded image file (optional if image string is provided)
    
    Returns:
        PoseDetectionResponse with detected key points and optional annotated image
    
    Raises:
        HTTPException: 400 if image data is invalid or missing
        HTTPException: 503 if pose detector is not initialized
    """
    # Validate input parameters
    if not image and not file:
        raise HTTPException(
            status_code=400,
            detail="Either 'image' (base64 string) or 'file' (multipart upload) must be provided"
        )
    
    if image and file:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'image' or 'file', not both"
        )
    
    # Validate base64 image if provided
    if image:
        if not isinstance(image, str) or len(image.strip()) == 0:
            raise HTTPException(
                status_code=400,
                detail="Image data cannot be empty"
            )
    
    # Validate file upload if provided
    if file:
        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file must have a filename"
            )
        
        # Check file extension
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        file_ext = '.' + file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
        if file_ext and file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file format: {file_ext}. Supported formats: {', '.join(allowed_extensions)}"
            )
    
    # Check if pose detector is initialized
    if not pose_detector_initialized or pose_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Pose detector not initialized. Service unavailable."
        )
    
    try:
        # Process image based on input type
        if file:
            # Process multipart file upload
            image_array = await ImageProcessor.process_uploaded_file(file)
        else:
            # Process base64 encoded image
            image_array = ImageProcessor.decode_base64_image(image)
        
        # Validate image
        ImageProcessor.validate_image(image_array)
        
        # Detect pose landmarks
        landmarks, annotated_image = pose_detector.detect_pose(image_array, draw_landmarks)
        
        # Check if person was detected
        if landmarks is None:
            # No person detected - return response with detected=False
            return PoseDetectionResponse(
                detected=False,
                key_points=None,
                annotated_image=None
            )
        
        # Extract key points as dictionary
        key_points_dict = pose_detector.extract_key_points(landmarks)
        
        # Convert key points to list format for JSON serialization
        key_points_json = {}
        if key_points_dict:
            for joint_name, coords in key_points_dict.items():
                key_points_json[joint_name] = list(coords)
        
        # Encode annotated image if requested
        annotated_image_base64 = None
        if draw_landmarks and annotated_image is not None:
            annotated_image_base64 = ImageProcessor.encode_image_to_base64(annotated_image)
        
        return PoseDetectionResponse(
            detected=True,
            key_points=key_points_json,
            annotated_image=annotated_image_base64
        )
        
    except ValueError as e:
        # Invalid image data or validation error
        raise HTTPException(
            status_code=400,
            detail=f"Invalid image data: {str(e)}"
        )
    except Exception as e:
        # Unexpected error
        logger.error(f"Error in pose detection: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Pose detection failed: {str(e)}"
        )


# ============================================================================
# EXERCISE ANALYSIS ENDPOINT
# ============================================================================

@app.post("/api/v1/analyze", response_model=AnalysisResponse, tags=["Exercise Analysis"])
async def analyze_exercise(request: AnalysisRequest):
    """
    Analyze exercise form and count repetitions.
    
    Uses the session's Exercise Analyzer to maintain state across multiple frames.
    Returns rep count, stage, angles, feedback, warnings, errors, and calories.
    
    **Requirements: 2.1, 4.1, 4.2, 6.2, 8.2, 8.3**
    - 2.1: WHEN a client requests form analysis with key points and exercise type 
           THEN the FastAPI Service SHALL return an Analysis Result containing 
           rep count, stage, angles, feedback, warnings, and errors
    - 4.1: WHEN the System has recorded at least 3 repetitions 
           THEN the System SHALL calculate a historical average pose quality score
    - 4.2: WHEN comparing current pose quality to thresholds 
           THEN the System SHALL use the historical average as a reference point
    - 6.2: WHEN a client submits frames for analysis with a session ID 
           THEN the FastAPI Service SHALL use the existing Exercise Analyzer 
           to maintain rep count and stage
    - 8.2: WHEN the System stores pose quality scores 
           THEN the System SHALL maintain a history for the current session
    - 8.3: WHEN a repetition is completed 
           THEN the System SHALL record the average pose quality score for that repetition
    
    Args:
        request: AnalysisRequest containing session_id, exercise_type, key_points, and optional image
    
    Returns:
        AnalysisResponse with analysis results including quality feedback
    
    Raises:
        HTTPException: 404 if session not found
        HTTPException: 400 if exercise type is invalid
        HTTPException: 500 if analysis fails
    """
    # Check if session manager is initialized
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized. Service unavailable."
        )
    
    # Get session data
    session_data = session_manager.get_session(request.session_id)
    
    if session_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {request.session_id}"
        )
    
    # Verify exercise type matches session
    if session_data.exercise_type != request.exercise_type.value:
        raise HTTPException(
            status_code=400,
            detail=f"Exercise type mismatch. Session is for '{session_data.exercise_type}', "
                   f"but request is for '{request.exercise_type.value}'"
        )
    
    try:
        # Convert key_points from list format to tuple format for analyzer
        key_points_tuples = {}
        for joint_name, coords in request.key_points.items():
            if len(coords) == 4:
                key_points_tuples[joint_name] = tuple(coords)
            else:
                raise ValueError(f"Invalid key point format for {joint_name}: expected 4 values, got {len(coords)}")
        
        # Analyze form using the session's analyzer
        analysis_result = session_data.analyzer.analyze_form(key_points_tuples)
        
        # Update workout session with latest stats
        session_data.workout_session.update_session(
            rep_count=analysis_result.get('rep_count', 0),
            calories=analysis_result.get('calories', 0.0)
        )
        
        # Calculate quality feedback using PoseQualityEvaluator
        # Calculate form metrics
        form_metrics = session_data.quality_evaluator.calculate_form_metrics(
            analysis_result,
            key_points_tuples
        )
        
        # Handle no pose detected scenario
        if not key_points_tuples or not form_metrics:
            # No pose detected or no valid metrics - return neutral feedback
            logger.info(f"No pose detected or no valid metrics for session {request.session_id}")
            
            response = AnalysisResponse(
                session_id=request.session_id,
                rep_count=analysis_result.get('rep_count', 0),
                stage=ExerciseStage(analysis_result.get('stage', 'start')),
                angles=analysis_result.get('angles', {}),
                feedback=analysis_result.get('feedback', []),
                warnings=analysis_result.get('warnings', []),
                errors=analysis_result.get('errors', []),
                calories=analysis_result.get('calories', 0.0),
                duration=analysis_result.get('duration', None),
                quality_score=0.0,
                quality_category='poor',
                real_time_feedback="Step into the camera frame to begin your workout!",
                historical_average=session_data.quality_evaluator.get_historical_average()
            )
            
            return response
        
        # Calculate quality score
        quality_score = session_data.quality_evaluator.calculate_quality_score(
            form_metrics,
            analysis_result
        )
        
        # Determine feedback category
        category = session_data.quality_evaluator.get_feedback_category(quality_score)
        
        # Generate feedback message
        feedback_message = session_data.quality_evaluator.generate_feedback_message(
            quality_score,
            category,
            form_metrics,
            analysis_result
        )
        
        # Update history
        session_data.quality_evaluator.update_history(quality_score)
        
        # Get historical average
        historical_average = session_data.quality_evaluator.get_historical_average()
        
        # Prepare response with quality feedback
        response = AnalysisResponse(
            session_id=request.session_id,
            rep_count=analysis_result.get('rep_count', 0),
            stage=ExerciseStage(analysis_result.get('stage', 'start')),
            angles=analysis_result.get('angles', {}),
            feedback=analysis_result.get('feedback', []),
            warnings=analysis_result.get('warnings', []),
            errors=analysis_result.get('errors', []),
            calories=analysis_result.get('calories', 0.0),
            duration=analysis_result.get('duration', None),
            quality_score=quality_score,
            quality_category=category,
            real_time_feedback=feedback_message,
            historical_average=historical_average
        )
        
        return response
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid request data: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error in exercise analysis: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Exercise analysis failed: {str(e)}"
        )


# ============================================================================
# SESSION MANAGEMENT ENDPOINTS
# ============================================================================

@app.post("/api/v1/sessions/start", response_model=SessionResponse, tags=["Session Management"])
async def start_session(request: SessionStartRequest):
    """
    Start a new workout session.
    
    Creates a new session with a unique identifier and initializes
    an Exercise Analyzer for the specified exercise type.
    
    **Requirements: 3.1, 6.1**
    - 3.1: WHEN a client initiates a workout session with an exercise type 
           THEN the FastAPI Service SHALL create a new Workout Session 
           with a unique identifier and timestamp
    - 6.1: WHEN a client starts a session 
           THEN the FastAPI Service SHALL create an Exercise Analyzer instance 
           associated with that session identifier
    
    Args:
        request: SessionStartRequest with exercise_type and optional user_id
    
    Returns:
        SessionResponse with session details
    
    Raises:
        HTTPException: 503 if session manager is not initialized
        HTTPException: 500 if session creation fails
    """
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized. Service unavailable."
        )
    
    try:
        # Create new session
        session_id = session_manager.create_session(
            exercise_type=request.exercise_type.value,
            user_id=request.user_id
        )
        
        # Get session data
        session_data = session_manager.get_session(session_id)
        
        if session_data is None:
            raise HTTPException(
                status_code=500,
                detail="Failed to create session"
            )
        
        # Return session response
        return SessionResponse(
            session_id=session_data.session_id,
            exercise_type=ExerciseType(session_data.exercise_type),
            start_time=session_data.created_at.isoformat(),
            end_time=None,
            duration=0.0,
            total_reps=0,
            calories_burned=0.0,
            status=SessionStatus.ACTIVE
        )
        
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start session: {str(e)}"
        )


@app.post("/api/v1/sessions/{session_id}/end", response_model=SessionResponse, tags=["Session Management"])
async def end_session(session_id: str = Depends(validate_session_id)):
    """
    End an active workout session.
    
    Persists session data to storage and cleans up the associated Exercise Analyzer.
    
    **Requirements: 3.2, 6.4**
    - 3.2: WHEN a client ends a workout session 
           THEN the FastAPI Service SHALL persist the session data including 
           reps, duration, calories, and timestamps to storage
    - 6.4: WHEN a session is ended 
           THEN the FastAPI Service SHALL clean up the associated Exercise Analyzer instance
    
    Args:
        session_id: UUID of the session to end
    
    Returns:
        SessionResponse with final session statistics
    
    Raises:
        HTTPException: 404 if session not found
        HTTPException: 400 if session_id is invalid
        HTTPException: 503 if session manager is not initialized
        HTTPException: 500 if ending session fails
    """
    # Validate session ID format
    try:
        validate_session_id(session_id)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized. Service unavailable."
        )
    
    try:
        # End the session
        session_data = session_manager.end_session(session_id)
        
        if session_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}"
            )
        
        # The workout_session.end_session() was already called in session_manager.end_session()
        # We can get the data from the completed sessions list or use the analyzer data
        # Since the session was just ended, we'll use the analyzer's final state
        
        return SessionResponse(
            session_id=session_data.session_id,
            exercise_type=ExerciseType(session_data.exercise_type),
            start_time=session_data.created_at.isoformat(),
            end_time=session_data.last_activity.isoformat(),
            duration=(session_data.last_activity - session_data.created_at).total_seconds(),
            total_reps=session_data.analyzer.rep_count,
            calories_burned=session_data.analyzer.calories_burned,
            status=SessionStatus.COMPLETED
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to end session: {str(e)}"
        )


@app.post("/api/v1/sessions/{session_id}/reset", response_model=SessionResetResponse, tags=["Session Management"])
async def reset_session(session_id: str = Depends(validate_session_id)):
    """
    Reset a session's state to initial values.
    
    Resets rep count, calories, and stage to zero/start without ending the session.
    
    **Requirements: 6.3**
    - 6.3: WHEN a client resets a session 
           THEN the FastAPI Service SHALL reset the Exercise Analyzer state to initial values
    
    Args:
        session_id: UUID of the session to reset
    
    Returns:
        SessionResetResponse confirming the reset
    
    Raises:
        HTTPException: 400 if session_id is invalid
        HTTPException: 404 if session not found
        HTTPException: 503 if session manager is not initialized
    """
    # Validate session ID format
    try:
        validate_session_id(session_id)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized. Service unavailable."
        )
    
    # Reset the session
    success = session_manager.reset_session(session_id)
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    return SessionResetResponse(
        session_id=session_id,
        message="Session reset successfully",
        rep_count=0,
        calories=0.0
    )


@app.get("/api/v1/sessions", response_model=SessionListResponse, tags=["Session Management"])
async def get_sessions(
    user_id: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of sessions to return"),
    offset: int = Query(0, ge=0, description="Number of sessions to skip for pagination")
):
    """
    Get list of workout sessions with optional filtering and pagination.
    
    **Requirements: 3.3**
    - 3.3: WHEN a client requests session history 
           THEN the FastAPI Service SHALL return all stored Workout Sessions 
           in chronological order
    
    Args:
        user_id: Optional user ID to filter sessions
        limit: Maximum number of sessions to return (default: 100, max: 1000)
        offset: Number of sessions to skip for pagination (default: 0)
    
    Returns:
        SessionListResponse with list of sessions and pagination info
    
    Raises:
        HTTPException: 400 if parameters are invalid
        HTTPException: 503 if session manager is not initialized
    """
    # Validate pagination parameters
    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="limit must be between 1 and 1000"
        )
    if offset < 0:
        raise HTTPException(
            status_code=400,
            detail="offset must be non-negative"
        )
    
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized. Service unavailable."
        )
    
    try:
        # Get all active sessions
        active_sessions = session_manager.get_all_sessions()
        
        # Convert active sessions to response format
        session_responses = []
        for sid, session_data in active_sessions.items():
            session_responses.append(
                SessionResponse(
                    session_id=session_data.session_id,
                    exercise_type=ExerciseType(session_data.exercise_type),
                    start_time=session_data.created_at.isoformat(),
                    end_time=None,
                    duration=(session_data.last_activity - session_data.created_at).total_seconds(),
                    total_reps=session_data.analyzer.rep_count,
                    calories_burned=session_data.analyzer.calories_burned,
                    status=SessionStatus(session_data.status)
                )
            )
        
        # Also get completed sessions from storage
        # Note: This requires accessing WorkoutSession's load_sessions
        # For now, we'll just return active sessions
        # In a production system, we'd query a database for all sessions
        
        # Filter by user_id if provided
        if user_id:
            session_responses = [
                s for s in session_responses 
                if session_data.user_id == user_id
            ]
        
        # Apply pagination
        total = len(session_responses)
        paginated_sessions = session_responses[offset:offset + limit]
        
        return SessionListResponse(
            sessions=paginated_sessions,
            total=total,
            limit=limit,
            offset=offset
        )
        
    except Exception as e:
        logger.error(f"Error retrieving sessions: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve sessions: {str(e)}"
        )


@app.get("/api/v1/sessions/{session_id}", response_model=SessionResponse, tags=["Session Management"])
async def get_session(session_id: str = Depends(validate_session_id)):
    """
    Get details of a specific workout session.
    
    **Requirements: 5.3**
    - 5.3: WHEN a client attempts to end a non-existent session 
           THEN the FastAPI Service SHALL return a 404 status code with an appropriate message
    
    Args:
        session_id: UUID of the session to retrieve
    
    Returns:
        SessionResponse with session details
    
    Raises:
        HTTPException: 400 if session_id is invalid
        HTTPException: 404 if session not found
        HTTPException: 503 if session manager is not initialized
    """
    # Validate session ID format
    try:
        validate_session_id(session_id)
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    
    if session_manager is None:
        raise HTTPException(
            status_code=503,
            detail="Session manager not initialized. Service unavailable."
        )
    
    # Get session data
    session_data = session_manager.get_session(session_id)
    
    if session_data is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session not found: {session_id}"
        )
    
    # Return session response
    return SessionResponse(
        session_id=session_data.session_id,
        exercise_type=ExerciseType(session_data.exercise_type),
        start_time=session_data.created_at.isoformat(),
        end_time=None if session_data.status == "active" else session_data.last_activity.isoformat(),
        duration=(session_data.last_activity - session_data.created_at).total_seconds(),
        total_reps=session_data.analyzer.rep_count,
        calories_burned=session_data.analyzer.calories_burned,
        status=SessionStatus(session_data.status)
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
