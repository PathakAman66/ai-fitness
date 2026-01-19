# Streamlit Interface Setup

The Streamlit interface has been refactored to use the FastAPI backend instead of running pose detection locally.

## Architecture

- **Frontend**: Streamlit web interface (this file)
- **Backend**: FastAPI REST API (handles pose detection and exercise analysis)
- **Communication**: HTTP requests with base64-encoded video frames

## Setup Instructions

### 1. Start the FastAPI Backend

First, start the backend server:

```bash
# From the project root directory
uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 2. Start the Streamlit Frontend

In a separate terminal, start the Streamlit interface:

```bash
# From the project root directory
streamlit run frontend/streamlit_interface.py
```

The Streamlit app will open in your browser at `http://localhost:8501`

## Features

### Backend Integration

- **Health Check**: Automatically checks if backend is available on startup
- **Session Management**: Creates and manages workout sessions via API
- **Pose Detection**: Sends video frames to backend for pose landmark detection
- **Exercise Analysis**: Receives real-time form feedback, rep counting, and calorie tracking

### Supported Exercises

- 💪 Bicep Curls
- 🦵 Squats
- 🏃 Push-ups

### Real-time Feedback

- ✅ Positive feedback for good form
- ⚠️ Warnings for minor form issues
- ❌ Errors for major form problems
- 📊 Live rep counting and calorie tracking

## API Endpoints Used

- `GET /health` - Check backend availability
- `POST /api/v1/sessions/start` - Start workout session
- `POST /api/v1/sessions/{session_id}/end` - End workout session
- `POST /api/v1/pose/detect` - Detect pose landmarks in frame
- `POST /api/v1/analyze` - Analyze exercise form and count reps

## Configuration

The API base URL can be changed in `streamlit_interface.py`:

```python
API_BASE_URL = "http://localhost:8000"
```

## Troubleshooting

### Backend Not Available

If you see "Backend API is not available", ensure:
1. FastAPI server is running on port 8000
2. No firewall blocking localhost connections
3. All dependencies are installed (`pip install -r config/requirements.txt`)

### Camera Not Working

- Ensure your webcam is not being used by another application
- Check browser permissions for camera access
- Try restarting the Streamlit app

### Slow Performance

- The interface sends frames to the backend every ~50ms
- Adjust the `time.sleep(0.05)` value in the code to change frame rate
- Consider running backend and frontend on the same machine to reduce latency
