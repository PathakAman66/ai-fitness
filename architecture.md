# 🏋️ AI Fitness Trainer — Project Overview & Architecture

## 1. Project Overview
The **AI Fitness Trainer** is a computer-vision–based application that uses a webcam and pose estimation to analyze human exercises in real time. It provides live feedback, repetition counting, form correction, calorie estimation, and workout session persistence.

The project is designed to be:
- Beginner-friendly
- Modular and extensible
- Suitable for desktop, enhanced desktop, and web-based interfaces

---

## 2. High-Level Architecture

```
+-------------------+
|   User / Webcam   |
+---------+---------+
          |
          v
+-------------------+
| Pose Detection    |  (MediaPipe + OpenCV)
+---------+---------+
          |
          v
+-------------------+
| Exercise Analyzer |  (Reps, angles, form checks)
+---------+---------+
          |
          v
+-------------------+
| UI / Overlay      |  (Visual + Console feedback)
+---------+---------+
          |
          v
+-------------------+
| Session Manager   |  (Workout lifecycle)
+---------+---------+
          |
          v
+-------------------+
| Report Manager    |  (JSON persistence)
+-------------------+
```

---

## 3. Execution Flow

### 3.1 Application Startup
1. Entry point is launched (`fixed_main.py`, `enhanced_trainer.py`, or via launcher).
2. Camera is initialized using OpenCV.
3. Pose detector and exercise analyzer are created.
4. User selects an exercise (enhanced mode) or starts directly (simple mode).

---

### 3.2 Runtime Loop
1. Webcam frame is captured.
2. Frame is passed to the **Pose Detector**.
3. Landmarks are extracted.
4. **Exercise Analyzer**:
   - Calculates joint angles
   - Detects movement stages
   - Counts repetitions or duration
   - Performs form validation
5. Results are rendered on screen as overlays.
6. Feedback, warnings, and errors are shown in real time.

---

### 3.3 Session End & Cleanup
1. User exits the session or application.
2. Workout session is finalized.
3. Report Manager saves workout data to disk.
4. Camera and UI resources are released cleanly.

---

## 4. Core Components

### 4.1 Pose Detector
**Responsibility:**
- Detects human pose using MediaPipe
- Extracts landmark coordinates and visibility

**Key Files:**
- `SimplePoseDetector`
- `EnhancedPoseDetector`

---

### 4.2 Exercise Analyzer
**Responsibility:**
- Interprets pose landmarks
- Calculates joint angles
- Tracks repetitions or hold duration
- Detects incorrect form

**Supported Exercises:**
- Bicep curls
- Squats
- Push-ups
- Shoulder press
- Lunges
- Plank

---

### 4.3 UI / Overlay System
**Responsibility:**
- Displays real-time metrics
- Shows feedback, warnings, and errors
- Renders angles, reps, calories, and timers

---

### 4.4 Workout Session Manager
**Responsibility:**
- Manages workout lifecycle
- Tracks session start and end
- Stores reps, duration, and calories

**Key Class:**
- `WorkoutSession`

---

### 4.5 Report Manager
**Responsibility:**
- Persists workout data
- Saves session summaries as JSON
- Ensures data is saved on clean or early exit

**Output Example:**
```json
{
  "exercise": "squat",
  "reps": 15,
  "duration": 82,
  "calories": 15.0,
  "timestamp": "2026-01-17 22:10:45"
}
```

---

## 5. Folder Structure

```
ai_fitness_trainer/
│
├── fixed_main.py            # Simple desktop trainer
├── enhanced_trainer.py      # Advanced trainer with UI & analytics
├── report_manager.py        # Workout persistence logic
├── workout_data/            # Saved workout sessions
│   └── sessions.json
├── reports/                 # JSON workout reports
├── web/                     # Web interface (optional)
├── tools/                   # Utility scripts
└── README.md
```

---

## 6. How to Extend the Project

### Add a New Exercise
1. Add exercise metadata to the config.
2. Implement a new `analyze_<exercise>()` method.
3. Register the exercise in the analyzer switch.

---

### Improve Analytics
- Add weekly/monthly aggregation
- Export reports to CSV
- Visualize progress using dashboards

---

## 7. Intended Audience
- Beginners learning computer vision
- Students building AI projects
- Contributors interested in fitness tech
- Developers exploring MediaPipe & OpenCV

---

## 8. Contribution Notes
- Follow modular design principles
- Keep persistence logic separate from UI
- Ensure cleanup always saves session data
- Prefer clarity over complexity

---

## 9. Summary
This project demonstrates how computer vision, real-time feedback, and structured data persistence can be combined to create a practical AI-powered fitness application. The architecture is intentionally modular to make learning, debugging, and extending the system easy for new contributors.

