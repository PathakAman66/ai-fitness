# 🏋️‍♂️ AI Fitness Trainer with Real-Time Pose Estimation

An intelligent AI-powered fitness trainer that uses **Computer Vision**, **MediaPipe**, and **OpenCV** to provide real-time posture feedback, count exercise repetitions, and track progress — all through a simple webcam.  
This project aims to make at-home fitness training more accessible, interactive, and accurate.

---

## ✨ Features

- **Real-time Pose Detection** using MediaPipe (33 body landmarks)
- **Exercise Form Analysis** with angle-based posture validation
- **Repetition Counting** for multiple exercises
- **Audio Feedback** for correction & guidance
- **Web Dashboard Support** using Streamlit
- **Lightweight & CPU-friendly** (no GPU required)
- **Extensible Exercise Modules**

Supported exercise categories include:

✔ Bicep curls  
✔ Squats  
✔ Push-ups  
✔ Shoulder press  
✔ Extendable for more exercises

---

## 🎥 Demo

### 📸 Screenshots

#### 🏠 Home Page
![Home Page](Screenshots/home.png)

#### 💪 Bicep Curls Mode
![Bicep Curls](Screenshots/bicep.png)

#### 🏋️ Squats Mode
![Squats](Screenshots/squat.png)

#### 🤸 Push-ups Mode
![Push-ups](Screenshots/pushup.png)

### 🎬 GIF / Video
(Add a GIF or video link here)


> Tip: You can record a short demo video and upload it to YouTube/Drive and paste the link here.

## 🚀 Quick Start

### **1. Clone the Repository**

```bash
git clone https://github.com/PathakAman66/ai-fitness-trainer.git
cd ai-fitness-trainer
```

### **2. Install Dependencies**

Recommended installation:

```bash
pip install -r requirements.txt
```

If this fails, fallback:

```bash
pip install mediapipe opencv-python numpy streamlit pyttsx3
```

---

## 🏃‍♂️ Running the Fitness Trainer

### **Option A — Webcam Fitness Trainer (OpenCV)**

```bash
python core/run_fitness_trainer.py
```

This launches the webcam and starts pose detection, rep counting, and feedback.

---

## 🌐 Running the Web Dashboard

This provides a more interactive UI for exercise monitoring.

### **Step 1 — Launch Web App**

```bash
python web/run_website.py
```

### **Step 2 — Open in Browser**

Default Streamlit URL:

```
http://localhost:8501
```

Features include:

✔ Camera feed  
✔ Exercise selection  
✔ Real-time feedback overlay  
✔ Progress display  

---

## 🌍 Alternative HTML Web Server

If Streamlit is not preferred:

```bash
python web/web_server.py
```

This exposes an HTML interface via a lightweight server.

---

## 🧱 Project Structure

```text
ai-fitness-trainer/
│
├── core/                      # Core AI & fitness logic
│   ├── enhanced_trainer.py
│   ├── fixed_main.py
│   └── run_fitness_trainer.py
│
├── web/                       # Web interfaces & dashboards
│   ├── web_interface.py
│   ├── simple_web.py
│   ├── launch_web.py
│   └── progress_dashboard.py
│
├── scripts/                   # Setup & automation scripts
│   ├── clean_setup.py
│   ├── create_structure.py
│   ├── create_web_files.py
│   ├── install_dependencies.py
│   ├── install_web_dependencies.py
│   ├── fix_installation.ps1
│   └── install_and_run.bat
│
├── tests/                     # Test and validation files
│   ├── simple_test.py
│   └── test_setup.py
│
├── requirements/              # Dependency files
│   ├── requirements.txt
│   └── requirements-simple.txt
│
├── run.py                     # Main entry point
├── setup.py                   # Packaging/build
├── CODE_OF_CONDUCT.md
└── README.md
```

---

## 📊 Exercise Detection Logic (Overview)

| Exercise | Detection Metric | Key Angle |
|---|---|---|
| Bicep Curl | Elbow flexion | Shoulder → Elbow → Wrist |
| Squat | Hip/Knee flexion | Shoulder → Hip → Knee |
| Push-up | Chest vertical depth | Shoulder → Elbow |
| Shoulder Press | Vertical motion | Wrist → Elbow → Shoulder |

---

## 🧪 Testing

Run tests to verify environment:

```bash
python tests/test_setup.py
```

Or minimal test:

```bash
python tests/simple_test.py
```

---

## 📦 Requirements

- **Python:** 3.8+
- **Camera:** Any 720p webcam
- **OS:** Windows / Linux / macOS
- **CPU:** Runs without GPU

Optional performance boost if GPU exists.

---

## 🧩 Extending the System

New exercises can be added by:

1. Adding angle logic in `core/`
2. Registering exercise in trainer
3. Updating web UI for exercise selection

---

## 🤝 Contributing

We welcome contributions including:

- New exercise models
- Pose detection improvements
- UI enhancements
- Documentation
- Bug fixes

Refer to `CONTRIBUTING.md` for contribution flow.

---

## ⭐ Show Support

If this project helped you, consider giving it a ⭐ on GitHub to support development!

