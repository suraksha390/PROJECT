# Video Based Driving Skill Validation System Using FaceNet, YOLOv8, MediaPipe, OpenCV, and Random Forest

An end-to-end Automated Driving Skill Validation System developed for Engineering Final Year Projects. The system authenticates candidates using **FaceNet CNN** facial embeddings, analyzes recorded driving test videos using **YOLOv8**, **OpenCV**, and **MediaPipe Pose**, predicts test outcomes using a **Random Forest** classifier, and automatically generates downloadable **PDF Driving Evaluation Reports**.

---

## 📌 Features & Architecture

```
                  ┌──────────────────────────────────────────────┐
                  │ 1. Candidate Registration & Passport Photo   │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 2. Real-Time FaceNet Webcam Authentication   │
                  └──────────────────────┬───────────────────────┘
                                         │ Passed
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 3. Upload & Process Driving Test Video       │
                  └──────────────────────┬───────────────────────┘
                                         │
         ┌───────────────────────────────┼───────────────────────────────┐
         │                               │                               │
         ▼                               ▼                               ▼
┌──────────────────┐           ┌──────────────────┐            ┌──────────────────┐
│ YOLOv8 Detection │           │ OpenCV Tracking  │            │ MediaPipe Pose   │
│ Helmet, Rider,   │           │ Track Lane Line  │            │ Foot-to-Ground   │
│ Motorcycle Zone  │           │ Boundary Cross   │            │ & Lean Stability │
└────────┬─────────┘           └────────┬─────────┘            └────────┬─────────┘
         │                               │                               │
         └───────────────────────────────┼───────────────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 4. Feature Extraction & Machine Learning     │
                  │    - Random Forest Classifier                │
                  └──────────────────────┬───────────────────────┘
                                         │
                                         ▼
                  ┌──────────────────────────────────────────────┐
                  │ 5. Automated Evaluation & PDF Report Gen.    │
                  │    Downloadable RTO Official Report          │
                  └──────────────────────────────────────────────┘
```

### Module 1: Candidate Registration & Authentication
- Demographics entry: Candidate Name, ID, Aadhaar Number, DOB, Gender, Address, Mobile, Email, and Passport Photo.
- SQLite Database candidate record storage.
- Real-time webcam face capture compared against registered photo using **FaceNet** (Cosine similarity threshold distance).

### Module 2: Driving Behaviour Computer Vision Analysis
- **Helmet Detection**: Detects helmet presence on rider's head region using YOLOv8 & OpenCV contour analysis.
- **Lane Following Detection**: Color segmentation (HSV), Canny edge detection & Hough line transform to track track boundary violations.
- **Foot Touch Detection**: MediaPipe Pose estimation tracking ankles/feet keypoints relative to ground level. Marks `Foot Touch = True` (FAIL) if ground distance $\le 0$.
- **Rider Balance & Posture**: Calculates spine/shoulder tilt angle ($\theta = \arctan2$) and stability variance across frame sequences. Returns `Stable` / `Unstable`.

### Module 3: Machine Learning Driving Performance Prediction
- Inputs aggregated feature matrix into a trained **Random Forest** classifier.
- Predicts overall result (`PASS` or `FAIL`) with score percentage (0-100%).

### Module 4: Automated PDF Report Generation
- Built using `ReportLab`.
- Renders candidate metadata, candidate photo, FaceNet auth status, computer vision telemetry breakdown, ML model predictions, identified mistakes (e.g. *Helmet not worn*, *Foot touched ground*, *Lane crossed*, *Rider imbalance*), and digital inspector signature.

---

## 🛠️ System Requirements & Dependencies

- **Python**: 3.8, 3.9, 3.10, 3.11, or 3.12
- **Web Browser**: Chrome, Edge, Firefox, or Safari (Webcam enabled)

### Required Libraries (`requirements.txt`)
- `flask`
- `opencv-python`
- `mediapipe`
- `ultralytics` (YOLOv8)
- `scikit-learn`
- `numpy`
- `pillow`
- `reportlab`
- `joblib`

---

## 🚀 Quick Setup & Installation Guide

### Step 1: Clone or Navigate to Project Folder
```bash
cd "c:/Users/SURAKSHA SHETTY/OneDrive/Documents/suraksha"
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Train Machine Learning Model (Random Forest)
Run the automated dataset generator and ML model training script:
```bash
python scripts/train_ml_models.py
```
This outputs trained `rf_model.pkl` and `scaler.pkl` inside `static/saved_models/`.

### Step 4: Run Flask Web Application
```bash
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📂 Project Directory Structure

```
.
├── app.py                      # Flask Application Entry Point
├── config.py                   # Storage Paths & Threshold Parameters
├── requirements.txt            # Python Dependencies
├── README.md                   # System Documentation
│
├── database/
│   └── db.py                   # SQLite Schema & Operations
│
├── models/
│   ├── facenet_auth.py         # Candidate FaceNet Authenticator
│   ├── yolo_detector.py        # YOLOv8 Rider, Helmet & Ground Detector
│   ├── lane_detector.py        # OpenCV Lane Boundary Segmentation
│   ├── pose_analyzer.py        # MediaPipe Pose Foot Touch & Balance Tracker
│   └── ml_evaluator.py         # Random Forest ML Engine
│
├── scripts/
│   ├── generate_sample_data.py # Dataset Synthesizer
│   └── train_ml_models.py      # RF Model Training Script
│
├── utils/
│   └── pdf_generator.py        # ReportLab PDF Report Generator
│
├── dataset/
│   └── driving_test_dataset.csv
│
├── static/
│   ├── css/
│   │   └── style.css           # Glassmorphism Modern UI
│   ├── js/
│   │   └── main.js             # Frontend Camera & AJAX Scripts
│   ├── uploads/                # Photos, Uploaded Videos & Processed Frames
│   ├── reports/                # Generated PDF Reports
│   └── saved_models/           # Serialized RF Model (.pkl)
│
└── templates/
    ├── base.html               # Base Template Layout
    ├── index.html              # System Dashboard
    ├── register.html           # Candidate Registration Form
    ├── authenticate.html       # FaceNet Verification Interface
    ├── analysis.html           # Video Upload & Telemetry Analytics
    └── report.html             # Detailed Digital & PDF Report View
```

---

## 📜 Final Year Project Presentation Summary
- **Title**: Video based Driving Skill Validation System Using FaceNet, YOLOv8, MediaPipe, OpenCV, and Random Forest
- **Domain**: Artificial Intelligence, Computer Vision, Machine Learning, Web Development
- **Backend Framework**: Flask (Python)
- **Computer Vision Frameworks**: OpenCV, YOLOv8, MediaPipe Pose
- **Deep Learning / Face Verification**: FaceNet CNN
- **Machine Learning Algorithms**: Random Forest
- **Database**: SQLite3
# AI_based_driving_skill_validation_system
# AI_based_driving_skill_validation_system
# new_Driving
# driving1
