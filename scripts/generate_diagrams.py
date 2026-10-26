import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'static', 'diagrams')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set global font family to Arial for clean professional rendering
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['font.family'] = 'sans-serif'

def create_system_architecture():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')
    
    # Outer frame
    ax.add_patch(patches.Rectangle((0.2, 0.2), 11.6, 7.1, facecolor='#ffffff', edgecolor='#1e293b', linewidth=2.5))
    ax.text(6.0, 6.8, "SYSTEM ARCHITECTURE DIAGRAM", fontsize=16, fontweight='bold', ha='center', color='#0f172a')

    # Presentation Layer
    ax.add_patch(patches.Rectangle((0.6, 4.7), 3.3, 1.7, facecolor='#eff6ff', edgecolor='#2563eb', linewidth=2))
    ax.add_patch(patches.Rectangle((0.6, 5.9), 3.3, 0.5, facecolor='#2563eb', edgecolor='#2563eb'))
    ax.text(2.25, 6.15, "Presentation Layer", fontsize=12, fontweight='bold', ha='center', va='center', color='#ffffff')
    ax.text(2.25, 5.25, "HTML5 / CSS3 / JavaScript\nBootstrap 5 Web Interface\nWebcam Capture & Video Upload", fontsize=10, fontweight='bold', ha='center', va='center', color='#1e3a8a')

    # Application Layer
    ax.add_patch(patches.Rectangle((4.35, 4.7), 3.3, 1.7, facecolor='#f5f3ff', edgecolor='#6d28d9', linewidth=2))
    ax.add_patch(patches.Rectangle((4.35, 5.9), 3.3, 0.5, facecolor='#6d28d9', edgecolor='#6d28d9'))
    ax.text(6.0, 6.15, "Application Layer", fontsize=12, fontweight='bold', ha='center', va='center', color='#ffffff')
    ax.text(6.0, 5.25, "Flask Web Server (Python)\nRESTful API Endpoints\nController Request Manager", fontsize=10, fontweight='bold', ha='center', va='center', color='#4c1d95')

    # Data Storage Layer
    ax.add_patch(patches.Rectangle((8.1, 4.7), 3.3, 1.7, facecolor='#fffbe6', edgecolor='#d97706', linewidth=2))
    ax.add_patch(patches.Rectangle((8.1, 5.9), 3.3, 0.5, facecolor='#d97706', edgecolor='#d97706'))
    ax.text(9.75, 6.15, "Data Storage Layer", fontsize=12, fontweight='bold', ha='center', va='center', color='#ffffff')
    ax.text(9.75, 5.25, "SQLite Database File\nCandidate Demographics & Vectors\nPDF Evaluation Storage", fontsize=10, fontweight='bold', ha='center', va='center', color='#78350f')

    # Computer Vision & ML Pipeline Layer
    ax.add_patch(patches.Rectangle((0.6, 0.6), 10.8, 3.5, facecolor='#f0fdf4', edgecolor='#16a34a', linewidth=2))
    ax.add_patch(patches.Rectangle((0.6, 3.6), 10.8, 0.5, facecolor='#16a34a', edgecolor='#16a34a'))
    ax.text(6.0, 3.85, "Computer Vision & Machine Learning Pipeline Engine", fontsize=13, fontweight='bold', ha='center', va='center', color='#ffffff')

    # Modules inside CV Layer
    cv_modules = [
        ("FaceNet CNN", "Webcam Face Verification\n512-d Cosine Matching", (0.9, 0.9)),
        ("YOLOv8 Detector", "Rider, Motorcycle, Helmet\n& Ground Detection", (3.5, 0.9)),
        ("OpenCV Track", "HSV / Canny / Hough\nLane Boundary Tracking", (6.1, 0.9)),
        ("MediaPipe Pose", "Foot Touch Distance &\nBody Lean Stability", (8.7, 0.9)),
    ]
    for name, desc, (x, y) in cv_modules:
        ax.add_patch(patches.Rectangle((x, y), 2.4, 2.4, facecolor='#ffffff', edgecolor='#16a34a', linewidth=1.8))
        ax.add_patch(patches.Rectangle((x, y+1.8), 2.4, 0.6, facecolor='#dcfce7', edgecolor='#16a34a', linewidth=1.5))
        ax.text(x+1.2, y+2.1, name, fontsize=11, fontweight='bold', ha='center', va='center', color='#14532d')
        ax.text(x+1.2, y+0.9, desc, fontsize=9.5, fontweight='bold', ha='center', va='center', color='#166534')

    # Horizontal & Vertical Connectors
    arrowprops = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color="#334155")
    ax.annotate("", xy=(4.35, 5.55), xytext=(3.9, 5.55), arrowprops=arrowprops)
    ax.annotate("", xy=(8.1, 5.55), xytext=(7.65, 5.55), arrowprops=arrowprops)
    ax.annotate("", xy=(6.0, 4.1), xytext=(6.0, 4.7), arrowprops=arrowprops)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'system_architecture.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional system_architecture.png")

def create_use_case():
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # System boundary box
    ax.add_patch(patches.Rectangle((3.0, 0.4), 6.0, 7.2, facecolor='#f8fafc', edgecolor='#334155', linewidth=2.5, linestyle='--'))
    ax.text(6.0, 7.3, "Driving Validation System Boundary", fontsize=13, fontweight='bold', ha='center', color='#0f172a')

    # Actor: Candidate
    ax.plot(1.2, 4.6, 'o', ms=26, color='#2563eb')
    ax.plot([1.2, 1.2], [3.5, 4.3], color='#2563eb', lw=3)
    ax.plot([0.5, 1.9], [4.0, 4.0], color='#2563eb', lw=3)
    ax.plot([1.2, 0.7], [3.5, 2.6], color='#2563eb', lw=3)
    ax.plot([1.2, 1.7], [3.5, 2.6], color='#2563eb', lw=3)
    ax.text(1.2, 2.2, "Candidate", fontsize=13, fontweight='bold', ha='center', color='#1e40af')

    # Actor: Admin / Inspector
    ax.plot(10.8, 4.6, 'o', ms=26, color='#059669')
    ax.plot([10.8, 10.8], [3.5, 4.3], color='#059669', lw=3)
    ax.plot([10.1, 11.5], [4.0, 4.0], color='#059669', lw=3)
    ax.plot([10.8, 10.3], [3.5, 2.6], color='#059669', lw=3)
    ax.plot([10.8, 11.3], [3.5, 2.6], color='#059669', lw=3)
    ax.text(10.8, 2.2, "System Inspector", fontsize=13, fontweight='bold', ha='center', color='#047857')

    # Use Cases (Ellipses)
    use_cases = [
        ("UC-1: Register Demographics & Photo", (6.0, 6.4)),
        ("UC-2: Real-Time FaceNet Verification", (6.0, 5.3)),
        ("UC-3: Upload Driving Test Video", (6.0, 4.2)),
        ("UC-4: Analyze Helmet, Lane, Foot & Balance", (6.0, 3.1)),
        ("UC-5: Predict Result (Random Forest & SVM)", (6.0, 2.0)),
        ("UC-6: Download Official PDF Report", (6.0, 0.9))
    ]

    for label, (x, y) in use_cases:
        ellipse = patches.Ellipse((x, y), 5.2, 0.8, facecolor='#e0f2fe', edgecolor='#0284c7', linewidth=2.0)
        ax.add_patch(ellipse)
        ax.text(x, y, label, fontsize=10.5, fontweight='bold', ha='center', va='center', color='#0369a1')

        # Connect candidate
        if "UC-1" in label or "UC-2" in label or "UC-3" in label or "UC-6" in label:
            ax.plot([1.9, x - 2.6], [4.0, y], color='#64748b', linestyle='-', lw=1.8)

        # Connect admin
        if "UC-4" in label or "UC-5" in label or "UC-6" in label:
            ax.plot([10.1, x + 2.6], [4.0, y], color='#64748b', linestyle='-', lw=1.8)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'use_case_diagram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional use_case_diagram.png")

def create_activity_diagram():
    fig, ax = plt.subplots(figsize=(9, 11), dpi=300)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 11)
    ax.axis('off')

    ax.text(4.5, 10.6, "SYSTEM ACTIVITY DIAGRAM", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

    # Initial state
    ax.add_patch(patches.Circle((4.5, 10.0), 0.25, facecolor='#0f172a'))

    steps = [
        ("Candidate Registration & Passport Photo Upload", 9.0),
        ("Store Candidate Metadata & FaceNet Embeddings in DB", 8.0),
        ("Activate Live Webcam & Capture Candidate Face Frame", 7.0),
        ("Extract FaceNet Features & Compare Cosine Distance", 6.0),
    ]

    for text, y in steps:
        ax.add_patch(patches.Rectangle((1.5, y-0.4), 6.0, 0.8, facecolor='#e0e7ff', edgecolor='#4338ca', linewidth=2))
        ax.text(4.5, y, text, fontsize=10.5, fontweight='bold', ha='center', va='center', color='#3730a3')

    # Decision Diamond
    diamond = patches.Polygon([[4.5, 5.0], [5.8, 4.4], [4.5, 3.8], [3.2, 4.4]], facecolor='#fef08a', edgecolor='#ca8a04', lw=2)
    ax.add_patch(diamond)
    ax.text(4.5, 4.4, "Authenticated?\n(Dist < 0.50)", fontsize=10, fontweight='bold', ha='center', va='center', color='#854d0e')

    # Reject state
    ax.add_patch(patches.Rectangle((0.3, 4.0), 2.2, 0.8, facecolor='#fee2e2', edgecolor='#dc2626', linewidth=2))
    ax.text(1.4, 4.4, "Reject Candidate\n(Auth Failed)", fontsize=9.5, fontweight='bold', ha='center', va='center', color='#991b1b')

    # Success Flow
    steps_after = [
        ("Upload Recorded Driving Test Video File", 3.0),
        ("Run YOLOv8, OpenCV Lane & MediaPipe Pose Analytics", 2.0),
        ("Classify Driving Outcome via Random Forest & SVM", 1.0)
    ]

    for text, y in steps_after:
        ax.add_patch(patches.Rectangle((1.5, y-0.4), 6.0, 0.8, facecolor='#dcfce7', edgecolor='#15803d', linewidth=2))
        ax.text(4.5, y, text, fontsize=10.5, fontweight='bold', ha='center', va='center', color='#166534')

    # End State
    ax.add_patch(patches.Circle((4.5, 0.25), 0.25, facecolor='#15803d', edgecolor='#ffffff', lw=2))
    ax.text(4.5, 0.25, "PDF", fontsize=8, fontweight='bold', ha='center', va='center', color='#ffffff')

    arrowprops = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.0, color="#334155")
    ax.annotate("", xy=(4.5, 9.4), xytext=(4.5, 9.75), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 8.4), xytext=(4.5, 8.6), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 7.4), xytext=(4.5, 7.6), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 6.4), xytext=(4.5, 6.6), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 5.0), xytext=(4.5, 5.6), arrowprops=arrowprops)

    # Branches
    ax.annotate("", xy=(2.5, 4.4), xytext=(3.2, 4.4), arrowprops=arrowprops)
    ax.text(2.8, 4.6, "No", fontsize=11, fontweight='bold', color='#dc2626')

    ax.annotate("", xy=(4.5, 3.4), xytext=(4.5, 3.8), arrowprops=arrowprops)
    ax.text(4.7, 3.6, "Yes", fontsize=11, fontweight='bold', color='#16a34a')

    ax.annotate("", xy=(4.5, 2.4), xytext=(4.5, 2.6), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 1.4), xytext=(4.5, 1.6), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 0.5), xytext=(4.5, 0.6), arrowprops=arrowprops)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'activity_diagram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional activity_diagram.png")

def create_sequence_diagram():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    ax.text(6.0, 7.0, "SYSTEM SEQUENCE DIAGRAM", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

    actors = [
        ("Candidate", 1.2),
        ("Flask Controller", 3.8),
        ("FaceNet Authenticator", 6.4),
        ("CV & ML Pipeline", 9.0),
        ("SQLite & ReportLab", 11.2)
    ]

    for name, x in actors:
        ax.add_patch(patches.Rectangle((x-1.0, 6.0), 2.0, 0.6, facecolor='#1e293b', edgecolor='#0f172a', linewidth=2))
        ax.text(x, 6.3, name, fontsize=10.5, fontweight='bold', ha='center', va='center', color='#ffffff')
        ax.plot([x, x], [0.6, 6.0], linestyle='--', color='#94a3b8', lw=1.8)

    messages = [
        (1.2, 3.8, 5.4, "1: Submit Candidate Form & Photo", "->"),
        (3.8, 11.2, 4.8, "2: Save Record & 512-d Face Embedding", "->"),
        (1.2, 3.8, 4.1, "3: Initiate Live Webcam Verification", "->"),
        (3.8, 6.4, 3.4, "4: Match Webcam Frame (Cosine Distance)", "->"),
        (6.4, 3.8, 2.7, "5: Auth Result (PASS / FAIL)", "<--"),
        (1.2, 3.8, 2.0, "6: Upload Test Video File", "->"),
        (3.8, 9.0, 1.4, "7: Extract CV Metrics & Classify (RF & SVM)", "->"),
        (9.0, 11.2, 0.8, "8: Save Evaluation & Compile PDF Report", "->")
    ]

    for x1, x2, y, label, style in messages:
        if style == "->":
            arrow = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=1.8, color="#2563eb")
        else:
            arrow = dict(arrowstyle="<-,head_width=0.4,head_length=0.6", lw=1.8, color="#16a34a", linestyle="--")
        
        ax.annotate("", xy=(x2, y), xytext=(x1, y), arrowprops=arrow)
        mid_x = (x1 + x2) / 2.0
        ax.text(mid_x, y + 0.15, label, fontsize=9.5, fontweight='bold', ha='center', color='#0f172a')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'sequence_diagram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional sequence_diagram.png")

def create_class_diagram():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    ax.text(6.0, 7.0, "UML CLASS DIAGRAM", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

    classes = [
        ("Candidate", 0.6, 3.8, 3.2, 2.8, [
            "+ candidate_id: str [PK]",
            "+ name: str",
            "+ aadhaar: str [UNIQUE]",
            "+ photo_path: str",
            "+ face_embedding: list"
        ], ["+ register_candidate()", "+ get_candidate_details()"]),
        
        ("FaceNetAuthenticator", 4.4, 3.8, 3.4, 2.8, [
            "+ match_threshold: float = 0.50",
            "- resnet_model: InceptionResnetV1",
            "- mtcnn_detector: MTCNN"
        ], ["+ extract_embedding(image)", "+ compare_faces(emb1, emb2)"]),
        
        ("CVTestProcessor", 8.2, 3.8, 3.2, 2.8, [
            "+ yolo_detector: YoloDetector",
            "+ lane_detector: LaneDetector",
            "+ pose_analyzer: PoseAnalyzer"
        ], ["+ analyze_video(video_file)", "+ process_frame_telemetry()"]),
        
        ("DrivingMLEvaluator", 4.4, 0.5, 3.4, 2.7, [
            "+ rf_model: RandomForestClassifier",
            "+ svm_model: SupportVectorMachine",
            "+ scaler: StandardScaler"
        ], ["+ evaluate(feature_vector)", "+ predict_score_and_result()"]),

        ("PDFReportGenerator", 8.2, 0.5, 3.2, 2.7, [
            "+ reports_dir: str",
            "+ pdf_styles: SampleStyleSheet"
        ], ["+ generate_pdf_report(eval, cand)"])
    ]

    for title, x, y, w, h, attrs, methods in classes:
        # Main Box
        ax.add_patch(patches.Rectangle((x, y), w, h, facecolor='#ffffff', edgecolor='#1e293b', lw=2))
        # Header Box
        ax.add_patch(patches.Rectangle((x, y+h-0.6), w, 0.6, facecolor='#38bdf8', edgecolor='#1e293b', lw=2))
        ax.text(x+w/2, y+h-0.3, title, fontsize=11, fontweight='bold', ha='center', va='center', color='#0f172a')

        # Attributes
        attr_text = "\n".join(attrs)
        ax.text(x+0.15, y+h-0.8, attr_text, fontsize=9, fontweight='bold', va='top', color='#334155')

        # Divider line
        ax.plot([x, x+w], [y+0.9, y+0.9], color='#94a3b8', lw=1.5)

        # Methods
        meth_text = "\n".join(methods)
        ax.text(x+0.15, y+0.75, meth_text, fontsize=9, fontweight='bold', va='top', color='#0369a1')

    # Association lines
    ax.plot([3.8, 4.4], [5.2, 5.2], color='#475569', lw=2) # Candidate -> FaceNet
    ax.plot([7.8, 8.2], [5.2, 5.2], color='#475569', lw=2) # FaceNet -> CVProcessor
    ax.plot([9.8, 9.8], [3.8, 3.2], color='#475569', lw=2) # CVProcessor -> PDFGenerator
    ax.plot([6.1, 6.1], [3.8, 3.2], color='#475569', lw=2) # FaceNet -> MLEvaluator

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'class_diagram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional class_diagram.png")

def create_er_diagram():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    ax.text(5.5, 6.0, "ENTITY-RELATIONSHIP (ER) DIAGRAM", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

    # Entity 1: Candidate
    ax.add_patch(patches.Rectangle((0.8, 1.8), 3.4, 3.2, facecolor='#dbeafe', edgecolor='#1d4ed8', lw=2.2))
    ax.add_patch(patches.Rectangle((0.8, 4.4), 3.4, 0.6, facecolor='#1d4ed8', edgecolor='#1d4ed8'))
    ax.text(2.5, 4.7, "CANDIDATE ENTITY", fontsize=12, fontweight='bold', ha='center', color='#ffffff')
    cand_attrs = "• candidate_id [PRIMARY KEY]\n• name: TEXT\n• aadhaar: TEXT [UNIQUE]\n• dob, gender, mobile, email\n• photo_path: TEXT\n• face_embedding: JSON Vector"
    ax.text(0.95, 4.2, cand_attrs, fontsize=9.5, fontweight='bold', va='top', color='#1e3a8a')

    # Relationship: Undergoes Test
    rel_diamond = patches.Polygon([[5.5, 3.9], [6.5, 3.4], [5.5, 2.9], [4.5, 3.4]], facecolor='#fef08a', edgecolor='#ca8a04', lw=2)
    ax.add_patch(rel_diamond)
    ax.text(5.5, 3.4, "UNDERGOES\nTEST", fontsize=10, fontweight='bold', ha='center', va='center', color='#854d0e')

    # Entity 2: Evaluation
    ax.add_patch(patches.Rectangle((7.0, 0.8), 3.4, 4.2, facecolor='#dcfce7', edgecolor='#15803d', lw=2.2))
    ax.add_patch(patches.Rectangle((7.0, 4.4), 3.4, 0.6, facecolor='#15803d', edgecolor='#15803d'))
    ax.text(8.7, 4.7, "EVALUATION ENTITY", fontsize=12, fontweight='bold', ha='center', color='#ffffff')
    eval_attrs = "• id [PRIMARY KEY]\n• candidate_id [FOREIGN KEY]\n• auth_status (0/1)\n• helmet_status (0/1)\n• lane_status (0/1)\n• foot_touch_status (0/1)\n• balance_status (0/1)\n• overall_result (PASS/FAIL)\n• rf_prediction & svm_pred\n• overall_score: REAL\n• mistakes_json & report_path"
    ax.text(7.15, 4.2, eval_attrs, fontsize=9.5, fontweight='bold', va='top', color='#166534')

    # Connections (1 to N)
    ax.plot([4.2, 4.5], [3.4, 3.4], color='#334155', lw=2.2)
    ax.plot([6.5, 7.0], [3.4, 3.4], color='#334155', lw=2.2)
    ax.text(4.3, 3.65, "1", fontsize=13, fontweight='bold', color='#1d4ed8')
    ax.text(6.7, 3.65, "N", fontsize=13, fontweight='bold', color='#15803d')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'er_diagram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional er_diagram.png")

def create_db_schema():
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 6.5)
    ax.axis('off')

    ax.text(5.5, 6.0, "DATABASE RELATIONAL SCHEMA (SQLITE)", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

    # Table 1: candidates
    ax.add_patch(patches.Rectangle((0.8, 1.2), 4.2, 4.2, facecolor='#ffffff', edgecolor='#0284c7', lw=2.2))
    ax.add_patch(patches.Rectangle((0.8, 4.8), 4.2, 0.6, facecolor='#0284c7', edgecolor='#0284c7'))
    ax.text(2.9, 5.1, "TABLE: candidates", fontsize=12, fontweight='bold', ha='center', color='#ffffff')

    cand_cols = [
        "id: INTEGER [PK, AUTOINCREMENT]",
        "candidate_id: TEXT [UNIQUE, NOT NULL]",
        "name: TEXT [NOT NULL]",
        "aadhaar: TEXT [UNIQUE, NOT NULL]",
        "dob, gender, mobile, email",
        "photo_path: TEXT [NOT NULL]",
        "face_embedding: TEXT (JSON)"
    ]
    ax.text(0.95, 4.6, "\n".join(cand_cols), fontsize=9.5, fontweight='bold', va='top', color='#0f172a')

    # Table 2: evaluations
    ax.add_patch(patches.Rectangle((6.0, 0.5), 4.4, 4.9, facecolor='#ffffff', edgecolor='#059669', lw=2.2))
    ax.add_patch(patches.Rectangle((6.0, 4.8), 4.4, 0.6, facecolor='#059669', edgecolor='#059669'))
    ax.text(8.2, 5.1, "TABLE: evaluations", fontsize=12, fontweight='bold', ha='center', color='#ffffff')

    eval_cols = [
        "id: INTEGER [PK, AUTOINCREMENT]",
        "candidate_id: TEXT [FK -> candidates]",
        "auth_status: INTEGER [NOT NULL]",
        "helmet_status: INTEGER [NOT NULL]",
        "lane_status: INTEGER [NOT NULL]",
        "foot_touch_status: INTEGER [NOT NULL]",
        "balance_status: INTEGER [NOT NULL]",
        "overall_result: TEXT [PASS / FAIL]",
        "rf_prediction: TEXT",
        "svm_prediction: TEXT",
        "overall_score: REAL [NOT NULL]",
        "mistakes_json: TEXT",
        "report_path: TEXT"
    ]
    ax.text(6.15, 4.6, "\n".join(eval_cols), fontsize=9.5, fontweight='bold', va='top', color='#0f172a')

    # FK relation arrow
    arrowprops = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.2, color="#dc2626")
    ax.annotate("", xy=(5.0, 4.3), xytext=(6.0, 4.3), arrowprops=arrowprops)
    ax.text(5.5, 4.55, "FK Link", fontsize=10, fontweight='bold', ha='center', color='#dc2626')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'database_schema.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional database_schema.png")

def create_system_flowchart():
    fig, ax = plt.subplots(figsize=(9, 11), dpi=300)
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 11)
    ax.axis('off')

    ax.text(4.5, 10.6, "SYSTEM EXECUTION FLOWCHART", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

    # Start
    ax.add_patch(patches.FancyBboxPatch((3.2, 9.9), 2.6, 0.5, boxstyle="round,pad=0.1", facecolor='#1e293b', edgecolor='#0f172a'))
    ax.text(4.5, 10.15, "START APPLICATION", fontsize=10, fontweight='bold', ha='center', va='center', color='#ffffff')

    nodes = [
        ("Register Candidate & Upload Passport Photo", 9.1, "process"),
        ("Extract & Store FaceNet Feature Vectors in SQLite", 8.1, "process"),
        ("Open Webcam & Capture Live Candidate Face", 7.1, "process"),
        ("Compare Face Embeddings (Cosine Distance < 0.50?)", 6.0, "decision"),
        ("Upload Recorded Driving Test Video", 4.7, "process"),
        ("Run YOLOv8, OpenCV Lane & MediaPipe Analytics", 3.7, "process"),
        ("Predict Test Outcome using Random Forest & SVM", 2.7, "process"),
        ("Compile Results & Generate PDF Evaluation Report", 1.7, "process"),
    ]

    for text, y, ntype in nodes:
        if ntype == "process":
            ax.add_patch(patches.Rectangle((1.2, y-0.4), 6.6, 0.8, facecolor='#dbeafe', edgecolor='#2563eb', linewidth=2))
            ax.text(4.5, y, text, fontsize=10.5, fontweight='bold', ha='center', va='center', color='#1e40af')
        elif ntype == "decision":
            diamond = patches.Polygon([[4.5, y+0.55], [6.0, y], [4.5, y-0.55], [3.0, y]], facecolor='#fef08a', edgecolor='#ca8a04', lw=2)
            ax.add_patch(diamond)
            ax.text(4.5, y, text, fontsize=9.5, fontweight='bold', ha='center', va='center', color='#854d0e')

    # End Node
    ax.add_patch(patches.FancyBboxPatch((3.2, 0.5), 2.6, 0.5, boxstyle="round,pad=0.1", facecolor='#15803d', edgecolor='#14532d'))
    ax.text(4.5, 0.75, "END / DOWNLOAD REPORT", fontsize=9.5, fontweight='bold', ha='center', va='center', color='#ffffff')

    # Stop Node for rejection
    ax.add_patch(patches.FancyBboxPatch((0.2, 5.75), 2.2, 0.5, boxstyle="round,pad=0.1", facecolor='#dc2626', edgecolor='#991b1b'))
    ax.text(1.3, 6.0, "REJECT CANDIDATE", fontsize=9, fontweight='bold', ha='center', va='center', color='#ffffff')

    arrowprops = dict(arrowstyle="->,head_width=0.4,head_length=0.6", lw=2.0, color="#334155")
    ax.annotate("", xy=(4.5, 9.5), xytext=(4.5, 9.9), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 8.5), xytext=(4.5, 8.7), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 7.5), xytext=(4.5, 7.7), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 6.55), xytext=(4.5, 6.7), arrowprops=arrowprops)
    
    # Branches
    ax.annotate("", xy=(2.4, 6.0), xytext=(3.0, 6.0), arrowprops=arrowprops) # No
    ax.text(2.7, 6.2, "No", fontsize=11, fontweight='bold', color='#dc2626')

    ax.annotate("", xy=(4.5, 5.1), xytext=(4.5, 5.45), arrowprops=arrowprops) # Yes
    ax.text(4.7, 5.3, "Yes", fontsize=11, fontweight='bold', color='#16a34a')

    ax.annotate("", xy=(4.5, 4.1), xytext=(4.5, 4.3), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 3.1), xytext=(4.5, 3.3), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 2.1), xytext=(4.5, 2.3), arrowprops=arrowprops)
    ax.annotate("", xy=(4.5, 1.0), xytext=(4.5, 1.3), arrowprops=arrowprops)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'system_flowchart.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional system_flowchart.png")

def create_ui_diagram():
    fig, ax = plt.subplots(figsize=(12, 7.5), dpi=300)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.5)
    ax.axis('off')

    ax.text(6.0, 7.0, "USER INTERFACE (UI) LAYOUT & DASHBOARD MOCKUP", fontsize=15, fontweight='bold', ha='center', color='#0f172a')

    # Browser Frame
    ax.add_patch(patches.Rectangle((0.6, 0.5), 10.8, 6.1, facecolor='#0f172a', edgecolor='#334155', lw=2.5))
    
    # Nav bar
    ax.add_patch(patches.Rectangle((0.6, 5.9), 10.8, 0.7, facecolor='#1e293b', edgecolor='#334155', lw=1.5))
    ax.text(1.0, 6.25, "AI Driving Skill Validation System", fontsize=11, fontweight='bold', color='#38bdf8', va='center')
    ax.text(7.2, 6.25, "Dashboard | Candidate Register | Face Verification | Video Analysis", fontsize=9.5, fontweight='bold', color='#94a3b8', va='center')

    # Hero Card Mockup
    ax.add_patch(patches.Rectangle((1.0, 4.3), 10.0, 1.3, facecolor='#1e3a8a', edgecolor='#2563eb', lw=1.8))
    ax.text(1.3, 5.2, "Automated Driving Skill Validation & Evaluation Dashboard", fontsize=12, fontweight='bold', color='#ffffff')
    ax.text(1.3, 4.7, "FaceNet CNN • YOLOv8 Detector • MediaPipe Pose • Random Forest & SVM Classifiers", fontsize=9.5, fontweight='bold', color='#93c5fd')

    # Telemetry Cards Mockup
    cards = [
        ("HELMET DETECTION", "PASS (Worn)", "#15803d", 1.0),
        ("LANE FOLLOWING", "PASS (Inside)", "#15803d", 3.6),
        ("FOOT TOUCH", "PASS (No Touch)", "#15803d", 6.2),
        ("RIDER BALANCE", "PASS (Stable)", "#15803d", 8.8),
    ]

    for title, val, color, x in cards:
        ax.add_patch(patches.Rectangle((x, 2.6), 2.2, 1.4, facecolor='#1e293b', edgecolor='#475569', lw=1.5))
        ax.text(x+1.1, 3.5, title, fontsize=8.5, fontweight='bold', ha='center', color='#94a3b8')
        ax.text(x+1.1, 3.0, val, fontsize=10.5, fontweight='bold', ha='center', color=color)

    # Video Player & Results Card
    ax.add_patch(patches.Rectangle((1.0, 0.8), 6.0, 1.6, facecolor='#1e293b', edgecolor='#475569', lw=1.5))
    ax.text(4.0, 1.6, "[ Annotated Video Analytics Live Telemetry Preview ]", fontsize=10.5, fontweight='bold', ha='center', color='#38bdf8')

    ax.add_patch(patches.Rectangle((7.3, 0.8), 3.7, 1.6, facecolor='#14532d', edgecolor='#22c55e', lw=1.8))
    ax.text(9.15, 1.9, "FINAL RESULT: PASS", fontsize=12, fontweight='bold', ha='center', color='#ffffff')
    ax.text(9.15, 1.4, "Score: 95.0% | RF: PASS | SVM: PASS", fontsize=9.5, fontweight='bold', ha='center', color='#dcfce7')
    ax.text(9.15, 1.05, "[ Download Official PDF Report ]", fontsize=9, fontweight='bold', ha='center', color='#fef08a')

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, 'ui_diagram.png')
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print("Generated professional ui_diagram.png")

if __name__ == '__main__':
    create_system_architecture()
    create_use_case()
    create_activity_diagram()
    create_sequence_diagram()
    create_class_diagram()
    create_er_diagram()
    create_db_schema()
    create_system_flowchart()
    create_ui_diagram()
    print("All 9 professional diagrams generated successfully!")
