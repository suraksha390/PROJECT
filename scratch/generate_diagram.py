import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def create_architecture_diagram(output_path, artifact_path=None):
    # Set high DPI and large figure for exceptional clarity
    fig, ax = plt.subplots(figsize=(20, 15), dpi=300)
    fig.patch.set_facecolor('#FFFFFF')
    ax.set_facecolor('#FFFFFF')
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 15)
    ax.axis('off')

    # Color Palette - Professional Modern Tech Palette
    C_BG_CANVAS = '#FFFFFF'
    C_TEXT_MAIN = '#0F172A'
    C_TEXT_MUTED = '#475569'
    C_TEXT_ACCENT = '#1E3A8A'

    # Layer container colors (very subtle backgrounds with crisp borders)
    LAYERS = [
        {
            "name": "1. PRESENTATION & CLIENT UI LAYER",
            "y": 12.0, "h": 1.7,
            "bg": "#F8FAFC", "border": "#94A3B8",
            "tag_color": "#1E293B", "tag_bg": "#E2E8F0"
        },
        {
            "name": "2. FLASK WEB APPLICATION & API CONTROLLER LAYER",
            "y": 9.6, "h": 1.7,
            "bg": "#F0FDF4", "border": "#86EFAC",
            "tag_color": "#14532D", "tag_bg": "#DCFCE7"
        },
        {
            "name": "3. COMPUTER VISION & MULTI-MODAL AI INFERENCE PIPELINE",
            "y": 5.4, "h": 3.5,
            "bg": "#F5F3FF", "border": "#C4B5FD",
            "tag_color": "#4C1D95", "tag_bg": "#EDE9FE"
        },
        {
            "name": "4. TEMPORAL AGGREGATION & MACHINE LEARNING EVALUATION LAYER",
            "y": 2.8, "h": 1.9,
            "bg": "#FEF3C7", "border": "#FCD34D",
            "tag_color": "#78350F", "tag_bg": "#FDE68A"
        },
        {
            "name": "5. DATA PERSISTENCE & REPORT GENERATION LAYER",
            "y": 0.4, "h": 1.7,
            "bg": "#EFF6FF", "border": "#93C5FD",
            "tag_color": "#1E3A8A", "tag_bg": "#DBEAFE"
        }
    ]

    # Draw Title Header
    ax.text(10.0, 14.55, "TWO-WHEELER DRIVING SKILL VALIDATION SYSTEM",
            ha='center', va='center', fontsize=18, fontweight='bold', color='#0F172A',
            fontfamily='sans-serif')
    ax.text(10.0, 14.2, "High-Level System Architecture & End-to-End AI/CV Inference Pipeline",
            ha='center', va='center', fontsize=12, fontweight='medium', color='#64748B',
            fontfamily='sans-serif')
    
    # Draw horizontal rule below title
    ax.plot([1.0, 19.0], [14.0, 14.0], color='#E2E8F0', lw=1.5)

    # Function to draw container box
    def draw_container(x, y, w, h, bg, border, label, tag_color, tag_bg):
        box = FancyBboxPatch((x, y), w, h,
                             boxstyle="round,pad=0.08,rounding_size=0.18",
                             facecolor=bg, edgecolor=border, lw=1.5, zorder=1)
        ax.add_patch(box)
        # Header Badge
        badge = FancyBboxPatch((x + 0.3, y + h - 0.32), 6.0, 0.28,
                               boxstyle="round,pad=0.04,rounding_size=0.08",
                               facecolor=tag_bg, edgecolor=border, lw=1.0, zorder=2)
        ax.add_patch(badge)
        ax.text(x + 0.4, y + h - 0.18, label,
                ha='left', va='center', fontsize=8.5, fontweight='bold', color=tag_color,
                fontfamily='sans-serif', zorder=3)

    # Draw Layer Containers
    for l in LAYERS:
        draw_container(0.8, l["y"], 18.4, l["h"], l["bg"], l["border"],
                       l["name"], l["tag_color"], l["tag_bg"])

    # Function to draw inner module cards
    def draw_card(x, y, w, h, title, items, badge_text=None, border_color="#3B82F6", fill_color="#FFFFFF", title_color="#1E293B"):
        # Card shadow/border
        card = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.06,rounding_size=0.12",
                              facecolor=fill_color, edgecolor=border_color, lw=1.4, zorder=3)
        ax.add_patch(card)

        # Title
        ax.text(x + 0.18, y + h - 0.22, title,
                ha='left', va='center', fontsize=9.2, fontweight='bold', color=title_color,
                fontfamily='sans-serif', zorder=4)

        # Badge if present
        if badge_text:
            bw = len(badge_text) * 0.08 + 0.25
            badge = FancyBboxPatch((x + w - bw - 0.15, y + h - 0.28), bw, 0.18,
                                   boxstyle="round,pad=0.02,rounding_size=0.05",
                                   facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=0.8, zorder=4)
            ax.add_patch(badge)
            ax.text(x + w - 0.15 - bw/2, y + h - 0.19, badge_text,
                    ha='center', va='center', fontsize=6.8, fontweight='bold', color="#1D4ED8",
                    fontfamily='sans-serif', zorder=5)

        # Divider line
        ax.plot([x + 0.15, x + w - 0.15], [y + h - 0.36, y + h - 0.36],
                color='#E2E8F0', lw=0.8, zorder=4)

        # Items / Description
        line_y = y + h - 0.54
        for item in items:
            ax.text(x + 0.18, line_y, f"• {item}",
                    ha='left', va='center', fontsize=7.6, color='#334155',
                    fontfamily='sans-serif', zorder=4)
            line_y -= 0.22

    # --- LAYER 1 MODULES (PRESENTATION & UI) ---
    draw_card(1.1, 12.15, 3.9, 1.15,
              "Candidate Registration",
              ["Webcam live face capture", "User metadata & photo storage", "Candidate ID generation"],
              badge_text="HTML5 / JS", border_color="#64748B")

    draw_card(5.4, 12.15, 4.0, 1.15,
              "Biometric Authentication",
              ["Real-time video auth stream", "Pre-test identity verification", "Face match confirmation"],
              badge_text="Webcam API", border_color="#64748B")

    draw_card(9.8, 12.15, 4.4, 1.15,
              "Video Analysis & Monitoring",
              ["Test video upload / recording", "Live bounding box overlay playback", "Real-time metrics & warnings HUD"],
              badge_text="HTML5 Canvas", border_color="#64748B")

    draw_card(14.6, 12.15, 4.3, 1.15,
              "Evaluation Dashboard & PDF",
              ["Skill score breakdown & mistakes", "Helmet, Lane, Feet, Balance status", "Downloadable PDF certificate"],
              badge_text="Dashboard UI", border_color="#64748B")

    # --- LAYER 2 MODULES (FLASK APPLICATION & API GATEWAY) ---
    draw_card(1.1, 9.75, 4.1, 1.15,
              "Authentication Controller",
              ["Route: /api/authenticate_face", "Candidate face frame ingestion", "Cosine similarity comparator"],
              badge_text="app.py", border_color="#059669")

    draw_card(5.6, 9.75, 4.6, 1.15,
              "Video Ingestion & Stream Dispatcher",
              ["Route: /api/analyze_video", "Video file reader & frame decoder", "Sequential frame batch scheduler"],
              badge_text="Flask Core", border_color="#059669")

    draw_card(10.6, 9.75, 4.4, 1.15,
              "Pipeline Orchestrator & Annotator",
              ["Synchronizes YOLO, Lane, Pose & Helmet", "Draws dynamic HUD & detection bounding boxes", "Encodes output MP4 annotated video"],
              badge_text="OpenCV / VideoWriter", border_color="#059669")

    draw_card(15.4, 9.75, 3.5, 1.15,
              "Reporting & Export API",
              ["Route: /report/<candidate_id>", "Aggregated evaluation query", "PDF generation trigger"],
              badge_text="Report Engine", border_color="#059669")

    # --- LAYER 3 MODULES (COMPUTER VISION & AI INFERENCE PIPELINE) ---
    # 5 specialized AI pipelines
    draw_card(1.1, 5.55, 3.2, 2.9,
              "Face Verification Engine",
              [
                  "Module: face_detector.py",
                  "Inception-Resnet-V1 / FaceNet",
                  "128-d face embedding vectors",
                  "Euclidean / Cosine metric",
                  "Verification Threshold (0.60)",
                  "Anti-spoof / Multi-face check",
                  "Outputs: Auth Pass / Fail"
              ],
              badge_text="PyTorch / FaceNet", border_color="#7C3AED")

    draw_card(4.6, 5.55, 3.4, 2.9,
              "Object & Rider Detector",
              [
                  "Module: yolo_detector.py",
                  "YOLOv8 / YOLOv8-Nano Model",
                  "Person (Rider) detection",
                  "Motorcycle bounding box",
                  "Spatial overlap association",
                  "Rider head ROI extraction",
                  "Confidence threshold: 0.50",
                  "Outputs: Rider & Head Coordinates"
              ],
              badge_text="Ultralytics YOLOv8", border_color="#7C3AED")

    draw_card(8.3, 5.55, 3.6, 2.9,
              "Helmet Decision Engine",
              [
                  "Enhanced Rule & ROI Classifier",
                  "Visor & brim edge curvature",
                  "Dome specular highlight check",
                  "Bare skin / hair color filter",
                  "Non-protective cap exclusion",
                  "Configurable Threshold (0.85)",
                  "• >= 95%: Strong Helmet (PASS)",
                  "• 85%-94%: Accepted Helmet",
                  "• < 85%: Needs Review",
                  "• Cap / Uncovered: FAIL"
              ],
              badge_text="Config: 0.85 Conf", border_color="#D97706", fill_color="#FFFBEB")

    draw_card(12.2, 5.55, 3.3, 2.9,
              "Lane Tracking Engine",
              [
                  "Module: lane_detector.py",
                  "Color masking (White / Yellow)",
                  "Canny Edge + Hough Transform",
                  "Perspective warp / Bird's eye",
                  "Polynomial curve fitting",
                  "Lateral lane deviation calculation",
                  "Boundary breach detector",
                  "Outputs: Lane discipline status"
              ],
              badge_text="OpenCV Lane", border_color="#7C3AED")

    draw_card(15.8, 5.55, 3.1, 2.9,
              "Pose & Balance Engine",
              [
                  "Module: pose_detector.py",
                  "MediaPipe BlazePose 33 landmarks",
                  "Rider torso angle tracking",
                  "Lateral tilt & balance score",
                  "Ankle / foot ground touch check",
                  "Foot-down event count",
                  "Braking control posture",
                  "Outputs: Balance & Foot Touches"
              ],
              badge_text="MediaPipe Pose", border_color="#7C3AED")

    # --- LAYER 4 MODULES (EVALUATION, AGGREGATION & ML) ---
    draw_card(1.1, 2.95, 4.3, 1.45,
              "Temporal Frame Aggregator",
              [
                  "Multi-frame sliding window voting",
                  "Rejects isolated blurred / uncertain frames",
                  "Helmet confidence persistence tracking",
                  "Accumulates foot touch & lane deviation counts"
              ],
              badge_text="Voting Logic", border_color="#D97706")

    draw_card(5.8, 2.95, 4.6, 1.45,
              "Deterministic Rule Auditor",
              [
                  "Mistake categorization & logging into JSON",
                  "Helmet compliance verification (Pass/Fail/Review)",
                  "Lane breach penalty calculation",
                  "Excessive foot-down penalty deduction"
              ],
              badge_text="Rules Engine", border_color="#D97706")

    draw_card(10.8, 2.95, 4.3, 1.45,
              "ML Skill Score Classifier",
              [
                  "Module: ml_evaluator.py",
                  "Random Forest / SVM Classifier",
                  "Trained on composite driving metrics",
                  "Generates composite Skill Score (0 - 100%)"
              ],
              badge_text="Scikit-Learn", border_color="#D97706")

    draw_card(15.5, 2.95, 3.4, 1.45,
              "Final Verdict Generator",
              [
                  "Overall Status: PASS / FAIL",
                  "Conditional certificate grant",
                  "Detailed sub-skill grades",
                  "Remarks & retry recommendation"
              ],
              badge_text="Decision Hub", border_color="#D97706")

    # --- LAYER 5 MODULES (DATA STORAGE & REPORTING) ---
    draw_card(1.1, 0.55, 4.8, 1.3,
              "SQLite Relational Database",
              [
                  "database.db: candidates & evaluations tables",
                  "Candidate credentials, registration timestamp & photo path",
                  "Test metrics: helmet status, lane deviations, foot touches, score, mistakes_json"
              ],
              badge_text="SQLite 3", border_color="#2563EB")

    draw_card(6.3, 0.55, 4.5, 1.3,
              "File & Video Asset Storage",
              [
                  "static/uploads/candidates/ (Reference biometric photos)",
                  "static/uploads/videos/ (Raw input driving videos)",
                  "static/uploads/annotated/ (Processed videos with HUD bounding boxes)"
              ],
              badge_text="Local FileSystem", border_color="#2563EB")

    draw_card(11.2, 0.55, 4.0, 1.3,
              "Report Lab PDF Generator",
              [
                  "Module: pdf_generator.py",
                  "Generates tamper-evident Driving Test Certificate",
                  "Includes candidate details, score card, and timestamps"
              ],
              badge_text="ReportLab PDF", border_color="#2563EB")

    draw_card(15.6, 0.55, 3.3, 1.3,
              "Configuration Manager",
              [
                  "config.py & Environment variables",
                  "Configurable HELMET_CONF_THRESHOLD (0.85)",
                  "Detection thresholds & weights paths"
              ],
              badge_text="Config Engine", border_color="#2563EB")

    # --- CONNECTING ARROWS & DATA FLOWS ---
    def draw_arrow(start, end, text=None, text_offset=(0, 0), color="#475569", rad=0.0, lw=1.6):
        arrow = FancyArrowPatch(start, end,
                                arrowstyle="-|>",
                                mutation_scale=14,
                                connectionstyle=f"arc3,rad={rad}",
                                color=color,
                                lw=lw,
                                zorder=10)
        ax.add_patch(arrow)
        if text:
            mid_x = (start[0] + end[0]) / 2 + text_offset[0]
            mid_y = (start[1] + end[1]) / 2 + text_offset[1]
            ax.text(mid_x, mid_y, text,
                    ha='center', va='center', fontsize=7.2, fontweight='bold',
                    color='#1E293B', backgroundcolor='#FFFFFF',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='#FFFFFF', edgecolor='#CBD5E1', lw=0.6),
                    zorder=11)

    # UI -> API Layer Arrows
    draw_arrow((3.1, 12.15), (3.1, 10.9), "Registration Data", lw=1.5)
    draw_arrow((7.4, 12.15), (7.4, 10.9), "Webcam Stream", lw=1.5)
    draw_arrow((12.0, 12.15), (12.0, 10.9), "Video Stream / File", lw=1.5)
    draw_arrow((16.8, 10.9), (16.8, 12.15), "Results & PDF Link", color="#059669", lw=1.5)

    # API -> AI Core Arrows
    draw_arrow((3.1, 9.75), (2.7, 8.45), "Candidate Frame", lw=1.5)
    draw_arrow((7.9, 9.75), (6.3, 8.45), "Raw Frame Feed", lw=1.5)
    draw_arrow((7.9, 9.75), (10.1, 8.45), "Head Crop / Frame", lw=1.5)
    draw_arrow((7.9, 9.75), (13.8, 8.45), "Road ROI Frame", lw=1.5)
    draw_arrow((7.9, 9.75), (17.3, 8.45), "Rider Skeleton", lw=1.5)

    # AI Pipeline Inter-connection (YOLO Head crop -> Helmet Engine)
    draw_arrow((8.0, 7.0), (8.3, 7.0), "Head ROI", color="#D97706", lw=1.8)

    # AI Core -> Layer 4 (Aggregation & ML) Arrows
    draw_arrow((2.7, 5.55), (3.2, 4.4), "Face Match Result", lw=1.5)
    draw_arrow((6.3, 5.55), (3.2, 4.4), "Rider Presence", lw=1.5)
    draw_arrow((10.1, 5.55), (8.1, 4.4), "Helmet Detections & Confidence", color="#D97706", lw=1.8)
    draw_arrow((13.8, 5.55), (8.1, 4.4), "Lane Crossings", lw=1.5)
    draw_arrow((17.3, 5.55), (12.9, 4.4), "Foot Touches & Tilt", lw=1.5)

    # Layer 4 internal flow
    draw_arrow((5.4, 3.65), (5.8, 3.65), lw=1.5)
    draw_arrow((10.4, 3.65), (10.8, 3.65), lw=1.5)
    draw_arrow((15.1, 3.65), (15.5, 3.65), lw=1.5)

    # Layer 4 -> Layer 5 (Storage & PDF) Arrows
    draw_arrow((3.5, 2.95), (3.5, 1.85), "Candidate & Score Sync", lw=1.5)
    draw_arrow((8.1, 2.95), (8.5, 1.85), "Video Metadata & Logs", lw=1.5)
    draw_arrow((17.2, 2.95), (13.2, 1.85), "Final Assessment Record", color="#2563EB", lw=1.5)

    # Config manager flow upwards to Helmet and YOLO
    draw_arrow((17.25, 1.85), (17.25, 2.95), "Thresholds", color="#475569", lw=1.2)
    draw_arrow((17.25, 4.4), (10.1, 5.55), "0.85 Threshold", color="#D97706", rad=0.2, lw=1.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight')
    if artifact_path:
        plt.savefig(artifact_path, dpi=300, facecolor='#FFFFFF', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Architecture diagram successfully saved to {output_path}")

if __name__ == "__main__":
    out_dir = r"c:\Users\SURAKSHA SHETTY\OneDrive\Desktop\project"
    out_file = os.path.join(out_dir, "architecture_diagram.png")
    art_file = r"C:\Users\SURAKSHA SHETTY\.gemini\antigravity\brain\e0a66d50-9dd4-4411-9c28-92b9be0b5225\architecture_diagram.png"
    create_architecture_diagram(out_file, art_file)
