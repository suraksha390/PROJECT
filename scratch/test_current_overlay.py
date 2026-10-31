import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector
from models.pose_analyzer import pose_analyzer

video_path = "static/uploads/videos/test_CAND-2026-070_suraksha1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

# Read to frame 60
for _ in range(60):
    ret, frame = cap.read()
    if not ret:
        break

lane_detector.reset()
yolo_dets, yolo_frame = yolo_detector.detect_objects(frame)
lane_viol, lane_frame = lane_detector.analyze_lane(
    yolo_frame,
    rider_bbox=yolo_dets['rider_bbox'],
    motorcycle_bbox=yolo_dets['motorcycle_bbox'],
    frame_idx=60,
    fps=fps
)
pose_info, final_frame = pose_analyzer.analyze_pose(lane_frame, yolo_dets['ground_line_y'])
cap.release()

cv2.imwrite("scratch/test_current_overlay.jpg", final_frame)
print("Saved scratch/test_current_overlay.jpg")
print("YOLO dets:", yolo_dets['rider_found'], yolo_dets.get('helmet_class'), yolo_dets.get('helmet_status'), yolo_dets.get('helmet_conf'))
print("Lane info:", lane_viol)
print("Pose info:", pose_info)
