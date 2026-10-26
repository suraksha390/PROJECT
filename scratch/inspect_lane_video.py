import cv2
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.lane_detector import LaneDetector
from models.yolo_detector import yolo_detector

video_path = 'static/uploads/videos/test_CAND-2026-063_sinch.mp4'
if not os.path.exists(video_path):
    video_path = 'static/uploads/videos/test_CAND-2026-062_sinch.mp4'

cap = cv2.VideoCapture(video_path)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Video: {video_path}, Total Frames: {total_frames}, FPS: {fps}, Dimensions: {w}x{h}")

os.makedirs('scratch/lane_inspection', exist_ok=True)

detector = LaneDetector()
frame_idx = 0
violations = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    dets, _ = yolo_detector.detect_objects(frame)
    is_viol, ann = detector.analyze_lane(
        frame,
        rider_bbox=dets.get('rider_bbox'),
        motorcycle_bbox=dets.get('motorcycle_bbox'),
        frame_idx=frame_idx,
        fps=fps
    )
    if is_viol:
        violations.append(frame_idx)
        
    if frame_idx in [15, 30, 60, 90, 120, 150, 180, 200]:
        cv2.imwrite(f'scratch/lane_inspection/frame_{frame_idx}.jpg', frame)
        cv2.imwrite(f'scratch/lane_inspection/ann_{frame_idx}.jpg', ann)

cap.release()
print(f"Total violations frames: {len(violations)}, list of frames: {violations[:20]} ... (total {len(violations)})")
summary = detector.get_summary()
print("Detector summary:", summary)
