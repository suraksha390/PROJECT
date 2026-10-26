import cv2
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.lane_detector import LaneDetector
from models.yolo_detector import yolo_detector

cap = cv2.VideoCapture('static/uploads/videos/test_CAND-2026-063_sinch.mp4')
detector = LaneDetector()

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    
    dets, _ = yolo_detector.detect_objects(frame)
    moto_bbox = dets.get('motorcycle_bbox')
    rider_bbox = dets.get('rider_bbox')
    
    # Analyze
    boundaries = detector.detect_lane_boundaries(frame)
    tracked_moto = detector.update_motorcycle_track(motorcycle_bbox=moto_bbox, rider_bbox=rider_bbox)
    contact = detector.check_boundary_contact(tracked_moto, boundaries, frame_idx=frame_idx, fps=24.0)
    
    if 105 <= frame_idx <= 125:
        h, w = frame.shape[:2]
        mx1, my1, mx2, my2 = tracked_moto if tracked_moto else (0,0,0,0)
        xl = boundaries['x_left_fn'](my2)
        xr = boundaries['x_right_fn'](my2)
        print(f"Frame {frame_idx:3d}: Moto=({mx1},{my1},{mx2},{my2}) | xl={xl:.1f}, xr={xr:.1f} | dist_l={contact['dist_left']:.1f}, dist_r={contact['dist_right']:.1f} | viol={contact['is_violating']} ({contact['boundary_touched']})")

cap.release()
