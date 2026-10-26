import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from scratch.test_lane_logic import AdvancedLaneDetector
from models.yolo_detector import yolo_detector

cap = cv2.VideoCapture('static/uploads/videos/test_CAND-2026-063_sinch.mp4')
det = AdvancedLaneDetector()
f = 0
while True:
    ret, fr = cap.read()
    if not ret:
        break
    f += 1
    dets, _ = yolo_detector.detect_objects(fr)
    v, _ = det.analyze_lane(fr, motorcycle_bbox=dets.get('motorcycle_bbox'), rider_bbox=dets.get('rider_bbox'), frame_idx=f, fps=24.0)
    if v:
        bbox = det.last_motorcycle_bbox
        bounds = det.detect_lane_boundaries(fr)
        mx1, my1, mx2, my2 = bbox
        xl = bounds['x_left_fn'](my2)
        xr = bounds['x_right_fn'](my2)
        print(f"Frame {f:3d}: VIOLATION on {det.active_violation.get('boundary')} | Moto=({mx1},{my1},{mx2},{my2}) | xl={xl:.1f}, xr={xr:.1f}")
cap.release()
