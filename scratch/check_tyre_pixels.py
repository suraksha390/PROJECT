import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
from models.yolo_detector import yolo_detector

cap = cv2.VideoCapture('static/uploads/videos/test_CAND-2026-063_sinch.mp4')
f_idx = 0

print("Frame | Moto Bbox | Bottom Center | HSV Under Tyre | Underneath Category")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    f_idx += 1
    if f_idx % 10 == 0 or f_idx in [108, 115, 120, 195]:
        dets, _ = yolo_detector.detect_objects(frame)
        if dets['motorcycle_found']:
            mx1, my1, mx2, my2 = dets['motorcycle_bbox']
            # Tyre contact point is at the bottom of the motorcycle bbox
            # Bounding box bottom center
            cx = (mx1 + mx2) // 2
            cy = min(frame.shape[0] - 1, my2)
            
            # Sample 10x10 patch under tyre
            y1 = max(0, cy - 10)
            y2 = min(frame.shape[0], cy + 2)
            x1 = max(0, cx - 15)
            x2 = min(frame.shape[1], cx + 15)
            patch = frame[y1:y2, x1:x2]
            
            hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            mean_hsv = hsv_patch.mean(axis=(0,1)).astype(int)
            h, s, v = mean_hsv
            
            # Check if grass (H in 32-85, S > 40)
            is_grass = (32 <= h <= 85 and s > 40)
            cat = "GRASS/VEG" if is_grass else "ASPHALT/ROAD"
            print(f"F{f_idx:3d} | ({mx1:3d},{my1:3d},{mx2:3d},{my2:3d}) | cx={cx:3d}, cy={cy:3d} | HSV=[{h:2d},{s:3d},{v:3d}] | {cat}")
cap.release()
