import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.yolo_detector import yolo_detector

cap = cv2.VideoCapture('static/uploads/videos/test_CAND-2026-060_cap_video.mp4')
for i in [15, 45, 90, 120]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if ret:
        dets, _ = yolo_detector.detect_objects(frame)
        print(f"Cap video Frame {i}: rider_found={dets['rider_found']}, rider_bbox={dets['rider_bbox']}, cls={dets['helmet_class']}, stat={dets['helmet_status']}")
cap.release()
