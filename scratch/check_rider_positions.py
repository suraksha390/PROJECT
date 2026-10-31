import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
from models.yolo_detector import yolo_detector

cap = cv2.VideoCapture("static/uploads/videos/test_CAND-2026-070_suraksha1.mp4")
frame_idx = 0
coords = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 15 == 0:
        dets, _ = yolo_detector.detect_objects(frame)
        if dets['rider_found']:
            coords.append((frame_idx, dets['rider_bbox'], dets['helmet_bbox']))

cap.release()
print(f"Total sampled frames with rider: {len(coords)}")
for c in coords[:10]:
    print("Frame", c[0], "Rider:", c[1], "Helmet:", c[2])
