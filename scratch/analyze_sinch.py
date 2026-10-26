import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from models.yolo_detector import yolo_detector

video_path = 'static/uploads/videos/test_CAND-2026-062_sinch.mp4'
if not os.path.exists(video_path):
    video_path = '../sinch.mp4'

cap = cv2.VideoCapture(video_path)
frame_idx = 0
rider_frames = 0
os.makedirs('scratch/sinch_frames', exist_ok=True)

stats = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    dets, _ = yolo_detector.detect_objects(frame)
    if dets['rider_found']:
        rider_frames += 1
        cls = dets['helmet_class']
        stat = dets['helmet_status']
        stats[stat] = stats.get(stat, 0) + 1
        
        if frame_idx % 20 == 0 or frame_idx in [30, 60, 90, 120, 150, 180]:
            print(f"Frame {frame_idx:3d}: cls={cls:12s} stat={stat:12s} conf={dets['helmet_conf']:.2f} details={dets['helmet_details']}")
            rx1, ry1, rx2, ry2 = dets['rider_bbox']
            head_h = max(1, int((ry2 - ry1) * 0.28))
            head_roi = frame[ry1:min(ry2, ry1 + head_h), rx1:rx2]
            cv2.imwrite(f'scratch/sinch_frames/head_{frame_idx}.jpg', head_roi)

cap.release()
print(f"\nTotal frames: {frame_idx}, Rider frames: {rider_frames}")
print("Status counts across video:", stats)
