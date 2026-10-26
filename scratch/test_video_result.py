import os
import sys
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.yolo_detector import yolo_detector
from config import VIDEO_HELMET_PASS_RATIO, VIDEO_VIOLATION_RATIO

video_path = 'static/uploads/videos/test_CAND-2026-061_sinchana_driving_with_helmet.mp4'
cap = cv2.VideoCapture(video_path)

frame_count = 0
stride = 3
total_rider_frames = 0
helmet_pass_count = 0
no_helmet_count = 0
other_headwear_count = 0
helmet_review_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % stride != 0:
        continue
        
    dets, _ = yolo_detector.detect_objects(frame)
    if dets['rider_found']:
        total_rider_frames += 1
        h_cls = dets.get('helmet_class', 'none')
        h_stat = dets.get('helmet_status', 'FAIL')

        if h_cls == 'helmet' and h_stat == 'PASS':
            helmet_pass_count += 1
        elif h_cls == 'other_headwear':
            other_headwear_count += 1
        elif h_cls == 'no_helmet':
            no_helmet_count += 1
        elif h_stat == 'Needs Review':
            helmet_review_count += 1

cap.release()

helmet_pass_ratio = helmet_pass_count / total_rider_frames
violation_ratio = (other_headwear_count + no_helmet_count) / total_rider_frames
review_ratio = helmet_review_count / total_rider_frames

print(f"Total processed rider frames: {total_rider_frames}")
print(f"Helmet PASS: {helmet_pass_count} ({helmet_pass_ratio*100:.1f}%)")
print(f"Other headwear: {other_headwear_count}")
print(f"No helmet: {no_helmet_count}")
print(f"Review count: {helmet_review_count}")

if other_headwear_count >= 2 and (other_headwear_count / total_rider_frames) >= VIDEO_VIOLATION_RATIO:
    state = 'FAIL (Other Headwear)'
elif no_helmet_count >= 2 and (no_helmet_count / total_rider_frames) >= VIDEO_VIOLATION_RATIO:
    state = 'FAIL (No Helmet)'
elif helmet_pass_ratio >= VIDEO_HELMET_PASS_RATIO and violation_ratio < 0.15:
    state = 'PASS'
elif review_ratio >= 0.35:
    state = 'Needs Review'
else:
    state = 'Needs Review (Inconsistent)'

print(f"FINAL HELMET RESULT: {state}")
