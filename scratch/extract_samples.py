import cv2
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.yolo_detector import yolo_detector

def extract_head_samples(video_path, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        idx += 1
        if idx % 15 == 0 and saved < 8:
            dets, _ = yolo_detector.detect_objects(frame)
            if dets['rider_found']:
                rx1, ry1, rx2, ry2 = dets['rider_bbox']
                head_h = max(1, int((ry2 - ry1) * 0.28))
                head_roi = frame[ry1:min(ry2, ry1 + head_h), rx1:rx2]
                cv2.imwrite(os.path.join(out_dir, f"{prefix}_{idx}.jpg"), head_roi)
                saved += 1
    cap.release()
    print(f"Extracted {saved} samples from {video_path}")

extract_head_samples('static/uploads/videos/test_CAND-2026-062_sinch.mp4', 'scratch/samples', 'sinch_helmet')
extract_head_samples('static/uploads/videos/test_CAND-2026-060_cap_video.mp4', 'scratch/samples', 'cap_video')
