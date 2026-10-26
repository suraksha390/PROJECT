import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
import cv2
from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector

cap = cv2.VideoCapture('static/uploads/videos/suraksha_with_helmet.mp4')
cap.set(cv2.CAP_PROP_POS_FRAMES, 118)
for f in range(118, 130):
    ret, fr = cap.read()
    if not ret: break
    dets, _ = yolo_detector.detect_objects(fr)
    mb = dets.get('motorcycle_bbox')
    rb = dets.get('rider_bbox')
    b = lane_detector.detect_lane_boundaries(fr, mb)
    print("Frame", f, "moto:", mb, "rider:", rb, "helmet:", dets.get('helmet_class'), dets.get('helmet_status'), "l_bot:", b['left_bot'], "r_bot:", b['right_bot'])
cap.release()
