import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
import cv2
import numpy as np

cap = cv2.VideoCapture('static/uploads/videos/suraksha_with_helmet.mp4')
f = 0
violations = 0
while True:
    ret, fr = cap.read()
    if not ret: break
    f += 1
    h, w = fr.shape[:2]
    
    # Let's use simple ground-level road boundary evaluation at my2
    # First, let's find non-drivable vegetation
    full_hsv = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    veg = ((full_hsv[:, :, 0] >= 28) & (full_hsv[:, :, 0] <= 88) &
           (full_hsv[:, :, 1] >= 35) & (full_hsv[:, :, 2] >= 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    veg_clean = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # YOLO detector
    from models.yolo_detector import yolo_detector
    dets, _ = yolo_detector.detect_objects(fr)
    mb = dets.get('motorcycle_bbox')
    if mb:
        mx1, my1, mx2, my2 = mb
        cx = int((mx1 + mx2) / 2)
        tyre_y = min(h - 1, my2)
        
        # Check non-drivable row at tyre_y
        # Exclude motorcycle box from veg mask
        row_veg = veg_clean[tyre_y, :].copy()
        row_veg[max(0, mx1):min(w, mx2)] = 0
        
        # Left road edge at this y: rightmost veg pixel < cx
        left_non_road = np.where(row_veg[:cx] > 0)[0]
        l_edge = float(np.max(left_non_road)) if len(left_non_road) > 2 else 0.0
        
        # Right road edge at this y: leftmost veg pixel > cx
        right_non_road = np.where(row_veg[cx:] > 0)[0]
        r_edge = float(cx + np.min(right_non_road)) if len(right_non_road) > 2 else float(w)
        
        # Footprint
        fp_left = float(mx1)
        fp_right = float(mx2)
        
        if fp_left <= l_edge or fp_right >= r_edge:
            violations += 1
            print(f"Frame {f}: moto=[{mx1}, {mx2}], road=[{l_edge:.1f}, {r_edge:.1f}], tyre_y={tyre_y}")
cap.release()
print(f"Total frames: {f}, Ground-level violations: {violations}")
