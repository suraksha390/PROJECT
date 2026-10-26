import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.yolo_detector import yolo_detector

def extract_road_boundaries(frame):
    h, w = frame.shape[:2]
    y_top = int(h * 0.35)
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Non-drivable surroundings masks
    # 1. Vegetation / grass mask (green hues)
    grass_mask = cv2.inRange(hsv, np.array([30, 45, 30]), np.array([85, 255, 255]))
    
    # 2. Road surface candidate mask:
    # Road is asphalt/concrete: low-to-medium saturation, not grass, in lower 65% of frame
    road_mask = np.zeros((h, w), dtype=np.uint8)
    
    # Exclude grass and sky
    road_candidate = (grass_mask == 0)
    road_candidate[:y_top, :] = False
    
    # Morphological clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    clean_road = cv2.morphologyEx(road_candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    
    # Scan horizontal profiles from y_top to h to find left and right road edges
    left_pts = []
    right_pts = []
    
    scan_step = max(5, int(h * 0.03))
    for y in range(y_top, h, scan_step):
        row = clean_road[y, :]
        # Find largest continuous segment of road containing center or near center
        road_indices = np.where(row > 0)[0]
        if len(road_indices) > (w * 0.20):  # At least 20% of width is road
            l_edge = np.min(road_indices)
            r_edge = np.max(road_indices)
            left_pts.append((float(l_edge), float(y)))
            right_pts.append((float(r_edge), float(y)))
            
    if len(left_pts) >= 4 and len(right_pts) >= 4:
        # Fit linear or smooth boundary: x = m * y + c
        y_vals = np.array([p[1] for p in left_pts])
        l_x = np.array([p[0] for p in left_pts])
        r_x = np.array([p[0] for p in right_pts])
        
        # Robust polynomial/linear fit: x(y)
        p_l = np.polyfit(y_vals, l_x, 1)
        p_r = np.polyfit(y_vals, r_x, 1)
        
        l_top_x = int(np.polyval(p_l, y_top))
        l_bot_x = int(np.polyval(p_l, h))
        r_top_x = int(np.polyval(p_r, y_top))
        r_bot_x = int(np.polyval(p_r, h))
        
        # Clamp to reasonable frame boundaries
        l_top_x = max(0, min(w - 50, l_top_x))
        l_bot_x = max(0, min(w - 50, l_bot_x))
        r_top_x = min(w, max(l_top_x + 50, r_top_x))
        r_bot_x = min(w, max(l_bot_x + 50, r_bot_x))
        
        return {
            'is_reliable': True,
            'left_top': (l_top_x, y_top),
            'left_bot': (l_bot_x, h),
            'right_top': (r_top_x, y_top),
            'right_bot': (r_bot_x, h),
            'x_left_fn': lambda y_val: np.polyval(p_l, y_val),
            'x_right_fn': lambda y_val: np.polyval(p_r, y_val),
            'clean_road': clean_road
        }
    return {'is_reliable': False}

# Test on 5 frames of sinch
cap = cv2.VideoCapture('static/uploads/videos/test_CAND-2026-063_sinch.mp4')
for f_idx in [20, 60, 100, 120, 150, 180]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
    ret, frame = cap.read()
    if ret:
        res = extract_road_boundaries(frame)
        print(f"Frame {f_idx:3d}: reliable={res['is_reliable']}, Left=[{res['left_bot'][0]},{res['left_top'][0]}], Right=[{res['right_bot'][0]},{res['right_top'][0]}]")
cap.release()
