import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
import cv2
import numpy as np
from models.yolo_detector import yolo_detector

def detect_road_corridor(frame, moto_bbox=None):
    h, w = frame.shape[:2]
    y_top = int(h * 0.35)

    if frame is None or np.mean(frame) < 5:
        return None, False

    roi = frame[y_top:, :]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # 1. Check for painted lines (test track)
    white_mask = cv2.inRange(hsv_roi, np.array([0, 0, 200]), np.array([180, 45, 255]))
    yellow_mask = cv2.inRange(hsv_roi, np.array([15, 80, 90]), np.array([35, 255, 255]))
    paint_mask = cv2.bitwise_or(white_mask, yellow_mask)

    min_line_len = max(35, int(h * 0.12))
    lines = cv2.HoughLinesP(paint_mask, 1, np.pi / 180, 40, minLineLength=min_line_len, maxLineGap=30)
    painted_left = []
    painted_right = []
    if lines is not None:
        for l in lines:
            x1, y1_rel, x2, y2_rel = l[0]
            y1 = y1_rel + y_top
            y2 = y2_rel + y_top
            dy = y2 - y1
            dx = x2 - x1
            if abs(dy) < 1.2 * abs(dx) or dy == 0:
                continue
            inv_slope = dx / float(dy)
            x_bot = x1 + (h - y1) * inv_slope
            x_top_p = x1 + (y_top - y1) * inv_slope
            if 0.04 * w <= x_bot <= 0.48 * w:
                painted_left.append((x_bot, x_top_p))
            elif 0.52 * w <= x_bot <= 0.96 * w:
                painted_right.append((x_bot, x_top_p))

    if painted_left and painted_right:
        l_bot = float(np.median([c[0] for c in painted_left]))
        l_top = float(np.median([c[1] for c in painted_left]))
        r_bot = float(np.median([c[0] for c in painted_right]))
        r_top = float(np.median([c[1] for c in painted_right]))
        if r_bot > l_bot + 0.15 * w:
            return (l_bot, r_bot, l_top, r_top), True

    # 2. Physical road segmentation: transitions to vegetation or soil shoulder
    full_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Vegetation / green grass mask
    veg = ((full_hsv[:, :, 0] >= 28) & (full_hsv[:, :, 0] <= 88) &
           (full_hsv[:, :, 1] >= 35) & (full_hsv[:, :, 2] >= 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    veg_clean = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # Mask out motorcycle
    if moto_bbox is not None:
        mx1, my1, mx2, my2 = moto_bbox
        veg_clean[max(0, my1):min(h, my2 + 5), max(0, mx1):min(w, mx2)] = 0
        ref_x = int((mx1 + mx2) / 2)
    else:
        ref_x = w // 2

    road_lefts = []
    road_rights = []
    scan_step = max(6, int(h * 0.03))

    for y in range(y_top, h, scan_step):
        row = veg_clean[y, :]

        # Look for non-drivable pixels left of ref_x
        left_non_road = np.where(row[:ref_x] > 0)[0]
        if len(left_non_road) >= 3:
            # Significant cluster of non-road pixels
            l_edge = float(np.max(left_non_road))
        else:
            l_edge = 0.0

        # Look for non-drivable pixels right of ref_x
        right_non_road = np.where(row[ref_x:] > 0)[0]
        if len(right_non_road) >= 3:
            r_edge = float(ref_x + np.min(right_non_road))
        else:
            r_edge = float(w)

        road_lefts.append((l_edge, y))
        road_rights.append((r_edge, y))

    if len(road_lefts) >= 4 and len(road_rights) >= 4:
        y_pts = np.array([p[1] for p in road_lefts])
        p_l = np.polyfit(y_pts, [p[0] for p in road_lefts], 1)
        p_r = np.polyfit(y_pts, [p[0] for p in road_rights], 1)

        l_bot = max(0.0, float(np.polyval(p_l, h)))
        l_top = max(0.0, float(np.polyval(p_l, y_top)))
        r_bot = min(float(w), float(np.polyval(p_r, h)))
        r_top = min(float(w), float(np.polyval(p_r, y_top)))

        if r_bot > l_bot + 0.15 * w:
            return (l_bot, r_bot, l_top, r_top), True

    return (0.20 * w, 0.80 * w, 0.30 * w, 0.70 * w), False

# Test across all frames of suraksha_with_helmet.mp4
cap = cv2.VideoCapture('static/uploads/videos/suraksha_with_helmet.mp4')
f = 0
violations = 0
while True:
    ret, fr = cap.read()
    if not ret: break
    f += 1
    dets, _ = yolo_detector.detect_objects(fr)
    mb = dets.get('motorcycle_bbox')
    corridor, ok = detect_road_corridor(fr, mb)
    if mb and corridor:
        l_bot, r_bot, l_top, r_top = corridor
        h, w = fr.shape[:2]
        y_top = int(h * 0.35)
        my2 = mb[3]
        ratio = (float(my2) - y_top) / float(h - y_top + 1e-5)
        xl = l_top + ratio * (l_bot - l_top)
        xr = r_top + ratio * (r_bot - r_top)
        mx1, my1, mx2, my2 = mb
        if mx1 <= xl or mx2 >= xr:
            violations += 1
            print(f"Frame {f}: moto={mb}, xl={xl:.1f}, xr={xr:.1f}")
cap.release()
print(f"suraksha_with_helmet: {f} frames, violations={violations}")
