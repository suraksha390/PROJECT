import cv2
import numpy as np
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.yolo_detector import yolo_detector

class DrivableRoadDetector:
    def __init__(self):
        self.lane_color = (140, 180, 210)  # Light brown BGR
        self.reset()
        
    def reset(self):
        self.total_frames = 0
        self.road_detected_frames = 0
        self.motorcycle_detected_frames = 0
        self.violations = []
        self.consecutive_violations = 0
        self.active_violation = None
        self.smoothed_left_bot = None
        self.smoothed_left_top = None
        self.smoothed_right_bot = None
        self.smoothed_right_top = None
        self.ema_alpha = 0.20
        self.last_moto_bbox = None
        self.frames_since_moto = 999
        
    def detect_drivable_road(self, frame):
        h, w = frame.shape[:2]
        y_top = int(h * 0.35)
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Non-drivable surroundings: Vegetation / Grass mask
        # Green hue range: 30 to 85, Saturation > 40, Value > 30
        grass_mask = cv2.inRange(hsv, np.array([30, 42, 30], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8))
        
        # 2. Road surface candidate
        road_candidate = (grass_mask == 0)
        road_candidate[:y_top, :] = False
        
        # Morphological filtering to close internal cracks, tyre marks, shadows
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        clean_road = cv2.morphologyEx(road_candidate.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        
        # 3. Also check for painted solid road boundary lines (if road is lined)
        lower_white = np.array([0, 0, 175], dtype=np.uint8)
        upper_white = np.array([180, 50, 255], dtype=np.uint8)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        lower_yellow = np.array([15, 80, 90], dtype=np.uint8)
        upper_yellow = np.array([35, 255, 255], dtype=np.uint8)
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
        paint_mask = cv2.bitwise_or(white_mask, yellow_mask)
        paint_mask[:y_top, :] = 0
        
        # Scan horizontal profile across road height
        scan_step = max(5, int(h * 0.03))
        left_samples = []
        right_samples = []
        
        for y in range(y_top, h, scan_step):
            row_road = clean_road[y, :]
            road_idx = np.where(row_road > 0)[0]
            if len(road_idx) >= int(w * 0.20):
                l_bound = float(np.min(road_idx))
                r_bound = float(np.max(road_idx))
                left_samples.append((l_bound, float(y)))
                right_samples.append((r_bound, float(y)))
                
        is_reliable = False
        if len(left_samples) >= 5 and len(right_samples) >= 5:
            y_pts = np.array([p[1] for p in left_samples])
            l_x = np.array([p[0] for p in left_samples])
            r_x = np.array([p[0] for p in right_samples])
            
            p_l = np.polyfit(y_pts, l_x, 1)
            p_r = np.polyfit(y_pts, r_x, 1)
            
            cur_l_bot = float(np.polyval(p_l, h))
            cur_l_top = float(np.polyval(p_l, y_top))
            cur_r_bot = float(np.polyval(p_r, h))
            cur_r_top = float(np.polyval(p_r, y_top))
            
            # Constrain to valid corridor
            cur_l_bot = max(0.0, min(float(w - 60), cur_l_bot))
            cur_l_top = max(0.0, min(float(w - 60), cur_l_top))
            cur_r_bot = min(float(w), max(cur_l_bot + 60.0, cur_r_bot))
            cur_r_top = min(float(w), max(cur_l_top + 60.0, cur_r_top))
            
            if cur_r_bot > cur_l_bot + 50 and cur_r_top > cur_l_top + 40:
                is_reliable = True
                self.road_detected_frames += 1
                
                if self.smoothed_left_bot is None:
                    self.smoothed_left_bot = cur_l_bot
                    self.smoothed_left_top = cur_l_top
                    self.smoothed_right_bot = cur_r_bot
                    self.smoothed_right_top = cur_r_top
                else:
                    a = self.ema_alpha
                    self.smoothed_left_bot = a * cur_l_bot + (1.0 - a) * self.smoothed_left_bot
                    self.smoothed_left_top = a * cur_l_top + (1.0 - a) * self.smoothed_left_top
                    self.smoothed_right_bot = a * cur_r_bot + (1.0 - a) * self.smoothed_right_bot
                    self.smoothed_right_top = a * cur_r_top + (1.0 - a) * self.smoothed_right_top
                    
        # Fallback if no road was ever detected
        if self.smoothed_left_bot is None:
            self.smoothed_left_bot = 0.05 * w
            self.smoothed_left_top = 0.15 * w
            self.smoothed_right_bot = 0.95 * w
            self.smoothed_right_top = 0.85 * w
            
        l_bot, l_top = self.smoothed_left_bot, self.smoothed_left_top
        r_bot, r_top = self.smoothed_right_bot, self.smoothed_right_top
        
        def x_left_fn(y_val):
            ratio = (float(y_val) - y_top) / float(h - y_top + 1e-5)
            return l_top + ratio * (l_bot - l_top)
            
        def x_right_fn(y_val):
            ratio = (float(y_val) - y_top) / float(h - y_top + 1e-5)
            return r_top + ratio * (r_bot - r_top)
            
        return {
            'is_reliable': is_reliable,
            'y_top': y_top,
            'left_top': (int(l_top), y_top),
            'left_bot': (int(l_bot), h),
            'right_top': (int(r_top), y_top),
            'right_bot': (int(r_bot), h),
            'x_left_fn': x_left_fn,
            'x_right_fn': x_right_fn,
            'grass_mask': grass_mask
        }

    def evaluate_motorcycle_footprint(self, moto_bbox, road, frame_idx=0, fps=24.0):
        timestamp_sec = frame_idx / max(1.0, float(fps))
        time_str = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:04.1f}"
        
        if moto_bbox is None:
            return {
                'status': 'INSIDE ROAD (PASS)',
                'is_violating': False,
                'boundary': None,
                'timestamp_str': time_str,
                'footprint': None
            }
            
        mx1, my1, mx2, my2 = moto_bbox
        moto_w = mx2 - mx1
        
        # Tyre contact patch footprint (central ~60% of base where tyres make ground contact)
        fp_left = mx1 + 0.20 * moto_w
        fp_right = mx2 - 0.20 * moto_w
        base_y = min(road['left_bot'][1] - 1, max(road['y_top'], my2))
        
        road_l = road['x_left_fn'](base_y)
        road_r = road['x_right_fn'](base_y)
        
        # Check off-road violation:
        # Motorcycle clearly crosses actual road edge into non-drivable area
        margin = max(5, int(moto_w * 0.05))
        left_crossing = (fp_left < road_l - margin)
        right_crossing = (fp_right > road_r + margin)
        
        # Also verify with grass/non-drivable mask under the tyre footprint
        grass_mask = road['grass_mask']
        tyre_cx = int((fp_left + fp_right) / 2)
        tyre_y = int(base_y)
        y1 = max(0, tyre_y - 12)
        y2 = min(grass_mask.shape[0], tyre_y + 2)
        x1 = max(0, tyre_cx - 15)
        x2 = min(grass_mask.shape[1], tyre_cx + 15)
        tyre_patch = grass_mask[y1:y2, x1:x2]
        grass_under_tyre = (np.mean(tyre_patch > 0) > 0.40) if tyre_patch.size > 0 else False
        
        violating = False
        side = None
        if left_crossing:
            side = 'LEFT'
            violating = True
        elif right_crossing or grass_under_tyre:
            side = 'RIGHT'
            violating = True
            
        if violating:
            self.consecutive_violations += 1
        else:
            self.consecutive_violations = max(0, self.consecutive_violations - 1)
            
        is_confirmed = (self.consecutive_violations >= 2)
        
        if is_confirmed and self.active_violation is None:
            viol_rec = {
                'type': 'LANE_BOUNDARY_VIOLATION',
                'boundary': side,
                'timestamp': time_str,
                'timestamp_sec': round(timestamp_sec, 2),
                'frame': frame_idx,
                'details': f"OUTSIDE ROAD (VIOLATION): Motorcycle crossed {side} road edge at {time_str}"
            }
            self.violations.append(viol_rec)
            self.active_violation = viol_rec
        elif not is_confirmed:
            self.active_violation = None
            
        status_text = "OUTSIDE ROAD (VIOLATION)" if is_confirmed else "INSIDE ROAD (PASS)"
        
        return {
            'status': status_text,
            'is_violating': is_confirmed,
            'boundary': side if is_confirmed else None,
            'timestamp_str': time_str,
            'footprint': (int(fp_left), int(fp_right), int(base_y)),
            'road_bounds_at_y': (int(road_l), int(road_r))
        }

# Run simulation on test_CAND-2026-063_sinch.mp4
detector = DrivableRoadDetector()
cap = cv2.VideoCapture('static/uploads/videos/test_CAND-2026-063_sinch.mp4')
f_idx = 0
viol_frames = 0
inside_frames = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    f_idx += 1
    dets, _ = yolo_detector.detect_objects(frame)
    road = detector.detect_drivable_road(frame)
    eval_res = detector.evaluate_motorcycle_footprint(dets.get('motorcycle_bbox'), road, frame_idx=f_idx, fps=24.0)
    if eval_res['is_violating']:
        viol_frames += 1
    else:
        inside_frames += 1

cap.release()
print(f"Total frames: {f_idx}")
print(f"Inside Road (PASS) frames: {inside_frames} ({inside_frames/f_idx*100:.1f}%)")
print(f"Outside Road (VIOLATION) frames: {viol_frames} ({viol_frames/f_idx*100:.1f}%)")
print(f"Recorded violations count: {len(detector.violations)}")
