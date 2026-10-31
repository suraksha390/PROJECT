import os
import cv2
import numpy as np

try:
    from config import (
        LANE_LINE_COLOR,
        LANE_MIN_CONFIRMATION_FRAMES,
        LANE_MIN_TRACKING_RATIO,
        LANE_MIN_BOUNDARY_RATIO
    )
except ImportError:
    LANE_LINE_COLOR = (140, 180, 210)       # Light brown in OpenCV BGR
    LANE_MIN_CONFIRMATION_FRAMES = 2
    LANE_MIN_TRACKING_RATIO = 0.20
    LANE_MIN_BOUNDARY_RATIO = 0.20


class LaneDetector:
    """
    OpenCV Straight Lane & Drivable Road Surface Detector and Motorcycle Tracker
    for Two-Wheeler Driving Skill Validation.

    Key Features:
    1. Supports EVERY uploaded video without hardcoding or assuming fixed filenames.
    2. Treats the actual drivable road surface as the permitted riding area.
       The rider passes lane discipline when remaining on the road, even if
       there are no painted lane markings.
    3. Handles diverse camera viewpoints: center-aligned roads, roads on the left,
       roads on the right, and wide perspective corridors.
    4. Detects solid painted test-track lane markings when present (e.g. driving test tracks).
    5. Ignores internal cracks, shadows, patches, or internal lane dividers.
    6. Tracks the road region across successive frames using Exponential Moving Average (EMA).
    7. Tracks the motorcycle throughout the entire video from first to last frame.
    8. Estimates motorcycle tyre-contact points and ground footprint relative to road edges.
    9. Confirms boundary contact/crossing across video frames (filtering 1-frame jitter),
       recording LANE_BOUNDARY_VIOLATION with timestamp and side (LEFT or RIGHT).
    10. Does NOT use minimum-distance / safe-distance thresholds: the only violation
        is actual contact with or crossing of either lane/road boundary into non-drivable area.
    11. Draws left and right boundaries in light brown (BGR: 140, 180, 210) and subtle
        transparent drivable road overlay.
    12. Marks result as NEEDS_REVIEW if boundaries or motorcycle cannot be detected reliably.
    """

    def __init__(self,
                 lane_color=None,
                 min_confirmation_frames=None,
                 min_tracking_ratio=None,
                 min_boundary_ratio=None):

        self.lane_color = lane_color or LANE_LINE_COLOR
        self.min_confirmation_frames = min_confirmation_frames or LANE_MIN_CONFIRMATION_FRAMES
        self.min_tracking_ratio = min_tracking_ratio or LANE_MIN_TRACKING_RATIO
        self.min_boundary_ratio = min_boundary_ratio or LANE_MIN_BOUNDARY_RATIO

        # Exponential moving average smoothing factor for lane boundaries
        self.ema_alpha = 0.25

        # Initialize tracking state
        self.reset()

    def reset(self):
        """Resets per-video tracking and boundary state."""
        self.total_frames_processed = 0
        self.motorcycle_detected_frames = 0
        self.lane_detected_frames = 0

        # Smoothed boundary endpoints: (x_bot, x_top)
        self.smoothed_left_bot = None
        self.smoothed_left_top = None
        self.smoothed_right_bot = None
        self.smoothed_right_top = None
        self.roi_y_top = None

        # Motorcycle tracking state
        self.last_motorcycle_bbox = None
        self.last_motorcycle_width = 0
        self.frames_since_motorcycle = 999

        # Boundary contact confirmation state
        self.consecutive_left_touches = 0
        self.consecutive_right_touches = 0
        self.consecutive_inside_frames = 0
        self.active_violation = None

        # Confirmed violations list:
        # [{ 'type': 'LANE_BOUNDARY_VIOLATION', 'boundary': 'LEFT'|'RIGHT', 'timestamp': '00:03.4', 'timestamp_sec': 3.4, 'frame': 82 }]
        self.confirmed_violations = []

    def detect_lane_boundaries(self, frame, motorcycle_bbox=None):
        """
        Detects left and right lane or road boundaries using OpenCV.
        Handles both formal painted test-track corridors and natural unlined road surfaces.

        Returns:
            dict containing:
                - 'left_top': (x, y)
                - 'left_bot': (x, y)
                - 'right_top': (x, y)
                - 'right_bot': (x, y)
                - 'x_left_fn': callable(y) -> x
                - 'x_right_fn': callable(y) -> x
                - 'is_detected': bool (detected in this frame)
                - 'drivable_polygon': np.ndarray of shape (4, 2)
        """
        h, w = frame.shape[:2]
        y_top = int(h * 0.35)
        self.roi_y_top = y_top

        # Blank or solid black frame handling
        if frame is None or np.mean(frame) < 5:
            return {
                'left_top': (int(0.20 * w), y_top),
                'left_bot': (int(0.20 * w), h),
                'right_top': (int(0.80 * w), y_top),
                'right_bot': (int(0.80 * w), h),
                'x_left_fn': lambda y: float(0.20 * w),
                'x_right_fn': lambda y: float(0.80 * w),
                'is_detected': False,
                'drivable_polygon': None
            }

        # Road surface Region of Interest (ROI)
        roi = frame[y_top:, :]
        roi_h = h - y_top
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # -------------------------------------------------------------
        # STEP 1: Check for Solid Painted Test-Track Lane Boundaries
        # -------------------------------------------------------------
        # High-brightness white painted markings mask (e.g., test track)
        white_mask = cv2.inRange(hsv_roi, np.array([0, 0, 200], dtype=np.uint8),
                                 np.array([180, 45, 255], dtype=np.uint8))
        # High-saturation yellow painted markings mask
        yellow_mask = cv2.inRange(hsv_roi, np.array([15, 80, 90], dtype=np.uint8),
                                  np.array([35, 255, 255], dtype=np.uint8))
        paint_mask = cv2.bitwise_or(white_mask, yellow_mask)

        # Detect steep line segments directly from painted markings
        min_line_len = max(35, int(h * 0.12))
        lines = cv2.HoughLinesP(paint_mask, 1, np.pi / 180, threshold=40,
                                minLineLength=min_line_len, maxLineGap=30)

        painted_left = []
        painted_right = []

        if lines is not None:
            for line in lines:
                x1, y1_rel, x2, y2_rel = line[0]
                y1 = y1_rel + y_top
                y2 = y2_rel + y_top

                dy = y2 - y1
                dx = x2 - x1

                # Steep corridor lines only (filter out horizontal noise/crossings)
                if abs(dy) < 1.2 * abs(dx) or dy == 0:
                    continue

                inv_slope = dx / float(dy)
                x_bot = x1 + (h - y1) * inv_slope
                x_top = x1 + (y_top - y1) * inv_slope

                if 0.04 * w <= x_bot <= 0.48 * w and 0.02 * w <= x_top <= 0.50 * w:
                    painted_left.append((x_bot, x_top))
                elif 0.52 * w <= x_bot <= 0.96 * w and 0.50 * w <= x_top <= 0.98 * w:
                    painted_right.append((x_bot, x_top))

        # -------------------------------------------------------------
        # STEP 2: Physical Road Surface Segmentation (Asphalt vs Surroundings)
        # -------------------------------------------------------------
        # Non-drivable surroundings: green vegetation / grass verge / trees
        full_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        veg_mask = ((full_hsv[:, :, 0] >= 28) & (full_hsv[:, :, 0] <= 88) &
                    (full_hsv[:, :, 1] >= 35) & (full_hsv[:, :, 2] >= 30))

        # Morphological opening to eliminate isolated pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        veg_clean = cv2.morphologyEx(veg_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

        # Exclude motorcycle bounding box so its shadow/body is never mistaken for roadside grass
        ref_x = w // 2
        if motorcycle_bbox is not None:
            mx1, my1, mx2, my2 = motorcycle_bbox
            veg_clean[max(0, my1):min(h, my2 + 5), max(0, mx1):min(w, mx2)] = 0
            ref_x = int(np.clip((mx1 + mx2) // 2, int(w * 0.15), int(w * 0.85)))

        # Scan rows for road boundaries (asphalt to vegetation / non-drivable transitions)
        road_lefts = []
        road_rights = []
        scan_step = max(6, int(h * 0.03))

        for y in range(y_top, h, scan_step):
            row_veg = veg_clean[y, :]

            # Search left of ref_x for non-drivable surroundings
            left_non_road = np.where(row_veg[:ref_x] > 0)[0]
            if len(left_non_road) >= 3:
                l_edge = float(np.max(left_non_road))
            else:
                l_edge = 0.0

            # Search right of ref_x for non-drivable surroundings
            right_non_road = np.where(row_veg[ref_x:] > 0)[0]
            if len(right_non_road) >= 3:
                r_edge = float(ref_x + np.min(right_non_road))
            else:
                r_edge = float(w)

            road_lefts.append((l_edge, y))
            road_rights.append((r_edge, y))

        # -------------------------------------------------------------
        # STEP 3: Boundary Determination & Temporal Smoothing
        # -------------------------------------------------------------
        detected_in_frame = False
        cur_l_bot, cur_l_top = None, None
        cur_r_bot, cur_r_top = None, None

        # Priority 1: Solid painted test track lines
        if painted_left and painted_right:
            cur_l_bot = float(np.median([c[0] for c in painted_left]))
            cur_l_top = float(np.median([c[1] for c in painted_left]))
            cur_r_bot = float(np.median([c[0] for c in painted_right]))
            cur_r_top = float(np.median([c[1] for c in painted_right]))
            if cur_r_bot > cur_l_bot + (0.15 * w):
                detected_in_frame = True

        # Priority 2: Physical road surface transitions
        if not detected_in_frame and len(road_lefts) >= 4 and len(road_rights) >= 4:
            y_pts = np.array([p[1] for p in road_lefts])
            p_l = np.polyfit(y_pts, [p[0] for p in road_lefts], 1)
            p_r = np.polyfit(y_pts, [p[0] for p in road_rights], 1)

            cur_l_bot = max(0.0, float(np.polyval(p_l, h)))
            cur_l_top = max(0.0, float(np.polyval(p_l, y_top)))
            cur_r_bot = min(float(w), float(np.polyval(p_r, h)))
            cur_r_top = min(float(w), float(np.polyval(p_r, y_top)))

            if cur_r_bot > cur_l_bot + (0.15 * w):
                detected_in_frame = True

        # Temporal EMA smoothing across video frames
        if detected_in_frame and cur_l_bot is not None and cur_r_bot is not None:
            self.lane_detected_frames += 1
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

        # Default fallback corridor if no boundaries detected yet
        if self.smoothed_left_bot is None:
            self.smoothed_left_bot = 0.20 * w
            self.smoothed_left_top = 0.30 * w
            self.smoothed_right_bot = 0.80 * w
            self.smoothed_right_top = 0.70 * w

        l_bot, l_top = self.smoothed_left_bot, self.smoothed_left_top
        r_bot, r_top = self.smoothed_right_bot, self.smoothed_right_top

        def x_left_fn(y_val):
            ratio = (float(y_val) - y_top) / float(h - y_top + 1e-5)
            return l_top + ratio * (l_bot - l_top)

        def x_right_fn(y_val):
            ratio = (float(y_val) - y_top) / float(h - y_top + 1e-5)
            return r_top + ratio * (r_bot - r_top)

        drivable_polygon = np.array([
            [int(l_top), y_top],
            [int(r_top), y_top],
            [int(r_bot), h],
            [int(l_bot), h]
        ], dtype=np.int32)

        return {
            'left_top': (int(l_top), y_top),
            'left_bot': (int(l_bot), h),
            'right_top': (int(r_top), y_top),
            'right_bot': (int(r_bot), h),
            'x_left_fn': x_left_fn,
            'x_right_fn': x_right_fn,
            'is_detected': detected_in_frame,
            'drivable_polygon': drivable_polygon
        }

    def update_motorcycle_track(self, motorcycle_bbox=None, rider_bbox=None):
        """
        Tracks motorcycle position and estimates width relative to lane boundaries.
        Prefers motorcycle detection from YOLO; falls back to rider lower body if needed.
        """
        bbox = motorcycle_bbox
        if bbox is None and rider_bbox is not None:
            # Fallback estimation from rider bounding box if motorcycle class momentarily missed
            rx1, ry1, rx2, ry2 = rider_bbox
            bbox = (rx1, int(ry1 + (ry2 - ry1) * 0.4), rx2, ry2)

        if bbox is not None:
            mx1, my1, mx2, my2 = bbox
            if mx2 > mx1 and my2 > my1:
                self.last_motorcycle_bbox = bbox
                self.last_motorcycle_width = mx2 - mx1
                self.motorcycle_detected_frames += 1
                self.frames_since_motorcycle = 0
                return bbox

        # Coasting: retain last known position for up to 3 frames of temporary dropout
        if self.frames_since_motorcycle <= 3 and self.last_motorcycle_bbox is not None:
            self.frames_since_motorcycle += 1
            return self.last_motorcycle_bbox

        self.frames_since_motorcycle += 1
        return None

    def check_boundary_contact(self, motorcycle_bbox, boundaries, frame_idx=0, fps=25.0):
        """
        Checks if the motorcycle touches or crosses either lane/road boundary.

        STRICT DECISION RULES:
        1. INSIDE ROAD (TRUE): The motorcycle remains within the drivable road surface.
        2. OUTSIDE ROAD (FALSE): A meaningful part of the motorcycle clearly crosses the
           actual road edge into a non-drivable area (soil, grass, outside boundary).
        3. Crossing internal painted lines, pavement joints, or lane dividers while remaining
           on the road is NOT a violation.
        4. No minimum-distance / safe-distance thresholds: the only lane violation is actual
           contact with or crossing of either boundary.
        5. Confirms boundary contact using multi-frame persistence (>= 2 frames) to filter 1-frame jitter.

        Returns:
            dict containing:
                - 'is_violating': bool (active confirmed violation on this frame)
                - 'boundary_touched': 'LEFT' | 'RIGHT' | None
                - 'dist_left': float (distance to left boundary, negative if crossing)
                - 'dist_right': float (distance to right boundary, negative if crossing)
                - 'motorcycle_width': int
                - 'timestamp_str': str
                - 'status_label': 'INSIDE ROAD (PASS)' | 'OUTSIDE ROAD (VIOLATION)'
        """
        timestamp_sec = frame_idx / max(1.0, float(fps))
        time_str = f"{int(timestamp_sec // 60):02d}:{timestamp_sec % 60:04.1f}"

        if motorcycle_bbox is None:
            return {
                'is_violating': False,
                'boundary_touched': None,
                'dist_left': 0.0,
                'dist_right': 0.0,
                'motorcycle_width': 0,
                'timestamp_str': time_str,
                'status_label': 'INSIDE ROAD (PASS)'
            }

        mx1, my1, mx2, my2 = motorcycle_bbox
        moto_w = mx2 - mx1
        moto_h = my2 - my1

        # Check contact at bottom wheel/ground contact level (my2)
        x_left = boundaries['x_left_fn'](my2)
        x_right = boundaries['x_right_fn'](my2)

        # Footprint contact estimation:
        # In standard straight driving (aspect ratio w/h < 1.0): footprint spans mx1 to mx2.
        # In wide turning/U-turn orientations (aspect ratio w/h >= 1.0):
        # Ground contact tyre footprint is centered at cx with physical ground span.
        cx = (mx1 + mx2) / 2.0
        if moto_w > 1.2 * moto_h and moto_w > 200:
            # Side-view turning: tyres contact near base centered around cx
            fp_left = max(float(mx1), cx - (0.35 * moto_w))
            fp_right = min(float(mx2), cx + (0.35 * moto_w))
        else:
            fp_left = float(mx1)
            fp_right = float(mx2)

        # Distances: positive = inside, <= 0 = touched/crossed boundary
        dist_left = fp_left - x_left
        dist_right = x_right - fp_right

        # Boundary contact conditions:
        # A boundary breach occurs if:
        # - Left boundary exists (x_left > 0) and motorcycle crosses it (dist_left <= 0)
        # - Right boundary exists (x_right < w) and motorcycle crosses it (dist_right <= 0)
        # - Or on painted corridors where both lines are within frame
        left_touched = (x_left > 0 and dist_left <= 0)
        right_touched = (dist_right <= 0)

        boundary_contact = None
        if left_touched:
            boundary_contact = 'LEFT'
            self.consecutive_left_touches += 1
            self.consecutive_right_touches = 0
            self.consecutive_inside_frames = 0
        elif right_touched:
            boundary_contact = 'RIGHT'
            self.consecutive_right_touches += 1
            self.consecutive_left_touches = 0
            self.consecutive_inside_frames = 0
        else:
            # Completely inside without touching either boundary
            self.consecutive_inside_frames += 1
            if self.consecutive_inside_frames >= 2:
                self.consecutive_left_touches = 0
                self.consecutive_right_touches = 0
                self.active_violation = None

        is_confirmed_violation = False

        # Confirm boundary contact using reliable multi-frame persistence (filters 1-frame noise)
        if boundary_contact == 'LEFT' and self.consecutive_left_touches >= self.min_confirmation_frames:
            is_confirmed_violation = True
            if self.active_violation is None or self.active_violation.get('boundary') != 'LEFT':
                start_frame = frame_idx - (self.consecutive_left_touches - 1)
                start_sec = start_frame / max(1.0, float(fps))
                start_time_str = f"{int(start_sec // 60):02d}:{start_sec % 60:04.1f}"
                violation_record = {
                    'type': 'LANE_BOUNDARY_VIOLATION',
                    'boundary': 'LEFT',
                    'timestamp': start_time_str,
                    'timestamp_sec': round(start_sec, 2),
                    'frame': start_frame,
                    'details': f"OUTSIDE ROAD (VIOLATION): Motorcycle crossed LEFT road edge at {start_time_str}"
                }
                self.confirmed_violations.append(violation_record)
                self.active_violation = violation_record

        elif boundary_contact == 'RIGHT' and self.consecutive_right_touches >= self.min_confirmation_frames:
            is_confirmed_violation = True
            if self.active_violation is None or self.active_violation.get('boundary') != 'RIGHT':
                start_frame = frame_idx - (self.consecutive_right_touches - 1)
                start_sec = start_frame / max(1.0, float(fps))
                start_time_str = f"{int(start_sec // 60):02d}:{start_sec % 60:04.1f}"
                violation_record = {
                    'type': 'LANE_BOUNDARY_VIOLATION',
                    'boundary': 'RIGHT',
                    'timestamp': start_time_str,
                    'timestamp_sec': round(start_sec, 2),
                    'frame': start_frame,
                    'details': f"OUTSIDE ROAD (VIOLATION): Motorcycle crossed RIGHT road edge at {start_time_str}"
                }
                self.confirmed_violations.append(violation_record)
                self.active_violation = violation_record

        status_label = "OUTSIDE ROAD (FAIL)" if is_confirmed_violation else "INSIDE ROAD (PASS)"

        return {
            'is_violating': is_confirmed_violation,
            'boundary_touched': boundary_contact if is_confirmed_violation else None,
            'dist_left': round(dist_left, 1),
            'dist_right': round(dist_right, 1),
            'motorcycle_width': moto_w,
            'timestamp_str': time_str,
            'status_label': status_label
        }

    def draw_lane_overlay(self, frame, boundaries, motorcycle_bbox, contact_info):
        """
        Draws clean road status banner on the video frame.
        Road boundary lines, diagonal lines, and edge labels are removed as per requirements.
        """
        annotated = frame.copy()
        h, w = frame.shape[:2]

        is_violating = contact_info['is_violating']

        # Driving Road Status HUD Banner (INSIDE ROAD (PASS) or OUTSIDE ROAD (FAIL))
        if is_violating:
            status_text = "OUTSIDE ROAD (FAIL)"
            bg_color = (0, 0, 200)
            box_w = 295
        else:
            status_text = "INSIDE ROAD (PASS)"
            bg_color = (0, 100, 0)
            box_w = 275

        cv2.rectangle(annotated, (15, 18), (15 + box_w, 54), bg_color, -1)
        cv2.putText(annotated, status_text, (22, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

        return annotated

    def analyze_lane(self, frame, rider_bbox=None, motorcycle_bbox=None, frame_idx=0, fps=25.0):
        """
        Main processing method called per frame.
        Maintains backward compatibility with original analyze_lane(frame, rider_bbox).
        """
        self.total_frames_processed += 1

        # 1. Detect straight lane / road boundaries via OpenCV
        boundaries = self.detect_lane_boundaries(frame, motorcycle_bbox=motorcycle_bbox)

        # 2. Track motorcycle position throughout video
        tracked_moto = self.update_motorcycle_track(motorcycle_bbox=motorcycle_bbox, rider_bbox=rider_bbox)

        # 3. Check for boundary contact
        contact_info = self.check_boundary_contact(tracked_moto, boundaries, frame_idx=frame_idx, fps=fps)

        # 4. Draw light brown road boundaries, drivable overlay, and motorcycle position
        annotated_frame = self.draw_lane_overlay(frame, boundaries, tracked_moto, contact_info)

        return contact_info['is_violating'], annotated_frame

    def get_summary(self):
        """
        Evaluates the entire video sequence and returns the final lane discipline verdict.

        Rules:
        - If boundaries or motorcycle position could not be detected reliably -> NEEDS_REVIEW.
        - If motorcycle remained completely inside without touching either boundary -> PASS.
        - If confirmed boundary violations occurred -> FAIL (with list of incidents).
        """
        total = max(1, self.total_frames_processed)
        moto_coverage = self.motorcycle_detected_frames / float(total)
        lane_coverage = self.lane_detected_frames / float(total)

        is_reliable = True
        remarks = []

        if moto_coverage < self.min_tracking_ratio:
            is_reliable = False
            remarks.append("Motorcycle could not be tracked reliably across video frames.")

        if lane_coverage < self.min_boundary_ratio:
            is_reliable = False
            remarks.append("Road boundaries could not be detected reliably across video frames.")

        if not is_reliable:
            return {
                'lane_state': 'NEEDS_REVIEW',
                'lane_status': 0,
                'violation_count': len(self.confirmed_violations),
                'violations': self.confirmed_violations,
                'motorcycle_tracking_coverage': round(moto_coverage, 2),
                'lane_detection_coverage': round(lane_coverage, 2),
                'is_reliable': False,
                'remarks': " ".join(remarks) if remarks else "Drivable road or motorcycle position could not be tracked reliably across video frames."
            }

        if len(self.confirmed_violations) == 0:
            return {
                'lane_state': 'PASS',
                'lane_status': 1,
                'violation_count': 0,
                'violations': [],
                'motorcycle_tracking_coverage': round(moto_coverage, 2),
                'lane_detection_coverage': round(lane_coverage, 2),
                'is_reliable': True,
                'remarks': "Motorcycle remained completely inside the drivable road boundaries throughout test."
            }
        else:
            return {
                'lane_state': 'FAIL',
                'lane_status': 0,
                'violation_count': len(self.confirmed_violations),
                'violations': self.confirmed_violations,
                'motorcycle_tracking_coverage': round(moto_coverage, 2),
                'lane_detection_coverage': round(lane_coverage, 2),
                'is_reliable': True,
                'remarks': f"Recorded {len(self.confirmed_violations)} confirmed road edge violation(s)."
            }


lane_detector = LaneDetector()
