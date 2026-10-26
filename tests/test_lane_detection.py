"""
Comprehensive Test Suite for Lane Boundary Detection & Motorcycle Tracking
==========================================================================

Verifies all requirements:
  1. OpenCV detection of left and right straight driving-test lane boundaries.
  2. YOLO motorcycle detection, position, and width estimation.
  3. Motorcycle tracking throughout video.
  4. Left and right boundary contact/crossing -> LANE_BOUNDARY_VIOLATION.
  5. Motorcycle inside lane without touching -> PASS.
  6. Minor detection noise & 1-frame jitter filtering vs confirmed contact.
  7. Exact timestamp and boundary side recording (LEFT / RIGHT).
  8. Light brown line drawing (BGR: 140, 180, 210) & motorcycle visualization.
  9. Low importance evaluation (reduces score slightly, doesn't reject whole test).
  10. Unreliable detection -> NEEDS_REVIEW (never assumes a pass).
  11. Strict absence of minimum-distance thresholds (only actual contact/crossing).
  12. Test execution on real sample video.
"""

import os
import sys
import unittest
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.lane_detector import LaneDetector, LANE_LINE_COLOR
from models.ml_evaluator import DrivingMLEvaluator


def create_synthetic_road_frame(w=1280, h=720, left_x=250, right_x=1030):
    """
    Creates a synthetic straight lane test frame with dark asphalt road surface
    and clear white lane boundary lines at left_x and right_x.
    """
    frame = np.full((h, w, 3), 60, dtype=np.uint8)  # Dark asphalt

    # Draw straight left boundary line (white line)
    cv2.line(frame, (left_x, int(h * 0.35)), (left_x, h), (255, 255, 255), 6)

    # Draw straight right boundary line (white line)
    cv2.line(frame, (right_x, int(h * 0.35)), (right_x, h), (255, 255, 255), 6)

    return frame


class TestLaneDetectionSystem(unittest.TestCase):
    """Unit and Integration Tests for Straight Lane Boundary Detection & Motorcycle Tracking."""

    def setUp(self):
        self.detector = LaneDetector()
        self.evaluator = DrivingMLEvaluator()

    # -------------------------------------------------------------------------
    # TEST 1: OpenCV Lane Boundary Detection & Light Brown Color
    # -------------------------------------------------------------------------
    def test_lane_boundary_detection_and_light_brown_color(self):
        """Test 1: Boundaries detected with OpenCV and drawn in light brown (140, 180, 210)."""
        frame = create_synthetic_road_frame(1280, 720, left_x=260, right_x=1020)
        self.detector.reset()

        boundaries = self.detector.detect_lane_boundaries(frame)
        self.assertTrue(boundaries['is_detected'], "OpenCV must detect lane boundaries on road frame")

        # Check left and right boundary position
        x_left = boundaries['x_left_fn'](700)
        x_right = boundaries['x_right_fn'](700)
        self.assertAlmostEqual(x_left, 260, delta=40, msg="Left boundary should be near 260")
        self.assertAlmostEqual(x_right, 1020, delta=40, msg="Right boundary should be near 1020")

        # Verify drawing uses light brown BGR color (140, 180, 210)
        contact_info = {
            'is_violating': False,
            'boundary_touched': None,
            'dist_left': 100.0,
            'dist_right': 100.0,
            'motorcycle_width': 120,
            'timestamp_str': '00:01.0'
        }
        annotated = self.detector.draw_lane_overlay(frame, boundaries, (400, 300, 520, 600), contact_info)

        # Check for presence of light brown color pixels in annotated frame
        expected_color = np.array([140, 180, 210], dtype=np.uint8)
        # Find pixels close to expected light brown color
        diff = np.abs(annotated.astype(np.int16) - expected_color.astype(np.int16))
        close_pixels = np.all(diff <= 5, axis=-1)
        self.assertTrue(np.any(close_pixels), "Lane boundaries must be drawn in light brown (140, 180, 210)")

    # -------------------------------------------------------------------------
    # TEST 2: Motorcycle Completely Inside Lane -> PASS
    # -------------------------------------------------------------------------
    def test_motorcycle_inside_lane_passes(self):
        """Test 2: When motorcycle stays strictly inside both boundaries -> PASS."""
        frame = create_synthetic_road_frame(1280, 720, left_x=260, right_x=1020)
        self.detector.reset()

        # Motorcycle centered: mx1=500, mx2=650, base my2=600 (well within [260, 1020])
        for f in range(1, 15):
            viol, _ = self.detector.analyze_lane(
                frame,
                motorcycle_bbox=(500, 350, 650, 600),
                frame_idx=f,
                fps=25.0
            )
            self.assertFalse(viol, f"Motorcycle inside lane should not violate at frame {f}")

        summary = self.detector.get_summary()
        self.assertEqual(summary['lane_state'], 'PASS', "Lane state must be PASS")
        self.assertEqual(summary['lane_status'], 1, "Lane status must be 1")
        self.assertEqual(summary['violation_count'], 0, "No violations should be recorded")

    # -------------------------------------------------------------------------
    # TEST 3: Actual Contact with LEFT Boundary -> LANE_BOUNDARY_VIOLATION
    # -------------------------------------------------------------------------
    def test_left_boundary_contact_records_violation(self):
        """Test 3: Motorcycle touches/crosses left boundary (mx1 <= x_left) -> violation."""
        frame = create_synthetic_road_frame(1280, 720, left_x=260, right_x=1020)
        self.detector.reset()

        # Frame 1-5: Inside lane
        for f in range(1, 6):
            self.detector.analyze_lane(frame, motorcycle_bbox=(500, 350, 650, 600), frame_idx=f, fps=25.0)

        # Frame 6-8: Left edge mx1=250 crosses left boundary (x_left ~ 260) -> 3 consecutive frames
        viol_flags = []
        for f in range(6, 9):
            v, _ = self.detector.analyze_lane(frame, motorcycle_bbox=(250, 350, 420, 600), frame_idx=f, fps=25.0)
            viol_flags.append(v)

        # Frame 7 (2nd consecutive touch) must confirm the violation
        self.assertTrue(viol_flags[1], "Second consecutive frame must confirm violation")

        summary = self.detector.get_summary()
        self.assertEqual(summary['lane_state'], 'FAIL')
        self.assertEqual(summary['violation_count'], 1)
        v_record = summary['violations'][0]
        self.assertEqual(v_record['type'], 'LANE_BOUNDARY_VIOLATION')
        self.assertEqual(v_record['boundary'], 'LEFT', "Boundary touched must be LEFT")
        self.assertEqual(v_record['timestamp'], '00:00.2', "Timestamp should be correctly formatted")

    # -------------------------------------------------------------------------
    # TEST 4: Actual Contact with RIGHT Boundary -> LANE_BOUNDARY_VIOLATION
    # -------------------------------------------------------------------------
    def test_right_boundary_contact_records_violation(self):
        """Test 4: Motorcycle touches/crosses right boundary (mx2 >= x_right) -> violation."""
        frame = create_synthetic_road_frame(1280, 720, left_x=260, right_x=1020)
        self.detector.reset()

        # Frame 1-10: Inside lane
        for f in range(1, 11):
            self.detector.analyze_lane(frame, motorcycle_bbox=(500, 350, 650, 600), frame_idx=f, fps=25.0)

        # Frame 11-13: Right edge mx2=1040 touches/crosses right boundary (~1020)
        for f in range(11, 14):
            self.detector.analyze_lane(frame, motorcycle_bbox=(850, 350, 1040, 600), frame_idx=f, fps=25.0)

        summary = self.detector.get_summary()
        self.assertEqual(summary['lane_state'], 'FAIL')
        self.assertGreaterEqual(summary['violation_count'], 1)
        v_record = summary['violations'][0]
        self.assertEqual(v_record['boundary'], 'RIGHT', "Boundary touched must be RIGHT")
        self.assertEqual(v_record['timestamp'], '00:00.4')

    # -------------------------------------------------------------------------
    # TEST 5: Minor Detection Noise (1-frame jitter) is Ignored
    # -------------------------------------------------------------------------
    def test_one_frame_detection_jitter_ignored(self):
        """Test 5: A single isolated 1-frame boundary touch is ignored as noise."""
        frame = create_synthetic_road_frame(1280, 720, left_x=260, right_x=1020)
        self.detector.reset()

        # Frame 1-5: Inside lane
        for f in range(1, 6):
            self.detector.analyze_lane(frame, motorcycle_bbox=(500, 350, 650, 600), frame_idx=f, fps=25.0)

        # Frame 6: 1-frame transient touch on left boundary
        v6, _ = self.detector.analyze_lane(frame, motorcycle_bbox=(250, 350, 420, 600), frame_idx=6, fps=25.0)
        self.assertFalse(v6, "1-frame touch must NOT trigger confirmed violation immediately")

        # Frame 7-10: Immediately back inside lane
        for f in range(7, 11):
            v, _ = self.detector.analyze_lane(frame, motorcycle_bbox=(500, 350, 650, 600), frame_idx=f, fps=25.0)
            self.assertFalse(v)

        summary = self.detector.get_summary()
        self.assertEqual(summary['lane_state'], 'PASS', "Isolated 1-frame jitter must not fail lane discipline")
        self.assertEqual(summary['violation_count'], 0)

    # -------------------------------------------------------------------------
    # TEST 6: Continuous Contact Over Multiple Frames is Single Incident
    # -------------------------------------------------------------------------
    def test_continuous_crossing_grouped_as_single_incident(self):
        """Test 6: Continuous boundary crossing over 10 consecutive frames recorded as 1 incident."""
        frame = create_synthetic_road_frame(1280, 720, left_x=260, right_x=1020)
        self.detector.reset()

        # Inside frames 1-4
        for f in range(1, 5):
            self.detector.analyze_lane(frame, motorcycle_bbox=(500, 350, 650, 600), frame_idx=f, fps=25.0)

        # Continuous contact across frames 5 to 15 (11 frames)
        for f in range(5, 16):
            self.detector.analyze_lane(frame, motorcycle_bbox=(240, 350, 400, 600), frame_idx=f, fps=25.0)

        # Back inside frames 16 to 20
        for f in range(16, 21):
            self.detector.analyze_lane(frame, motorcycle_bbox=(500, 350, 650, 600), frame_idx=f, fps=25.0)

        summary = self.detector.get_summary()
        self.assertEqual(summary['violation_count'], 1, "Continuous crossing should be grouped as 1 incident")
        self.assertEqual(summary['violations'][0]['boundary'], 'LEFT')

    # -------------------------------------------------------------------------
    # TEST 7: Unreliable Detection Returns NEEDS_REVIEW
    # -------------------------------------------------------------------------
    def test_unreliable_detection_returns_needs_review(self):
        """Test 7: If motorcycle or boundaries cannot be detected reliably -> NEEDS_REVIEW."""
        self.detector.reset()
        # Blank black frame (no lane markings)
        blank = np.zeros((720, 1280, 3), dtype=np.uint8)

        # Process frames without motorcycle detection
        for f in range(1, 15):
            self.detector.analyze_lane(blank, motorcycle_bbox=None, frame_idx=f, fps=25.0)

        summary = self.detector.get_summary()
        self.assertEqual(summary['lane_state'], 'NEEDS_REVIEW', "Must return NEEDS_REVIEW when unreliable")
        self.assertEqual(summary['lane_status'], 0, "Must not assume pass")
        self.assertFalse(summary['is_reliable'])

    # -------------------------------------------------------------------------
    # TEST 8: Strict Absence of Minimum-Distance Threshold
    # -------------------------------------------------------------------------
    def test_no_minimum_distance_threshold(self):
        """Test 8: Getting very close to the line (e.g. 5px away) without touching must NOT violate."""
        frame = create_synthetic_road_frame(1280, 720, left_x=260, right_x=1020)
        self.detector.reset()

        # Left boundary is at ~260. Motorcycle leftmost point mx1 = 265 (5px inside line, NOT touching)
        for f in range(1, 10):
            viol, _ = self.detector.analyze_lane(
                frame,
                motorcycle_bbox=(265, 350, 420, 600),
                frame_idx=f,
                fps=25.0
            )
            self.assertFalse(viol, "Close proximity without touching must NEVER be a violation")

        summary = self.detector.get_summary()
        self.assertEqual(summary['lane_state'], 'PASS', "Must pass if motorcycle didn't actually touch the line")

    # -------------------------------------------------------------------------
    # TEST 9: Low Importance Evaluation Impact
    # -------------------------------------------------------------------------
    def test_lane_violation_does_not_reject_whole_test(self):
        """Test 9: Lane discipline is low importance: violations deduct points but don't fail test alone."""
        # Candidate passed compulsory helmet (35), balance (25), foot touch (25), braking (5) = 90
        # Lane failed -> lane points = 0. Total score = 90 / 100 >= 70.
        res = self.evaluator.evaluate(
            auth_status=1,
            helmet_status=1,
            lane_status=0,
            foot_touch_status=1,
            balance_status=1,
            braking_status=1,
            lane_state='FAIL',
            lane_violation_count=2,
            helmet_state='PASS'
        )
        self.assertEqual(res['final_result'], 'PASS',
                         "Candidate with good overall driving must still PASS despite lane violation")
        self.assertEqual(res['overall_score'], 90.0)

    # -------------------------------------------------------------------------
    # TEST 10: Execution on Real Video Sample
    # -------------------------------------------------------------------------
    def test_sample_video_execution(self):
        """Test 10: Run lane detector on actual video test_CAND-2026-046_1000394971.mp4."""
        video_path = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'videos', 'test_CAND-2026-046_1000394971.mp4')
        if not os.path.exists(video_path):
            self.skipTest("Sample video file not found")

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.detector.reset()

        frame_count = 0
        while cap.isOpened() and frame_count < 40:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Mock or YOLO motorcycle bbox in middle
            viol, annotated = self.detector.analyze_lane(
                frame,
                motorcycle_bbox=(450, 300, 750, 600),
                frame_idx=frame_count,
                fps=fps
            )
            self.assertIsNotNone(annotated)
            self.assertEqual(annotated.shape, frame.shape)
        cap.release()

        summary = self.detector.get_summary()
        self.assertIn(summary['lane_state'], ['PASS', 'FAIL', 'NEEDS_REVIEW'])
        print(f"\n[Real Video Test] Processed {frame_count} frames. Verdict: {summary['lane_state']}")

    # -------------------------------------------------------------------------
    # TEST 11: Real Unlined Road Execution (sinch(1).mp4) -> PASS
    # -------------------------------------------------------------------------
    def test_sinch_unlined_road_execution(self):
        """Test 11: Rider remaining on drivable road surface in sinch(1).mp4 must PASS."""
        video_path = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'videos', 'test_CAND-2026-063_sinch.mp4')
        if not os.path.exists(video_path):
            alt_path = os.path.join(PROJECT_ROOT, '..', 'sinch.mp4')
            if os.path.exists(alt_path):
                video_path = alt_path
            else:
                self.skipTest("sinch.mp4 video file not found")

        from models.yolo_detector import yolo_detector
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        self.detector.reset()

        frame_count = 0
        while cap.isOpened() and frame_count < 60:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            dets, _ = yolo_detector.detect_objects(frame)
            viol, annotated = self.detector.analyze_lane(
                frame,
                motorcycle_bbox=dets.get('motorcycle_bbox'),
                rider_bbox=dets.get('rider_bbox'),
                frame_idx=frame_count,
                fps=fps
            )
            self.assertFalse(viol, f"Frame {frame_count} should not violate while remaining on road")
        cap.release()

        summary = self.detector.get_summary()
        self.assertEqual(summary['lane_state'], 'PASS', "Unlined road driving should PASS")
        self.assertEqual(summary['violation_count'], 0)


if __name__ == '__main__':
    unittest.main()
