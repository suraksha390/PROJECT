"""
Comprehensive Test Suite for Two-Wheeler Safety Helmet Detection
================================================================

Tests all 7 required cases specified in requirements:
  Case 1: A rider wearing a genuine motorcycle safety helmet.
  Case 2: A rider wearing a baseball cap (must never pass as helmet).
  Case 3: A rider wearing a hat, towel, turban, or other headwear.
  Case 4: A rider with an uncovered head (bare face/hair).
  Case 5: A partially hidden or blurred head (must return 'Needs Review').
  Case 6: No rider detected in frame.
  Case 7: The dedicated helmet model or its weights are missing (fallback safety).
  Case 8: Driving evaluation rule integration (compulsory override & review).
"""

import os
import sys
import unittest
from unittest.mock import MagicMock
import numpy as np
import cv2

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.yolo_detector import YoloDetector
from models.ml_evaluator import DrivingMLEvaluator


def create_synthetic_skin_head(w=80, h=80):
    """Generates a synthetic bare head image with skin tones in HSV."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # BGR values approximating human skin tone
    img[:, :] = (150, 180, 230)
    # Add slight facial texture
    cv2.circle(img, (w // 2, h // 2), 25, (130, 160, 215), -1)
    return img


def create_synthetic_fabric_head(w=80, h=80):
    """Generates synthetic towel/turban with heavy fabric folds (high edge density)."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (200, 200, 200)
    # Draw many wrinkle fold lines
    for i in range(5, h - 5, 4):
        cv2.line(img, (10, i), (w - 10, i + (i % 3)), (30, 30, 30), 2)
    return img


def create_synthetic_cap_head(w=80, h=80):
    """Generates synthetic head with baseball cap and horizontal brim."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Dark blue cap crown
    cv2.ellipse(img, (w // 2, int(h * 0.45)), (w // 3, h // 3), 0, 180, 360, (180, 40, 20), -1)
    # Horizontal visor/brim line
    brim_y = int(h * 0.55)
    cv2.rectangle(img, (5, brim_y), (w - 5, brim_y + 6), (20, 20, 20), -1)
    # Lower face skin
    cv2.rectangle(img, (int(w * 0.25), brim_y + 7), (int(w * 0.75), h - 5), (150, 180, 230), -1)
    return img


def create_synthetic_dome_head(w=80, h=80):
    """Generates a covered head with rounded dome shape."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, :] = (50, 50, 50)
    # Smooth glossy dome
    cv2.circle(img, (w // 2, h // 2), 32, (220, 30, 30), -1)
    cv2.ellipse(img, (w // 2, int(h * 0.4)), (20, 10), 0, 0, 360, (255, 120, 120), -1)
    return img


class MockBox:
    def __init__(self, cls_id, conf, xyxy):
        self.cls = [cls_id]
        self.conf = [conf]
        self.xyxy = [xyxy]


class MockYoloResult:
    def __init__(self, boxes, names):
        self.boxes = boxes
        self.names = names


def build_mock_detector(class_id=0, conf=0.90, class_names=None):
    """Creates a YoloDetector instance with a mocked dedicated helmet model."""
    if class_names is None:
        class_names = {0: 'helmet', 1: 'no_helmet', 2: 'other_headwear'}

    detector = YoloDetector(helmet_model_path=None)  # Start without loading file
    # Attach mock model
    mock_model = MagicMock()
    mock_box = MockBox(class_id, conf, [0, 0, 60, 60])
    mock_res = MockYoloResult([mock_box], class_names)
    mock_model.return_value = [mock_res]
    detector.helmet_model = mock_model
    return detector


class TestHelmetDetectionSystem(unittest.TestCase):
    """Unit and Integration Tests for Safety Helmet Detection."""

    def setUp(self):
        self.evaluator = DrivingMLEvaluator()

    # -------------------------------------------------------------------------
    # CASE 1: Rider wearing a genuine motorcycle safety helmet
    # -------------------------------------------------------------------------
    def test_case_1_rider_wearing_genuine_motorcycle_helmet(self):
        """Case 1: Genuine helmet detected with high confidence -> PASS."""
        detector = build_mock_detector(class_id=0, conf=0.92)  # 0: helmet
        head_img = create_synthetic_dome_head(80, 80)

        res = detector.classify_head_roi(head_img)
        self.assertEqual(res['class'], 'helmet', "Should classify as 'helmet'")
        self.assertEqual(res['status'], 'PASS', "Helmet status must be PASS")
        self.assertGreaterEqual(res['confidence'], 0.50, "Confidence should meet threshold")
        self.assertTrue(detector.check_helmet_roi(head_img), "check_helmet_roi must return True")

    # -------------------------------------------------------------------------
    # CASE 2: Rider wearing a baseball cap
    # -------------------------------------------------------------------------
    def test_case_2_rider_wearing_baseball_cap(self):
        """Case 2: Rider wearing baseball cap -> other_headwear, FAIL (never accepted as helmet)."""
        detector = build_mock_detector(class_id=2, conf=0.88)  # 2: other_headwear
        cap_img = create_synthetic_cap_head(80, 80)

        res = detector.classify_head_roi(cap_img)
        self.assertEqual(res['class'], 'other_headwear', "Cap must be classified as 'other_headwear'")
        self.assertEqual(res['status'], 'FAIL', "Cap status must be FAIL")
        self.assertFalse(detector.check_helmet_roi(cap_img), "check_helmet_roi must reject cap")

    # -------------------------------------------------------------------------
    # CASE 3: Rider wearing a hat or other headwear (towel, turban, scarf)
    # -------------------------------------------------------------------------
    def test_case_3_rider_wearing_hat_or_other_headwear(self):
        """Case 3: Towel / turban / cloth -> other_headwear, FAIL."""
        # 3A: Tested via dedicated model
        detector = build_mock_detector(class_id=2, conf=0.82)
        fabric_img = create_synthetic_fabric_head(80, 80)

        res = detector.classify_head_roi(fabric_img)
        self.assertEqual(res['class'], 'other_headwear')
        self.assertEqual(res['status'], 'FAIL')
        self.assertFalse(detector.check_helmet_roi(fabric_img))

        # 3B: Tested via CV fallback on high edge density fabric texture
        cv_detector = YoloDetector(helmet_model_path="non_existent_weights.pt")
        cv_res = cv_detector.classify_head_roi(fabric_img)
        self.assertEqual(cv_res['class'], 'other_headwear')
        self.assertEqual(cv_res['status'], 'FAIL')
        self.assertFalse(cv_detector.check_helmet_roi(fabric_img))

    # -------------------------------------------------------------------------
    # CASE 4: Rider with an uncovered head (bare face/hair)
    # -------------------------------------------------------------------------
    def test_case_4_rider_with_uncovered_head(self):
        """Case 4: Bare head -> no_helmet, FAIL."""
        # 4A: Dedicated model
        detector = build_mock_detector(class_id=1, conf=0.91)  # 1: no_helmet
        skin_img = create_synthetic_skin_head(80, 80)

        res = detector.classify_head_roi(skin_img)
        self.assertEqual(res['class'], 'no_helmet', "Uncovered head must be classified as 'no_helmet'")
        self.assertEqual(res['status'], 'FAIL', "Bare head status must be FAIL")
        self.assertFalse(detector.check_helmet_roi(skin_img))

        # 4B: CV fallback via skin tone detection
        cv_detector = YoloDetector(helmet_model_path="non_existent_weights.pt")
        cv_res = cv_detector.classify_head_roi(skin_img)
        self.assertEqual(cv_res['class'], 'no_helmet')
        self.assertEqual(cv_res['status'], 'FAIL')
        self.assertFalse(cv_detector.check_helmet_roi(skin_img))

    # -------------------------------------------------------------------------
    # CASE 5: Partially hidden or blurred head
    # -------------------------------------------------------------------------
    def test_case_5_partially_hidden_or_blurred_head(self):
        """Case 5: Blurred or tiny head ROI -> Needs Review (never assume helmet)."""
        detector = build_mock_detector(class_id=0, conf=0.90)

        # 5A: Heavily blurred image (Laplacian variance < 35.0)
        base_img = create_synthetic_dome_head(80, 80)
        blurred_img = cv2.GaussianBlur(base_img, (31, 31), 15.0)
        blur_res = detector.classify_head_roi(blurred_img)
        self.assertEqual(blur_res['status'], 'Needs Review', "Blurred head must return 'Needs Review'")
        self.assertFalse(detector.check_helmet_roi(blurred_img), "Blurred head must NOT pass check_helmet_roi")

        # 5B: Tiny / Distant head (width or height < 25 px)
        tiny_img = cv2.resize(base_img, (15, 15))
        tiny_res = detector.classify_head_roi(tiny_img)
        self.assertEqual(tiny_res['status'], 'Needs Review', "Tiny head must return 'Needs Review'")
        self.assertFalse(detector.check_helmet_roi(tiny_img), "Tiny head must NOT pass check_helmet_roi")

        # 5C: Empty / None head ROI
        empty_res = detector.classify_head_roi(None)
        self.assertEqual(empty_res['status'], 'FAIL', "None ROI must fail")
        self.assertFalse(detector.check_helmet_roi(None))

    # -------------------------------------------------------------------------
    # CASE 6: No rider detected
    # -------------------------------------------------------------------------
    def test_case_6_no_rider_detected(self):
        """Case 6: Empty scene / no rider detected -> rider_found is False, helmet_detected is False."""
        detector = YoloDetector()
        # Blank black frame with no person
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        detections, _ = detector.detect_objects(blank_frame)
        self.assertFalse(detections['rider_found'], "rider_found must be False when no rider exists")
        self.assertIsNone(detections['rider_bbox'], "rider_bbox must be None")
        self.assertFalse(detections['helmet_detected'], "helmet_detected must be False")
        self.assertEqual(detections['helmet_class'], 'none')
        self.assertEqual(detections['helmet_status'], 'FAIL')

    # -------------------------------------------------------------------------
    # CASE 7: Helmet model or its weights are missing
    # -------------------------------------------------------------------------
    def test_case_7_helmet_model_weights_missing(self):
        """Case 7: Dedicated weights missing -> Calibrated multi-cue computer vision classifier."""
        detector = YoloDetector(helmet_model_path="non_existent_helmet_weights.pt")
        self.assertIsNone(detector.helmet_model, "helmet_model should be None when weights are missing")

        # 7A: Bare head -> FAIL (no_helmet)
        skin_img = create_synthetic_skin_head(80, 80)
        res_skin = detector.classify_head_roi(skin_img)
        self.assertEqual(res_skin['class'], 'no_helmet')
        self.assertEqual(res_skin['status'], 'FAIL')
        self.assertFalse(detector.check_helmet_roi(skin_img))

        # 7B: Cap / Fabric -> FAIL (other_headwear)
        fabric_img = create_synthetic_fabric_head(80, 80)
        res_fab = detector.classify_head_roi(fabric_img)
        self.assertEqual(res_fab['class'], 'other_headwear')
        self.assertEqual(res_fab['status'], 'FAIL')
        self.assertFalse(detector.check_helmet_roi(fabric_img))

        # 7C: Genuine helmet dome -> PASS (helmet) (Resolves false Needs Review)
        dome_img = create_synthetic_dome_head(80, 80)
        res_dome = detector.classify_head_roi(dome_img)
        self.assertEqual(res_dome['class'], 'helmet')
        self.assertEqual(res_dome['status'], 'PASS')
        self.assertTrue(detector.check_helmet_roi(dome_img), "Genuine helmet must pass check_helmet_roi")

        # 7D: Ambiguous / borderline head -> Needs Review
        ambiguous_img = np.zeros((80, 80, 3), dtype=np.uint8)
        ambiguous_img[:25, :] = (150, 180, 230)  # ~30% skin patch
        ambiguous_img[25:, :] = (60, 60, 60)
        res_ambig = detector.classify_head_roi(ambiguous_img)
        self.assertEqual(res_ambig['status'], 'Needs Review', "Ambiguous image must return 'Needs Review'")
        self.assertFalse(detector.check_helmet_roi(ambiguous_img))

    # -------------------------------------------------------------------------
    # CASE 8: Driving Evaluation Integration (Compulsory Requirement)
    # -------------------------------------------------------------------------
    def test_case_8_driving_evaluation_rules(self):
        """Case 8: Compulsory helmet validation impact on overall test outcome."""
        # 8A: Confirmed helmet FAIL makes overall test FAIL even if other metrics are perfect
        res_fail = self.evaluator.evaluate(
            auth_status=1,
            helmet_status=0,
            lane_status=1,
            foot_touch_status=1,
            balance_status=1,
            helmet_state='FAIL'
        )
        self.assertEqual(res_fail['final_result'], 'FAIL',
                         "Failing compulsory helmet compliance must fail the candidate")

        # 8B: Helmet 'Needs Review' must result in 'NEEDS REVIEW', not an automatic pass
        res_review = self.evaluator.evaluate(
            auth_status=1,
            helmet_status=0,
            lane_status=1,
            foot_touch_status=1,
            balance_status=1,
            helmet_state='Needs Review'
        )
        self.assertEqual(res_review['final_result'], 'NEEDS REVIEW',
                         "Uncertain helmet detection must require review, never automatic PASS")

        # 8C: Confirmed helmet PASS with good driving score qualifies
        res_pass = self.evaluator.evaluate(
            auth_status=1,
            helmet_status=1,
            lane_status=1,
            foot_touch_status=1,
            balance_status=1,
            helmet_state='PASS'
        )
        self.assertEqual(res_pass['final_result'], 'PASS',
                         "Confirmed helmet + stable metrics must PASS")

    # -------------------------------------------------------------------------
    # CASE 9: Real Test Videos Evaluation (Supplied sinch.mp4 & cap_video.mp4)
    # -------------------------------------------------------------------------
    def test_case_9_real_video_evaluations(self):
        """Case 9: Verify user's sinch.mp4 evaluates to Helmet PASS and cap video fails."""
        sinch_path = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'videos', 'test_CAND-2026-062_sinch.mp4')
        if not os.path.exists(sinch_path):
            sinch_path = os.path.abspath(os.path.join(PROJECT_ROOT, '..', 'sinch.mp4'))

        if os.path.exists(sinch_path):
            cap = cv2.VideoCapture(sinch_path)
            detector = YoloDetector()
            helmet_pass = 0
            total_rider = 0
            for _ in range(60):
                ret, frame = cap.read()
                if not ret:
                    break
                dets, _ = detector.detect_objects(frame)
                if dets['rider_found']:
                    total_rider += 1
                    if dets['helmet_detected']:
                        helmet_pass += 1
            cap.release()
            self.assertGreater(total_rider, 0, "Rider should be detected in sinch.mp4")
            pass_ratio = helmet_pass / total_rider
            self.assertGreaterEqual(pass_ratio, 0.65, f"sinch.mp4 helmet pass ratio ({pass_ratio:.2f}) must meet 0.65 threshold")

        cap_video_path = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'videos', 'test_CAND-2026-060_cap_video.mp4')
        if os.path.exists(cap_video_path):
            cap = cv2.VideoCapture(cap_video_path)
            detector = YoloDetector()
            helmet_pass = 0
            total_rider = 0
            for _ in range(60):
                ret, frame = cap.read()
                if not ret:
                    break
                dets, _ = detector.detect_objects(frame)
                if dets['rider_found']:
                    total_rider += 1
                    if dets['helmet_detected']:
                        helmet_pass += 1
            cap.release()
            if total_rider > 0:
                pass_ratio = helmet_pass / total_rider
                self.assertLess(pass_ratio, 0.20, f"Cap video should not pass helmet validation (got {pass_ratio:.2f})")


if __name__ == '__main__':
    unittest.main()
