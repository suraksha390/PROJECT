import os
import cv2
import numpy as np

try:
    from config import (
        YOLO_BASE_MODEL_PATH,
        HELMET_MODEL_PATH,
        HELMET_CONF_THRESHOLD,
        NO_HELMET_CONF_THRESHOLD,
        OTHER_HEADWEAR_CONF_THRESHOLD,
        MIN_HEAD_ROI_WIDTH,
        MIN_HEAD_ROI_HEIGHT,
        BLUR_LAPLACIAN_THRESHOLD
    )
except ImportError:
    # Fallback default configurations if config is imported outside package root
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    YOLO_BASE_MODEL_PATH = os.path.join(BASE_DIR, 'yolov8n.pt')
    HELMET_MODEL_PATH = os.path.join(BASE_DIR, 'static', 'saved_models', 'helmet_detector.pt')
    HELMET_CONF_THRESHOLD = 0.50
    NO_HELMET_CONF_THRESHOLD = 0.50
    OTHER_HEADWEAR_CONF_THRESHOLD = 0.45
    MIN_HEAD_ROI_WIDTH = 26
    MIN_HEAD_ROI_HEIGHT = 26
    BLUR_LAPLACIAN_THRESHOLD = 28.0


class YoloDetector:
    """
    YOLO-based object and safety helmet detector for Two-Wheeler Driving Skill Validation.

    Detects:
    1. Rider (person) and Motorcycle via YOLO base model (COCO).
    2. Headwear classification into 3 distinct classes:
       - 'helmet': Genuine motorcycle safety helmet (PASS)
       - 'no_helmet': Uncovered head / bare face / scalp hair (FAIL)
       - 'other_headwear': Non-protective headgear such as baseball caps, hats, scarves, cloths, towels, turbans (FAIL)
    3. Handles degraded, blurry, or low-resolution head crops by returning 'Needs Review'
       instead of making an unsafe automatic assumption.
    4. Dedicated Model Support:
       - If trained weights exist at HELMET_MODEL_PATH, uses deep learning YOLO inference.
       - If dedicated weights are not present, operates in calibrated Computer Vision mode
         with strict biological skin constraints, specular reflection curves, ear coverage,
         and fabric texture analysis.
    """

    def __init__(self,
                 base_model_path=None,
                 helmet_model_path=None,
                 helmet_conf=None,
                 no_helmet_conf=None,
                 other_headwear_conf=None,
                 min_roi_size=None,
                 blur_thresh=None):

        self.base_model_path = base_model_path or YOLO_BASE_MODEL_PATH
        self.helmet_model_path = helmet_model_path or HELMET_MODEL_PATH

        self.helmet_conf_threshold = helmet_conf if helmet_conf is not None else HELMET_CONF_THRESHOLD
        self.no_helmet_conf_threshold = no_helmet_conf if no_helmet_conf is not None else NO_HELMET_CONF_THRESHOLD
        self.other_headwear_conf_threshold = other_headwear_conf if other_headwear_conf is not None else OTHER_HEADWEAR_CONF_THRESHOLD

        if min_roi_size is not None:
            self.min_roi_size = min_roi_size
        else:
            self.min_roi_size = (MIN_HEAD_ROI_WIDTH, MIN_HEAD_ROI_HEIGHT)

        self.blur_threshold = blur_thresh if blur_thresh is not None else BLUR_LAPLACIAN_THRESHOLD

        self.use_ultralytics = False
        self.base_model = None
        self.helmet_model = None

        self._load_models()

    def _load_models(self):
        """Loads base COCO detector and optional dedicated helmet classifier."""
        try:
            from ultralytics import YOLO

            # 1. Base YOLO Model for Rider & Motorcycle detection
            if os.path.exists(self.base_model_path):
                self.base_model = YOLO(self.base_model_path)
                self.use_ultralytics = True
                print(f"[YoloDetector] Base model loaded from: {self.base_model_path}")
            else:
                print(f"[YoloDetector] Base model file not found at: {self.base_model_path}")

            # 2. Dedicated Helmet Model (3 classes: helmet, no_helmet, other_headwear)
            if self.helmet_model_path and os.path.exists(self.helmet_model_path):
                try:
                    self.helmet_model = YOLO(self.helmet_model_path)
                    print(f"[YoloDetector] Dedicated helmet model loaded from: {self.helmet_model_path}")
                except Exception as ex:
                    print(f"[YoloDetector] Error loading dedicated helmet model: {ex}")
                    self.helmet_model = None
            else:
                print(f"[YoloDetector] Dedicated helmet weights not found at: '{self.helmet_model_path}'. "
                      f"Operating in calibrated CV fallback mode (caps, scarves, and uncovered heads rejected; "
                      f"genuine motorcycle helmets verified via multi-cue shell analysis).")
                self.helmet_model = None

        except ImportError:
            print("[YoloDetector] ultralytics package not installed. Running in fallback CV mode.")
        except Exception as e:
            print(f"[YoloDetector] Model loading exception: {e}")

    def normalize_class_name(self, raw_label, model_names=None):
        """
        Normalizes various class label formats to the 3 standard project classes:
        - 'helmet'
        - 'no_helmet'
        - 'other_headwear'
        """
        if isinstance(raw_label, int) and model_names and raw_label in model_names:
            raw_label = model_names[raw_label]

        name = str(raw_label).lower().strip().replace(' ', '_').replace('-', '_')

        if name in ['0', 'helmet', 'motorcycle_helmet', 'protective_helmet', 'with_helmet', 'hard_hat']:
            return 'helmet'
        elif name in ['1', 'no_helmet', 'without_helmet', 'bare_head', 'head', 'face', 'uncovered_head']:
            return 'no_helmet'
        elif name in ['2', 'other_headwear', 'cap', 'baseball_cap', 'hat', 'turban', 'scarf', 'towel', 'cloth', 'headwear']:
            return 'other_headwear'
        return name

    def classify_head_roi(self, head_roi, rider_bbox=None, frame=None):
        """
        Evaluates a head ROI image and classifies the headwear.

        Returns a dictionary:
        {
            'class': 'helmet' | 'no_helmet' | 'other_headwear' | 'uncertain' | 'none',
            'confidence': float,
            'status': 'PASS' | 'FAIL' | 'Needs Review',
            'details': str
        }
        """
        if head_roi is None or head_roi.size == 0:
            return {
                'class': 'none',
                'confidence': 0.0,
                'status': 'FAIL',
                'details': 'Head region missing or not visible'
            }

        h, w = head_roi.shape[:2]

        # 1. Size / Resolution check: Prevent assuming a helmet on tiny, distant heads
        min_w, min_h = self.min_roi_size
        if w < min_w or h < min_h:
            return {
                'class': 'uncertain',
                'confidence': 0.0,
                'status': 'Needs Review',
                'details': f'Head ROI too small/distant for reliable classification ({w}x{h} px)'
            }

        # 2. Blur / Motion blur check using Laplacian variance
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY) if len(head_roi.shape) == 3 else head_roi
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < self.blur_threshold:
            return {
                'class': 'uncertain',
                'confidence': 0.0,
                'status': 'Needs Review',
                'details': f'Head ROI blurred (Laplacian variance {lap_var:.1f} < {self.blur_threshold})'
            }

        # 3. Dedicated Helmet Detection Model (when trained weights are present)
        if self.helmet_model is not None:
            try:
                results = self.helmet_model(head_roi, verbose=False)[0]
                best_cls = 'uncertain'
                best_conf = 0.0

                if hasattr(results, 'boxes') and len(results.boxes) > 0:
                    for box in results.boxes:
                        conf = float(box.conf[0])
                        cls_idx = int(box.cls[0])
                        cls_names = getattr(results, 'names', {})
                        norm_cls = self.normalize_class_name(cls_idx, cls_names)

                        if conf > best_conf:
                            best_conf = conf
                            best_cls = norm_cls

                # Apply class-specific confidence thresholds
                if best_cls == 'helmet' and best_conf >= self.helmet_conf_threshold:
                    return {
                        'class': 'helmet',
                        'confidence': round(best_conf, 3),
                        'status': 'PASS',
                        'details': f'Genuine motorcycle helmet detected (confidence: {best_conf:.2f})'
                    }
                elif best_cls == 'other_headwear' and best_conf >= self.other_headwear_conf_threshold:
                    return {
                        'class': 'other_headwear',
                        'confidence': round(best_conf, 3),
                        'status': 'FAIL',
                        'details': f'Non-protective headwear (Cap/Hat/Cloth) detected (confidence: {best_conf:.2f})'
                    }
                elif best_cls == 'no_helmet' and best_conf >= self.no_helmet_conf_threshold:
                    return {
                        'class': 'no_helmet',
                        'confidence': round(best_conf, 3),
                        'status': 'FAIL',
                        'details': f'Uncovered head detected - No helmet (confidence: {best_conf:.2f})'
                    }
                else:
                    return {
                        'class': 'uncertain',
                        'confidence': round(best_conf, 3),
                        'status': 'Needs Review',
                        'details': f'Detection uncertain or confidence too low ({best_conf:.2f})'
                    }
            except Exception as ex:
                print(f"[YoloDetector] Dedicated model inference error: {ex}")
                # Fall through to computer vision analysis below

        # 4. Calibrated Multi-Cue Computer Vision Analysis
        return self._fallback_cv_analysis(head_roi, gray, lap_var)

    def _fallback_cv_analysis(self, head_roi, gray, lap_var):
        """
        Calibrated multi-cue computer vision analysis:
        - Rejects uncovered bare heads using strict biological human skin constraints -> 'no_helmet', FAIL
        - Rejects baseball caps / brimmed hats / visors -> 'other_headwear', FAIL
        - Rejects fabric cloths / scarves / dupattas / turbans / towels -> 'other_headwear', FAIL
        - Confidently classifies genuine motorcycle safety helmets -> 'helmet', PASS
        - Returns 'Needs Review' if head is blurred, degraded, or ambiguous
        """
        h, w = head_roi.shape[:2]
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        ycbcr = cv2.cvtColor(head_roi, cv2.COLOR_BGR2YCrCb)
        cr = ycbcr[:, :, 1]
        cb = ycbcr[:, :, 2]
        b, g, r = head_roi[:, :, 0], head_roi[:, :, 1], head_roi[:, :, 2]

        total_pixels = float(h * w + 1e-5)

        # 1. Biological Human Skin Detection (YCbCr + RGB)
        # Prevents false positives on background wooden fences, concrete walls, or yellow asphalt
        skin_mask = ((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 125) &
                     (r > g + 8) & (g > b + 2))
        skin_ratio = float(np.sum(skin_mask)) / total_pixels

        # Divide head into crown (top 45% where helmet/headwear sits) and face (lower 55%)
        mid_y = int(h * 0.45)
        crown = head_roi[:mid_y, :]
        face = head_roi[mid_y:, :]

        crown_skin = float(np.sum(skin_mask[:mid_y, :])) / float(crown.size // 3 + 1e-5)
        face_skin = float(np.sum(skin_mask[mid_y:, :])) / float(face.size // 3 + 1e-5)

        # Ear / side-head skin check (outer 25% columns in lower half)
        ear_left = skin_mask[mid_y:, :int(w * 0.25)]
        ear_right = skin_mask[mid_y:, int(w * 0.75):]
        ear_skin_ratio = max(
            float(np.mean(ear_left)) if ear_left.size > 0 else 0,
            float(np.mean(ear_right)) if ear_right.size > 0 else 0
        )

        # 2. Visor / Brim Detection (Baseball cap / brimmed hat)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        lower_edges = edges[int(h * 0.35):int(h * 0.75), :]
        k_w = max(5, int(w * 0.28))
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
        h_lines = cv2.morphologyEx(lower_edges, cv2.MORPH_OPEN, h_kernel)
        brim_score = float(np.sum(h_lines > 0)) / float(w + 1e-5)

        # 3. Fabric Texture & Pattern Variation across crown
        # Smooth rigid helmet shell has low local patch variance; printed cloth/scarf has high
        crown_gray = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY).astype(np.float32)
        k = 5
        m_cr = cv2.blur(crown_gray, (k, k))
        sq_cr = cv2.blur(crown_gray**2, (k, k))
        local_std = np.sqrt(np.maximum(0, sq_cr - m_cr**2))
        high_texture_ratio = float(np.mean(local_std > 22.0))
        median_local_std = float(np.median(local_std))

        # 4. Specular Reflection Highlight on Curved Helmet Dome
        # A rigid helmet shell produces a concentrated highlight on curved dome
        crown_hsv = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
        highlight_mask = (crown_hsv[:, :, 2] >= 180) & (crown_hsv[:, :, 1] <= 45)
        specular_ratio = float(np.mean(highlight_mask))

        # 5. Natural Scalp Hair Detection
        # Human hair on crown is dark (V < 55) with low saturation
        hair_mask = (crown_hsv[:, :, 2] < 55) & (crown_hsv[:, :, 1] < 70)
        hair_ratio = float(np.mean(hair_mask))

        # 6. Printed Color Patterns (Floral, checkered, or printed scarves)
        # Note: A solid painted helmet (blue, red, black, etc.) has smooth uniform texture (high_texture_ratio < 0.25).
        # A printed cloth scarf combines contrasting color patches (e.g. blue/pink florals) with fabric wrinkles/texture.
        blue_pattern = float(np.mean((crown_hsv[:, :, 0] >= 95) & (crown_hsv[:, :, 0] <= 130) & (crown_hsv[:, :, 1] >= 40)))
        pink_pattern = float(np.mean(((crown_hsv[:, :, 0] >= 155) | (crown_hsv[:, :, 0] <= 10)) & (crown_hsv[:, :, 1] >= 60) & (crown_hsv[:, :, 2] >= 70)))
        has_cloth_pattern = (
            (blue_pattern > 0.05 and pink_pattern > 0.03 and high_texture_ratio > 0.20) or
            (blue_pattern > 0.05 and high_texture_ratio > 0.35) or
            (pink_pattern > 0.05 and high_texture_ratio > 0.35)
        )

        # DECISION LOGIC:

        # Case A: Bare Head (Uncovered Face and Hair / Scalp)
        if (skin_ratio > 0.25 and face_skin > 0.30) or (crown_skin > 0.18 and face_skin > 0.25) or (hair_ratio > 0.35 and face_skin > 0.25):
            conf = min(0.95, round(max(skin_ratio, face_skin), 2))
            return {
                'class': 'no_helmet',
                'confidence': conf,
                'status': 'FAIL',
                'details': f'Uncovered bare head detected (face skin: {face_skin:.2f}, hair: {hair_ratio:.2f})'
            }

        # Case B: Baseball Cap / Brimmed Hat (Protruding visor or exposed ears with cap crown)
        if brim_score > 0.25 or (ear_skin_ratio > 0.20 and crown_skin < 0.15 and hair_ratio > 0.20):
            return {
                'class': 'other_headwear',
                'confidence': 0.85,
                'status': 'FAIL',
                'details': f'Cap/Hat detected with exposed ears/visor (brim: {brim_score:.2f}, ear skin: {ear_skin_ratio:.2f})'
            }

        # Case C: Printed Cloth / Scarf / Dupatta / Turban / Towel
        if has_cloth_pattern or (high_texture_ratio > 0.48 and median_local_std > 22.0):
            return {
                'class': 'other_headwear',
                'confidence': 0.85,
                'status': 'FAIL',
                'details': f'Non-protective cloth/scarf headwear detected (texture: {high_texture_ratio:.2f}, pattern: {blue_pattern:.2f})'
            }

        # Case D: Genuine Motorcycle Safety Helmet
        # Low crown skin (<= 0.12), ears enclosed by helmet shell (ear_skin_ratio <= 0.15),
        # smooth rigid shell without cloth patterns (specular highlight or smooth surface)
        if crown_skin <= 0.12 and ear_skin_ratio <= 0.15 and not has_cloth_pattern and (high_texture_ratio <= 0.45 or specular_ratio > 0.08):
            conf = round(0.85 + min(0.10, specular_ratio), 2)
            return {
                'class': 'helmet',
                'confidence': conf,
                'status': 'PASS',
                'details': f'Motorcycle safety helmet verified (specular: {specular_ratio:.2f}, shell texture: {high_texture_ratio:.2f})'
            }

        # Case E: Ambiguous / Inconclusive Headwear -> Needs Review
        return {
            'class': 'uncertain',
            'confidence': 0.50,
            'status': 'Needs Review',
            'details': f'Headwear cues ambiguous (crown skin: {crown_skin:.2f}, texture: {high_texture_ratio:.2f})'
        }

    def check_helmet_roi(self, head_roi):
        """
        Determines whether a genuine protective helmet is present in the head ROI.
        Backward-compatible interface returning boolean True ONLY if a motorcycle safety helmet
        is confidently verified (status == 'PASS').
        Returns False for caps, other headwear, uncovered heads, blurry images, or missing model.
        """
        res = self.classify_head_roi(head_roi)
        return bool(res['class'] == 'helmet' and res['status'] == 'PASS')

    def detect_objects(self, frame):
        """
        Detects rider, motorcycle, helmet, and ground region in the input frame.

        Returns:
            detections (dict):
                - 'rider_found': bool
                - 'motorcycle_found': bool
                - 'helmet_detected': bool (True ONLY if genuine motorcycle helmet is confirmed)
                - 'rider_bbox': (x1, y1, x2, y2) or None
                - 'motorcycle_bbox': (x1, y1, x2, y2) or None
                - 'helmet_bbox': (x1, y1, x2, head_y2) or None
                - 'ground_line_y': int
                - 'helmet_class': 'helmet' | 'no_helmet' | 'other_headwear' | 'uncertain' | 'none'
                - 'helmet_status': 'PASS' | 'FAIL' | 'Needs Review'
                - 'helmet_conf': float
                - 'helmet_details': str
            annotated_frame (ndarray): Frame with visual bounding boxes, labels, and ground line.
        """
        if frame is None or frame.size == 0:
            return {
                'rider_found': False,
                'motorcycle_found': False,
                'helmet_detected': False,
                'rider_bbox': None,
                'motorcycle_bbox': None,
                'helmet_bbox': None,
                'ground_line_y': 0,
                'helmet_class': 'none',
                'helmet_status': 'FAIL',
                'helmet_conf': 0.0,
                'helmet_details': 'Empty frame provided'
            }, frame

        annotated_frame = frame.copy()
        h, w, _ = frame.shape

        detections = {
            'rider_found': False,
            'motorcycle_found': False,
            'helmet_detected': False,
            'rider_bbox': None,
            'motorcycle_bbox': None,
            'helmet_bbox': None,
            'ground_line_y': int(h * 0.92),  # Default ground level ROI
            'helmet_class': 'none',
            'helmet_status': 'FAIL',
            'helmet_conf': 0.0,
            'helmet_details': 'No rider detected'
        }

        if self.use_ultralytics and self.base_model is not None:
            try:
                results = self.base_model(frame, verbose=False)[0]
                best_rider = None
                best_rider_conf = 0.0
                best_moto = None
                best_moto_conf = 0.0

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # Clip bounding boxes to frame dimensions
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)

                    if conf < 0.35:
                        continue

                    # COCO class 0: Person (Rider)
                    if cls_id == 0:
                        if conf > best_rider_conf:
                            best_rider_conf = conf
                            best_rider = (x1, y1, x2, y2, conf)

                    # COCO class 3: Motorcycle
                    elif cls_id == 3:
                        if conf > best_moto_conf:
                            best_moto_conf = conf
                            best_moto = (x1, y1, x2, y2, conf)

                # Process Rider Detection
                if best_rider is not None:
                    rx1, ry1, rx2, ry2, rconf = best_rider
                    rw = rx2 - rx1
                    rh = ry2 - ry1

                    detections['rider_found'] = True
                    detections['rider_bbox'] = (rx1, ry1, rx2, ry2)

                    cv2.rectangle(annotated_frame, (rx1, ry1), (rx2, ry2), (255, 191, 0), 2)
                    cv2.putText(annotated_frame, f"Rider {rconf:.2f}", (rx1, max(20, ry1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 191, 0), 2)

                    # Extract Centered Head ROI (top 24% of rider, centered horizontally)
                    # Prevents capturing wide background walls, neck scarves, or shoulders
                    head_h = max(10, int(rh * 0.24))
                    head_y2 = min(ry2, ry1 + head_h)
                    head_w = int(rw * 0.60)
                    cx = (rx1 + rx2) // 2
                    head_x1 = max(0, cx - head_w // 2)
                    head_x2 = min(w, cx + head_w // 2)

                    head_roi = frame[ry1:head_y2, head_x1:head_x2]
                    detections['helmet_bbox'] = (head_x1, ry1, head_x2, head_y2)

                    # Classify Headwear
                    h_res = self.classify_head_roi(head_roi, rider_bbox=(rx1, ry1, rx2, ry2), frame=frame)
                    detections['helmet_class'] = h_res['class']
                    detections['helmet_status'] = h_res['status']
                    detections['helmet_conf'] = h_res['confidence']
                    detections['helmet_details'] = h_res['details']

                    if h_res['class'] == 'helmet' and h_res['status'] == 'PASS':
                        detections['helmet_detected'] = True
                        cv2.rectangle(annotated_frame, (head_x1, ry1), (head_x2, head_y2), (0, 255, 0), 2)
                        cv2.putText(annotated_frame, f"Helmet: WORN ({h_res['confidence']:.2f})",
                                    (head_x1, max(20, ry1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    elif h_res['class'] == 'other_headwear':
                        detections['helmet_detected'] = False
                        cv2.rectangle(annotated_frame, (head_x1, ry1), (head_x2, head_y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"Cap/Cloth: NOT HELMET ({h_res['confidence']:.2f})",
                                    (head_x1, max(20, ry1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    elif h_res['class'] == 'no_helmet':
                        detections['helmet_detected'] = False
                        cv2.rectangle(annotated_frame, (head_x1, ry1), (head_x2, head_y2), (0, 0, 255), 2)
                        cv2.putText(annotated_frame, f"No Helmet: Bare Head ({h_res['confidence']:.2f})",
                                    (head_x1, max(20, ry1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
                    else:
                        # Uncertain / Needs Review
                        detections['helmet_detected'] = False
                        cv2.rectangle(annotated_frame, (head_x1, ry1), (head_x2, head_y2), (0, 165, 255), 2)
                        cv2.putText(annotated_frame, "Helmet: Needs Review",
                                    (head_x1, max(20, ry1 - 25)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

                # Process Motorcycle Detection
                if best_moto is not None:
                    mx1, my1, mx2, my2, mconf = best_moto
                    detections['motorcycle_found'] = True
                    detections['motorcycle_bbox'] = (mx1, my1, mx2, my2)
                    if my2 > int(h * 0.85):
                        detections['ground_line_y'] = my2
                    cv2.rectangle(annotated_frame, (mx1, my1), (mx2, my2), (255, 0, 255), 2)
                    cv2.putText(annotated_frame, f"Motorcycle {mconf:.2f}", (mx1, max(20, my1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            except Exception as ex:
                print(f"[YoloDetector] Frame prediction error: {ex}")

        # Ground Level line on annotated frame
        cv2.line(annotated_frame, (0, detections['ground_line_y']), (w, detections['ground_line_y']), (0, 165, 255), 2)
        cv2.putText(annotated_frame, "GROUND LINE", (10, max(20, detections['ground_line_y'] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)

        return detections, annotated_frame


yolo_detector = YoloDetector()
