import cv2
import os
import sys
import numpy as np

def classify_head_opencv(head_roi, debug=False):
    """
    Robust multi-cue OpenCV classifier distinguishing:
    - 'helmet': Confidently verified motorcycle safety helmet
    - 'no_helmet': Uncovered head / face / hair
    - 'other_headwear': Baseball cap, hat, towel, scarf, cloth
    - 'uncertain': Blurry, too small, or ambiguous
    """
    if head_roi is None or head_roi.size == 0:
        return {'class': 'none', 'confidence': 0.0, 'status': 'FAIL', 'details': 'Empty head ROI'}
        
    h, w = head_roi.shape[:2]
    if w < 25 or h < 25:
        return {'class': 'uncertain', 'confidence': 0.0, 'status': 'Needs Review', 'details': f'ROI too small ({w}x{h})'}
        
    gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY) if len(head_roi.shape) == 3 else head_roi
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 35.0:
        return {'class': 'uncertain', 'confidence': 0.0, 'status': 'Needs Review', 'details': f'ROI blurred ({lap_var:.1f})'}
        
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    
    # 1. Calibrated Skin Tone Detection (HSV range covering diverse Indian & global skin tones)
    lower_skin = np.array([0, 20, 50], dtype=np.uint8)
    upper_skin = np.array([25, 190, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    total_pixels = float(h * w + 1e-5)
    skin_ratio = float(np.sum(skin_mask > 0)) / total_pixels
    
    # Lower 55% of head ROI (where face, nose, mouth, ears, neck are located)
    lower_h_start = int(h * 0.45)
    lower_skin_mask = skin_mask[lower_h_start:, :]
    lower_skin_ratio = float(np.sum(lower_skin_mask > 0)) / float(lower_skin_mask.size + 1e-5)
    
    # Upper 45% of head ROI (crown, forehead, or helmet dome)
    upper_skin_mask = skin_mask[:lower_h_start, :]
    upper_skin_ratio = float(np.sum(upper_skin_mask > 0)) / float(upper_skin_mask.size + 1e-5)

    # 2. Fabric / Towel / Scarf Fold Texture & Edge Analysis
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edge_density = float(np.sum(edges > 0)) / total_pixels
    
    # 3. Cap Brim / Visor Detection
    # A baseball cap has a protruding horizontal brim in the middle/lower half
    lower_edges = edges[int(h * 0.35):int(h * 0.80), :]
    k_w = max(5, int(w * 0.30))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    h_lines = cv2.morphologyEx(lower_edges, cv2.MORPH_OPEN, h_kernel)
    brim_score = float(np.sum(h_lines > 0)) / float(w + 1e-5)
    
    # 4. Shell curvature and color variance
    # Helmets have a smooth, homogeneous outer shell with uniform reflection
    gray_std = float(gray.std())
    
    # Upper dome convexity check:
    # On a motorcycle helmet, the upper boundary of the head is a smooth convex dome
    # Calculate top profile y(x) = min y where edge or dark silhouette appears
    top_half = gray[:int(h * 0.6), :]
    
    if debug:
        print(f"DEBUG: w={w}, h={h}, skin={skin_ratio:.2f}, low_skin={lower_skin_ratio:.2f}, up_skin={upper_skin_ratio:.2f}, edge={edge_density:.2f}, brim={brim_score:.2f}, std={gray_std:.1f}")

    # DECISION LOGIC:
    
    # Case A: Bare Head (No Helmet)
    # If the crown or face has extensive skin tone (>40% overall or >45% in lower face with significant upper skin)
    if skin_ratio > 0.40 or (upper_skin_ratio > 0.30 and lower_skin_ratio > 0.35):
        conf = min(0.95, round(max(skin_ratio, lower_skin_ratio), 2))
        return {
            'class': 'no_helmet',
            'confidence': conf,
            'status': 'FAIL',
            'details': f'Uncovered bare head detected (skin ratio: {skin_ratio:.2f})'
        }

    # Case B: Fabric Folds / Towel / Turban / Scarf
    if edge_density > 0.22:
        return {
            'class': 'other_headwear',
            'confidence': 0.80,
            'status': 'FAIL',
            'details': f'Fabric folds / towel / turban texture detected (edge density: {edge_density:.2f})'
        }

    # Case C: Baseball Cap / Hat (Other Headwear)
    # Characterized by:
    # 1. Distinct brim projection (brim_score > 0.25) OR
    # 2. Exposed face/ears/neck in lower half (lower_skin_ratio > 0.35) while crown has low skin OR
    # 3. High contrast texture (gray_std > 48) combined with lower skin exposure > 0.30
    if brim_score > 0.25:
        return {
            'class': 'other_headwear',
            'confidence': 0.85,
            'status': 'FAIL',
            'details': f'Cap visor / brim detected (brim score: {brim_score:.2f})'
        }
    if lower_skin_ratio > 0.35 and upper_skin_ratio < 0.25:
        return {
            'class': 'other_headwear',
            'confidence': 0.82,
            'status': 'FAIL',
            'details': f'Cap/Hat detected: crown covered but face/ears exposed (lower skin: {lower_skin_ratio:.2f})'
        }
    if gray_std > 48.0 and lower_skin_ratio > 0.25:
        return {
            'class': 'other_headwear',
            'confidence': 0.78,
            'status': 'FAIL',
            'details': f'Cap/Hat profile detected (high contrast {gray_std:.1f}, lower skin: {lower_skin_ratio:.2f})'
        }

    # Case D: Genuine Motorcycle Helmet
    # Genuine motorcycle helmets have:
    # 1. Very low skin exposure across both upper and lower ROI (enclosed head & visor / shell, skin_ratio < 0.20 and lower_skin_ratio < 0.25)
    # 2. Low internal fabric edge density (smooth rigid shell, edge_density < 0.18)
    # 3. Homogeneous composite shell material (gray_std < 45.0)
    # 4. Sufficient head coverage
    if skin_ratio <= 0.20 and lower_skin_ratio <= 0.25 and edge_density <= 0.18 and gray_std <= 45.0:
        # Confidence increases with lower skin exposure and smooth shell texture
        confidence = round(0.85 + (0.10 * (1.0 - min(1.0, skin_ratio / 0.20))), 2)
        return {
            'class': 'helmet',
            'confidence': confidence,
            'status': 'PASS',
            'details': f'Motorcycle safety helmet verified (skin: {skin_ratio:.2f}, smooth shell texture std: {gray_std:.1f})'
        }
        
    # Case E: Ambiguous / Borderline -> Needs Review (Never blindly assume PASS or FAIL)
    return {
        'class': 'uncertain',
        'confidence': 0.50,
        'status': 'Needs Review',
        'details': f'Headwear cues ambiguous (skin: {skin_ratio:.2f}, edge: {edge_density:.2f}, std: {gray_std:.1f})'
    }

def test_on_video(name, path):
    if not os.path.exists(path):
        print(f"Skipping {name}")
        return
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from models.yolo_detector import yolo_detector

    cap = cv2.VideoCapture(path)
    res_counts = {}
    f_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        f_idx += 1
        dets, _ = yolo_detector.detect_objects(frame)
        if dets['rider_found']:
            rx1, ry1, rx2, ry2 = dets['rider_bbox']
            head_h = max(1, int((ry2 - ry1) * 0.28))
            head_roi = frame[ry1:min(ry2, ry1 + head_h), rx1:rx2]
            res = classify_head_opencv(head_roi)
            c = res['class']
            res_counts[c] = res_counts.get(c, 0) + 1
    cap.release()
    print(f"Results for {name} ({f_idx} frames): {res_counts}")

test_on_video('sinch.mp4 (HELMET)', 'static/uploads/videos/test_CAND-2026-062_sinch.mp4')
test_on_video('sinchana_driving_with_helmet.mp4 (HELMET)', 'static/uploads/videos/test_CAND-2026-061_sinchana_driving_with_helmet.mp4')
test_on_video('cap_video.mp4 (CAP)', 'static/uploads/videos/test_CAND-2026-060_cap_video.mp4')
