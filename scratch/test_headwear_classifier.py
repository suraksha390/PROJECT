import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
import cv2
import numpy as np

def extract_head_crop(rider_crop):
    """Extracts centered head ROI from rider bounding box crop."""
    rh, rw = rider_crop.shape[:2]
    head_h = max(10, int(rh * 0.28))
    # Head is centered horizontally within the rider crop
    head_w = int(rw * 0.65)
    cx = rw // 2
    x1 = max(0, cx - head_w // 2)
    x2 = min(rw, cx + head_w // 2)
    return rider_crop[:head_h, x1:x2]

def classify_head_features(head_roi):
    if head_roi is None or head_roi.size == 0:
        return {'class': 'none', 'status': 'FAIL', 'conf': 0.0, 'details': 'Empty'}
    
    h, w = head_roi.shape[:2]
    if w < 20 or h < 20:
        return {'class': 'uncertain', 'status': 'Needs Review', 'conf': 0.0, 'details': 'Small'}
    
    gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 25.0:
        return {'class': 'uncertain', 'status': 'Needs Review', 'conf': 0.0, 'details': f'Blurred ({lap_var:.1f})'}

    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(head_roi, cv2.COLOR_BGR2YCrCb)
    cr = ycbcr[:, :, 1]
    cb = ycbcr[:, :, 2]
    
    # 1. Biological human skin (YCbCr + RGB)
    # Strictly avoids false positives on brown wood, concrete, or yellow asphalt
    b, g, r = head_roi[:,:,0], head_roi[:,:,1], head_roi[:,:,2]
    skin_mask = ((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 125) &
                 (r > g + 10) & (g > b + 2))
    
    total_px = float(h * w + 1e-5)
    skin_ratio = float(np.sum(skin_mask)) / total_px
    
    # Split into crown (top 50%) and face (lower 50%)
    mid_y = int(h * 0.50)
    crown = head_roi[:mid_y, :]
    face = head_roi[mid_y:, :]
    
    crown_skin = float(np.sum(skin_mask[:mid_y, :])) / float(crown.size // 3 + 1e-5)
    face_skin = float(np.sum(skin_mask[mid_y:, :])) / float(face.size // 3 + 1e-5)
    
    # 2. Local texture variation across crown
    # Smooth rigid helmet shell has low local patch std; printed fabric/scarf has high
    crown_gray = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = 5
    m_cr = cv2.blur(crown_gray, (k, k))
    sq_cr = cv2.blur(crown_gray**2, (k, k))
    local_std = np.sqrt(np.maximum(0, sq_cr - m_cr**2))
    high_texture_ratio = float(np.mean(local_std > 22.0))
    median_local_std = float(np.median(local_std))
    
    # 3. Specular highlight on smooth curved helmet shell
    crown_hsv = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
    highlight_mask = (crown_hsv[:, :, 2] >= 170) & (crown_hsv[:, :, 1] <= 45)
    specular_ratio = float(np.mean(highlight_mask))
    
    # 4. Cap brim detection (horizontal edge line in middle-lower third of head)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    lower_edges = edges[int(h * 0.35):int(h * 0.80), :]
    k_w = max(5, int(w * 0.30))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    h_lines = cv2.morphologyEx(lower_edges, cv2.MORPH_OPEN, h_kernel)
    brim_score = float(np.sum(h_lines > 0)) / float(w + 1e-5)

    # 5. Natural human hair detection (dark fibrous texture on crown)
    # Hair is dark (V < 55 or gray < 55) with low saturation
    hair_mask = (crown_hsv[:, :, 2] < 55) & (crown_hsv[:, :, 1] < 70)
    hair_ratio = float(np.mean(hair_mask))
    
    # DECISION RULES:
    # A. Uncovered bare head: High face skin and hair on crown
    if (skin_ratio > 0.30 and face_skin > 0.35) or (crown_skin > 0.25 and face_skin > 0.30) or (hair_ratio > 0.35 and face_skin > 0.25):
        conf = min(0.95, round(max(skin_ratio, face_skin), 2))
        return {'class': 'no_helmet', 'status': 'FAIL', 'conf': conf, 'details': f'Bare head (face skin: {face_skin:.2f}, hair: {hair_ratio:.2f})'}
        
    # B. Cap / Hat: Visor / brim detected or exposed lower face with covered crown and high brim
    if brim_score > 0.25:
        return {'class': 'other_headwear', 'status': 'FAIL', 'conf': 0.85, 'details': f'Cap brim/visor (brim score: {brim_score:.2f})'}
        
    # C. Printed / woven cloth scarf / towel / dupatta / turban
    # High local texture variation across crown (fabric folds/print) and low specular shell shine
    if high_texture_ratio > 0.50 and median_local_std > 25.0:
        return {'class': 'other_headwear', 'status': 'FAIL', 'conf': 0.85, 'details': f'Fabric scarf/cloth texture (texture ratio: {high_texture_ratio:.2f}, std: {median_local_std:.1f})'}

    # D. Genuine motorcycle safety helmet
    # Low crown skin (<= 0.15), smooth shell surface (high_texture_ratio <= 0.45 or specular shine)
    if crown_skin <= 0.15 and (high_texture_ratio <= 0.45 or specular_ratio > 0.08):
        conf = round(0.85 + min(0.10, specular_ratio), 2)
        return {'class': 'helmet', 'status': 'PASS', 'conf': conf, 'details': f'Motorcycle safety helmet (specular: {specular_ratio:.2f}, texture: {high_texture_ratio:.2f})'}

    return {'class': 'uncertain', 'status': 'Needs Review', 'conf': 0.50, 'details': f'Ambiguous (skin: {skin_ratio:.2f}, text: {high_texture_ratio:.2f})'}


from models.yolo_detector import yolo_detector

videos = {
    'suraksha_with_helmet': 'static/uploads/videos/suraksha_with_helmet.mp4',
    'suraksha_driving (scarf)': 'static/uploads/videos/suraksha_driving.mp4',
    'sinch_helmet': 'static/uploads/videos/test_CAND-2026-063_sinch.mp4',
    'sinch_without_helmet': 'static/uploads/videos/test_CAND-2026-064_sinch_without_helmet.mp4',
    'cap_video': 'static/uploads/videos/test_CAND-2026-060_cap_video.mp4'
}

for name, path in videos.items():
    if not os.path.exists(path):
        print(f"Skipping {name}")
        continue
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    results = {}
    f = 0
    while True:
        ret, fr = cap.read()
        if not ret: break
        f += 1
        if f % 2 != 0: continue  # sample every 2 frames for speed
        dets, _ = yolo_detector.detect_objects(fr)
        rb = dets.get('rider_bbox')
        if rb:
            rx1, ry1, rx2, ry2 = rb
            head_crop = extract_head_crop(fr[ry1:ry2, rx1:rx2])
            res = classify_head_features(head_crop)
            cls = res['class']
            results[cls] = results.get(cls, 0) + 1
    cap.release()
    print(f"[{name}] {f} frames -> {results}")
