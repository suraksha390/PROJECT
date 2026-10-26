import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
import cv2
import numpy as np
from models.yolo_detector import yolo_detector

def extract_refined_head(frame, rider_bbox):
    rx1, ry1, rx2, ry2 = rider_bbox
    rw = rx2 - rx1
    rh = ry2 - ry1
    
    # Head height: top 26% of rider
    head_h = max(10, int(rh * 0.26))
    head_y1 = ry1
    head_y2 = min(ry2, ry1 + head_h)
    
    # Head width: centered horizontally within rider box, approx 60% of rider width
    head_w = int(rw * 0.60)
    cx = (rx1 + rx2) // 2
    head_x1 = max(0, cx - head_w // 2)
    head_x2 = min(frame.shape[1], cx + head_w // 2)
    
    return frame[head_y1:head_y2, head_x1:head_x2], (head_x1, head_y1, head_x2, head_y2)

def evaluate_head_sample(head_roi):
    h, w = head_roi.shape[:2]
    
    # 1. Distant / Low-resolution check
    if w < 28 or h < 28:
        return 'uncertain', 0.0, f'Too small ({w}x{h})'
        
    gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 30.0:
        return 'uncertain', 0.0, f'Blurred (Laplacian {lap_var:.1f})'
        
    hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
    ycbcr = cv2.cvtColor(head_roi, cv2.COLOR_BGR2YCrCb)
    cr = ycbcr[:, :, 1]
    cb = ycbcr[:, :, 2]
    b, g, r = head_roi[:,:,0], head_roi[:,:,1], head_roi[:,:,2]
    
    # True biological human skin mask
    skin_mask = ((cr >= 135) & (cr <= 175) & (cb >= 85) & (cb <= 125) &
                 (r > g + 10) & (g > b + 2))
    total_px = float(h * w + 1e-5)
    skin_ratio = float(np.sum(skin_mask)) / total_px
    
    mid_y = int(h * 0.45)
    crown = head_roi[:mid_y, :]
    face = head_roi[mid_y:, :]
    
    crown_skin = float(np.sum(skin_mask[:mid_y, :])) / float(crown.size // 3 + 1e-5)
    face_skin = float(np.sum(skin_mask[mid_y:, :])) / float(face.size // 3 + 1e-5)
    
    # Ear / side-head skin check (columns on outer 25% of left and right, in lower half)
    ear_left = skin_mask[mid_y:, :int(w * 0.25)]
    ear_right = skin_mask[mid_y:, int(w * 0.75):]
    ear_skin_ratio = max(
        float(np.mean(ear_left)) if ear_left.size > 0 else 0,
        float(np.mean(ear_right)) if ear_right.size > 0 else 0
    )
    
    # Cap visor / horizontal brim detection
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    lower_edges = edges[int(h * 0.35):int(h * 0.75), :]
    k_w = max(5, int(w * 0.28))
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    h_lines = cv2.morphologyEx(lower_edges, cv2.MORPH_OPEN, h_kernel)
    brim_score = float(np.sum(h_lines > 0)) / float(w + 1e-5)
    
    # Fabric pattern / local texture in crown
    crown_gray = cv2.cvtColor(crown, cv2.COLOR_BGR2GRAY).astype(np.float32)
    k = 5
    m_cr = cv2.blur(crown_gray, (k, k))
    sq_cr = cv2.blur(crown_gray**2, (k, k))
    local_std = np.sqrt(np.maximum(0, sq_cr - m_cr**2))
    high_texture_ratio = float(np.mean(local_std > 22.0))
    median_local_std = float(np.median(local_std))
    
    # Specular reflection highlight on rigid curved helmet dome
    crown_hsv = cv2.cvtColor(crown, cv2.COLOR_BGR2HSV)
    highlight_mask = (crown_hsv[:, :, 2] >= 170) & (crown_hsv[:, :, 1] <= 45)
    specular_ratio = float(np.mean(highlight_mask))
    
    # Hair detection on crown
    hair_mask = (crown_hsv[:, :, 2] < 55) & (crown_hsv[:, :, 1] < 70)
    hair_ratio = float(np.mean(hair_mask))
    
    # Decision tree:
    # 1. Bare head (no helmet):
    if (skin_ratio > 0.25 and face_skin > 0.30) or (crown_skin > 0.20 and face_skin > 0.25) or (hair_ratio > 0.35 and face_skin > 0.25):
        return 'no_helmet', min(0.95, max(skin_ratio, face_skin)), 'Bare head'
        
    # 2. Baseball cap / hat:
    if brim_score > 0.25 or (ear_skin_ratio > 0.20 and crown_skin < 0.15 and hair_ratio > 0.20):
        return 'other_headwear', 0.85, 'Cap / Hat'
        
    # 3. Fabric cloth / scarf / turban:
    if high_texture_ratio > 0.48 and median_local_std > 22.0:
        return 'other_headwear', 0.85, 'Fabric scarf / cloth'
        
    # 4. Genuine helmet:
    # Crown has no skin (<= 0.12), ears covered (ear_skin_ratio <= 0.15),
    # smooth shell (high_texture_ratio <= 0.45 or specular highlight present)
    if crown_skin <= 0.12 and ear_skin_ratio <= 0.15 and (high_texture_ratio <= 0.45 or specular_ratio > 0.08):
        return 'helmet', 0.90, 'Motorcycle helmet'
        
    return 'uncertain', 0.50, 'Ambiguous'

# Let's test this on our 5 test videos!
videos = [
    ('suraksha_with_helmet', 'static/uploads/videos/suraksha_with_helmet.mp4', 'EXPECT HELMET PASS'),
    ('suraksha_driving (scarf)', 'static/uploads/videos/suraksha_driving.mp4', 'EXPECT OTHER_HEADWEAR FAIL'),
    ('sinch_helmet', 'static/uploads/videos/test_CAND-2026-063_sinch.mp4', 'EXPECT HELMET PASS'),
    ('sinch_without_helmet', 'static/uploads/videos/test_CAND-2026-064_sinch_without_helmet.mp4', 'EXPECT NO_HELMET FAIL'),
    ('cap_video', 'static/uploads/videos/test_CAND-2026-060_cap_video.mp4', 'EXPECT OTHER_HEADWEAR FAIL')
]

for label, vpath, expected in videos:
    cap = cv2.VideoCapture(vpath)
    counts = {}
    f = 0
    while True:
        ret, fr = cap.read()
        if not ret: break
        f += 1
        if f % 2 != 0: continue
        dets, _ = yolo_detector.detect_objects(fr)
        rb = dets.get('rider_bbox')
        if rb:
            head_roi, _ = extract_refined_head(fr, rb)
            cls, conf, det_str = evaluate_head_sample(head_roi)
            counts[cls] = counts.get(cls, 0) + 1
    cap.release()
    print(f"[{label}] ({expected}) -> {counts}")
