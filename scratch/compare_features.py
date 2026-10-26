import cv2
import os
import glob
import numpy as np

def analyze_crop(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Skin detection in lower 60% of ROI (face, chin, neck)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([25, 180, 255], dtype=np.uint8)
    skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = float(np.sum(skin_mask > 0)) / float(h * w + 1e-5)
    
    # Lower half skin ratio (in cap/bare head, ears/face/neck are visible; in full/3-4 helmet, visor or shell covers it)
    lower_half = skin_mask[int(h * 0.4):, :]
    lower_skin_ratio = float(np.sum(lower_half > 0)) / float(lower_half.size + 1e-5)

    # 2. Edges
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    edge_density = float(np.sum(edges > 0)) / float(h * w + 1e-5)

    # 3. Contour geometry
    # Threshold to find main head/headwear silhouette
    # Otsu threshold or adaptive
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Also gradient magnitude for silhouette
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt = None
    max_area = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > max_area:
            max_area = area
            best_cnt = cnt
            
    circularity = 0.0
    solidity = 0.0
    aspect_ratio = float(w) / float(h)
    extent = 0.0
    if best_cnt is not None and max_area > 0:
        perimeter = cv2.arcLength(best_cnt, True)
        if perimeter > 0:
            circularity = (4 * np.pi * max_area) / (perimeter * perimeter)
        hull = cv2.convexHull(best_cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = max_area / hull_area
        extent = max_area / float(w * h)

    # 4. Color variation (standard deviation of hue and saturation)
    h_std = float(hsv[:, :, 0].std())
    s_std = float(hsv[:, :, 1].std())
    v_std = float(hsv[:, :, 2].std())
    
    # 5. Cap brim check (horizontal protrusion in lower middle)
    lower_edges = edges[int(h * 0.35):int(h * 0.8), :]
    k_w = max(5, w // 3)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_w, 1))
    h_lines = cv2.morphologyEx(lower_edges, cv2.MORPH_OPEN, h_kernel)
    brim_score = float(np.sum(h_lines > 0)) / float(w + 1e-5)

    basename = os.path.basename(img_path)
    print(f"{basename:20s} | {w}x{h:3d} | AR:{aspect_ratio:.2f} | Skin:{skin_ratio:.2f} (Low:{lower_skin_ratio:.2f}) | Edge:{edge_density:.2f} | Circ:{circularity:.2f} | Sol:{solidity:.2f} | Brim:{brim_score:.2f} | H_std:{h_std:.1f} V_std:{v_std:.1f}")

print("=== SINCH (HELMET) SAMPLES ===")
for p in sorted(glob.glob('scratch/samples/sinch_helmet_*.jpg')):
    analyze_crop(p)

print("\n=== CAP VIDEO SAMPLES ===")
for p in sorted(glob.glob('scratch/samples/cap_video_*.jpg')):
    analyze_crop(p)
