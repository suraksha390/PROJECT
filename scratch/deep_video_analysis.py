import cv2
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.yolo_detector import yolo_detector

videos = {
    'sinch_helmet': 'static/uploads/videos/test_CAND-2026-062_sinch.mp4',
    'cap_video': 'static/uploads/videos/test_CAND-2026-060_cap_video.mp4',
    'sinchana_driving_helmet': 'static/uploads/videos/test_CAND-2026-061_sinchana_driving_with_helmet.mp4',
}

def analyze_video(name, path):
    if not os.path.exists(path):
        print(f"Skipping {name}: not found")
        return
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"\n=================== {name} ({total} frames) ===================")
    
    skin_ratios = []
    face_skin_ratios = []
    edge_densities = []
    circularities = []
    solidities = []
    aspect_ratios = []
    color_stds = []
    lap_vars = []
    
    frame_idx = 0
    sampled = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 5 != 0:
            continue
            
        dets, _ = yolo_detector.detect_objects(frame)
        if not dets['rider_found']:
            continue
            
        rx1, ry1, rx2, ry2 = dets['rider_bbox']
        head_h = max(1, int((ry2 - ry1) * 0.28))
        head_roi = frame[ry1:min(ry2, ry1 + head_h), rx1:rx2]
        h, w = head_roi.shape[:2]
        if w < 20 or h < 20:
            continue
            
        sampled += 1
        gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
        
        # Skin in entire head ROI
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([25, 180, 255], dtype=np.uint8)
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        s_ratio = float(np.sum(skin_mask > 0)) / float(h * w)
        skin_ratios.append(s_ratio)
        
        # Skin in lower half of head ROI (where face/ears/chin appear)
        lower_half = skin_mask[int(h * 0.45):, :]
        f_ratio = float(np.sum(lower_half > 0)) / float(lower_half.size)
        face_skin_ratios.append(f_ratio)
        
        # Edges
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 40, 120)
        e_dens = float(np.sum(edges > 0)) / float(h * w)
        edge_densities.append(e_dens)
        
        # Laplacian
        lap = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lap_vars.append(lap)
        
        # Contour
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            max_c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(max_c)
            perimeter = cv2.arcLength(max_c, True)
            if perimeter > 0 and area > 50:
                circ = (4 * np.pi * area) / (perimeter * perimeter)
                circularities.append(circ)
            hull = cv2.convexHull(max_c)
            h_area = cv2.contourArea(hull)
            if h_area > 0 and area > 50:
                sol = area / h_area
                solidities.append(sol)
        
        aspect_ratios.append(float(w) / float(h))
        color_stds.append(float(gray.std()))
        
    cap.release()
    print(f"Sampled {sampled} rider frames.")
    if sampled > 0:
        print(f"Skin Ratio:      mean={np.mean(skin_ratios):.3f}, min={np.min(skin_ratios):.3f}, max={np.max(skin_ratios):.3f}")
        print(f"Face Skin Ratio: mean={np.mean(face_skin_ratios):.3f}, min={np.min(face_skin_ratios):.3f}, max={np.max(face_skin_ratios):.3f}")
        print(f"Edge Density:    mean={np.mean(edge_densities):.3f}, min={np.min(edge_densities):.3f}, max={np.max(edge_densities):.3f}")
        print(f"Laplacian Var:   mean={np.mean(lap_vars):.1f}, min={np.min(lap_vars):.1f}, max={np.max(lap_vars):.1f}")
        print(f"Aspect Ratio:    mean={np.mean(aspect_ratios):.2f}, min={np.min(aspect_ratios):.2f}, max={np.max(aspect_ratios):.2f}")
        if circularities:
            print(f"Circularity:     mean={np.mean(circularities):.3f}, min={np.min(circularities):.3f}, max={np.max(circularities):.3f}")
        if solidities:
            print(f"Solidity:        mean={np.mean(solidities):.3f}, min={np.min(solidities):.3f}, max={np.max(solidities):.3f}")
        print(f"Gray Std (Tex):  mean={np.mean(color_stds):.2f}, min={np.min(color_stds):.2f}, max={np.max(color_stds):.2f}")

for name, path in videos.items():
    analyze_video(name, path)
