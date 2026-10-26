import os, sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)
import cv2
import numpy as np
from models.yolo_detector import yolo_detector

for vname in ['suraksha_driving.mp4', 'suraksha_with_helmet.mp4']:
    cap = cv2.VideoCapture(os.path.join(PROJECT_ROOT, 'static', 'uploads', 'videos', vname))
    print(f"=== Analyzing {vname} ===")
    for f in [30, 60, 90, 120, 150]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, fr = cap.read()
        if not ret: break
        dets, _ = yolo_detector.detect_objects(fr)
        rb = dets.get('rider_bbox')
        if rb:
            rx1, ry1, rx2, ry2 = rb
            head_h = max(1, int((ry2 - ry1) * 0.28))
            head_roi = fr[ry1:min(ry2, ry1 + head_h), rx1:rx2]
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            lower_skin = np.array([0, 20, 50], dtype=np.uint8)
            upper_skin = np.array([25, 190, 255], dtype=np.uint8)
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            skin_ratio = float(np.sum(skin_mask > 0)) / float(head_roi.shape[0] * head_roi.shape[1] + 1e-5)
            gray = cv2.cvtColor(head_roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blur, 40, 120)
            edge_density = float(np.sum(edges > 0)) / float(head_roi.shape[0] * head_roi.shape[1] + 1e-5)
            mean_bgr = head_roi.mean(axis=(0,1)).astype(int)
            print(f"Frame {f}: h_cls={dets.get('helmet_class')}, conf={dets.get('helmet_conf')}, skin_ratio={skin_ratio:.3f}, edge_density={edge_density:.3f}, mean_bgr={mean_bgr}, std={gray.std():.1f}")
            out_name = f"scratch/{vname[:-4]}_head_f{f}.jpg"
            cv2.imwrite(out_name, head_roi)
    cap.release()
