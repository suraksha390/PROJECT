import cv2
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector

def inspect_violating_frames():
    vpath = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'videos', 'suraksha_with_helmet.mp4')
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lane_detector.reset()
    f = 0
    while True:
        ret, fr = cap.read()
        if not ret: break
        f += 1
        dets, _ = yolo_detector.detect_objects(fr)
        v, _ = lane_detector.analyze_lane(fr, rider_bbox=dets.get('rider_bbox'), motorcycle_bbox=dets.get('motorcycle_bbox'), frame_idx=f, fps=fps)
        if v or lane_detector.consecutive_left_touches > 0 or lane_detector.consecutive_right_touches > 0:
            mb = dets.get('motorcycle_bbox')
            b = lane_detector.detect_lane_boundaries(fr, mb)
            my2 = mb[3] if mb else 0
            xl = b['x_left_fn'](my2)
            xr = b['x_right_fn'](my2)
            print(f"Frame {f}: viol={v}, left_touches={lane_detector.consecutive_left_touches}, right_touches={lane_detector.consecutive_right_touches}, moto={mb}, xl={xl:.1f}, xr={xr:.1f}")
    cap.release()
    print("Violations:", lane_detector.confirmed_violations)

if __name__ == '__main__':
    inspect_violating_frames()
