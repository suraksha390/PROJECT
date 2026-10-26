import cv2
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector

def test_video(vname):
    vpath = os.path.join(PROJECT_ROOT, 'static', 'uploads', 'videos', vname)
    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    lane_detector.reset()
    f = 0
    rider_f = 0
    helmet_classes = {}
    lane_viols = 0
    while True:
        ret, fr = cap.read()
        if not ret:
            break
        f += 1
        dets, _ = yolo_detector.detect_objects(fr)
        if dets['rider_found']:
            rider_f += 1
            h_cls = dets.get('helmet_class', 'none')
            helmet_classes[h_cls] = helmet_classes.get(h_cls, 0) + 1
        v, _ = lane_detector.analyze_lane(fr, rider_bbox=dets.get('rider_bbox'), motorcycle_bbox=dets.get('motorcycle_bbox'), frame_idx=f, fps=fps)
        if v:
            lane_viols += 1
    cap.release()
    l_sum = lane_detector.get_summary()
    print("=" * 40)
    print("VIDEO:", vname)
    print("Total frames:", f, "Rider frames:", rider_f)
    print("Helmet classes distribution:", helmet_classes)
    print("Lane state:", l_sum['lane_state'], "Violations:", l_sum['violation_count'], "Violating frames:", lane_viols)
    print("=" * 40)

if __name__ == '__main__':
    for v in ['suraksha_driving.mp4', 'suraksha_with_helmet.mp4']:
        test_video(v)
