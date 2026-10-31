import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector
from models.pose_analyzer import pose_analyzer

video_path = "static/uploads/videos/test_CAND-2026-070_suraksha1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

lane_detector.reset()

test_frames = [30, 60, 90, 120, 135]
saved = []

frame_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1

    if frame_idx in test_frames:
        yolo_dets, yolo_frame = yolo_detector.detect_objects(frame)
        lane_viol, lane_frame = lane_detector.analyze_lane(
            yolo_frame,
            rider_bbox=yolo_dets['rider_bbox'],
            motorcycle_bbox=yolo_dets['motorcycle_bbox'],
            frame_idx=frame_idx,
            fps=fps
        )
        pose_info, final_frame = pose_analyzer.analyze_pose(lane_frame, yolo_dets['ground_line_y'])
        
        out_name = f"scratch/verified_frame_{frame_idx}.jpg"
        cv2.imwrite(out_name, final_frame)
        saved.append(out_name)
        print(f"Processed frame {frame_idx}: Rider={yolo_dets['rider_found']}, Helmet={yolo_dets.get('helmet_class')}, Status={yolo_dets.get('helmet_status')}")

cap.release()
print("Saved frames:", saved)
