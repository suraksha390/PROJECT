import sys, os
sys.path.insert(0, os.path.abspath('.'))
import cv2
import numpy as np
from models.yolo_detector import yolo_detector
from models.lane_detector import lane_detector
from models.pose_analyzer import pose_analyzer
from models.ml_evaluator import ml_evaluator

video_path = "static/uploads/videos/test_CAND-2026-070_suraksha1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
lane_detector.reset()

frame_count = 0
total_rider_frames = 0
reliable_rider_frames = 0
helmet_pass_count = 0
no_helmet_count = 0
other_headwear_count = 0
helmet_review_count = 0
helmet_confidences = []

lane_violation_count = 0
foot_touch_count = 0
unstable_balance_count = 0
max_lean_angle = 0.0

stride = 3
last_annotated_frame = None
best_annotated_frame = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % stride != 0:
        continue

    # 1. YOLO
    yolo_dets, yolo_frame = yolo_detector.detect_objects(frame)
    if yolo_dets['rider_found']:
        total_rider_frames += 1
        h_cls = yolo_dets.get('helmet_class', 'none')
        h_stat = yolo_dets.get('helmet_status', 'FAIL')
        h_conf = yolo_dets.get('helmet_conf', 0.0)

        if h_cls == 'helmet' and h_stat == 'PASS':
            helmet_pass_count += 1
            reliable_rider_frames += 1
            helmet_confidences.append(h_conf)
        elif h_cls == 'other_headwear':
            other_headwear_count += 1
            reliable_rider_frames += 1
        elif h_cls == 'no_helmet':
            no_helmet_count += 1
            reliable_rider_frames += 1
        elif h_stat == 'Needs Review':
            helmet_review_count += 1
            if h_cls == 'helmet' and h_conf > 0:
                helmet_confidences.append(h_conf)

    # 2. Lane
    lane_viol, lane_frame = lane_detector.analyze_lane(
        yolo_frame,
        rider_bbox=yolo_dets['rider_bbox'],
        motorcycle_bbox=yolo_dets['motorcycle_bbox'],
        frame_idx=frame_count,
        fps=fps
    )
    if lane_viol:
        lane_violation_count += 1

    # 3. Pose
    pose_info, final_frame = pose_analyzer.analyze_pose(lane_frame, yolo_dets['ground_line_y'])
    if pose_info['foot_touch']:
        foot_touch_count += 1
    if pose_info['balance_status'] == 'Unstable':
        unstable_balance_count += 1
    if abs(pose_info['lean_angle']) > max_lean_angle:
        max_lean_angle = abs(pose_info['lean_angle'])

    if yolo_dets['rider_found']:
        best_annotated_frame = final_frame
    last_annotated_frame = final_frame

cap.release()

saved_frame = best_annotated_frame if best_annotated_frame is not None else last_annotated_frame
cv2.imwrite("scratch/full_pipeline_sample.jpg", saved_frame)
print("Pipeline run completed!")
print(f"Rider frames: {total_rider_frames}, Helmet pass: {helmet_pass_count}, Review: {helmet_review_count}")
print(f"Lane violations: {lane_violation_count}, Foot touches: {foot_touch_count}, Unstable balance: {unstable_balance_count}")
print(f"Saved sample frame to scratch/full_pipeline_sample.jpg")
