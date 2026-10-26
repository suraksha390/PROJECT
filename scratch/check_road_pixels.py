import cv2
import numpy as np

for f_idx in [30, 90, 120, 150, 180]:
    frame = cv2.imread(f'scratch/lane_inspection/frame_{f_idx}.jpg')
    h, w = frame.shape[:2]
    # Check bottom half of the frame: road surface vs surroundings
    # Analyze HSV and color of road (asphalt/concrete) vs surroundings (soil/grass/shoulder)
    road_sample = frame[int(h * 0.7):int(h * 0.9), int(w * 0.3):int(w * 0.7)]
    left_side = frame[int(h * 0.7):int(h * 0.9), :int(w * 0.25)]
    right_side = frame[int(h * 0.7):int(h * 0.9), int(w * 0.75):]
    
    print(f"\n--- Frame {f_idx} ---")
    print(f"Road center BGR mean: {road_sample.mean(axis=(0,1)).astype(int)}, HSV mean: {cv2.cvtColor(road_sample, cv2.COLOR_BGR2HSV).mean(axis=(0,1)).astype(int)}")
    print(f"Left side   BGR mean: {left_side.mean(axis=(0,1)).astype(int)}, HSV mean: {cv2.cvtColor(left_side, cv2.COLOR_BGR2HSV).mean(axis=(0,1)).astype(int)}")
    print(f"Right side  BGR mean: {right_side.mean(axis=(0,1)).astype(int)}, HSV mean: {cv2.cvtColor(right_side, cv2.COLOR_BGR2HSV).mean(axis=(0,1)).astype(int)}")
