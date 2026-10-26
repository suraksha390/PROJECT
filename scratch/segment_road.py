import cv2
import numpy as np

def inspect_frame_scene(f_idx):
    frame = cv2.imread(f'scratch/lane_inspection/frame_{f_idx}.jpg')
    if frame is None:
        return
    h, w = frame.shape[:2]
    
    # Analyze color segmentation across the frame
    # Convert to HSV and Lab
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    
    # Road characteristics:
    # Asphalt / concrete pavement has low-to-medium saturation and neutral hue
    # Grass / vegetation has high Green (Hue ~ 35-85) and distinct saturation
    # Soil / mud shoulder has reddish-brown or yellowish-brown hue
    # Sky / background is at top (y < 0.35 * h)
    
    # Measure vegetation mask (Hue 32 to 85, Sat 60 to 255)
    veg_mask = cv2.inRange(hsv, np.array([30, 50, 40]), np.array([85, 255, 255]))
    
    # Measure soil / shoulder mask
    # Find horizontal profile of road across multiple y levels: y = 0.5*h, 0.65*h, 0.8*h, 0.95*h
    print(f"\n================ FRAME {f_idx} ================")
    for y_frac in [0.5, 0.65, 0.8, 0.92]:
        y = int(y_frac * h)
        row_hsv = hsv[y, :, :]
        # Check vegetation presence on row
        row_veg = veg_mask[y, :]
        veg_indices = np.where(row_veg > 0)[0]
        veg_left = [x for x in veg_indices if x < w // 2]
        veg_right = [x for x in veg_indices if x >= w // 2]
        
        left_bound = max(veg_left) if veg_left else 0
        right_bound = min(veg_right) if veg_right else w
        print(f"y={y:3d} ({y_frac*100:.0f}% height): Drivable Road width ~ [{left_bound:3d} to {right_bound:3d}] (Road width = {right_bound - left_bound}px out of {w}px)")

for idx in [15, 30, 60, 90, 120, 150, 180, 200]:
    inspect_frame_scene(idx)
