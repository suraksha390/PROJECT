import cv2
import numpy as np

for f_idx in [15, 60, 110, 150, 190]:
    frame = cv2.imread(f'scratch/lane_inspection/frame_{f_idx}.jpg')
    h, w = frame.shape[:2]
    # Sample horizontal scanline across bottom at y = 0.85 * h
    y_scan = int(h * 0.85)
    scanline_bgr = frame[y_scan, :, :]
    scanline_hsv = cv2.cvtColor(frame[y_scan:y_scan+1, :, :], cv2.COLOR_BGR2HSV)[0]
    
    # Calculate gradients / transitions across the horizontal scanline
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    row_grad = np.abs(grad_x[y_scan, :])
    
    print(f"\nFrame {f_idx}:")
    print(f"Top 5 gradient peaks at y={y_scan}: {np.argsort(row_grad)[-5:]}")
    # Print color values across x at 10%, 25%, 50%, 75%, 90% width
    for pct in [0.05, 0.15, 0.30, 0.50, 0.70, 0.85, 0.95]:
        x = int(pct * w)
        print(f"  x={x:3d} ({pct*100:2.0f}%): BGR={scanline_bgr[x]} HSV={scanline_hsv[x]}")
