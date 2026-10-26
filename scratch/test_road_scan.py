import cv2
import numpy as np

def test_road_scanning(image_path_or_frame, moto_x=None):
    if isinstance(image_path_or_frame, str):
        frame = cv2.imread(image_path_or_frame)
    else:
        frame = image_path_or_frame

    h, w = frame.shape[:2]
    y_top = int(h * 0.35)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 1. Non-drivable vegetation/grass mask
    veg = ((hsv[:, :, 0] >= 28) & (hsv[:, :, 0] <= 88) &
           (hsv[:, :, 1] >= 35) & (hsv[:, :, 2] >= 30))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    veg_clean = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    # 2. Reference x center of the road
    if moto_x is not None:
        ref_x = int(np.clip(moto_x, w * 0.2, w * 0.8))
    else:
        ref_x = w // 2

    road_lefts = []
    road_rights = []

    for y in range(y_top, h, 10):
        row_veg = veg_clean[y, :]

        # Scan left from ref_x to 0
        left_non_road = np.where(row_veg[:ref_x] > 0)[0]
        if len(left_non_road) > 2:
            l_edge = float(np.max(left_non_road))
        else:
            l_edge = 0.0

        # Scan right from ref_x to w
        right_non_road = np.where(row_veg[ref_x:] > 0)[0]
        if len(right_non_road) > 2:
            r_edge = float(ref_x + np.min(right_non_road))
        else:
            r_edge = float(w)

        road_lefts.append(l_edge)
        road_rights.append(r_edge)

    l_bot = float(np.median(road_lefts[-5:]))
    r_bot = float(np.median(road_rights[-5:]))
    l_top = float(np.median(road_lefts[:5]))
    r_top = float(np.median(road_rights[:5]))

    return (l_bot, r_bot, l_top, r_top)

print("Testing road scanning on frame 120 of suraksha_with_helmet:")
res_helmet = test_road_scanning('scratch/frame_suraksha_with_helmet_120.jpg', moto_x=260)
print(f"l_bot: {res_helmet[0]:.1f}, r_bot: {res_helmet[1]:.1f}, l_top: {res_helmet[2]:.1f}, r_top: {res_helmet[3]:.1f}")

print("\nTesting road scanning on frame 120 of suraksha_driving:")
res_driving = test_road_scanning('scratch/frame_suraksha_driving_120.jpg', moto_x=400)
print(f"l_bot: {res_driving[0]:.1f}, r_bot: {res_driving[1]:.1f}, l_top: {res_driving[2]:.1f}, r_top: {res_driving[3]:.1f}")
