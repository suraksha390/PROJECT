import os
import sys
import cv2

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from app import app
from database.db import get_all_candidates

with app.test_client() as client:
    # Check if we can run video analysis on sinch.mp4
    video_path = 'static/uploads/videos/test_CAND-2026-062_sinch.mp4'
    if not os.path.exists(video_path):
        video_path = '../sinch.mp4'
        
    print(f"Testing video at: {video_path}")
    print(f"File exists: {os.path.exists(video_path)}, Size: {os.path.getsize(video_path)}")
