import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
from config import BASE_DIR

def generate_dataset(num_samples=1000, random_state=42):
    np.random.seed(random_state)
    
    # Feature vectors:
    # 0: auth_status (0/1)
    # 1: helmet_status (0/1)
    # 2: lane_status (0/1)
    # 3: foot_touch_status (0/1)
    # 4: balance_status (0/1)
    # 5: lane_violation_count (int)
    # 6: foot_touch_count (int)
    # 7: max_lean_angle (float)

    auth = np.random.choice([1, 0], size=num_samples, p=[0.85, 0.15])
    helmet = np.random.choice([1, 0], size=num_samples, p=[0.80, 0.20])
    lane = np.random.choice([1, 0], size=num_samples, p=[0.75, 0.25])
    foot_touch = np.random.choice([1, 0], size=num_samples, p=[0.82, 0.18])
    balance = np.random.choice([1, 0], size=num_samples, p=[0.88, 0.12])

    lane_violations = np.where(lane == 1, 0, np.random.randint(1, 5, size=num_samples))
    foot_touches = np.where(foot_touch == 1, 0, np.random.randint(1, 4, size=num_samples))
    
    lean_angle = np.where(balance == 1, 
                          np.random.uniform(2.0, 15.0, size=num_samples), 
                          np.random.uniform(23.0, 45.0, size=num_samples))

    # Target label: 1 = PASS, 0 = FAIL
    # Strict rule: Must pass authentication, wear helmet, no foot touch, no lane violation, stable balance
    target = np.where(
        (auth == 1) & (helmet == 1) & (lane == 1) & (foot_touch == 1) & (balance == 1),
        1, 0
    )

    df = pd.DataFrame({
        'auth_status': auth,
        'helmet_status': helmet,
        'lane_status': lane,
        'foot_touch_status': foot_touch,
        'balance_status': balance,
        'lane_violation_count': lane_violations,
        'foot_touch_count': foot_touches,
        'max_lean_angle': np.round(lean_angle, 2),
        'result_label': target
    })

    output_dir = os.path.join(BASE_DIR, 'dataset')
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'driving_test_dataset.csv')
    df.to_csv(csv_path, index=False)
    print(f"Generated synthetic dataset with {num_samples} samples at {csv_path}")
    return csv_path

if __name__ == '__main__':
    generate_dataset()
