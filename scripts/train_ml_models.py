import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score

from config import BASE_DIR, SAVED_MODELS_DIR
from scripts.generate_sample_data import generate_dataset

def train_and_save_models():
    dataset_path = os.path.join(BASE_DIR, 'dataset', 'driving_test_dataset.csv')
    if not os.path.exists(dataset_path):
        dataset_path = generate_dataset()

    df = pd.read_csv(dataset_path)

    X = df.drop(columns=['result_label'])
    y = df['result_label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_pred = rf.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    print(f"--- Random Forest Accuracy: {rf_acc * 100:.2f}% ---")
    print(classification_report(y_test, rf_pred))

    # Save model and scaler
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    joblib.dump(rf, os.path.join(SAVED_MODELS_DIR, 'rf_model.pkl'))
    joblib.dump(scaler, os.path.join(SAVED_MODELS_DIR, 'scaler.pkl'))

    print(f"Random Forest model successfully trained & saved to {SAVED_MODELS_DIR}")

if __name__ == '__main__':
    train_and_save_models()
