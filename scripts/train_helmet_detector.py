"""
Two-Wheeler Safety Helmet & Headwear Model Training / Fine-Tuning Script
========================================================================

PURPOSE:
Trains or fine-tunes a YOLOv8 model to reliably distinguish between:
  0: helmet           - Genuine motorcycle protective helmet
  1: no_helmet        - Uncovered head / bare face / hair
  2: other_headwear   - Non-protective headwear (baseball cap, hat, towel, turban, scarf)

WHY PRETRAINED YOLO (COCO) IS INSUFFICIENT:
Pretrained 'yolov8n.pt' is trained on the 80 COCO classes (person, motorcycle, car, dog, etc.).
It does NOT contain classes for motorcycle helmets, baseball caps, or bare heads.
Classical computer-vision heuristics (dome circularity, edge density) can reject bare skin or
fabric texture, but CANNOT reliably distinguish a baseball cap or turban from a safety helmet.
A dedicated model trained on labelled examples of these 3 classes is required for reliable
distinction without false approvals.

DATASET PREPARATION INSTRUCTIONS:
1. Recommended Data Sources:
   - Roboflow Universe: Search for 'motorcycle-helmet-detection' or 'helmet-vest-headwear'.
   - Kaggle: 'Two-Wheeler Helmet vs Cap Dataset', 'Safety Helmet Wearing Dataset (SHWD)'.
   - Custom Camera Footage: Annotate using tools like Roboflow, CVAT, or LabelImg in YOLO format.

2. Required Dataset Directory Structure:
   dataset_helmet/
     ├── data.yaml
     ├── train/
     │   ├── images/  (*.jpg, *.png)
     │   └── labels/  (*.txt, YOLO format: <class_id> <x_center> <y_center> <width> <height>)
     └── val/
         ├── images/
         └── labels/

3. Format of 'data.yaml':
   path: ./dataset_helmet
   train: train/images
   val: val/images
   names:
     0: helmet
     1: no_helmet
     2: other_headwear

USAGE EXAMPLES:
  # Generate sample folder structure & template data.yaml:
  python scripts/train_helmet_detector.py --create-template

  # Train the model once dataset is placed:
  python scripts/train_helmet_detector.py --data dataset/helmet_data.yaml --epochs 50 --batch 16
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

# Add project root to Python module path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import YOLO_BASE_MODEL_PATH, HELMET_MODEL_PATH, SAVED_MODELS_DIR


def create_dataset_template(target_dir='dataset/helmet_dataset'):
    """Creates a scaffold directory structure and template data.yaml file."""
    base_path = Path(PROJECT_ROOT) / target_dir
    for sub in ['train/images', 'train/labels', 'val/images', 'val/labels']:
        (base_path / sub).mkdir(parents=True, exist_ok=True)

    yaml_path = base_path / 'data.yaml'
    yaml_content = f"""# YOLOv8 Safety Helmet & Headwear Dataset Config
path: {base_path.resolve().as_posix()}
train: train/images
val: val/images

# Number of classes
nc: 3

# Class names
names:
  0: helmet           # Genuine motorcycle protective helmet
  1: no_helmet        # Rider head is bare / uncovered
  2: other_headwear   # Cap, hat, towel, turban, scarf, non-protective cloth
"""
    with open(yaml_path, 'w', encoding='utf-8') as f:
        f.write(yaml_content)

    print(f"\n[INFO] Created dataset directory scaffold at: {base_path}")
    print(f"[INFO] Created template config at: {yaml_path}")
    print("\nNext Steps:")
    print("  1. Add your annotated images and labels into train/ and val/ folders.")
    print(f"  2. Run: python scripts/train_helmet_detector.py --data \"{yaml_path}\"")
    return yaml_path


def train_model(data_yaml, epochs=50, batch_size=16, imgsz=640,
                base_model=YOLO_BASE_MODEL_PATH, output_path=HELMET_MODEL_PATH, device=None):
    """Fine-tunes YOLOv8 on the 3-class helmet & headwear dataset."""
    if not os.path.exists(data_yaml):
        print(f"\n[ERROR] Dataset configuration file not found at: '{data_yaml}'")
        print("To generate a template dataset structure, run:")
        print("  python scripts/train_helmet_detector.py --create-template")
        sys.exit(1)

    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n[ERROR] 'ultralytics' package not installed. Install via: pip install ultralytics")
        sys.exit(1)

    print("=" * 75)
    print("  Two-Wheeler Safety Helmet Model Training (YOLOv8)")
    print(f"  Dataset Config: {data_yaml}")
    print(f"  Base Model:     {base_model}")
    print(f"  Epochs:         {epochs}")
    print(f"  Batch Size:     {batch_size}")
    print(f"  Image Size:     {imgsz}")
    print(f"  Target Output:  {output_path}")
    print("=" * 75)

    # Load base pretrained model (transfers weights from COCO features)
    model = YOLO(base_model)

    train_kwargs = {
        'data': data_yaml,
        'epochs': epochs,
        'batch': batch_size,
        'imgsz': imgsz,
        'save': True,
        'verbose': True,
        'project': os.path.join(PROJECT_ROOT, 'runs', 'helmet_train'),
        'name': 'helmet_run'
    }
    if device is not None:
        train_kwargs['device'] = device

    print("\n[INFO] Starting training...")
    train_results = model.train(**train_kwargs)

    # Validate trained model
    print("\n[INFO] Evaluating model on validation split...")
    val_results = model.val()

    # Locate and copy the best weights to target path
    best_weights_path = Path(train_results.save_dir) / 'weights' / 'best.pt'
    if best_weights_path.exists():
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        shutil.copy(best_weights_path, output_path)
        print(f"\n[SUCCESS] Best trained model successfully saved to: {output_path}")
        print(f"The driving validation system will now automatically use this model for helmet validation.")
    else:
        print(f"\n[WARNING] Could not locate 'best.pt' at {best_weights_path}. Check runs directory.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Helmet & Headwear Classifier")
    parser.add_argument('--data', type=str, default='dataset/helmet_dataset/data.yaml',
                        help="Path to data.yaml dataset config file")
    parser.add_argument('--epochs', type=int, default=50,
                        help="Number of training epochs (default: 50)")
    parser.add_argument('--batch', type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="Input image resolution (default: 640)")
    parser.add_argument('--base-model', type=str, default=YOLO_BASE_MODEL_PATH,
                        help=f"Base weights to fine-tune (default: {YOLO_BASE_MODEL_PATH})")
    parser.add_argument('--output', type=str, default=HELMET_MODEL_PATH,
                        help=f"Path to save best weights (default: {HELMET_MODEL_PATH})")
    parser.add_argument('--device', type=str, default=None,
                        help="Device to use ('cpu', '0', etc.)")
    parser.add_argument('--create-template', action='store_true',
                        help="Generate a template dataset directory structure & data.yaml")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.create_template:
        create_dataset_template()
    else:
        train_model(
            data_yaml=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            imgsz=args.imgsz,
            base_model=args.base_model,
            output_path=args.output,
            device=args.device
        )
