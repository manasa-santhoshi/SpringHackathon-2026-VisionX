"""
Fine-tune YOLOv11 on the VisDrone drone detection dataset.

VisDrone provides diverse aerial/drone footage for training a model that generalizes
to drone parking lot scenarios like DLP. The DLP data is used only as an external test set.

Usage:
    python -m src.detection.train
    python -m src.detection.train --epochs 20 --model yolo11s.pt --imgsz 1280
"""

import argparse
from pathlib import Path

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = PROJECT_ROOT / "models"


def main():
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv11 on VisDrone dataset")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model (default: yolo11n.pt)")
    parser.add_argument("--data", default="VisDrone.yaml",
                        help="Dataset YAML (default: VisDrone.yaml, auto-downloads ~1.8GB)")
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="mps", help="Device (mps, cuda, cpu)")
    parser.add_argument("--name", default="yolo11n-visdrone", help="Run name")
    args = parser.parse_args()

    model = YOLO(args.model)
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(DEFAULT_OUTPUT),
        name=args.name,
        exist_ok=True,
        pretrained=True,
    )

    best_path = Path(results.save_dir) / "weights" / "best.pt"
    print(f"\nBest model saved to: {best_path}")
    print(f"\nTo run the pipeline with this model:")
    print(f"  python -m src.pipeline.run --video data/raw/DLP/raw/DJI_0012.MOV --model {best_path}")


if __name__ == "__main__":
    main()
