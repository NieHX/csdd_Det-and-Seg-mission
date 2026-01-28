import argparse
from ultralytics import YOLO

from config import DET_YAML, ensure_configs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 on CSDD detection.")
    parser.add_argument("--model", default="yolov8l.pt", help="Base model or weights.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--project", default="runs/csdd_det")
    parser.add_argument("--name", default="yolov8_det")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_configs()
    model = YOLO(args.model)
    model.train(
        data=str(DET_YAML),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        workers=args.workers,
        project=args.project,
        name=args.name,
    )


if __name__ == "__main__":
    main()
