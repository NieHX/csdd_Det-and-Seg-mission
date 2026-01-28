from pathlib import Path
import random
from typing import List, Optional

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp")


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        return []
    images = [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    return sorted(images)


def sample_images(images: List[Path], limit: Optional[int], seed: int = 0) -> List[Path]:
    if not limit or limit <= 0 or limit >= len(images):
        return images
    rng = random.Random(seed)
    return rng.sample(images, limit)


def print_det_metrics(metrics, class_names) -> None:
    print("Detection metrics")
    print(f"mAP50-95: {metrics.box.map:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    for i, ap in enumerate(metrics.box.maps):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"{name} AP: {ap:.4f}")


def print_seg_metrics(metrics, class_names) -> None:
    print("Segmentation metrics (mask)")
    print(f"Mask mAP50-95: {metrics.seg.map:.4f}")
    print(f"Mask mAP50: {metrics.seg.map50:.4f}")
    for i, ap in enumerate(metrics.seg.maps):
        name = class_names[i] if i < len(class_names) else str(i)
        print(f"{name} Mask AP: {ap:.4f}")
