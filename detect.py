import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "best.pt"
OUTPUT_DIR = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run defect detection on an image, video, or folder.")
    parser.add_argument("--source", required=True, help="Path to an image, video, folder, or webcam index such as 0.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Path to trained YOLOv8 model.")
    parser.add_argument("--conf", type=float, default=0.25, help="Minimum confidence threshold.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = Path(args.model)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. Train first with python train.py, "
            "or pass --model with another .pt file."
        )

    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(str(model_path))
    results = model.predict(
        source=args.source,
        conf=args.conf,
        save=True,
        project=str(OUTPUT_DIR),
        name="predictions",
        exist_ok=True,
    )

    for result in results:
        defect_count = len(result.boxes) if result.boxes is not None else 0
        print(f"{Path(result.path).name}: {defect_count} possible defect(s)")


if __name__ == "__main__":
    main()
