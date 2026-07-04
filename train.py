from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_CONFIG = ROOT / "data.yaml"
MODEL_DIR = ROOT / "models"
TRAIN_IMAGES = ROOT / "dataset" / "images" / "train"
VAL_IMAGES = ROOT / "dataset" / "images" / "val"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for file_path in folder.iterdir() if file_path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_dataset() -> None:
    train_count = count_images(TRAIN_IMAGES)
    val_count = count_images(VAL_IMAGES)

    if train_count == 0 or val_count == 0:
        raise SystemExit(
            "Dataset is not ready for training yet.\n\n"
            f"Training images found: {train_count} in {TRAIN_IMAGES}\n"
            f"Validation images found: {val_count} in {VAL_IMAGES}\n\n"
            "Add annotated images before running training:\n"
            "1. Put training images in dataset/images/train\n"
            "2. Put matching YOLO label .txt files in dataset/labels/train\n"
            "3. Put validation images in dataset/images/val\n"
            "4. Put matching YOLO label .txt files in dataset/labels/val\n\n"
            "Each label file must have the same filename as its image, for example:\n"
            "dataset/images/train/part_001.jpg\n"
            "dataset/labels/train/part_001.txt"
        )


def main() -> None:
    validate_dataset()
    MODEL_DIR.mkdir(exist_ok=True)

    model = YOLO(str(ROOT / "yolov8n.pt") if (ROOT / "yolov8n.pt").exists() else "yolov8n.pt")
    model.train(
        data=str(DATA_CONFIG),
        epochs=50,
        imgsz=640,
        batch=8,
        project=str(ROOT / "runs"),
        name="defect_detector",
        exist_ok=True,
        patience=10,
    )

    best_model = ROOT / "runs" / "defect_detector" / "weights" / "best.pt"
    target_model = MODEL_DIR / "best.pt"
    if best_model.exists():
        target_model.write_bytes(best_model.read_bytes())
        print(f"Saved best model to {target_model}")


if __name__ == "__main__":
    main()
