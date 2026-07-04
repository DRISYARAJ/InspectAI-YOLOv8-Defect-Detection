from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "NEU-DET"
DATASET_ROOT = ROOT / "dataset"

CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def find_image(split: str, filename: str) -> Path:
    images_root = SOURCE_ROOT / split / "images"
    matches = list(images_root.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"Could not find image {filename} in {images_root}")
    return matches[0]


def convert_box(width: int, height: int, box: ET.Element) -> tuple[float, float, float, float]:
    xmin = float(box.findtext("xmin", "0"))
    ymin = float(box.findtext("ymin", "0"))
    xmax = float(box.findtext("xmax", "0"))
    ymax = float(box.findtext("ymax", "0"))

    x_center = ((xmin + xmax) / 2) / width
    y_center = ((ymin + ymax) / 2) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def prepare_split(source_split: str, target_split: str) -> tuple[int, int]:
    annotation_dir = SOURCE_ROOT / source_split / "annotations"
    target_image_dir = DATASET_ROOT / "images" / target_split
    target_label_dir = DATASET_ROOT / "labels" / target_split
    target_image_dir.mkdir(parents=True, exist_ok=True)
    target_label_dir.mkdir(parents=True, exist_ok=True)

    image_count = 0
    box_count = 0

    for annotation_file in sorted(annotation_dir.glob("*.xml")):
        tree = ET.parse(annotation_file)
        root = tree.getroot()

        filename = root.findtext("filename")
        if not filename:
            continue

        width = int(root.findtext("size/width", "0"))
        height = int(root.findtext("size/height", "0"))
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size in {annotation_file}")

        yolo_lines = []
        for item in root.findall("object"):
            class_name = item.findtext("name")
            if class_name not in CLASSES:
                raise ValueError(f"Unknown class '{class_name}' in {annotation_file}")

            box = item.find("bndbox")
            if box is None:
                continue

            class_id = CLASSES.index(class_name)
            x_center, y_center, box_width, box_height = convert_box(width, height, box)
            yolo_lines.append(
                f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}"
            )
            box_count += 1

        source_image = find_image(source_split, filename)
        shutil.copy2(source_image, target_image_dir / filename)
        (target_label_dir / f"{Path(filename).stem}.txt").write_text(
            "\n".join(yolo_lines),
            encoding="utf-8",
        )
        image_count += 1

    return image_count, box_count


def write_data_yaml() -> None:
    names = "\n".join(f"  {index}: {name}" for index, name in enumerate(CLASSES))
    (ROOT / "data.yaml").write_text(
        "path: dataset\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        f"names:\n{names}\n",
        encoding="utf-8",
    )


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise SystemExit("NEU-DET folder was not found.")

    train_images, train_boxes = prepare_split("train", "train")
    val_images, val_boxes = prepare_split("validation", "val")
    write_data_yaml()

    print("NEU-DET dataset prepared for YOLO training.")
    print(f"Training images: {train_images}, defect boxes: {train_boxes}")
    print(f"Validation images: {val_images}, defect boxes: {val_boxes}")
    print("Updated data.yaml with NEU-DET defect classes.")


if __name__ == "__main__":
    main()
