"""
convert_neu_det.py

Converts the NEU-DET steel surface defect dataset (PASCAL VOC .xml
annotations) into YOLO format, and copies everything into this
project's expected folder layout:

    dataset/images/train
    dataset/images/val
    dataset/labels/train
    dataset/labels/val

NEU-DET's usual download layout looks like one of:

    NEU-DET/
      IMAGES/            *.jpg
      ANNOTATIONS/       *.xml
or
    NEU-DET/
      train/
        images/   *.jpg
        annotations/  *.xml
      validation/
        images/   *.jpg
        annotations/  *.xml

This script auto-detects images + matching xml files anywhere under
the folder you point it at, so it should work regardless of which
exact zip layout you downloaded.

USAGE
-----
    python convert_neu_det.py --src "E:\\Internship project\\NEU-DET"
    python convert_neu_det.py --src "E:\\Internship project\\NEU-DET" --ratio 0.85

After it finishes, just run:
    python train.py
"""

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# NEU-DET's 6 standard defect classes. Edit this list (and the matching
# `names:` section in data.yaml) if your download uses different labels.
DEFAULT_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]


def find_pairs(src: Path):
    """Find every (image, matching xml) pair anywhere under src."""
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXTS}
    xmls = {p.stem: p for p in src.rglob("*.xml")}

    pairs = []
    missing_xml = []
    for stem, img_path in images.items():
        xml_path = xmls.get(stem)
        if xml_path:
            pairs.append((img_path, xml_path))
        else:
            missing_xml.append(img_path)

    return pairs, missing_xml


def discover_classes(pairs):
    """Scan all XML files to build the class list, preserving first-seen order."""
    found = []
    for _, xml_path in pairs:
        tree = ET.parse(xml_path)
        for obj in tree.getroot().findall("object"):
            name_el = obj.find("name")
            if name_el is None:
                continue
            name = name_el.text.strip()
            if name not in found:
                found.append(name)
    return found


def voc_xml_to_yolo_lines(xml_path: Path, class_to_id: dict):
    """Parse one PASCAL VOC xml file and return a list of YOLO label lines."""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    img_w = float(size.find("width").text)
    img_h = float(size.find("height").text)

    lines = []
    for obj in root.findall("object"):
        name_el = obj.find("name")
        if name_el is None:
            continue
        cls_name = name_el.text.strip()
        if cls_name not in class_to_id:
            # Unknown class not in our list -> skip (shouldn't happen if
            # discover_classes ran first)
            continue
        cls_id = class_to_id[cls_name]

        bnd = obj.find("bndbox")
        xmin = float(bnd.find("xmin").text)
        ymin = float(bnd.find("ymin").text)
        xmax = float(bnd.find("xmax").text)
        ymax = float(bnd.find("ymax").text)

        # Clamp to image bounds just in case of slightly-off annotations
        xmin = max(0, min(xmin, img_w))
        xmax = max(0, min(xmax, img_w))
        ymin = max(0, min(ymin, img_h))
        ymax = max(0, min(ymax, img_h))

        x_center = ((xmin + xmax) / 2) / img_w
        y_center = ((ymin + ymax) / 2) / img_h
        width = (xmax - xmin) / img_w
        height = (ymax - ymin) / img_h

        if width <= 0 or height <= 0:
            continue

        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="Folder where you unzipped NEU-DET")
    parser.add_argument("--dest", default="dataset", help="Project dataset folder (default: dataset)")
    parser.add_argument("--ratio", type=float, default=0.85, help="Train split ratio (rest goes to val)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    src = Path(args.src)
    if not src.exists():
        raise SystemExit(f"Source folder not found: {src}")

    dest = Path(args.dest)
    for split in ("train", "val"):
        (dest / "images" / split).mkdir(parents=True, exist_ok=True)
        (dest / "labels" / split).mkdir(parents=True, exist_ok=True)

    print(f"Scanning {src} for images + matching XML annotations...")
    pairs, missing_xml = find_pairs(src)

    if not pairs:
        raise SystemExit(
            "No image+XML pairs found. Check that --src points at the folder "
            "that actually contains the .jpg/.png images and .xml files "
            "(they can be in subfolders, that's fine)."
        )

    print(f"Found {len(pairs)} image/annotation pairs.")
    if missing_xml:
        print(f"Note: {len(missing_xml)} images had no matching .xml and will be skipped.")

    # Discover actual class names from the data, fall back to the standard
    # NEU-DET list if discovery finds nothing (shouldn't happen normally).
    discovered = discover_classes(pairs)
    classes = discovered if discovered else DEFAULT_CLASSES
    class_to_id = {name: i for i, name in enumerate(classes)}

    print("\nClasses found in annotations:")
    for i, name in enumerate(classes):
        print(f"  {i}: {name}")

    # Shuffle + split
    random.seed(args.seed)
    random.shuffle(pairs)
    n_train = int(len(pairs) * args.ratio)
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:]

    def process(split_pairs, split_name):
        written = 0
        for img_path, xml_path in split_pairs:
            lines = voc_xml_to_yolo_lines(xml_path, class_to_id)
            if not lines:
                continue  # skip images with no valid boxes

            img_out = dest / "images" / split_name / img_path.name
            lbl_out = dest / "labels" / split_name / (img_path.stem + ".txt")

            shutil.copy2(img_path, img_out)
            lbl_out.write_text("\n".join(lines), encoding="utf-8")
            written += 1
        print(f"[{split_name}] wrote {written} image/label pairs")

    process(train_pairs, "train")
    process(val_pairs, "val")

    # Write/refresh data.yaml with the discovered classes
    data_yaml_path = Path("data.yaml")
    yaml_lines = [
        "# Auto-generated by convert_neu_det.py",
        "path: dataset",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ] + [f"  {i}: {name}" for i, name in enumerate(classes)]
    data_yaml_path.write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")
    print(f"\nUpdated {data_yaml_path} with {len(classes)} classes.")

    print("\nDone. Next step:")
    print("    python train.py")


if __name__ == "__main__":
    main()