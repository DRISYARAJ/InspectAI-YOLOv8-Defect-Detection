# InspectAI: Automated Steel Surface Defect Detection using YOLOv8

InspectAI is an AI-powered steel surface defect detection system that uses YOLOv8 to identify and localize defects such as crazing, inclusion, patches, pitted surface, rolled-in scale, and scratches.

## Features

- Train a YOLOv8 defect localization model.
- Run prediction on images, videos, folders, or webcam input.
- Use a Streamlit demo app for image upload and visual inspection.
- Keep dataset, model files, and prediction outputs organized for internship presentation.

## Project Structure

```text
.
|-- app.py
|-- run_app.py
|-- train.py
|-- detect.py
|-- data.yaml
|-- requirements.txt
|-- dataset/
|   |-- images/
|   |   |-- train/
|   |   |-- val/
|   |   `-- test/
|   `-- labels/
|       |-- train/
|       |-- val/
|       `-- test/
|-- models/
`-- outputs/
```

## Why Training Failed Earlier

The command `python train.py` failed because this folder was empty:

```text
dataset/images/train
```

YOLO cannot train until you add real images and matching annotation files. The project code now checks this first and prints a clear message instead of a long traceback.

## Dataset Format

Use YOLO annotation format:

```text
class_id x_center y_center width height
```

All coordinates must be normalized between `0` and `1`.

Example:

```text
0 0.512 0.431 0.120 0.084
```

Put images and labels in matching folders:

```text
dataset/images/train/image_001.jpg
dataset/labels/train/image_001.txt
dataset/images/val/image_002.jpg
dataset/labels/val/image_002.txt
```

## Defect Classes

- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

Change these names to match your dataset before training.

## Model Performance

| Metric | Value |
|--------|-------|
| Precision | 65.7% |
| Recall | 77.5% |
| mAP@50 | 77.8% |
| mAP@50-95 | 44.5% |

### Per-Class mAP50

| Class | mAP50 |
|-------|-------|
| Crazing | 46.1% |
| Inclusion | 87.7% |
| Patches | 94.7% |
| Pitted Surface | 78.1% |
| Rolled-in Scale | 64.9% |
| Scratches | 95.3% |

## Dataset

This project uses the NEU Surface Defect Dataset (NEU-DET), a benchmark dataset for steel surface defect detection.

Dataset classes:
- Crazing
- Inclusion
- Patches
- Pitted Surface
- Rolled-in Scale
- Scratches

### Per-Class mAP50

| Class | mAP50 |
|-------|-------|
| Crazing | 46.1% |
| Inclusion | 87.7% |
| Patches | 94.7% |
| Pitted Surface | 78.1% |
| Rolled-in Scale | 64.9% |
| Scratches | 95.3% |

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

If you did not create a virtual environment and installed packages directly for your user account, that is also okay.

## Train

If you are using the included `NEU-DET` dataset folder, prepare it once:

```bash
python prepare_neu_det.py
```

Then train:

```bash
python train.py
```

After training, the best model is copied to:

```text
models/best.pt
```

## Predict

Run detection on one image:

```bash
python detect.py --source path\to\image.jpg
```

Run detection on a webcam:

```bash
python detect.py --source 0
```

Prediction images are saved in:

```text
outputs/predictions/
```

## Demo App

Because Windows said `streamlit` is not on PATH, use this command:

```bash
python -m streamlit run app.py
```

## Demo

The Streamlit application allows users to upload steel surface images and receive defect predictions with bounding box visualization.

![InspectAI Demo](demo_screenshot.png)

Or use the included launcher:

```bash
python run_app.py
```

Open the local URL shown by Streamlit, upload an image, and view the defect result.

The app needs a trained model at:

```text
models/best.pt
```

## Suggested Datasets

- MVTec AD
- DAGM Dataset
- NEU Surface Defect Dataset
- KolektorSDD
- Custom phone or camera images

For YOLOv8 detection, every defect must have a bounding-box label. Tools such as Roboflow, CVAT, and LabelImg can export labels in YOLO format.

## Internship Report Points

- Problem statement
- Dataset source and class labels
- Annotation process
- Model architecture: YOLOv8
- Training settings: image size, epochs, batch size
- Metrics: precision, recall, mAP50, confusion matrix
- Demo screenshots
- Limitations and future improvements
