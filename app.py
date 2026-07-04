from pathlib import Path
from tempfile import NamedTemporaryFile

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "best.pt"


st.set_page_config(page_title="InspectAI", page_icon="QC", layout="wide")


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def draw_predictions(image: Image.Image, model: YOLO, confidence: float) -> tuple[np.ndarray, int]:
    with NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        image.convert("RGB").save(temp_file.name)
        results = model.predict(source=temp_file.name, conf=confidence)

    result = results[0]
    plotted = result.plot()
    plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
    count = len(result.boxes) if result.boxes is not None else 0
    return plotted, count


st.title("InspectAI")
st.caption("Automated quality inspection for detecting visible product defects with a trained YOLOv8 model.")

if not MODEL_PATH.exists():
    st.warning("No trained model found. Train one first with `python train.py`, then place it at `models/best.pt`.")
    st.stop()

model = load_model(str(MODEL_PATH))

left, right = st.columns([1, 1])

with left:
    uploaded_file = st.file_uploader("Upload inspection image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    confidence = st.slider("Confidence threshold", 0.05, 0.95, 0.25, 0.05)

with right:
    st.metric("Model", MODEL_PATH.name)

if uploaded_file:
    image = Image.open(uploaded_file)
    output_image, defect_count = draw_predictions(image, model, confidence)

    st.subheader("Inspection Result")
    status = "Defective" if defect_count else "Normal"
    st.metric("Status", status)
    st.metric("Detected defect regions", defect_count)
    st.image(output_image, caption="Predicted defect locations", use_container_width=True)
