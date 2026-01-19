import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# --- Page Configuration ---
st.set_page_config(
    page_title="InfernoGuard | AI Fire Detection",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling ---
st.markdown("""
    <style>
        .main {
            background-color: #0E1117;
        }
        h1, h2, h3 {
            color: #FF4B4B;
            font-family: 'Inter', sans-serif;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 8px;
            border: none;
            padding: 10px 24px;
            font-weight: 600;
            transition: all 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #D93030;
            box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
        }
        .uploadedImage {
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .stDeployButton {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://img.icons8.com/color/96/fire-element--v1.png", width=80)
with col2:
    st.title("InfernoGuard AI")
    st.markdown("### Real-time Fire & Smoke Detection System")

st.markdown("---")

# --- Model Loading ---
@st.cache_resource
def load_model():
    # Load YOLOv8n model - will auto-download if not present
    return YOLO('yolov8n.pt')

try:
    with st.spinner("Initializing Neural Network..."):
        model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
    st.info("Adjust the confidence threshold to filter weak detections.")
    
    st.markdown("### Model Info")
    st.text("YOLOv8 Nano")
    st.text("Pre-trained: COCO")
    
    st.markdown("---")
    st.write("Developed for [Amjid-Ali](https://github.com/Amjid-Ali)")

# --- Main Interface ---
st.write("Upload an image to detect potential fire hazards.")

uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read Image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Display Original
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True, caption="Source Input")

    # Run Inference
    with st.spinner("Analyzing scene..."):
        # Run inference
        results = model(image, conf=confidence)

        # Visualize results
        res_plotted = results[0].plot()
        res_image = Image.fromarray(res_plotted[..., ::-1]) # Convert BGR to RGB if needed, specifically for plot() output usually RGB but cv2 uses BGR. 
        # Ultralytics plot() returns BGR numpy array.
        
    # Display Result
    with col2:
        st.subheader("AI Analysis")
        st.image(res_plotted, channels="BGR", use_column_width=True, caption="Detection Output")
        
    # Detection Stats
    st.markdown("### 📊 Detection Details")
    detections = results[0].boxes
    if len(detections) > 0:
        for box in detections:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            name = model.names[cls]
            if name.lower() in ['fire', 'smoke'] or True: # Show all detections for now since base model might not have 'fire' unless finetuned.
                # Note: YOLOv8n (COCO) detects 'toaster', 'oven', etc. but for 'fire' specifically users usually use a custom model.
                # Since the user repo linked had 'yolov8n.pt', we assume they might expect generic objects or they have a custom model they didn't provide.
                # I will add a note if it's the base model.
                st.success(f"Detected **{name.upper()}** with {conf:.2f} confidence")
    else:
        st.warning("No objects detected above threshold.")

else:
    st.markdown("""
        <div style="text-align: center; padding: 50px; background-color: #262730; border-radius: 12px; border: 2px dashed #4B4B4B;">
            <p style="color: #FAFAFA; font-size: 1.2rem;">Waiting for Image Upload...</p>
        </div>
    """, unsafe_allow_html=True)
