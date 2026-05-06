import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 1. Konfigurasi Halaman (Centered & Industrial Theme)
st.set_page_config(
    page_title="WeldingDefect AI - KURNIYAWANTORO",
    page_icon="🏗️",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    .stHeader {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f59e0b;
        margin-bottom: 2rem;
    }
    h1 { color: #f3f4f6; font-family: 'Segoe UI', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# 2. Header
st.markdown('<div class="stHeader"><h1>🏗️ Welding Defect AI Inspection System</h1><p style="color:#9ca3af;">Hybrid Transformer & Residual-Dilated-Inception U-Net Implementation</p></div>', unsafe_allow_html=True)

# 3. Sidebar
st.sidebar.title("Control Panel")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45)
st.sidebar.markdown("---")
st.sidebar.code("Architecture: Hybrid U-Net\nFramework: YOLOv8-Seg\nInput Size: 640x640")

# 4. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt') 

try:
    model = load_model()
except Exception as e:
    st.error(f"Model 'best.pt' tidak ditemukan! Pastikan file ada di direktori yang sama.")

# 5. Main Area
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Input Image")
    uploaded_file = st.file_uploader("Unggah foto alur las...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_container_width=True)

with col2:
    st.subheader("🔍 Analysis Result")
    if uploaded_file is not None:
        with st.spinner('🔄 Analyzing Surface...'):
            # --- PROSES FIX KOORDINAT ---
            # 1. Convert ke OpenCV BGR
            img_cv2 = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            # 2. Prediksi dengan imgsz=640 untuk sinkronisasi koordinat
            results = model.predict(source=img_cv2, conf=conf_threshold, imgsz=640, save=False)
            
            # 3. Plotting Hasil (Sesuai gaya Gambar 2)
            # Parameter conf=True dan labels=True memastikan teks muncul di kotak
            res_plotted = results[0].plot(
                conf=True, 
                line_width=2, 
                font_size=1, 
                labels=True, 
                boxes=True
            )
            
            # 4. Tampilkan Kembali ke RGB
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="AI Segmentation Result", use_container_width=True)
            
            # --- STATUS BOX ---
            if len(results[0].boxes) > 0:
                class_name = model.names[int(results[0].boxes[0].cls)]
                conf_score = float(results[0].boxes[0].conf[0])
                color = "#064e3b" if "Good" in class_name else "#7f1d1d"
                
                st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px; text-align: center;">
                        <h3 style="margin:0;">HASIL: {class_name.upper()}</h3>
                        <p style="margin:5px 0 0 0;">Confidence Score: <b>{conf_score:.2f}</b></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.info("Objek tidak terdeteksi. Coba turunkan Confidence Threshold.")

# 6. Engineering Metrics
if uploaded_file and len(results[0].boxes) > 0:
    st.markdown("---")
    cols = st.columns(3)
    if results[0].masks is not None:
        area = (results[0].masks.data > 0).sum().item()
        cols[0].metric("Defect Area", f"{area} Px")
    cols[1].metric("Inference Time", f"{results[0].speed['inference']:.1f} ms")
    cols[2].metric("Image Res", f"{image.size[0]}x{image.size[1]}")
