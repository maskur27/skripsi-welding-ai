import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 1. Konfigurasi Halaman & Tema Industrial
st.set_page_config(
    page_title="WeldingDefect - AI Inspection",
    page_icon="🏗️",
    layout="wide"
)

# Custom CSS untuk tampilan Industrial Dark Theme
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stHeader {
        background-color: #1f2937;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #f59e0b; /* Aksen Kuning Industri */
        margin-bottom: 2rem;
    }
    h1 {
        color: #f3f4f6;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .stButton>button {
        background-color: #f59e0b;
        color: black;
        font-weight: bold;
        width: 100%;
        border-radius: 5px;
    }
    .stButton>button:hover {
        background-color: #d97706;
        color: white;
    }
    .status-box {
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Header Dashboard
with st.container():
    st.markdown('<div class="stHeader"><h1>🏗️ Welding Defect AI Inspection System</h1><p style="color:#9ca3af;">Deep Learning Implementation using U-Net Architecture for Industrial Quality Control</p></div>', unsafe_allow_html=True)

# 3. Sidebar (Control Panel)
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2610/2610332.png", width=100) # Icon Las
st.sidebar.title("Control Panel")
st.sidebar.info("Gunakan panel ini untuk mengatur sensitivitas deteksi model U-Net.")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45)
st.sidebar.markdown("---")
st.sidebar.write("**Model Info:**")
st.sidebar.code("Architecture: U-Net\nFramework: YOLOv8-Seg\nInput Size: 640x640")

# 4. Load Model
@st.cache_resource
def load_model():
    return YOLO('best.pt') 

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Pastikan file 'best.pt' ada di folder yang sama.")

# 5. Area Utama (Upload & Hasil)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Input Image")
    uploaded_file = st.file_uploader("Upload foto alur las...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Citra Masukan Mentah", use_container_width=True)

with col2:
    st.subheader("🔍 Analysis Result")
    if uploaded_file is not None:
        with st.spinner('🔄 Memproses AI Segmentation...'):
            # Jalankan Prediksi
            results = model.predict(source=image, conf=conf_threshold)
            
            # Plot Hasil (Masker Biru)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Visualisasi Masker Abnormalitas", use_container_width=True)
            
            # Metadata Hasil
            count = len(results[0].boxes)
            # Metadata Hasil Dinamis
            if len(results[0].boxes) > 0:
                # Ambil nama kelas dari hasil deteksi pertama
                class_id = int(results[0].boxes[0].cls)
                class_name = model.names[class_id] # Akan muncul 'Good Weld' atau 'Bad Weld'
                conf_score = results[0].boxes[0].conf[0]
                
                if "Bad" in class_name:
                    # Jika terdeteksi cacat
                    st.markdown(f"""
                        <div style="background-color: #7f1d1d; color: white; padding: 15px; border-radius: 5px; text-align: center;">
                            ⚠️ <b>HASIL ANALISIS: TERDETEKSI CACAT ({class_name})</b><br>
                            Sistem mendeteksi area abnormalitas dengan keyakinan {conf_score:.2f}.
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    # Jika terdeteksi las bagus
                    st.markdown(f"""
                        <div style="background-color: #064e3b; color: white; padding: 15px; border-radius: 5px; text-align: center;">
                            ✅ <b>HASIL ANALISIS: PENGELASAN BAIK ({class_name})</b><br>
                            Alur las terdeteksi normal dengan keyakinan {conf_score:.2f}.
                        </div>
                    """, unsafe_allow_html=True)
            else:
                # Jika sama sekali tidak ada yang terdeteksi
                st.info("Sistem tidak menemukan objek alur las yang dikenali. Coba sesuaikan Confidence Threshold.")

# 6. Footer Analisis Teknik
if uploaded_file:
    st.markdown("---")
    st.subheader("📊 Metrik Teknis")
    cols = st.columns(4)
    
    # Menghitung Luas Masker (Sangat bagus untuk Gap Penelitian)
    if count > 0 and results[0].masks is not None:
        # Menghitung jumlah piksel masker
        total_area = 0
        for mask in results[0].masks.data:
            total_area += (mask > 0).sum().item()
        
        cols[0].metric("Defect Area", f"{total_area} Px", "Pixel count")
    else:
        cols[0].metric("Defect Area", "0 Px", "Clean")
        
    cols[1].metric("Inference Time", f"{results[0].speed['inference']:.2f} ms")
    cols[2].metric("Pre-process", f"{results[0].speed['preprocess']:.2f} ms")
    cols[3].metric("Image Size", f"{image.size[0]}x{image.size[1]}")