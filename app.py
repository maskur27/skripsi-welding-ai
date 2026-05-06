import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# 1. Konfigurasi Halaman & Tema Industrial
st.set_page_config(
    page_title="WeldingDefect AI - Linda Marlinda",
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
        border-left: 5px solid #f59e0b;
        margin-bottom: 2rem;
    }
    h1 {
        color: #f3f4f6;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #f59e0b;
        color: black;
        font-weight: bold;
        width: 100%;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# 2. Header Dashboard
with st.container():
    st.markdown("""
        <div class="stHeader">
            <h1>🏗️ Welding Defect AI Inspection System</h1>
            <p style="color:#9ca3af;">Implementation of Hybrid Transformer & Residual-Dilated-Inception U-Net Architecture</p>
        </div>
    """, unsafe_allow_html=True)

# 3. Sidebar (Control Panel)
st.sidebar.title("Control Panel")
st.sidebar.info("Sesuaikan ambang batas deteksi untuk hasil segmentasi yang lebih akurat.")

conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45)
st.sidebar.markdown("---")
st.sidebar.write("**Model Technical Details:**")
st.sidebar.code("Base: YOLOv8-Segmentation\nDecoder: U-Net Structure\nFeature Extractor: Hybrid Transformer\nInput: 640x640")

# 4. Load Model
@st.cache_resource
def load_model():
    # Memuat weights terbaik hasil training kamu
    return YOLO('best.pt') 

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}. Pastikan file 'best.pt' tersedia.")

# 5. Area Utama (Upload & Hasil)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Input Image")
    uploaded_file = st.file_uploader("Unggah foto permukaan las (JPG/PNG)...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Input Image", use_container_width=True)

with col2:
    st.subheader("🔍 Analysis Result")
    if uploaded_file is not None:
        with st.spinner('🔄 Analyzing Surface with Hybrid AI...'):
            # --- TAHAP PRE-PROCESSING ---
            # Konversi PIL ke OpenCV (BGR) untuk stabilitas koordinat YOLO
            img_array = np.array(image)
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # --- TAHAP INFERENSI ---
            # Menjalankan model pada image yang sudah dikonversi
            results = model.predict(source=img_cv2, conf=conf_threshold, save=False)
            
            # --- TAHAP VISUALISASI ---
            # results[0].plot() secara otomatis menangani rescaling koordinat 640x640 kembali ke ukuran asli
            res_plotted = results[0].plot(line_width=3, font_size=3)
            
            # Konversi kembali ke RGB untuk tampilan Streamlit
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="AI-Generated Segmentation Mask", use_container_width=True)
            
            # --- ANALISIS HASIL ---
            if len(results[0].boxes) > 0:
                # Mengambil informasi dari deteksi pertama (paling dominan)
                box = results[0].boxes[0]
                class_id = int(box.cls)
                class_name = model.names[class_id]
                conf_score = float(box.conf[0])
                
                # Tampilan status berdasarkan klasifikasi
                if "Bad" in class_name:
                    color = "#7f1d1d" # Red dark
                    status = f"⚠️ TERDETEKSI CACAT: {class_name.upper()}"
                else:
                    color = "#064e3b" # Green dark
                    status = f"✅ KUALITAS BAIK: {class_name.upper()}"
                
                st.markdown(f"""
                    <div style="background-color: {color}; color: white; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid white;">
                        <h3 style="margin:0;">{status}</h3>
                        <p style="margin-top:10px;">Confidence Score: <b>{conf_score:.2f}</b></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Objek tidak terdeteksi. Silakan turunkan Confidence Threshold di panel kiri.")

# 6. Footer Metrik Teknik
if uploaded_file and len(results[0].boxes) > 0:
    st.markdown("---")
    st.subheader("📊 Engineering Metrics")
    m_cols = st.columns(3)
    
    # Menghitung Luas Masker dalam Pixel
    if results[0].masks is not None:
        pixel_area = (results[0].masks.data > 0).sum().item()
        m_cols[0].metric("Defect Surface Area", f"{pixel_area} Px")
    
    m_cols[1].metric("Inference Speed", f"{results[0].speed['inference']:.1f} ms")
    m_cols[2].metric("Processing Speed", f"{results[0].speed['preprocess']:.1f} ms")
