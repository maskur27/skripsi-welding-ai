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

# Custom CSS untuk Industrial Dark Theme
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
st.markdown("""
    <div class="stHeader">
        <h1>🏗️ Welding Defect AI Inspection System</h1>
        <p style="color:#9ca3af;">Hybrid Transformer & Residual-Dilated-Inception U-Net Implementation</p>
    </div>
""", unsafe_allow_html=True)

# 3. Sidebar (Control Panel)
st.sidebar.title("Control Panel")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.45)
st.sidebar.markdown("---")
st.sidebar.write("**Technical Specs:**")
st.sidebar.code("Base: YOLOv8-Seg\nInput: 640x640 Fixed\nTask: Instance Segmentation")

# 4. Load Model
@st.cache_resource
def load_model():
    # Pastikan file 'best.pt' ada di folder yang sama dengan skrip ini
    return YOLO('best.pt') 

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model: {e}")

# 5. Area Utama (Upload & Hasil)
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.subheader("📁 Input Image")
    uploaded_file = st.file_uploader("Unggah foto alur las...", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Original Input", use_container_width=True)

with col2:
    st.subheader("🔍 Analysis Result")
    if uploaded_file is not None:
        with st.spinner('🔄 Analyzing Surface...'):
            # --- TAHAP SAKTI: FORCED RESIZING ---
            # Kita paksa gambar ke 640x640 agar koordinat kotak/masker tidak lari ke pojok
            img_resized = raw_image.resize((640, 640))
            img_array = np.array(img_resized)
            
            # Konversi RGB ke BGR untuk standar YOLOv8
            img_cv2 = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            # Jalankan Prediksi
            results = model.predict(source=img_cv2, conf=conf_threshold, imgsz=640)
            
            # Plotting hasil deteksi ke gambar
            res_plotted = results[0].plot(line_width=2, font_size=1)
            
            # Konversi kembali ke RGB untuk Streamlit
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="AI Detection & Segmentation Result", use_container_width=True)
            
            # --- DINAMIS STATUS BOX ---
            if len(results[0].boxes) > 0:
                class_id = int(results[0].boxes[0].cls)
                class_name = model.names[class_id]
                conf_score = float(results[0].boxes[0].conf[0])
                
                # Warna box status (Hijau untuk Good, Merah untuk Bad)
                status_color = "#064e3b" if "good" in class_name.lower() else "#7f1d1d"
                
                st.markdown(f"""
                    <div style="background-color: {status_color}; color: white; padding: 20px; border-radius: 8px; text-align: center; border: 1px solid white;">
                        <h3 style="margin:0;">HASIL: {class_name.upper()}</h3>
                        <p style="margin-top:10px;">Confidence Score: <b>{conf_score:.2f}</b></p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("Sistem tidak mendeteksi objek. Coba turunkan Confidence Threshold.")

# 6. Engineering Metrics
if uploaded_file and len(results[0].boxes) > 0:
    st.markdown("---")
    m_cols = st.columns(3)
    
    # Estimasi Luas Area Cacat/Las
    if results[0].masks is not None:
        pixel_area = (results[0].masks.data > 0).sum().item()
        m_cols[0].metric("Segmentation Area", f"{pixel_area} Px")
    
    m_cols[1].metric("Inference Speed", f"{results[0].speed['inference']:.1f} ms")
    m_cols[2].metric("Model Input", "640x640 (Fixed)")
