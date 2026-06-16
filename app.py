import streamlit as st
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model
import time

# Page configuration
st.set_page_config(
    page_title="Face Mask Detection System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for professional corporate UI
st.markdown("""
<style>
/* Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font Override */
html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
}

/* App Background & Colors */
.stApp {
    background-color: #f8fafc !important;
    color: #0f172a !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #f8fafc !important;
}

[data-testid="stHeader"] {
    background-color: transparent !important;
}

/* Hide Default Streamlit Style Elements & Top Decoration Bar */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
[data-testid="stDecoration"] {display: none !important;}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}
::-webkit-scrollbar-track {
    background: #f1f5f9;
}
::-webkit-scrollbar-thumb {
    background: #cbd5e1;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #94a3b8;
}

/* Layout Constraints */
[data-testid="stAppViewBlockContainer"] {
    padding-top: 3rem !important;
    padding-bottom: 6rem !important;
    max-width: 1200px !important;
}

/* Hero Section */
.hero-section {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 20px;
    padding: 3.5rem 2.5rem;
    text-align: center;
    margin-bottom: 3rem;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.01), 0 10px 15px -3px rgba(0,0,0,0.02);
    position: relative;
    overflow: hidden;
}

.hero-section::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 6px;
    background: linear-gradient(90deg, #2563eb 0%, #3b82f6 50%, #10b981 100%);
}

.hero-badges {
    display: flex;
    justify-content: center;
    gap: 0.75rem;
    margin-top: 1.5rem;
    margin-bottom: 0 !important;
}

.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: #eff6ff;
    color: #2563eb;
    font-size: 0.85rem;
    font-weight: 700;
    padding: 0.35rem 1rem;
    border-radius: 9999px;
    border: 1px solid #bfdbfe;
    letter-spacing: 0.02em;
}

.hero-pill.secondary {
    background: #f0fdf4;
    color: #16a34a;
    border-color: #bbf7d0;
}

.hero-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 2.75rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0 0 1rem 0;
    letter-spacing: -0.03em;
    line-height: 1.15;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #475569;
    max-width: 750px;
    margin: 0 auto !important;
    line-height: 1.6;
    text-align: center !important;
}

/* Card Styling for Native Bordered Containers */
div[data-testid="stVerticalBlockBorder"] {
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 16px !important;
    padding: 2rem !important;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.01), 0 10px 15px -3px rgba(0,0,0,0.02) !important;
    margin-bottom: 1.5rem !important;
}

div[data-testid="stVerticalBlockBorder"]:hover {
    border-color: #cbd5e1 !important;
}

.card-title {
    color: #0f172a;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    margin-top: 0;
    margin-bottom: 1.25rem;
    letter-spacing: -0.02em;
}

/* Radio Selector (Tabs design) */
div[data-testid="stRadio"] {
    background-color: transparent !important;
}

div[data-testid="stRadio"] > label {
    display: none !important;
}

div[data-testid="stRadio"] > div[role="radiogroup"] {
    display: flex !important;
    flex-direction: row !important;
    gap: 0.75rem !important;
    width: 100% !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label {
    flex: 1 !important;
    background-color: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 10px !important;
    padding: 0.75rem 1rem !important;
    margin: 0 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    cursor: pointer !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.02) !important;
}

/* Hide the circle visual container specifically */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* Handle the text styles */
div[data-testid="stRadio"] label[data-baseweb="radio"] > div:nth-child(2) {
    display: block !important;
    width: 100% !important;
    text-align: center !important;
}

div[data-testid="stRadio"] label p,
div[data-testid="stRadio"] label span,
div[data-testid="stRadio"] label div[data-testid="stMarkdownContainer"] {
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: #64748b !important;
    margin: 0 !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
    border-color: #cbd5e1 !important;
    background-color: #f8fafc !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
}

div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] p,
div[data-testid="stRadio"] div[role="radiogroup"] > label[data-checked="true"] div[data-testid="stMarkdownContainer"] {
    color: #1d4ed8 !important;
}

/* Upload Area Visual Frame */
.upload-area {
    border: 2px dashed #cbd5e1;
    border-radius: 12px;
    padding: 2.25rem 1.5rem;
    text-align: center;
    background: #f8fafc;
    transition: all 0.2s ease;
    cursor: pointer;
    margin-bottom: 1.25rem;
}

.upload-area:hover {
    border-color: #2563eb;
    background: #eff6ff;
}

.upload-icon {
    font-size: 2.25rem;
    color: #2563eb;
    margin-bottom: 0.75rem;
    line-height: 1;
}

.upload-text {
    color: #334155;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0;
}

.upload-hint {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0.35rem 0 0 0;
}

/* Custom Success/Info Alert Badges */
.custom-badge {
    padding: 0.6rem 1rem;
    border-radius: 10px;
    font-size: 0.9rem;
    font-weight: 600;
    margin-top: 1rem;
    text-align: center;
    border: 1px solid transparent;
}

.badge-success {
    background-color: #f0fdf4;
    color: #16a34a;
    border-color: #bbf7d0;
}

/* Constrained Image styling */
div[data-testid="stVerticalBlockBorder"] img {
    border-radius: 10px !important;
    max-width: 260px !important;
    margin: 0 auto !important;
    display: block !important;
}

.image-caption {
    padding-top: 0.75rem;
    color: #64748b;
    font-size: 0.85rem;
    font-weight: 600;
    text-align: center;
    letter-spacing: 0.02em;
}

/* File Uploader override */
div[data-testid="stFileUploader"] {
    border: 1px dashed #e2e8f0 !important;
    border-radius: 12px !important;
    background-color: #f8fafc !important;
    padding: 1rem !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stFileUploader"]:hover {
    border-color: #2563eb !important;
    background-color: #eff6ff !important;
}

div[data-testid="stFileUploader"] > section {
    background-color: transparent !important;
    border: none !important;
    padding: 0 !important;
}

div[data-testid="stFileUploader"] button {
    background-color: #ffffff !important;
    border: 1px solid #cbd5e1 !important;
    border-radius: 8px !important;
    color: #334155 !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.25rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.05) !important;
}

div[data-testid="stFileUploader"] button:hover {
    background-color: #f1f5f9 !important;
    border-color: #94a3b8 !important;
    color: #0f172a !important;
}

/* Camera Input override */
div[data-testid="stCameraInput"] {
    border: 1px solid #e2e8f0 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.02) !important;
}

div[data-testid="stCameraInput"] button {
    background-color: #2563eb !important;
    color: #ffffff !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    transition: all 0.2s ease !important;
}

div[data-testid="stCameraInput"] button:hover {
    background-color: #1d4ed8 !important;
}

/* Action buttons styling */
div.stButton > button {
    background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
    color: #ffffff !important;
    border: none !important;
    padding: 0.85rem 2rem !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: -0.01em !important;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.15), 0 2px 4px -2px rgba(37, 99, 235, 0.15) !important;
    width: 100% !important;
    margin-top: 1rem !important;
    cursor: pointer !important;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(37, 99, 235, 0.25), 0 4px 6px -4px rgba(37, 99, 235, 0.25) !important;
    color: #ffffff !important;
}

div.stButton > button:active {
    transform: translateY(0) !important;
    box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.1) !important;
}

/* Result Display Styles */
.result-container {
    background: #ffffff;
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #e2e8f0;
    border-left: 6px solid #3b82f6;
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.01), 0 10px 15px -3px rgba(0,0,0,0.02);
    margin-bottom: 1.5rem;
}

.result-success {
    border-left-color: #10b981;
    background: linear-gradient(90deg, #ffffff 0%, #f0fdf4 100%);
}

.result-danger {
    border-left-color: #ef4444;
    background: linear-gradient(90deg, #ffffff 0%, #fef2f2 100%);
}

.result-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}

.result-title-success {
    color: #065f46;
}

.result-title-danger {
    color: #991b1b;
}

.result-description {
    color: #475569;
    font-size: 0.95rem;
    margin: 0 0 1.5rem 0;
    line-height: 1.5;
}

.confidence-section {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.25rem;
}

.confidence-label {
    color: #475569;
    font-size: 0.95rem;
    font-weight: 600;
}

.confidence-value {
    font-weight: 800;
    font-size: 1.35rem;
    font-family: 'Plus Jakarta Sans', sans-serif;
}

.confidence-value-success {
    color: #059669;
}

.confidence-value-danger {
    color: #dc2626;
}

.raw-score {
    color: #94a3b8;
    font-size: 0.8rem;
    margin: 0.75rem 0 0 0;
    text-align: right;
}

/* Empty / Awaiting States content */
.empty-state-content {
    text-align: center;
    padding: 3rem 1.5rem;
}

.empty-icon {
    font-size: 4rem;
    color: #94a3b8;
    margin-bottom: 1.25rem;
    line-height: 1;
}

.empty-state-content .empty-icon {
    color: #94a3b8;
}

.empty-state-content.info .empty-icon {
    color: #3b82f6;
}

.empty-title {
    color: #1e293b;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 0;
    letter-spacing: -0.01em;
}

.empty-subtitle {
    color: #64748b;
    font-size: 0.95rem;
    margin: 0.65rem 0 0 0;
    line-height: 1.5;
}

/* Info Cards Grid */
.info-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-top: 3.5rem;
    padding-top: 2.5rem;
    border-top: 1px solid #e2e8f0;
}

.info-item {
    text-align: center;
    padding: 1.25rem;
    background: #ffffff;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,0.01), 0 1px 2px -1px rgba(0,0,0,0.01);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.info-item:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05);
    border-color: #cbd5e1;
}

.info-icon {
    font-size: 1.75rem;
    color: #2563eb;
    margin-bottom: 0.75rem;
    line-height: 1;
}

.info-title {
    color: #0f172a;
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.01em;
}

.info-description {
    color: #64748b;
    font-size: 0.85rem;
    margin: 0;
    line-height: 1.4;
}

/* Footer */
.footer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.85rem;
    padding: 2.5rem 0 1.5rem 0;
    border-top: 1px solid #e2e8f0;
    margin-top: 3.5rem;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# Initialize Session States for cleaner workflows
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = None
if 'current_file_id' not in st.session_state:
    st.session_state['current_file_id'] = None

# Load model with cache
@st.cache_resource
def load_mask_model():
    try:
        model = load_model("face_mask_detector_v2.keras")
        return model
    except Exception as e:
        st.error(f"Model file not found. Please ensure 'face_mask_detector_v2.keras' is in the correct directory. Error: {e}")
        return None

model = load_mask_model()

# Hero Section
st.markdown("""
<div class="hero-section">
<h1 class="hero-title">Face Mask Detection System</h1>
<p class="hero-subtitle">Utilizing state-of-the-art computer vision and an optimized MobileNetV2 deep neural network for precise, real-time face mask compliance monitoring.</p>
<div class="hero-badges">
<span class="hero-pill">⚡ MobileNetV2 Architecture</span>
<span class="hero-pill secondary">👁️ Computer Vision</span>
</div>
</div>
""", unsafe_allow_html=True)

# Main layout split into Interactive Inputs (Col 1) and Analysis Outputs (Col 2)
col1, col2 = st.columns([1, 1], gap="large")

image = None

with col1:
    with st.container(border=True):
        st.markdown('<p class="card-title">Input Selection</p>', unsafe_allow_html=True)
        
        option = st.radio(
            "Select input method",
            ["Upload Image", "Use Camera"],
            index=0
        )
        
        if option == "Upload Image":
            st.markdown("""
            <div class="upload-area">
                <div class="upload-icon">📤</div>
                <p class="upload-text">Select a face image to analyze</p>
                <p class="upload-hint">Drag & drop or click "Browse files" below. Supports JPG, JPEG, PNG.</p>
            </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=["jpg", "jpeg", "png"],
                label_visibility="collapsed"
            )
            
            if uploaded_file:
                image = Image.open(uploaded_file).convert("RGB")
                # Auto-reset predictions when a new file is uploaded
                file_id = f"upload_{uploaded_file.name}_{uploaded_file.size}"
                if st.session_state['current_file_id'] != file_id:
                    st.session_state['prediction'] = None
                    st.session_state['current_file_id'] = file_id
                
                st.markdown('<div class="custom-badge badge-success">✓ Image loaded successfully</div>', unsafe_allow_html=True)
        
        else:
            camera_image = st.camera_input("Capture image", label_visibility="collapsed")
            
            if camera_image:
                image = Image.open(camera_image).convert("RGB")
                # Auto-reset predictions when a new photo is taken
                file_id = f"camera_{camera_image.size}"
                if st.session_state['current_file_id'] != file_id:
                    st.session_state['prediction'] = None
                    st.session_state['current_file_id'] = file_id
                    
                st.markdown('<div class="custom-badge badge-success">✓ Image captured successfully</div>', unsafe_allow_html=True)
    
    # Reset states when no image is loaded
    if image is None:
        st.session_state['prediction'] = None
        st.session_state['current_file_id'] = None

    # Group the Action Button with the Inputs in Column 1
    if image is not None:
        if st.button("⚡ Analyze Image", use_container_width=True):
            if model is not None:
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    img = np.array(image)
                    img = cv2.resize(img, (128, 128))
                    img = img / 255.0
                    img = np.expand_dims(img, axis=0)
                    
                    # Make prediction
                    pred = model.predict(img, verbose=0)[0][0]
                    st.session_state['prediction'] = pred
            else:
                st.error("Model not loaded. Please check the model file.")

with col2:
    if image is not None:
        # 1. Preview Container
        with st.container(border=True):
            st.markdown('<p class="card-title">Preview Image</p>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown('<div class="image-caption">Preview Image</div>', unsafe_allow_html=True)
        
        # 2. Results Container or Action Awaiting message
        if st.session_state['prediction'] is not None:
            pred = st.session_state['prediction']
            if pred > 0.5:
                confidence = pred * 100
                st.markdown(f"""
                <div class="result-container result-success">
                    <h3 class="result-title result-title-success">Mask Detected</h3>
                    <p class="result-description">The subject in the image is wearing a face mask</p>
                    <div class="confidence-section">
                        <span class="confidence-label">Confidence Score</span>
                        <span class="confidence-value confidence-value-success">{confidence:.1f}%</span>
                    </div>
                    <p class="raw-score">Raw prediction score: {pred:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                confidence = (1 - pred) * 100
                st.markdown(f"""
                <div class="result-container result-danger">
                    <h3 class="result-title result-title-danger">No Mask Detected</h3>
                    <p class="result-description">The subject in the image is not wearing a face mask</p>
                    <div class="confidence-section">
                        <span class="confidence-label">Confidence Score</span>
                        <span class="confidence-value confidence-value-danger">{confidence:.1f}%</span>
                    </div>
                    <p class="raw-score">Raw prediction score: {pred:.4f}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            with st.container(border=True):
                st.markdown("""
                <div class="empty-state-content info">
                    <div class="empty-icon">⚡</div>
                    <p class="empty-title">Awaiting Analysis</p>
                    <p class="empty-subtitle">Click the "Analyze Image" button on the left to run prediction.</p>
                </div>
                """, unsafe_allow_html=True)
            
    else:
        # Awaiting Input state
        with st.container(border=True):
            st.markdown("""
            <div class="empty-state-content">
                <div class="empty-icon">🖼️</div>
                <p class="empty-title">No image loaded</p>
                <p class="empty-subtitle">Upload an image or use camera on the left to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)

# Information section
st.markdown("""
<div class="info-grid">
    <div class="info-item">
        <div class="info-icon">🎯</div>
        <p class="info-title">High Accuracy</p>
        <p class="info-description">Advanced neural network trained on thousands of diverse images</p>
    </div>
    <div class="info-item">
        <div class="info-icon">⚡</div>
        <p class="info-title">Real-time Analysis</p>
        <p class="info-description">High-performance inference engine delivering instant feedback</p>
    </div>
    <div class="info-item">
        <div class="info-icon">🛡️</div>
        <p class="info-title">Privacy Centric</p>
        <p class="info-description">Local processing guarantees your media is never uploaded to external servers</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Face Mask Detection System • Powered by TensorFlow and Streamlit
</div>
""", unsafe_allow_html=True)