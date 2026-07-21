import os
import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# Page configuration
st.set_page_config(
    page_title="Real-Time Face Detection App",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
    <style>
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        background: linear-gradient(135deg, #10b981, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stApp {
        background-color: #0b0f19;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">👤 Real-Time Face Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Image Processing Assignment • Powered by OpenCV & Streamlit</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Detection Parameters")

scale_factor = st.sidebar.slider(
    "Scale Factor",
    min_value=1.05,
    max_value=1.50,
    value=1.10,
    step=0.05,
    help="Specifies how much the image size is reduced at each image scale."
)

min_neighbors = st.sidebar.slider(
    "Min Neighbors",
    min_value=1,
    max_value=10,
    value=5,
    step=1,
    help="Specifies how many neighbors each candidate rectangle should have to retain it."
)

box_color_choice = st.sidebar.selectbox(
    "Bounding Box Color",
    options=["Neon Green", "Electric Cyan", "Hot Pink", "Warm Amber"]
)

thickness = st.sidebar.slider(
    "Border Thickness",
    min_value=1,
    max_value=6,
    value=2,
    step=1
)

COLOR_MAP = {
    "Neon Green": (0, 255, 0),
    "Electric Cyan": (255, 255, 0),
    "Hot Pink": (203, 192, 255),
    "Warm Amber": (0, 191, 255)
}
chosen_color = COLOR_MAP[box_color_choice]

# Ultra-safe Haar Cascade initialization
def get_face_cascade():
    cascade = cv2.CascadeClassifier()
    
    # Candidate paths to locate XML file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base_dir, "haarcascade_frontalface_default.xml"),
        os.path.join(base_dir, "facedetection", "haarcascade_frontalface_default.xml"),
        "haarcascade_frontalface_default.xml",
        "facedetection/haarcascade_frontalface_default.xml"
    ]
    
    if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
        candidates.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))

    for path in candidates:
        if os.path.exists(path):
            if cascade.load(path):
                return cascade

    # Final fallback attempt
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        pass

    return cascade

face_cascade = get_face_cascade()

# WebRTC Video Processor Class
class FaceDetector(VideoProcessorBase):
    def __init__(self):
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.box_color = chosen_color
        self.thickness = thickness

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), self.box_color, self.thickness)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# STUN Server configuration for remote public hosting on Streamlit Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# App mode selector
mode = st.radio("Choose Input Mode:", ["📹 Live Webcam Stream", "📸 Camera Snapshot", "📁 Upload Image"], horizontal=True)

if mode == "📹 Live Webcam Stream":
    st.info("Click **START** below to grant webcam access and start real-time detection.")
    
    ctx = webrtc_streamer(
        key="face-detection",
        video_processor_factory=FaceDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    if ctx.video_processor:
        ctx.video_processor.scale_factor = scale_factor
        ctx.video_processor.min_neighbors = min_neighbors
        ctx.video_processor.box_color = chosen_color
        ctx.video_processor.thickness = thickness

elif mode == "📸 Camera Snapshot":
    img_file_buffer = st.camera_input("Take a photo to detect faces")
    
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(cv2_img, (x, y), (x + w, y + h), chosen_color, thickness)
                
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_container_width=True)
        else:
            st.error("Cascade classifier failed to load.")

elif mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image file (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=scale_factor,
                minNeighbors=min_neighbors,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in faces:
                cv2.rectangle(cv2_img, (x, y), (x + w, y + h), chosen_color, thickness)
                
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_container_width=True)
        else:
            st.error("Cascade classifier failed to load.")
