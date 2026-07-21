import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# Page configuration
st.set_page_config(
    page_title="Real-Time Edge Detection App",
    page_icon="⚡",
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
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
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

st.markdown('<div class="main-header">⚡ Real-Time Edge Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Image Processing Assignment • Powered by OpenCV & Streamlit</div>', unsafe_allow_html=True)

# Sidebar settings
st.sidebar.header("⚙️ Canny Edge Parameters")

blur_kernel = st.sidebar.slider(
    "Gaussian Blur Size",
    min_value=1,
    max_value=15,
    value=5,
    step=2,
    help="Kernel size for Gaussian blur to reduce noise."
)

canny_min = st.sidebar.slider(
    "Canny Low Threshold",
    min_value=0,
    max_value=255,
    value=50,
    step=1
)

canny_max = st.sidebar.slider(
    "Canny High Threshold",
    min_value=0,
    max_value=255,
    value=150,
    step=1
)

invert = st.sidebar.checkbox("Invert Colors (Dark Edges)", value=False)

def process_edges(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    k = blur_kernel
    if k % 2 == 0:
        k += 1
    blur = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blur, canny_min, canny_max)
    if invert:
        edges = cv2.bitwise_not(edges)
    return edges

# WebRTC Video Processor Class
class EdgeDetector(VideoProcessorBase):
    def __init__(self):
        self.blur_kernel = blur_kernel
        self.canny_min = canny_min
        self.canny_max = canny_max
        self.invert = invert

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        k = self.blur_kernel
        if k % 2 == 0:
            k += 1
        blur = cv2.GaussianBlur(gray, (k, k), 0)
        edges = cv2.Canny(blur, self.canny_min, self.canny_max)
        if self.invert:
            edges = cv2.bitwise_not(edges)

        # Convert grayscale back to BGR for video output stream
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return av.VideoFrame.from_ndarray(edges_bgr, format="bgr24")

# STUN Server configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# App mode selector
mode = st.radio("Choose Input Mode:", ["📹 Live Webcam Stream", "📸 Camera Snapshot", "📁 Upload Image"], horizontal=True)

if mode == "📹 Live Webcam Stream":
    st.info("Click **START** below to grant webcam access and start real-time edge detection.")
    
    ctx = webrtc_streamer(
        key="edge-detection",
        video_processor_factory=EdgeDetector,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    
    if ctx.video_processor:
        ctx.video_processor.blur_kernel = blur_kernel
        ctx.video_processor.canny_min = canny_min
        ctx.video_processor.canny_max = canny_max
        ctx.video_processor.invert = invert

elif mode == "📸 Camera Snapshot":
    img_file_buffer = st.camera_input("Take a photo to process edges")
    
    if img_file_buffer is not None:
        bytes_data = img_file_buffer.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        edges = process_edges(cv2_img)
        st.image(edges, caption="Processed Edge Detection", use_container_width=True)

elif mode == "📁 Upload Image":
    uploaded_file = st.file_uploader("Upload an image file (JPG, PNG, JPEG)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        edges = process_edges(cv2_img)
        st.image(edges, caption="Processed Edge Detection", use_container_width=True)
