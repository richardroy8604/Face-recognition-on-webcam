import cv2
import av
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, RTCConfiguration, VideoProcessorBase

# Page config
st.set_page_config(
    page_title="Image Processing Web Suite",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main-header {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.6rem;
        background: linear-gradient(135deg, #10b981, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #9ca3af;
        font-size: 1.05rem;
        margin-bottom: 1.8rem;
    }
    .stApp {
        background-color: #0b0f19;
    }
    </style>
""", unsafe_allow_html=True)

# Navigation in Sidebar
st.sidebar.title("📌 Navigation")
app_mode = st.sidebar.radio("Select Assignment Application:", ["👤 Face Detection", "⚡ Edge Detection"])

# STUN Server configuration for Streamlit Cloud deployment
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# -------------------------------------------------------------
# 1. FACE DETECTION APP
# -------------------------------------------------------------
if app_mode == "👤 Face Detection":
    st.markdown('<div class="main-header">👤 Real-Time Face Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Image Processing Assignment • Haar Cascade Face Detector</div>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Face Detection Parameters")
    scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.50, 1.10, 0.05)
    min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5, 1)
    box_color_choice = st.sidebar.selectbox("Bounding Box Color", ["Neon Green", "Electric Cyan", "Hot Pink", "Warm Amber"])
    thickness = st.sidebar.slider("Border Thickness", 1, 6, 2, 1)

    COLOR_MAP = {
        "Neon Green": (0, 255, 0),
        "Electric Cyan": (255, 255, 0),
        "Hot Pink": (203, 192, 255),
        "Warm Amber": (0, 191, 255)
    }
    chosen_color = COLOR_MAP[box_color_choice]

    @st.cache_resource
    def load_face_cascade():
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    face_cascade = load_face_cascade()

    class FaceDetector(VideoProcessorBase):
        def __init__(self):
            self.scale_factor = scale_factor
            self.min_neighbors = min_neighbors
            self.box_color = chosen_color
            self.thickness = thickness

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=self.scale_factor,
                minNeighbors=self.min_neighbors,
                minSize=(30, 30)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), self.box_color, self.thickness)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    mode = st.radio("Input Source:", ["📹 Live Webcam Stream", "📸 Camera Snapshot", "📁 Upload Image"], horizontal=True)

    if mode == "📹 Live Webcam Stream":
        st.info("Click **START** below to grant webcam access and launch real-time face detection.")
        ctx = webrtc_streamer(
            key="face-stream",
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
        img_buffer = st.camera_input("Take a photo")
        if img_buffer:
            cv2_img = cv2.imdecode(np.frombuffer(img_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(cv2_img, (x, y), (x + w, y + h), chosen_color, thickness)
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_container_width=True)

    elif mode == "📁 Upload Image":
        uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded:
            cv2_img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))
            for (x, y, w, h) in faces:
                cv2.rectangle(cv2_img, (x, y), (x + w, y + h), chosen_color, thickness)
            st.image(cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB), caption=f"Detected {len(faces)} face(s)", use_container_width=True)

# -------------------------------------------------------------
# 2. EDGE DETECTION APP
# -------------------------------------------------------------
else:
    st.markdown('<div class="main-header">⚡ Real-Time Edge Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Image Processing Assignment • Canny Edge Detector</div>', unsafe_allow_html=True)

    st.sidebar.header("⚙️ Edge Detection Parameters")
    blur_kernel = st.sidebar.slider("Gaussian Blur Size", 1, 15, 5, 2)
    canny_min = st.sidebar.slider("Canny Low Threshold", 0, 255, 50, 1)
    canny_max = st.sidebar.slider("Canny High Threshold", 0, 255, 150, 1)
    invert = st.sidebar.checkbox("Invert Colors", value=False)

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
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return av.VideoFrame.from_ndarray(edges_bgr, format="bgr24")

    mode = st.radio("Input Source:", ["📹 Live Webcam Stream", "📸 Camera Snapshot", "📁 Upload Image"], horizontal=True)

    if mode == "📹 Live Webcam Stream":
        st.info("Click **START** below to grant webcam access and launch real-time edge detection.")
        ctx = webrtc_streamer(
            key="edge-stream",
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
        img_buffer = st.camera_input("Take a photo")
        if img_buffer:
            cv2_img = cv2.imdecode(np.frombuffer(img_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            edges = process_edges(cv2_img)
            st.image(edges, caption="Processed Edge Detection", use_container_width=True)

    elif mode == "📁 Upload Image":
        uploaded = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])
        if uploaded:
            cv2_img = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            edges = process_edges(cv2_img)
            st.image(edges, caption="Processed Edge Detection", use_container_width=True)
