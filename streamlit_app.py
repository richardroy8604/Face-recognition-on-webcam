import os
import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(
    page_title="Real-Time Vision Portal",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust multi-STUN server configuration for cloud WebRTC streaming
RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302",
                    "stun:stun1.l.google.com:19302",
                    "stun:stun2.l.google.com:19302",
                    "stun:stun3.l.google.com:19302",
                    "stun:stun4.l.google.com:19302",
                    "stun:stun.services.mozilla.com"
                ]
            }
        ]
    }
)

# Cached helper to load cascade classifier safely across all OS environments
@st.cache_resource
def get_face_cascade():
    paths_to_try = [
        os.path.abspath("haarcascade_frontalface_default.xml"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "haarcascade_frontalface_default.xml"),
        os.path.join(os.getcwd(), "haarcascade_frontalface_default.xml"),
        "haarcascade_frontalface_default.xml",
    ]
    try:
        if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
            paths_to_try.append(os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml"))
    except Exception:
        pass

    for p in paths_to_try:
        try:
            if p and os.path.exists(p):
                c = cv2.CascadeClassifier(p)
                if hasattr(c, 'empty') and not c.empty():
                    return c
                c2 = cv2.CascadeClassifier()
                if hasattr(c2, 'load') and c2.load(p) and not c2.empty():
                    return c2
        except Exception:
            pass

    # Final fallback
    try:
        return cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    except Exception:
        return cv2.CascadeClassifier()


st.title("🎥 Real-Time Image Processing Web App")
st.markdown("Choose between **Edge Detection** and **Face Detection** in the sidebar.")

# App selection
app_mode = st.sidebar.selectbox(
    "Select Application",
    ["Edge Detection", "Face Detection"]
)

# Input method selection
input_mode = st.sidebar.radio(
    "Select Input Mode",
    ["Camera Snapshot / File Upload", "Live Video (Webcam)"]
)

# ----------------- EDGE DETECTION PROCESSORS -----------------
class EdgeDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.blur_kernel = 5
        self.canny_min = 50
        self.canny_max = 150
        self.invert = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            k = self.blur_kernel
            if k < 1: k = 1
            if k % 2 == 0: k += 1
            blur = cv2.GaussianBlur(gray, (k, k), 0)
            
            edges = cv2.Canny(blur, self.canny_min, self.canny_max)
            
            if self.invert:
                edges = cv2.bitwise_not(edges)
                
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            return av.VideoFrame.from_ndarray(edges_bgr, format="bgr24")
        except Exception:
            return frame

# ----------------- FACE DETECTION PROCESSORS -----------------
class FaceDetectionProcessor(VideoProcessorBase):
    def __init__(self):
        self.scale_factor = 1.1
        self.min_neighbors = 5
        self.color = (0, 255, 0) # BGR: Green
        self.thickness = 2
        self.face_cascade = get_face_cascade()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            if hasattr(self.face_cascade, 'empty') and not self.face_cascade.empty():
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=self.scale_factor,
                    minNeighbors=self.min_neighbors,
                    minSize=(30, 30)
                )
                for (x, y, w, h) in faces:
                    cv2.rectangle(img, (x, y), (x + w, y + h), self.color, self.thickness)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception:
            return frame

# Helper for static images
def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)


# =============================================================
# EDGE DETECTION VIEW
# =============================================================
if app_mode == "Edge Detection":
    st.header("⚡ Edge Detection")
    st.markdown("Adjust parameters in the sidebar to process images/video.")

    st.sidebar.markdown("### 🎛️ Parameters")
    blur_kernel = st.sidebar.slider("Gaussian Blur Kernel Size", 1, 15, 5, step=2)
    canny_min = st.sidebar.slider("Canny Low Threshold", 0, 255, 50)
    canny_max = st.sidebar.slider("Canny High Threshold", 0, 255, 150)
    invert = st.sidebar.checkbox("Invert Colors", value=False)

    if input_mode == "Live Video (Webcam)":
        st.info("Click **START** below to grant camera access. *(Note: If WebRTC is blocked on your cloud network, switch to Camera Snapshot mode in sidebar)*.")
        try:
            ctx = webrtc_streamer(
                key="edge-detection-live",
                video_processor_factory=EdgeDetectionProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )

            if ctx.video_processor:
                ctx.video_processor.blur_kernel = blur_kernel
                ctx.video_processor.canny_min = canny_min
                ctx.video_processor.canny_max = canny_max
                ctx.video_processor.invert = invert
        except Exception as e:
            st.error(f"Live WebRTC stream error: {e}. Please use 'Camera Snapshot / File Upload' mode.")

    else: # Snapshot / Upload
        source = st.radio("Choose Source", ["Camera Snapshot", "Upload Image File"])
        img_array = None

        if source == "Camera Snapshot":
            img_file = st.camera_input("Take a photo")
            if img_file:
                img_array = load_image(img_file)
        else:
            img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if img_file:
                img_array = load_image(img_file)

        if img_array is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(img_array, use_container_width=True)
            with col2:
                st.subheader("Processed Edges")
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                k = blur_kernel if blur_kernel % 2 != 0 else blur_kernel + 1
                blur = cv2.GaussianBlur(gray, (k, k), 0)
                edges = cv2.Canny(blur, canny_min, canny_max)
                if invert:
                    edges = cv2.bitwise_not(edges)
                st.image(edges, use_container_width=True)


# =============================================================
# FACE DETECTION VIEW
# =============================================================
elif app_mode == "Face Detection":
    st.header("🧑 Face Detection")
    st.markdown("Adjust parameters in the sidebar to detect faces.")

    st.sidebar.markdown("### 🎛️ Parameters")
    scale_factor = st.sidebar.slider("Scale Factor", 1.05, 1.50, 1.10, step=0.05)
    min_neighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5)
    color_choice = st.sidebar.selectbox("Box Color", ["Neon Green", "Electric Cyan", "Hot Pink", "Warm Amber"])
    thickness = st.sidebar.slider("Box Thickness", 1, 5, 2)

    COLOR_MAP_BGR = {
        "Neon Green": (0, 255, 0),
        "Electric Cyan": (255, 255, 0),
        "Hot Pink": (203, 192, 255),
        "Warm Amber": (0, 191, 255)
    }
    COLOR_MAP_RGB = {
        "Neon Green": (0, 255, 0),
        "Electric Cyan": (0, 255, 255),
        "Hot Pink": (255, 105, 180),
        "Warm Amber": (255, 191, 0)
    }

    if input_mode == "Live Video (Webcam)":
        st.info("Click **START** below to grant camera access. *(Note: If WebRTC is blocked on your cloud network, switch to Camera Snapshot mode in sidebar)*.")
        try:
            ctx = webrtc_streamer(
                key="face-detection-live",
                video_processor_factory=FaceDetectionProcessor,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False}
            )

            if ctx.video_processor:
                ctx.video_processor.scale_factor = scale_factor
                ctx.video_processor.min_neighbors = min_neighbors
                ctx.video_processor.color = COLOR_MAP_BGR[color_choice]
                ctx.video_processor.thickness = thickness
        except Exception as e:
            st.error(f"Live WebRTC stream error: {e}. Please use 'Camera Snapshot / File Upload' mode.")

    else: # Snapshot / Upload
        source = st.radio("Choose Source", ["Camera Snapshot", "Upload Image File"])
        img_array = None

        if source == "Camera Snapshot":
            img_file = st.camera_input("Take a photo")
            if img_file:
                img_array = load_image(img_file)
        else:
            img_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
            if img_file:
                img_array = load_image(img_file)

        if img_array is not None:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.image(img_array, use_container_width=True)
            with col2:
                st.subheader("Detected Faces")
                processed_img = img_array.copy()
                gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
                face_cascade = get_face_cascade()
                
                if hasattr(face_cascade, 'empty') and not face_cascade.empty():
                    faces = face_cascade.detectMultiScale(
                        gray,
                        scaleFactor=scale_factor,
                        minNeighbors=min_neighbors,
                        minSize=(30, 30)
                    )
                    for (x, y, w, h) in faces:
                        cv2.rectangle(processed_img, (x, y), (x + w, y + h), COLOR_MAP_RGB[color_choice], thickness)
                    st.image(processed_img, use_container_width=True)
                    st.success(f"Detected {len(faces)} face(s)!")
                else:
                    st.warning("Haar cascade face classifier model is loading...")
