import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
import cv2
import numpy as np
from ultralytics import YOLO
import mediapipe as mp
import threading
import time
import platform

if platform.system() == "Windows":
    import winsound
    def play_alarm():
        winsound.Beep(1000, 1000)
else:
    def play_alarm():
        pass  # atau play_sound via streamlit

class DrowsinessTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.detectyawn = YOLO("runs/detectyawn/train/weights/best.pt")
        self.detecteye = YOLO("runs/detecteye/train/weights/best.pt")
        # init state variables...
        
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        # --- Copy logic dari process_frames di kode asli ---
        # Deteksi wajah, mata, yawn, update counters
        # Tambahkan overlay teks atau bounding box jika perlu:
        cv2.putText(img, f"Blinks: {self.blinks}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return img

# Streamlit UI
st.title("Drowsiness Detector")
ctx = webrtc_streamer(
    key="drowsy",
    video_transformer_factory=DrowsinessTransformer,
    media_stream_constraints={"video": True, "audio": False},
)

if st.button("Play Alarm"):
    play_alarm()
