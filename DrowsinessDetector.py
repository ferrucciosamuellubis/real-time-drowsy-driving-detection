import streamlit as st
import cv2
import numpy as np
import mediapipe as mp

# Definisi indeks mata dari MediaPipe Face Mesh
LEFT_EYE = [362,385,387,263,373,380]
RIGHT_EYE = [33,160,158,133,153,144]

# Fungsi hitung EAR
def eye_aspect_ratio(lm, idxs, w, h):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idxs]
    A = np.linalg.norm(np.array(pts[1]) - pts[5])
    B = np.linalg.norm(np.array(pts[2]) - pts[4])
    C = np.linalg.norm(np.array(pts[0]) - pts[3])
    return (A + B) / (2.0 * C)

st.title("Deteksi Kantuk dari Foto 📸")
uploaded = st.file_uploader("Unggah foto wajah", type=["jpg","png","jpeg"])

if uploaded:
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img_rgb, caption="Foto Asli", use_column_width=True)

    mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
    res = mp_face.process(img_rgb)

    if res.multi_face_landmarks:
        lm = res.multi_face_landmarks[0].landmark
        h, w, _ = img.shape
        ear_l = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        ear_r = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        ear = (ear_l + ear_r) / 2.0

        st.write(f"EAR rata-rata: **{ear:.2f}**")

        EAR_THRESH = 0.18
        if ear < EAR_THRESH:
            st.warning("📌 Mata tertutup terlalu lama! Risiko kantuk.")

        else:
            st.success("Selang mata normal.")
    else:
        st.error("Tidak terdeteksi wajah.")

    mp_face.close()
