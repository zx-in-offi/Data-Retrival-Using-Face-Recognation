import streamlit as st
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace

# Paths
EMB_PATH = "embeddings/roll_embeddings.npy"
IDX_PATH = "embeddings/roll_index.csv"

MODEL_NAME = "ArcFace"
DET_BACKEND = "retinaface"
SIM_THRESHOLD = 0.45

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

# Load embeddings + index once
@st.cache_resource
def load_index():
    X = np.load(EMB_PATH).astype(np.float32)
    idx_df = pd.read_csv(IDX_PATH, dtype=str).fillna("")
    return X, idx_df

X, idx_df = load_index()

# --------------------- STREAMLIT UI ---------------------
st.title("🎭 Student Data Retrieval using Face Recognition")
st.write("Upload a photo or use webcam to recognize classmates and fetch their details.")

# File uploader
uploaded_file = st.file_uploader("📤 Upload a photo", type=["jpg","jpeg","png"])

# Webcam input
camera_input = st.camera_input("📸 Or capture a photo with webcam")

img_bytes = None
if uploaded_file is not None:
    img_bytes = uploaded_file.read()
elif camera_input is not None:
    img_bytes = camera_input.getvalue()

if img_bytes:
    # Convert bytes → OpenCV image
    img_array = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    st.image(frame, channels="BGR", caption="Uploaded Image")

    # Detect faces
    try:
        faces = DeepFace.extract_faces(img_path=frame, detector_backend=DET_BACKEND, align=True)
    except Exception as e:
        st.error(f"Error detecting faces: {e}")
        faces = []

    if not faces:
        st.warning("😢 No face detected.")
    else:
        for i, f in enumerate(faces):
            face_rgb = (f["face"] * 255).astype("uint8")
            reps = DeepFace.represent(img_path=face_rgb, model_name=MODEL_NAME, detector_backend="skip")
            emb = l2_normalize(np.array(reps[0]["embedding"], dtype=np.float32))

            sims = np.dot(X, emb)
            j = int(np.argmax(sims)); s = float(sims[j])

            if s >= SIM_THRESHOLD:
                r = idx_df.iloc[j]
                st.success(f"✅ {r['full_name']} ({r['roll_no']}), Age {r['age']} | sim={s:.2f}")
            else:
                st.error(f"❌ Unknown Face {i+1} | sim={s:.2f}")