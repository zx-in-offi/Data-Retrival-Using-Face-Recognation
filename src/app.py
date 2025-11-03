import os
import streamlit as st
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
import gdown

# --------------------- PATHS & CONFIG ---------------------
EMB_PATH = "embeddings/roll_embeddings.npy"
IDX_PATH = "embeddings/roll_index.csv"

os.makedirs("embeddings", exist_ok=True)

MODEL_NAME = "ArcFace"
DET_BACKEND = "retinaface"
SIM_THRESHOLD = 0.45

# --------------------- LOAD SECRETS ---------------------
try:
    EMB_FILE_ID = st.secrets["drive"]["embeddings_id"]
    IDX_FILE_ID = st.secrets["drive"]["index_id"]
except Exception as e:
    st.error("❌ Missing Google Drive file IDs in Streamlit secrets. Please configure them under 'Manage app → Secrets'.")
    st.stop()

# --------------------- DOWNLOAD IF NOT EXISTS ---------------------
def download_from_gdrive(file_id, dest_path):
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        gdown.download(url, dest_path, quiet=False)
    except Exception as e:
        st.error(f"❌ Failed to download {dest_path}: {e}")
        st.stop()

if not os.path.exists(EMB_PATH):
    st.info("📥 Downloading embeddings from Google Drive...")
    download_from_gdrive(EMB_FILE_ID, EMB_PATH)

if not os.path.exists(IDX_PATH):
    st.info("📥 Downloading index CSV from Google Drive...")
    download_from_gdrive(IDX_FILE_ID, IDX_PATH)

# --------------------- UTIL ---------------------
def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

# --------------------- LOAD EMBEDDINGS ---------------------
@st.cache_resource
def load_index():
    X = np.load(EMB_PATH).astype(np.float32)
    idx_df = pd.read_csv(IDX_PATH, dtype=str).fillna("")
