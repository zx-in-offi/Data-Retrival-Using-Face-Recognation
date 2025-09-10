import sys, numpy as np, pandas as pd, cv2
from deepface import DeepFace

# Paths
EMB_PATH = "embeddings/roll_embeddings.npy"
IDX_PATH = "embeddings/roll_index.csv"

# Model settings
MODEL_NAME = "ArcFace"
DET_BACKEND = "retinaface"
SIM_THRESHOLD = 0.45

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-12)

def main(img_path):
    # Load embeddings and metadata
    X = np.load(EMB_PATH).astype(np.float32)
    idx_df = pd.read_csv(IDX_PATH, dtype=str).fillna("")

    # Load original image for drawing
    orig_img = cv2.imread(img_path)
    if orig_img is None:
        print("Error: cannot read image", img_path)
        return

    # Detect and align faces
    faces = DeepFace.extract_faces(img_path=img_path, detector_backend=DET_BACKEND, align=True)
    if not faces:
        print("No face found.")
        return

    for i, f in enumerate(faces):
        # Extract embedding
        face_rgb = (f["face"] * 255).astype("uint8")
        reps = DeepFace.represent(img_path=face_rgb, model_name=MODEL_NAME, detector_backend="skip")
        emb = l2_normalize(np.array(reps[0]["embedding"], dtype=np.float32))

        # Compare with index
        sims = np.dot(X, emb)
        j = int(np.argmax(sims)); s = float(sims[j])

        # Facial area (for drawing boxes)
        fa = f["facial_area"]
        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

        if s >= SIM_THRESHOLD:
            r = idx_df.iloc[j]
            label = f"{r['full_name']} ({r['roll_no']}), Age {r['age']}"
            print(f"[Face {i}] {label} | sim={s:.2f}")
        else:
            label = "Unknown"
            print(f"[Face {i}] Unknown | sim={s:.2f}")

        # Draw on image
        cv2.rectangle(orig_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(orig_img, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Show result
    cv2.imshow("Recognition Result", orig_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/recognize_image.py path/to/photo.jpg")
    else:
        main(sys.argv[1])
