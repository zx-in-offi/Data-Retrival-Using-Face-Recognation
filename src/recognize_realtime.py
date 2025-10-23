import cv2, numpy as np, pandas as pd
from deepface import DeepFace

EMB_PATH = "embeddings/roll_embeddings.npy"
IDX_PATH = "embeddings/roll_index.csv"
MODEL_NAME = "ArcFace"
DET_BACKEND = "retinaface"

# Cosine similarity threshold (tune for your data)
# With L2-normalized ArcFace, genuine pairs often > 0.45–0.7 in small datasets.
SIM_THRESHOLD = 0.45

def cosine_sim(a, B):
    # a: (D,), B: (N,D)
    return np.dot(B, a)  # a is 1D, B is 2D

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def main():
    X = np.load(EMB_PATH).astype(np.float32) #(N,D)
    idx_df = pd.read_csv(IDX_PATH, dtype=str).fillna("")
    roll_nos = idx_df["roll_no"].tolist()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open camera.")
        return
    
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        #Detect faces (aligned crops returned as RGB float [0,1])
        try:
            faces = DeepFace.extract_faces(
                img_path = frame,
                detector_backend = DET_BACKEND,
                align = True,
                enforce_detection = False
            )
        except Exception:
            faces = []

        #Draw detections
        for f in faces:
            # Box from (x,y,w,h) if available, else skip drawing box
            region = f.get("facial_area", None)
            face_rgb = (f["face"] * 255).astype(np.uint8) # convert back to uint8 BGR
            #compute ebeddings
            reps = DeepFace.represent(
                img_path= face_rgb, # pass the crop directly
                model_name= MODEL_NAME,
                detector_backend = "skip", # already detected
                enforce_detection= False
            )
            emb = l2_normalize(np.array(reps[0]["embedding"], dtype=np.float32))
            sims = cosine_sim(emb, X) # (N,)
            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if region:
                x,y,w,h = region["x"], region["y"], region["w"], region["h"]
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

            if best_sim >= SIM_THRESHOLD:
                row = idx_df.iloc[best_idx]
                label = f'{row["full_name"]} | {row["roll_no"]} | Age {row["age"]} | sim {best_sim:.2f}'
            else:
                label = f"Unknown | sim {best_sim:.2f}"

            if region:
                cv2.putText(frame, label, (x, y-10 if y-10>10 else y+h+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            else:
                cv2.putText(frame, label, (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                
        cv2.imshow("Face Recognition (ArcFace)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()