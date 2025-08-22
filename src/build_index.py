import numpy as np
import pandas as pd
from pathlib import Path
from deepface import DeepFace

PROC_DIR = Path("data/processed")
EMB_DIR  = Path("embeddings"); EMB_DIR.mkdir(parents=True, exist_ok=True)
META_CSV = Path("metadata/students.csv")

MODEL_NAME = "ArcFace"  # robust & accurate

def l2_normalize(v):
    v = np.asarray(v, dtype=np.float32)
    n = np.linalg.norm(v) + 1e-12
    return v / n

def main():
    # Load student metadata
    meta = pd.read_csv(META_CSV, dtype=str).fillna("")

    # Standardize column names
    meta.rename(columns={
        "Id": "roll_no",
        "Name": "full_name",
        "Age": "age"
    }, inplace=True)

    # Drop unused 'Path' column if it exists
    if "Path" in meta.columns:
        meta.drop(columns=["Path"], inplace=True)
    # meta["roll_no"] = meta["roll_no"].str.strip()

    rows = []
    for roll_dir in sorted(PROC_DIR.iterdir()):
        if not roll_dir.is_dir(): 
            continue
        roll_no = roll_dir.name
        face_imgs = sorted(list(roll_dir.glob("*.jpg")))
        if not face_imgs:
            continue

        embs = []
        for img in face_imgs:
            try:
                # Already aligned, so skip detection for speed
                reps = DeepFace.represent(
                    img_path=str(img),
                    model_name=MODEL_NAME,
                    detector_backend="skip",
                    enforce_detection=False
                )
                emb = np.array(reps[0]["embedding"], dtype=np.float32)
                embs.append(l2_normalize(emb))
            except Exception as e:
                print(f"[ERR] {img}: {e}")

        if not embs:
            continue

        avg_emb = l2_normalize(np.mean(embs, axis=0))
        rows.append((roll_no, avg_emb))

    if not rows:
        print("No embeddings found. Did preprocess run successfully?")
        return

    # Save embeddings matrix and index CSV joined with metadata
    roll_nos = [r for r, _ in rows]
    X = np.stack([e for _, e in rows])  # shape: (N, D)
    np.save(EMB_DIR / "roll_embeddings.npy", X)

    # Join roll_no with full_name, age from metadata
    idx_df = pd.DataFrame({"roll_no": roll_nos})
    idx_df = idx_df.merge(meta, on="roll_no", how="left")
    idx_df.to_csv(EMB_DIR / "roll_index.csv", index=False)
    print(f"[OK] Saved {X.shape[0]} roll embeddings.")

if __name__ == "__main__":
    main()
