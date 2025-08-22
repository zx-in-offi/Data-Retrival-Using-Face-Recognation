import os, re, cv2
from pathlib import Path
from deepface import DeepFace

RAW_DIR = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)

def parse_roll_and_idx(filename):
    stem = Path(filename).stem
    #Matching pattern like : rollno_index eg: 2310080081_2
    if "_" in stem:
        roll, idx = stem.split("_", 1)
        return roll, idx
    
    #handling cases of "-copy"
    if "-copy" in stem:
        roll = stem.split("-copy", "")
        return roll, "copy"
    # Fallback: take whole as roll, idx=1
    return stem, "1"

def save_face_crop(img_bgr, out_path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img_bgr)

def main():
    images = [p for p in RAW_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    if not images:
        print("No images found in the raw directory. Add images then re-run")
        return
    
    for img_path in images:
        try:
            # Detect + align faces (RetinaFace is Robust)
            faces = DeepFace.extract_faces(
                img_path=str(img_path),
                detector_backend = "retinaface",
                enforce_detection=True,
                align = True
            )
            if not faces:
                print(f"[WARN] No faces detected in {img_path}")
                continue

            roll, idx = parse_roll_and_idx(img_path.name)
            #use first face (assuming single person per image)
            face_dict = faces[0]
            face_rgb = (face_dict["face"] * 255).astype("uint8")
            face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)

            out_dir = PROC_DIR / roll
            out_path = out_dir / f"{roll}_{idx}.jpg"
            save_face_crop(face_bgr, out_path)
            print(f"[OK] {img_path.name} -> {out_path}")
        except Exception as e:
            print(f"[ERR] {img_path.name}: {e}")

if __name__ == "__main__":
    main()



