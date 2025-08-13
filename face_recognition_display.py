import cv2
import face_recognition
import os
import pandas as pd

# Load details from CSV
details_df = pd.read_csv(r"C:\\Users\\venka\\OneDrive\\Desktop\\Video_Processing\\people_details.csv")


# Encode known faces
known_encodings = []
known_names = []

for index, row in details_df.iterrows():
    image = face_recognition.load_image_file(row["ImagePath"])
    encoding = face_recognition.face_encodings(image)
    if encoding:
        known_encodings.append(encoding[0])
        known_names.append(row["Name"])

# Start video capture
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for speed
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Detect faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            match_index = matches.index(True)
            name = known_names[match_index]

            # Get person details
            person_info = details_df[details_df["Name"] == name].iloc[0]
            info_text = f"Name: {person_info['Name']}, Age: {person_info['Age']}, ID: {person_info['ID']}"
        else:
            info_text = "Unknown Person"

        # Scale back face location
        top, right, bottom, left = [v * 4 for v in face_location]

        # Draw bounding box and label
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, info_text, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow('Video - Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
