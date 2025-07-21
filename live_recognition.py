import cv2
import face_recognition
import os
import argparse

# Argument parser to get camera index
parser = argparse.ArgumentParser()
parser.add_argument('--source', type=int, default=0, help='Camera index (default=0)')
args = parser.parse_args()

# Load known faces
known_face_encodings = []
known_face_names = []
known_dir = 'known_people'

for filename in os.listdir(known_dir):
    if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(known_dir, filename)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)
        else:
            print(f"[Warning] No face found in {filename}")

# Open webcam
video_capture = cv2.VideoCapture(args.source)


print("[INFO] Starting camera. Press 'q' to quit.")

while True:
    ret, frame = video_capture.read()
    if not ret:
        print("[ERROR] Failed to grab frame")
        break

    # Convert to RGB and resize for faster processing
    rgb_frame = frame[:, :, ::-1]
    small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.25, fy=0.25)

    # Detect faces
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Scale back up face locations
        top *= 4; right *= 4; bottom *= 4; left *= 4

        # Check if face matches known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
        name = "Unknown"
        color = (0, 0, 255)  # Red for unknown

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
            color = (0, 255, 0)  # Green for known


        # Draw rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.rectangle(frame, (left, bottom - 20), (right, bottom), color, cv2.FILLED)
        cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

        # ✅ Print detected name
        print(f"Detected: {name}")

        # ✅ Save detected name to file
        with open("detected_names.txt", "a") as f:
            f.write(name + "\n")

    cv2.imshow('Face Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()