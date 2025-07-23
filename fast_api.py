import os
import cv2
import numpy as np
import face_recognition
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directories
known_dir = 'known_faces'
captured_dir = 'captured_faces'
os.makedirs(known_dir, exist_ok=True)
os.makedirs(captured_dir, exist_ok=True)

# Serve static images
app.mount("/known_faces", StaticFiles(directory=known_dir), name="known_faces")
app.mount("/captured_faces", StaticFiles(directory=captured_dir), name="captured_faces")

# Known faces memory
known_face_encodings, known_face_names = [], []
saved_names = set()

def reload_known_faces():
    global known_face_encodings, known_face_names
    known_face_encodings, known_face_names = [], []
    for filename in os.listdir(known_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img = face_recognition.load_image_file(os.path.join(known_dir, filename))
            encs = face_recognition.face_encodings(img)
            if encs:
                known_face_encodings.append(encs[0])
                known_face_names.append(os.path.splitext(filename)[0])

reload_known_faces()

# Camera
camera_index = 0
video_capture = cv2.VideoCapture(camera_index)

@app.get("/")
def root():
    return {"message": "Face recognition API is running"}

@app.post("/add_known_face")
async def add_known_face(name: str = Form(...), file: UploadFile = Form(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb_img)
    if not encs:
        return {"message": "No face detected!"}
    new_encoding = encs[0]
    matches = face_recognition.compare_faces(known_face_encodings, new_encoding, tolerance=0.5)
    if True in matches:
        return {"message": "Face already exists!"}
    with open(os.path.join(known_dir, f"{name}.jpg"), "wb") as f:
        f.write(contents)
    reload_known_faces()
    return {"message": f"Added: {name}"}

@app.post("/capture_known_face")
async def capture_known_face(data: dict = Body(...)):
    name = data.get("name")
    if not name:
        return {"message": "Name required"}
    ret, frame = video_capture.read()
    if not ret:
        return {"message": "Failed to capture frame"}
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encs = face_recognition.face_encodings(rgb)
    if not encs:
        return {"message": "No face detected!"}
    new_encoding = encs[0]
    matches = face_recognition.compare_faces(known_face_encodings, new_encoding, tolerance=0.5)
    if True in matches:
        return {"message": "Face already exists!"}
    cv2.imwrite(os.path.join(known_dir, f"{name}.jpg"), frame)
    reload_known_faces()
    return {"message": f"Captured & saved: {name}"}

@app.delete("/delete_known_face/{name}")
async def delete_known_face(name: str):
    path = os.path.join(known_dir, f"{name}.jpg")
    if os.path.exists(path):
        os.remove(path)
        reload_known_faces()
        return {"message": f"Deleted: {name}"}
    return {"message": "Not found"}

@app.delete("/delete_captured_face/{filename}")
async def delete_captured_face(filename: str):
    path = os.path.join(captured_dir, filename)
    if os.path.exists(path):
        os.remove(path)
        return {"message": f"Deleted: {filename}"}
    return {"message": "Not found"}

@app.get("/list_known_faces")
async def list_known_faces():
    return sorted(os.listdir(known_dir))

@app.get("/list_captured_faces")
async def list_captured_faces():
    return sorted(os.listdir(captured_dir))

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/set_camera")
async def set_camera(index: int = Query(...)):
    global camera_index, video_capture
    camera_index = index
    if video_capture.isOpened():
        video_capture.release()
    video_capture = cv2.VideoCapture(camera_index)
    return {"message": f"Camera set to {camera_index}"}

def generate_frames():
    saved_names.clear()
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        rgb = frame[:, :, ::-1]
        small = cv2.resize(rgb, (0,0), fx=0.25, fy=0.25)
        locations = face_recognition.face_locations(small)
        encodings = face_recognition.face_encodings(small, locations)
        for (top, right, bottom, left), encoding in zip(locations, encodings):
            top *= 4; right *= 4; bottom *= 4; left *= 4
            matches = face_recognition.compare_faces(known_face_encodings, encoding, tolerance=0.5)
            name = "Unknown"
            color = (0, 0, 255)
            if True in matches:
                idx = matches.index(True)
                name = known_face_names[idx]
                color = (0, 255, 0)
            if name not in saved_names and not any(f.startswith(name+"_") for f in os.listdir(captured_dir)):
                face_img = frame[top:bottom, left:right]
                filename = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(os.path.join(captured_dir, filename), face_img)
                saved_names.add(name)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,255,255), 1)
        ret, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
