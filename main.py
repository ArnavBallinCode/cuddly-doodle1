import os
import json
import numpy as np
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from PIL import Image
from io import BytesIO

# Try to import InsightFace
INSIGHTFACE_AVAILABLE = False
try:
    from insightface.app.face_analysis import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except:
    pass

app = FastAPI()
face_app = None
face_database = {}

DB_PATH = "faces.json"

def setup_models():
    global face_app
    if INSIGHTFACE_AVAILABLE:
        face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        face_app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

def load_database():
    global face_database
    if os.path.exists(DB_PATH):
        with open(DB_PATH, 'r') as f:
            face_database.update(json.load(f))

def save_database():
    with open(DB_PATH, 'w') as f:
        json.dump(face_database, f)

def get_embedding(image: Image.Image):
    global face_app
    if not face_app or not image:
        return None, "No model or image"
    img_array = np.array(image.convert('RGB'))
    faces = face_app.get(img_array)
    if not faces:
        return None, "No face detected"
    face = faces[0]
    return face.embedding, f"Face detected (confidence: {face.det_score:.2f})"

@app.on_event("startup")
def startup_event():
    setup_models()
    load_database()

@app.post("/add_face")
async def add_face(image: UploadFile = File(...), name: str = Form(...)):
    if not name.strip():
        return JSONResponse({"result": "Please enter a name"}, status_code=400)
    img = Image.open(BytesIO(await image.read()))
    embedding, msg = get_embedding(img)
    if embedding is None:
        return JSONResponse({"result": f"Failed: {msg}"}, status_code=400)
    face_database[name.strip()] = embedding.tolist()
    save_database()
    return {"result": f"✓ Added {name} ({msg}). Database now has {len(face_database)} faces."}

@app.post("/match_face")
async def match_face(image: UploadFile = File(...)):
    if not face_database:
        return {"result": "Database is empty. Please add faces first."}
    img = Image.open(BytesIO(await image.read()))
    embedding, msg = get_embedding(img)
    if embedding is None:
        return {"result": f"Failed: {msg}"}
    best_match = None
    best_score = -1
    for name, stored_emb in face_database.items():
        stored_emb = np.array(stored_emb)
        score = np.dot(embedding, stored_emb) / (np.linalg.norm(embedding) * np.linalg.norm(stored_emb))
        if score > best_score:
            best_score = score
            best_match = name
    if best_score > 0.6:
        return {"result": f"✓ Match Found: {best_match} (confidence: {best_score*100:.1f}%)"}
    else:
        return {"result": f"❌ No match found. Best score: {best_score*100:.1f}% (threshold: 60%)"}

@app.get("/status")
def get_status():
    status = "✓ InsightFace loaded" if face_app else "Demo mode"
    db_count = len(face_database)
    return {"result": f"System: {status} | Database: {db_count} faces"}

@app.post("/clear_database")
def clear_database():
    global face_database
    face_database = {}
    save_database()
    return {"result": "Database cleared successfully"}
