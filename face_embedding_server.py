from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
from insightface.app import FaceAnalysis

app = FastAPI()

model = FaceAnalysis(name='buffalo_l', allowed_modules=['detection', 'recognition'])
model.prepare(ctx_id=-1)  # Use CPU

@app.post("/embed")
async def get_embedding(file: UploadFile = File(...)):
    contents = await file.read()
    img_array = np.asarray(bytearray(contents), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    print("Image shape:", img.shape)

    faces = model.get(img)
    if not faces:
        return {"error": "No face detected"}

    face = faces[0]
    embedding = face.embedding
    if embedding is None:
        embedding = model.models['recognition'].get(face)

    if embedding is None:
        return {"error": "Embedding could not be computed"}

    return {"embedding": embedding.tolist()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)