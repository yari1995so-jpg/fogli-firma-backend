from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json

app = FastAPI(title="Fogli Firma AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"status": "online", "message": "Fogli Firma AI Backend Operativo"}

def extract_signature_crops(image_np):
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Ridimensiona l'immagine se supera i 1200px per velocizzare il processo e risparmiare RAM
    h, w = gray.shape
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    crops = []
    curr_h, curr_w = gray.shape

    for cnt in contours:
        x, y, w_box, h_box = cv2.boundingRect(cnt)
        # Filtra i contorni troppo piccoli (rumore) o troppo grandi (intere tabelle)
        if w_box > 40 and h_box > 20 and w_box < curr_w * 0.75 and h_box < curr_h * 0.4:
            crop = gray[y:y+h_box, x:x+w_box]
            crops.append(crop)

    return crops

def get_grid_vector(gray_crop):
    resized = cv2.resize(gray_crop, (64, 64), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)
    return (binary > 0).astype(np.float32)

@app.post("/scan-sheet")
async def scan_sheet(
    file: UploadFile = File(...),
    db_signatures_json: str = Form(...)
):
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        db_signatures = json.loads(db_signatures_json)
        found_crops = extract_signature_crops(image)
        
        matches = []

        for crop in found_crops:
            crop_vec = get_grid_vector(crop)
            best_name = None
            max_score = 0.0

            for sig in db_signatures:
                try:
                    sample_data = sig["dataUrl"].split(",")[1]
                    sample_bytes = base64.b64decode(sample_data)
                    sample_np = np.frombuffer(sample_bytes, np.uint8)
                    sample_img = cv2.imdecode(sample_np, cv2.IMREAD_GRAYSCALE)
                    
                    sample_vec = get_grid_vector(sample_img)

                    intersection = np.sum(crop_vec * sample_vec)
                    total = np.sum(crop_vec) + np.sum(sample_vec)

                    if total > 0:
                        score = (2.0 * intersection) / total
                        if score > max_score:
                            max_score = score
                            best_name = sig["dipendente"]
                except Exception:
                    continue

            # Soglia del 65% per il riconoscimento da foglio intero
            if max_score >= 0.65 and best_name:
                matches.append({
                    "dipendente": best_name,
                    "score": round(float(max_score * 100), 1)
                })

        # Mantiene il punteggio più alto per dipendente
        unique = {}
        for m in matches:
            nome = m["dipendente"]
            if nome not in unique or m["score"] > unique[nome]["score"]:
                unique[nome] = m

        return {"status": "success", "matches": list(unique.values())}

    except Exception as e:
        return {"status": "error", "message": str(e)}
