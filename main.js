from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import base64
import json

app = FastAPI(title="Fogli Firma AI Engine")

# Abilita CORS per permettere le chiamate dalla PWA Netlify
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_signatures_from_sheet(image_np):
    """
    Rileva automaticamente i contorni della griglia o dei blocchi d'inchiostro 
    presenti nel foglio intero usando OpenCV.
    """
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    # Binarizzazione adattiva per isolare l'inchiostro dallo sfondo del foglio
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10
    )

    # Trova i contorni delle firme/scritte sul foglio
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    detected_crops = []
    height, width = gray.shape

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filtra rumore piccolo (es. macchie) o l'intero foglio
        if w > 40 and h > 20 and w < width * 0.8 and h < height * 0.5:
            crop = gray[y:y+h, x:x+w]
            detected_crops.append(crop)

    return detected_crops

def get_feature_vector(crop_gray):
    """Ridimensiona la firma isolata e genera un vettore normalizzato 64x64."""
    resized = cv2.resize(crop_gray, (64, 64), interpolation=cv2.INTER_AREA)
    _, binary = cv2.threshold(resized, 150, 255, cv2.THRESH_BINARY_INV)
    return (binary > 0).astype(np.float32)

@app.post("/scan-sheet")
async def scan_sheet(
    file: UploadFile = File(...),
    db_signatures_json: str = Form(...)
):
    """
    Riceve la foto del foglio intero e l'anagrafica firme in formato JSON,
    scansiona la pagina e restituisce le corrispondenze trovate.
    """
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Decodifica l'anagrafica firme inviata dalla PWA
    db_signatures = json.loads(db_signatures_json)
    
    # Estrai i ritagli d'inchiostro dal foglio
    found_crops = extract_signatures_from_sheet(image)
    
    matches_found = []

    for crop in found_crops:
        crop_vec = get_feature_vector(crop)
        
        best_match = None
        highest_score = 0.0

        for sig in db_signatures:
            # Decodifica l'immagine Base64 salvata in Firestore per il confronto
            sample_data = sig["dataUrl"].split(",")[1]
            sample_bytes = base64.b64decode(sample_data)
            sample_np = np.frombuffer(sample_bytes, np.uint8)
            sample_img = cv2.imdecode(sample_np, cv2.IMREAD_GRAYSCALE)
            
            sample_vec = get_feature_vector(sample_img)

            # Calcolo similarità (Indice di Dice / Overlap)
            intersection = np.sum(crop_vec * sample_vec)
            total = np.sum(crop_vec) + np.sum(sample_vec)
            
            if total > 0:
                score = (2.0 * intersection) / total
                if score > highest_score:
                    highest_score = score
                    best_match = sig["dipendente"]

        # Soglia di accettazione AI al 68%
        if highest_score >= 0.68 and best_match:
            matches_found.append({
                "dipendente": best_match,
                "score": round(highest_score * 100, 1)
            })

    # Rimuovi duplicati mantenendo il punteggio migliore
    unique_matches = {}
    for m in matches_found:
        nome = m["dipendente"]
        if nome not in unique_matches or m["score"] > unique_matches[nome]["score"]:
            unique_matches[nome] = m

    return {"status": "success", "matches": list(unique_matches.values())}