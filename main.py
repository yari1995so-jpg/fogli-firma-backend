from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps
import io
import base64

app = FastAPI(title="Fogli Firma AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.options("/{full_path:path}")
async def options_handler(full_path: str):
    return JSONResponse(
        content="OK",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )

@app.get("/")
def read_root():
    return {"status": "online", "message": "Fogli Firma AI Backend Operativo"}

def extract_signature_crops(pil_img):
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    gray = pil_img.convert("RGB").convert("L")
    
    w, h = gray.size
    if max(w, h) > 1200:
        scale = 1200.0 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)))
    
    img_np = np.array(gray)
    
    # Binarizzazione basata sul contrasto locale dell'immagine
    threshold = np.percentile(img_np, 25)
    binary = (img_np < threshold).astype(np.uint8)

    curr_h, curr_w = binary.shape
    crops = []

    # Scansione a finestre sovrapposte per catturare le firme intere
    win_h = max(40, int(curr_h / 12))
    win_w = max(100, int(curr_w / 3))
    step_y = int(win_h / 2)
    step_x = int(win_w / 2)

    for y in range(0, curr_h - win_h, step_y):
        for x in range(0, curr_w - win_w, step_x):
            sub_np = binary[y:y+win_h, x:x+win_w]
            pixel_count = np.sum(sub_np)
            
            # Filtra per garantire che ci sia del tratto a penna significativo
            if 100 < pixel_count < (win_h * win_w * 0.45):
                crops.append(sub_np)

    return crops

def get_feature_vector(binary_array):
    img = Image.fromarray((binary_array * 255).astype(np.uint8))
    img_resized = img.resize((64, 64), Image.Resampling.BILINEAR)
    vec = np.array(img_resized) > 80
    return vec.astype(np.float32)

@app.post("/scan-sheet")
async def scan_sheet(request: Request):
    try:
        data = await request.json()
        
        image_base64 = data.get("file_base64")
        db_signatures = data.get("signatures", [])

        if not image_base64 or not db_signatures:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Parametri mancanti nella richiesta."}
            )

        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]

        image_bytes = base64.b64decode(image_base64)
        
        try:
            pil_img = Image.open(io.BytesIO(image_bytes))
        except Exception as img_err:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Formato immagine non valido: {str(img_err)}"}
            )

        found_crops = extract_signature_crops(pil_img)
        matches = []

        for crop in found_crops:
            crop_vec = get_feature_vector(crop)
            best_name = None
            max_score = 0.0

            for sig in db_signatures:
                try:
                    raw_data = sig.get("dataUrl", "")
                    sample_data = raw_data.split(",")[1] if "," in raw_data else raw_data

                    sample_bytes = base64.b64decode(sample_data)
                    sample_pil = Image.open(io.BytesIO(sample_bytes))
                    sample_gray = sample_pil.convert("RGB").convert("L")
                    
                    sample_np = np.array(sample_gray) < 180
                    sample_vec = get_feature_vector(sample_np)

                    intersection = np.sum(crop_vec * sample_vec)
                    total = np.sum(crop_vec) + np.sum(sample_vec)

                    if total > 0:
                        score = (2.0 * intersection) / total
                        if score > max_score:
                            max_score = score
                            best_name = sig.get("dipendente", "Sconosciuto")
                except Exception:
                    continue

            # Soglia di affinità bilanciata al 52%
            if max_score >= 0.52 and best_name:
                matches.append({
                    "dipendente": best_name,
                    "score": round(float(max_score * 100), 1)
                })

        unique = {}
        for m in matches:
            nome = m["dipendente"]
            if nome not in unique or m["score"] > unique[nome]["score"]:
                unique[nome] = m

        return JSONResponse(
            status_code=200,
            content={"status": "success", "matches": list(unique.values())}
        )

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": f"Errore interno: {str(e)}"}
        )
