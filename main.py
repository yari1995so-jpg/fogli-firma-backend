from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import numpy as np
from PIL import Image, ImageOps
import io
import base64
import json

app = FastAPI(title="Fogli Firma AI Backend")

# Abilita CORS completo per qualsiasi origine (Netlify, localhost, PWA)
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
    if max(w, h) > 1000:
        scale = 1000.0 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)))
    
    img_np = np.array(gray)
    
    threshold = np.mean(img_np) * 0.85
    binary = (img_np < threshold).astype(np.uint8)

    curr_h, curr_w = binary.shape
    crops = []
    
    row_sums = np.sum(binary, axis=1)
    threshold_row = np.mean(row_sums) * 0.3
    active_rows = np.where(row_sums > threshold_row)[0]

    if len(active_rows) > 0:
        y_min, y_max = active_rows[0], active_rows[-1]
        step = max(30, int((y_max - y_min) / 6))
        
        for y in range(y_min, y_max, step):
            band = binary[y:min(y+step, curr_h), :]
            col_sums = np.sum(band, axis=0)
            active_cols = np.where(col_sums > 2)[0]
            
            if len(active_cols) > 0:
                x_min, x_max = active_cols[0], active_cols[-1]
                w_box = x_max - x_min
                h_box = band.shape[0]
                
                if w_box > 25 and h_box > 15:
                    crop_np = binary[y:y+h_box, x_min:x_max]
                    crops.append(crop_np)

    return crops

def get_feature_vector(binary_array):
    img = Image.fromarray((binary_array * 255).astype(np.uint8))
    img_resized = img.resize((64, 64), Image.Resampling.BILINEAR)
    vec = np.array(img_resized) > 100
    return vec.astype(np.float32)

@app.post("/scan-sheet")
async def scan_sheet(
    file: UploadFile = File(...),
    db_signatures_json: str = Form(...)
):
    try:
        contents = await file.read()
        
        try:
            pil_img = Image.open(io.BytesIO(contents))
        except Exception as img_err:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Formato immagine non valido: {str(img_err)}"}
            )

        try:
            db_signatures = json.loads(db_signatures_json)
        except Exception:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Anagrafica firme non valida."}
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
                    
                    sample_np = np.array(sample_gray) < 160
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

            if max_score >= 0.50 and best_name:
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
