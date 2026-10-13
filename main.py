from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageOps
from skimage.measure import label, regionprops
import io
import base64
import json
import traceback

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

def process_and_extract_crops(pil_img):
    # Applica orientamento EXIF per foto scattate da iPhone/Android
    try:
        pil_img = ImageOps.exif_transpose(pil_img)
    except Exception:
        pass

    # Converti sempre prima in RGB e poi in scala di grigi per evitare crash su PNG con Alpha o HEIC
    gray = pil_img.convert("RGB").convert("L")
    
    w, h = gray.size
    if max(w, h) > 1000:
        scale = 1000.0 / max(w, h)
        gray = gray.resize((int(w * scale), int(h * scale)))
    
    img_np = np.array(gray)
    
    threshold = np.mean(img_np) * 0.85
    binary = img_np < threshold

    labeled = label(binary)
    regions = regionprops(labeled)

    crops = []
    curr_h, curr_w = img_np.shape

    for reg in regions:
        min_y, min_x, max_y, max_x = reg.bbox
        w_box = max_x - min_x
        h_box = max_y - min_y

        if w_box > 30 and h_box > 15 and w_box < curr_w * 0.8 and h_box < curr_h * 0.5:
            crop_np = binary[min_y:max_y, min_x:max_x]
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
            return {"status": "error", "message": f"Formato immagine non supportato: {str(img_err)}"}

        try:
            db_signatures = json.loads(db_signatures_json)
        except Exception:
            return {"status": "error", "message": "Anagrafica firme non valida."}

        found_crops = process_and_extract_crops(pil_img)
        matches = []

        for crop in found_crops:
            crop_vec = get_feature_vector(crop)
            best_name = None
            max_score = 0.0

            for sig in db_signatures:
                try:
                    raw_data = sig.get("dataUrl", "")
                    if "," in raw_data:
                        sample_data = raw_data.split(",")[1]
                    else:
                        sample_data = raw_data

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

            # Soglia di corrispondenza al 55%
            if max_score >= 0.55 and best_name:
                matches.append({
                    "dipendente": best_name,
                    "score": round(float(max_score * 100), 1)
                })

        unique = {}
        for m in matches:
            nome = m["dipendente"]
            if nome not in unique or m["score"] > unique[nome]["score"]:
                unique[nome] = m

        return {"status": "success", "matches": list(unique.values())}

    except Exception as e:
        print("CRASH SERVER:", traceback.format_exc())
        return {"status": "error", "message": f"Errore interno del server: {str(e)}"}
