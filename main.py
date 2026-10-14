from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import json
import base64
import re
import os
import io
from PIL import Image

app = FastAPI(title="Fogli Firma AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

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
    return {"status": "online", "message": "Fogli Firma AI Backend Operativo con Gemini"}

def process_base64_image(base64_str):
    mime_type = "image/jpeg"
    clean_b64 = base64_str

    if "," in base64_str:
        header, clean_b64 = base64_str.split(",", 1)
        if "png" in header:
            mime_type = "image/png"
            
    try:
        img_bytes = base64.b64decode(clean_b64)
        img = Image.open(io.BytesIO(img_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        final_b64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return final_b64, "image/jpeg"
    except Exception:
        return clean_b64, mime_type

@app.post("/scan-sheet")
async def scan_sheet(request: Request):
    try:
        api_key = GEMINI_API_KEY.strip()
        if not api_key:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Chiave GEMINI_API_KEY non trovata nelle variabili d'ambiente di Render!"}
            )

        data = await request.json()
        image_base64 = data.get("file_base64", "")
        db_signatures = data.get("signatures", [])

        if not image_base64 or not db_signatures:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Parametri mancanti nella richiesta."}
            )

        dipendenti_list = [sig.get("dipendente", "") for sig in db_signatures if sig.get("dipendente")]
        b64_data, mime_type = process_base64_image(image_base64)

        prompt_text = (
            f"Sei un ispettore aziendale rigoroso. Analizza la foto del foglio di presenza/firme allegato.\n"
            f"Anagrafica dipendenti registrati: {', '.join(dipendenti_list)}.\n\n"
            f"REGOLE RIGIDE:\n"
            f"1. Cerca soltanto firme scritte a mano, sigle o tratti a penna nella pagina.\n"
            f"2. Se un dipendente ha una firma scritta a mano visibile vicino al suo nome o nella casella corrispondente, inseriscilo tra i 'presenti'.\n"
            f"3. Se il nome è solo stampato a macchina senza firma a penna, considera il dipendente ASSENTE.\n"
            f"4. Se non vedi alcuna firma a penna, la lista 'presenti' deve essere vuota.\n\n"
            f"Rispondi ESCLUSIVAMENTE con un oggetto JSON valido con questo formato preciso e nessun testo aggiuntivo:\n"
            f'{{"presenti": ["Nome Cognome"]}}'
        )

        # Prova sia Gemini 1.5 Flash che Gemini 2.0 Flash
        models_to_try = ["gemini-1.5-flash", "gemini-2.0-flash"]
        
        parsed_reply = None
        last_error = ""

        for model_name in models_to_try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt_text},
                            {
                                "inline_data": {
                                    "mime_type": mime_type,
                                    "data": b64_data
                                }
                            }
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.0,
                    "response_mime_type": "application/json"
                }
            }

            try:
                res = requests.post(url, json=payload, timeout=25)
                if res.status_code == 200:
                    res_json = res.json()
                    candidates = res_json.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            raw_text = parts[0].get("text", "").strip()
                            # Pulizia eventuale blocco markdown
                            if "```" in raw_text:
                                raw_text = re.sub(r'```(?:json)?\s*', '', raw_text)
                                raw_text = raw_text.replace('```', '').strip()
                            parsed_reply = json.loads(raw_text)
                            break
                else:
                    last_error = f"Model {model_name} HTTP {res.status_code}: {res.text[:150]}"
            except Exception as err:
                last_error = f"Error with {model_name}: {str(err)}"

        if not parsed_reply:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Errore risposta Gemini: {last_error}"}
            )

        presenti = parsed_reply.get("presenti", [])
        matches = []
        for nome in presenti:
            for dip_real in dipendenti_list:
                if dip_real.lower() in str(nome).lower() or str(nome).lower() in dip_real.lower():
                    matches.append({
                        "dipendente": dip_real,
                        "score": 98.0
                    })
                    break

        return JSONResponse(
            status_code=200,
            content={"status": "success", "matches": matches}
        )

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": f"Errore generale server: {str(e)}"}
        )
