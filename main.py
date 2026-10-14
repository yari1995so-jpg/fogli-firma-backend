from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import json
import re
import os

app = FastAPI(title="Fogli Firma AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

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

@app.post("/scan-sheet")
async def scan_sheet(request: Request):
    try:
        if not GEMINI_API_KEY:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Chiave GEMINI_API_KEY non impostata su Render!"}
            )

        data = await request.json()
        image_base64 = data.get("file_base64", "")
        db_signatures = data.get("signatures", [])

        if not image_base64 or not db_signatures:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Dati o immagine mancanti."}
            )

        dipendenti_list = [sig.get("dipendente", "") for sig in db_signatures if sig.get("dipendente")]

        # Pulizia stringa Base64
        mime_type = "image/jpeg"
        clean_b64 = image_base64
        if "," in image_base64:
            header, clean_b64 = image_base64.split(",", 1)
            if "png" in header:
                mime_type = "image/png"

        prompt_text = (
            f"Sei un ispettore di sicurezza. Analizza il foglio di presenza in foto.\n"
            f"Dipendenti in anagrafica: {', '.join(dipendenti_list)}.\n\n"
            f"REGOLE:\n"
            f"1. Cerca soltanto firme a mano o sigle a penna.\n"
            f"2. Se un dipendente ha firmato a penna, inseriscilo nella lista 'presenti'.\n"
            f"3. Se non c'è firma a penna, la lista deve essere vuota.\n\n"
            f"Rispondi SOLO ed ESCLUSIVAMENTE con questo JSON senza altro testo:\n"
            f'{{"presenti": ["Nome Cognome"]}}'
        )

        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"

        payload = {
            "contents": [{
                "parts": [
                    {"text": prompt_text},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": clean_b64
                        }
                    }
                ]
            }],
            "generationConfig": {
                "temperature": 0.0,
                "response_mime_type": "application/json"
            }
        }

        res = requests.post(url, json=payload, timeout=15)

        if res.status_code != 200:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Errore Gemini {res.status_code}: {res.text[:100]}"}
            )

        res_data = res.json()
        raw_text = res_data['candidates'][0]['content']['parts'][0]['text']
        parsed = json.loads(raw_text)

        presenti = parsed.get("presenti", [])
        matches = []
        for nome in presenti:
            for dip_real in dipendenti_list:
                if dip_real.lower() in str(nome).lower() or str(nome).lower() in dip_real.lower():
                    matches.append({"dipendente": dip_real, "score": 98.0})
                    break

        return JSONResponse(status_code=200, content={"status": "success", "matches": matches})

    except Exception as e:
        return JSONResponse(status_code=200, content={"status": "error", "message": f"Errore server: {str(e)}"})
