from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import requests
import json
import os

app = FastAPI(title="Fogli Firma AI Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

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
        if not GROQ_API_KEY:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Chiave GROQ_API_KEY non configurata su Render!"}
            )

        data = await request.json()
        image_base64 = data.get("file_base64", "")
        db_signatures = data.get("signatures", [])

        if not image_base64 or not db_signatures:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": "Parametri mancanti nella richiesta."}
            )

        # Prepara la lista di nomi ed esempi per il prompt vision
        dipendenti_list = [sig.get("dipendente", "") for sig in db_signatures if sig.get("dipendente")]
        
        # Formatta l'immagine base64 per la chiamata all'API Vision
        if not image_base64.startswith("data:image"):
            image_url = f"data:image/jpeg;base64,{image_base64}"
        else:
            image_url = image_base64

        prompt_text = (
            f"Sei un assistente AI specializzato nell'analisi di fogli di presenze e firme.\n"
            f"I dipendenti registrati nell'anagrafica sono: {', '.join(dipendenti_list)}.\n\n"
            f"Analizza l'immagine fornita (foglio di firma) ed individua quali tra questi dipendenti "
            f"hanno una firma apposta all'interno del foglio.\n"
            f"Rispondi ESCLUSIVAMENTE con un oggetto JSON valido con questo formato preciso e nessun testo aggiuntivo:\n"
            f'{{"presenti": ["Nome1", "Nome2"]}}'
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": "llama-3.2-11b-vision-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": image_url}}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 300,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=25
        )

        if response.status_code != 200:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Errore API Vision: {response.text}"}
            )

        res_json = response.json()
        raw_reply = res_json['choices'][0]['message']['content']
        parsed_reply = json.loads(raw_reply)

        presenti = parsed_reply.get("presenti", [])
        
        matches = []
        for nome in presenti:
            matches.append({
                "dipendente": nome,
                "score": 98.0
            })

        return JSONResponse(
            status_code=200,
            content={"status": "success", "matches": matches}
        )

    except Exception as e:
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": f"Errore analisi IA: {str(e)}"}
        )
