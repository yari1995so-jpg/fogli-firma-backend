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

        dipendenti_list = [sig.get("dipendente", "") for sig in db_signatures if sig.get("dipendente")]
        
        if not image_base64.startswith("data:image"):
            image_url = f"data:image/jpeg;base64,{image_base64}"
        else:
            image_url = image_base64

        prompt_text = (
            f"Sei un ispettore di sicurezza estremamente rigoroso nell'analisi di fogli di presenza aziendali.\n"
            f"Ecco la lista completa dell'anagrafica dipendenti da verificare: {', '.join(dipendenti_list)}.\n\n"
            f"REGOLE DI ANALISI SEVERISSIME:\n"
            f"1. Guarda attentamente l'immagine del foglio allegato.\n"
            f"2. Cerca SOLO le firme autografe, sigle o scritte a penna apposte sul foglio.\n"
            f"3. NON inserire un dipendente nella lista dei presenti solo perché il suo nome appare nell'anagrafica o nel testo stampato.\n"
            f"4. Se una casella è vuota o NON contiene una firma scritta a mano ben visibile, considera quel dipendente ASSENTE.\n"
            f"5. Se non vedi NESSUNA firma a penna, la lista 'presenti' deve essere vuota [].\n\n"
            f"Rispondi ESCLUSIVAMENTE con un oggetto JSON valido in questo formato preciso e senza alcun altro commento:\n"
            f'{{"presenti": ["Nome Cognome Esatto"]}}'
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Modelli Vision in ordine di priorità
        vision_models = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "meta-llama/llama-3.2-11b-vision-instruct"
        ]

        response = None
        last_err = ""

        for model_name in vision_models:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": image_url}}
                        ]
                    }
                ],
                "temperature": 0.0,  # Zero per garantire la massima determinazione senza invenzioni
                "max_tokens": 250,
                "response_format": {"type": "json_object"}
            }

            res = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=25
            )

            if res.status_code == 200:
                response = res
                break
            else:
                last_err = res.text

        if not response:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Errore API Vision: {last_err}"}
            )

        res_json = response.json()
        raw_reply = res_json['choices'][0]['message']['content']
        parsed_reply = json.loads(raw_reply)

        presenti = parsed_reply.get("presenti", [])
        
        matches = []
        for nome in presenti:
            # Filtra per garantire che il nome restituito faccia parte dell'anagrafica reale
            for dip_real in dipendenti_list:
                if dip_real.lower() in nome.lower() or nome.lower() in dip_real.lower():
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
            content={"status": "error", "message": f"Errore analisi IA: {str(e)}"}
        )
