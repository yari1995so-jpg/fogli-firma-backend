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
            f"Sei un ispettore di sicurezza rigoroso nell'analisi di fogli di presenza aziendali.\n"
            f"Anagrafica dipendenti: {', '.join(dipendenti_list)}.\n\n"
            f"REGOLE RIGIDE:\n"
            f"1. Analizza il foglio allegato.\n"
            f"2. Individua SOLO le firme autografe, sigle o scritte a mano presente sul foglio.\n"
            f"3. NON inserire un dipendente nella lista dei presenti se il suo nome è solo stampato o assente.\n"
            f"4. Se non vedi firme a mano, la lista dei presenti deve essere vuota.\n\n"
            f"Rispondi SOLTANTO con un JSON in questo formato preciso e nessun altro testo:\n"
            f'{{"presenti": ["Nome Cognome"]}}'
        )

        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        # Modelli Vision disponibili su Groq Cloud
        vision_models = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview",
            "qwen/qwen3.8-27b"
        ]

        parsed_reply = None
        last_error_msg = ""

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
                "temperature": 0.0,
                "max_tokens": 300
            }

            try:
                res = requests.post(
                    "[https://api.groq.com/openai/v1/chat/completions](https://api.groq.com/openai/v1/chat/completions)",
                    headers=headers,
                    json=payload,
                    timeout=25
                )

                if res.status_code == 200:
                    res_json = res.json()
                    raw_reply = res_json['choices'][0]['message']['content'].strip()
                    
                    # Pulizia da blocchi markdown ```json ... ```
                    if "```" in raw_reply:
                        raw_reply = re.sub(r'```(?:json)?\s*', '', raw_reply)
                        raw_reply = raw_reply.replace('```', '').strip()
                    
                    # Estrazione sicura della prima struttura JSON valida
                    json_match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
                    if json_match:
                        raw_reply = json_match.group(0)

                    parsed_reply = json.loads(raw_reply)
                    break
                else:
                    last_error_msg = f"HTTP {res.status_code}: {res.text}"
            except Exception as parse_err:
                last_error_msg = f"Parsing error su {model_name}: {str(parse_err)}"

        if parsed_reply is None:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Impossibile analizzare l'immagine: {last_error_msg}"}
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
            content={"status": "error", "message": f"Errore interno server: {str(e)}"}
        )
