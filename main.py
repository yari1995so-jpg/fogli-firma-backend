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

def compress_base64_image(base64_str, max_size=(1024, 1024)):
    """Ridimensiona e comprime l'immagine per rientrare nei limiti dell'API Groq."""
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
            
        img_data = base64.b64decode(base64_str)
        img = Image.open(io.BytesIO(img_data))
        
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=75)
        compressed_bytes = buffer.getvalue()
        
        encoded = base64.b64encode(compressed_bytes).decode('utf-8')
        return f"data:image/jpeg;base64,{encoded}"
    except Exception as e:
        if not base64_str.startswith("data:image"):
            return f"data:image/jpeg;base64,{base64_str}"
        return base64_str

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
        
        # Comprime l'immagine per l'invio sicuro via API
        optimized_image_url = compress_base64_image(image_base64)

        prompt_text = (
            f"Sei un ispettore di sicurezza nell'analisi di fogli di presenza aziendali.\n"
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
            "Authorization": f"Bearer {GROQ_API_KEY.strip()}",
            "Content-Type": "application/json"
        }

        # Modelli Vision correnti Groq
        vision_models = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview"
        ]

        parsed_reply = None
        last_error_details = ""

        for model_name in vision_models:
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt_text},
                            {"type": "image_url", "image_url": {"url": optimized_image_url}}
                        ]
                    }
                ],
                "temperature": 0.0,
                "max_tokens": 300
            }

            try:
                res = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=20
                )

                if res.status_code == 200:
                    res_json = res.json()
                    raw_reply = res_json['choices'][0]['message']['content'].strip()
                    
                    if "```" in raw_reply:
                        raw_reply = re.sub(r'```(?:json)?\s*', '', raw_reply)
                        raw_reply = raw_reply.replace('```', '').strip()
                    
                    json_match = re.search(r'\{.*\}', raw_reply, re.DOTALL)
                    if json_match:
                        raw_reply = json_match.group(0)

                    parsed_reply = json.loads(raw_reply)
                    break
                else:
                    last_error_details = f"Modello {model_name} -> HTTP {res.status_code}: {res.text[:200]}"
            except Exception as ex:
                last_error_details = f"Errore chiamata/parsing su {model_name}: {str(ex)}"

        if parsed_reply is None:
            return JSONResponse(
                status_code=200,
                content={"status": "error", "message": f"Dettagli errore Groq API: {last_error_details}"}
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
