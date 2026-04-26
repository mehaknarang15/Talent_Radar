from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import uvicorn
import os

# Import the core AI pipeline logic
from talent_scout import run_pipeline 

app = FastAPI(title="Talent Radar API", description="AI Scouting Engine Endpoint")
templates = Jinja2Templates(directory="templates")

# ==========================================
# DATABASE INITIALIZATION
# ==========================================
CANDIDATES_DB = {}

# Safely load candidates into memory on startup
try:
    if os.path.exists('candidates.json'):
        with open('candidates.json', 'r', encoding='utf-8') as f:
            candidates_list = json.load(f)
            # Convert list of dicts into the {"Name": "Resume Text"} format expected by the pipeline
            CANDIDATES_DB = {c['name']: c['resume_text'] for c in candidates_list}
            print(f"[*] Loaded {len(CANDIDATES_DB)} candidates into memory.")
    else:
        print("[!] Warning: candidates.json not found. Operating with an empty candidate database.")
except Exception as e:
    print(f"[!] Error loading candidates database: {e}")

# ==========================================
# DATA MODELS
# ==========================================
class PipelineRequest(BaseModel):
    job_description: str

# ==========================================
# ROUTES
# ==========================================
@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    """Serves the main frontend application interface."""
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/run-pipeline")
async def api_run_pipeline(payload: PipelineRequest):
    """
    Executes the real AI scouting pipeline.
    Expects a job description string and processes it against the loaded candidates.
    """
    try:
        # Execute the multi-agent AI pipeline
        results, ghost, jd_report = run_pipeline(payload.job_description, CANDIDATES_DB)
        
        return {
            "status": "success",
            "results": results,
            "jd_report": jd_report   
        }
    except Exception as e:
        print(f"[!] Pipeline Execution Error: {str(e)}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)