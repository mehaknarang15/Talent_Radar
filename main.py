from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import json
import uvicorn

# Import your real pipeline
from talent_scout import run_pipeline 

app = FastAPI(title="Talent Radar API")
templates = Jinja2Templates(directory="templates")

# Load candidates into memory on startup
with open('candidates.json', 'r') as f:
    candidates_list = json.load(f)
    # Convert list of dicts into the {"Name": "Resume Text"} format the pipeline expects
    CANDIDATES_DB = {c['name']: c['resume_text'] for c in candidates_list}

# We only need the JD from the frontend now
class PipelineRequest(BaseModel):
    job_description: str

@app.get("/", response_class=HTMLResponse)
async def serve_frontend(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/api/run-pipeline")
async def api_run_pipeline(payload: PipelineRequest):
    """Executes the real AI pipeline."""
    try:
        # Run the actual agents
        results, ghost, jd_report = run_pipeline(payload.job_description, CANDIDATES_DB)
        
        return {
            "status": "success",
            "results": results,
            "jd_report": jd_report   # ✅ ADD THIS LINE
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)