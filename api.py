import os, uuid, shutil, json
from pathlib import Path
from typing import Dict, Any
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from engine import CashSecurityEngine

# --- Config ---
API_KEY = "forte-hack-2026"
api_key_header = APIKeyHeader(name="X-API-Key")
UPLOAD_DIR = Path("tmp_uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR = Path("baseline/evidence"); EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

# --- App & Engine ---
app = FastAPI(title="ForteBank Cash Discipline API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

print("Initializing engine...")
engine = CashSecurityEngine(rules_csv="violation_rules_verified.csv")

# --- Job Store ---
# Schema: { job_id: { "status": "processing|done|failed", "result": {...} | None, "error": str | None } }
jobs: Dict[str, Dict[str, Any]] = {}

def run_engine_task(job_id: str, video_path: Path):
    """Background task to process the video."""
    try:
        # Point evidence dir to the shared static folder
        result = engine.analyze_video(video_path, sample_fps=2, evidence_dir=str(EVIDENCE_DIR))
        
        # Post-process evidence paths to be web URLs instead of local paths
        for v in result.get("violations", []):
            v["evidence_frames"] = [
                f"/evidence/{Path(p).name}" for p in v.get("evidence_frames", []) if p
            ]
        
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if video_path.exists():
            video_path.unlink()  # clean up temp file

# --- Auth ---
def verify_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid X-API-Key")
    return api_key

# --- Endpoints ---
@app.post("/api/v1/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...), auth: str = Depends(verify_key)):
    if not file.filename.endswith(".mov"):
        raise HTTPException(status_code=400, detail="Only .mov files are supported")
    
    job_id = str(uuid.uuid4())
    tmp_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = {"status": "processing", "result": None, "error": None}
    background_tasks.add_task(run_engine_task, job_id, tmp_path)
    
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str, auth: str = Depends(verify_key)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

# --- Static Evidence ---
@app.get("/evidence/{filename}")
async def get_evidence(filename: str):
    p = EVIDENCE_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Evidence image not found")
    return FileResponse(p)

# --- Simple Web UI ---
@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ForteBank Cash Control</title>
        <style>
            body { font-family: -apple-system, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; }
            .card { border: 1px solid #ccc; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
            .violation { border-left: 4px solid #d32f2f; background: #ffebee; padding: 1rem; margin-bottom: 1rem; }
            .clean { border-left: 4px solid #388e3c; background: #e8f5e9; padding: 1rem; margin-bottom: 1rem; }
            .evidence { max-height: 300px; border: 1px solid #ccc; margin-top: 10px; }
            button { padding: 10px 20px; font-size: 16px; cursor: pointer; }
        </style>
    </head>
    <body>
        <h1>ForteBank Cash Control API</h1>
        <div class="card">
            <h2>Upload Video</h2>
            <input type="file" id="fileInput" accept=".mov">
            <button onclick="uploadVideo()">Analyze</button>
            <p id="statusMsg" style="color: #666; margin-top: 10px;"></p>
        </div>
        <div id="resultContainer"></div>

        <script>
            const API_KEY = "forte-hack-2026";
            async function uploadVideo() {
                const file = document.getElementById('fileInput').files[0];
                if (!file) return alert("Select a .mov file first");
                
                document.getElementById('statusMsg').innerText = "Uploading...";
                document.getElementById('resultContainer').innerHTML = "";
                
                const formData = new FormData();
                formData.append("file", file);
                
                try {
                    const res = await fetch("/api/v1/analyze", {
                        method: "POST",
                        headers: {"X-API-Key": API_KEY},
                        body: formData
                    });
                    const data = await res.json();
                    if (res.ok) pollStatus(data.job_id);
                    else throw new Error(data.detail);
                } catch (e) {
                    document.getElementById('statusMsg').innerText = "Error: " + e.message;
                }
            }

            async function pollStatus(jobId) {
                document.getElementById('statusMsg').innerText = "Processing video (this takes ~10s)...";
                const interval = setInterval(async () => {
                    const res = await fetch(`/api/v1/jobs/${jobId}`, {headers: {"X-API-Key": API_KEY}});
                    const data = await res.json();
                    
                    if (data.status === "done") {
                        clearInterval(interval);
                        document.getElementById('statusMsg').innerText = "Done!";
                        renderResult(data.result);
                    } else if (data.status === "failed") {
                        clearInterval(interval);
                        document.getElementById('statusMsg').innerText = "Failed: " + data.error;
                    }
                }, 2000);
            }

            function renderResult(res) {
                let html = `<div class="card"><h3>Result for ${res.video}</h3><p>Duration: ${res.duration_sec}s</p>`;
                if (res.verdict === "нарушение") {
                    html += `<h4 style="color: #d32f2f;">Violations Found</h4>`;
                    res.violations.forEach(v => {
                        html += `<div class="violation">
                            <strong>${v.type}</strong> (${v.timecode_sec}s - ${v.timecode_end_sec}s)<br>
                            <em>${v.description}</em><br>
                            <a href="${v.adilet_url}" target="_blank">RK NBRK No. 231, ${v.article_number}</a>
                            ${v.evidence_frames.map(img => `<br><img src="${img}" class="evidence">`).join('')}
                        </div>`;
                    });
                } else {
                    html += `<div class="clean"><strong>No violations detected.</strong> Cashier maintained station.</div>`;
                }
                html += `</div>`;
                document.getElementById('resultContainer').innerHTML = html;
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
