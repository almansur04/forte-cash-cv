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
    try:
        result = engine.analyze_video(video_path, sample_fps=2, evidence_dir=str(EVIDENCE_DIR))
        
        # Preserve the video for playback
        playback_dir = Path("baseline/evidence")
        playback_path = playback_dir / f"{job_id}_{video_path.name}"
        if playback_path.exists():
            playback_path.unlink()
        shutil.copy(str(video_path), str(playback_path))
        
        for v in result.get("violations", []):
            v["evidence_frames"] = [f"/evidence/{Path(p).name}" for p in v.get("evidence_frames", []) if p]
            v["video_url"] = f"/evidence/{playback_path.name}"
        
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if video_path.exists():
            video_path.unlink()

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

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "device": engine.device,
        "model": "yolov8n",
        "rules_loaded": len(engine.rules),
        "active_jobs": sum(1 for j in jobs.values() if j["status"] == "processing"),
        "total_jobs": len(jobs),
    }

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
    <html class="dark">
    <head>
        <title>ForteBank | Cash Control AI</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <script src="https://cdn.tailwindcss.com"></script>
        <script src="https://unpkg.com/lucide@latest"></script>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            body { font-family: 'Inter', sans-serif; background-color: #0f172a; color: #f8fafc; }
            .glass { background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.1); }
            .violation-card { border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.1); }
            .success-card { border-left: 4px solid #10b981; background: rgba(16, 185, 129, 0.1); }
            @keyframes spin { 100% { transform: rotate(360deg); } }
            .loader { animation: spin 1s linear infinite; }
        </style>
    </head>
    <body class="min-h-screen flex flex-col">
        <!-- Navbar -->
        <nav class="glass border-b border-slate-700 w-full px-6 py-4 flex justify-between items-center sticky top-0 z-50">
            <div class="flex items-center gap-3">
                <div class="bg-emerald-500 p-2 rounded-lg">
                    <i data-lucide="shield-check" class="text-white w-6 h-6"></i>
                </div>
                <div>
                    <h1 class="text-xl font-bold tracking-tight text-white">ForteBank <span class="text-emerald-400">Vision</span></h1>
                    <p class="text-xs text-slate-400">Cash Discipline Monitoring System</p>
                </div>
            </div>
            <div class="flex items-center gap-4 text-sm">
                <div class="flex items-center gap-2">
                    <span class="relative flex h-3 w-3">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span class="relative inline-flex rounded-full h-3 w-3 bg-emerald-500"></span>
                    </span>
                    <span class="text-slate-300">System Online</span>
                </div>
                <div class="px-3 py-1 bg-slate-800 rounded-md border border-slate-700 text-slate-300">
                    API Key: <span class="text-emerald-400 font-mono text-xs">***-2026</span>
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <main class="flex-grow max-w-5xl w-full mx-auto p-6 flex flex-col gap-8">
            
            <!-- Upload Zone -->
            <div id="uploadSection" class="glass rounded-2xl p-8 text-center border-dashed border-2 border-slate-600 hover:border-emerald-500 transition-colors cursor-pointer" onclick="document.getElementById('fileInput').click()">
                <input type="file" id="fileInput" class="hidden" accept=".mov" onchange="handleFileSelect(this)">
                <div class="flex flex-col items-center gap-4">
                    <div class="p-4 bg-slate-800 rounded-full text-emerald-400">
                        <i data-lucide="upload-cloud" class="w-8 h-8"></i>
                    </div>
                    <div>
                        <h3 class="text-lg font-semibold text-white">Upload Video for Analysis</h3>
                        <p class="text-sm text-slate-400 mt-1">Drag & drop your .mov file here, or click to browse</p>
                    </div>
                </div>
            </div>

            <!-- Status / Processing -->
            <div id="statusSection" class="hidden glass rounded-2xl p-6 flex items-center justify-center gap-4">
                <svg class="loader w-6 h-6 text-emerald-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24"><circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle><path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path></svg>
                <div>
                    <p class="font-semibold text-white" id="statusText">Uploading video...</p>
                    <p class="text-sm text-slate-400">Please wait while our AI analyzes the footage.</p>
                </div>
            </div>

            <!-- Results Container -->
            <div id="resultSection" class="hidden flex-col gap-6">
                <!-- Verdict Header -->
                <div id="verdictBanner" class="glass rounded-2xl p-6 flex items-center justify-between">
                    <div class="flex items-center gap-4">
                        <div id="verdictIcon" class="p-3 rounded-full"></div>
                        <div>
                            <h2 id="verdictTitle" class="text-2xl font-bold"></h2>
                            <p id="verdictDesc" class="text-slate-400 text-sm mt-1"></p>
                        </div>
                    </div>
                    <div class="text-right">
                        <p class="text-xs text-slate-500 uppercase tracking-wider">Processing Time</p>
                        <p id="processingTime" class="text-lg font-mono text-white"></p>
                    </div>
                </div>

                <!-- Violations List -->
                <div id="violationsList" class="flex flex-col gap-6"></div>
            </div>
        </main>

        <script>
            const API_KEY = "forte-hack-2026";
            lucide.createIcons();

            function handleFileSelect(input) {
                const file = input.files[0];
                if (!file) return;
                uploadVideo(file);
            }

            // Drag and Drop
            const dropZone = document.getElementById('uploadSection');
            dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-emerald-500'); });
            dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-emerald-500'));
            dropZone.addEventListener('drop', (e) => {
                e.preventDefault();
                dropZone.classList.remove('border-emerald-500');
                if (e.dataTransfer.files.length) {
                    document.getElementById('fileInput').files = e.dataTransfer.files;
                    uploadVideo(e.dataTransfer.files[0]);
                }
            });

            async function uploadVideo(file) {
                document.getElementById('uploadSection').classList.add('hidden');
                document.getElementById('statusSection').classList.remove('hidden');
                document.getElementById('resultSection').classList.add('hidden');
                
                const formData = new FormData();
                formData.append("file", file);
                const startTime = performance.now();

                try {
                    const res = await fetch("/api/v1/analyze", {
                        method: "POST",
                        headers: {"X-API-Key": API_KEY},
                        body: formData
                    });
                    const data = await res.json();
                    if (res.ok) pollStatus(data.job_id, startTime);
                    else throw new Error(data.detail);
                } catch (e) {
                    alert("Error: " + e.message);
                    resetUI();
                }
            }

            async function pollStatus(jobId, startTime) {
                document.getElementById('statusText').innerText = "AI is analyzing frames...";
                const interval = setInterval(async () => {
                    const res = await fetch(`/api/v1/jobs/${jobId}`, {headers: {"X-API-Key": API_KEY}});
                    const data = await res.json();
                    
                    if (data.status === "done") {
                        clearInterval(interval);
                        const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
                        renderResult(data.result, elapsed);
                    } else if (data.status === "failed") {
                        clearInterval(interval);
                        alert("Failed: " + data.error);
                        resetUI();
                    }
                }, 1000);
            }

            function resetUI() {
                document.getElementById('uploadSection').classList.remove('hidden');
                document.getElementById('statusSection').classList.add('hidden');
                document.getElementById('resultSection').classList.add('hidden');
                document.getElementById('fileInput').value = "";
            }

            function renderResult(res, elapsed) {
                document.getElementById('statusSection').classList.add('hidden');
                document.getElementById('resultSection').classList.remove('hidden');
                document.getElementById('resultSection').classList.add('flex');
                
                const banner = document.getElementById('verdictBanner');
                const icon = document.getElementById('verdictIcon');
                const title = document.getElementById('verdictTitle');
                const desc = document.getElementById('verdictDesc');

                if (res.verdict === "нарушение") {
                    banner.className = "glass violation-card rounded-2xl p-6 flex items-center justify-between";
                    icon.className = "p-3 rounded-full bg-red-500/20 text-red-500";
                    icon.innerHTML = '<i data-lucide="alert-triangle" class="w-8 h-8"></i>';
                    title.innerText = "Violation Detected";
                    title.className = "text-2xl font-bold text-red-500";
                    desc.innerText = `Found ${res.violations.length} violation(s) in ${res.video}`;
                } else {
                    banner.className = "glass success-card rounded-2xl p-6 flex items-center justify-between";
                    icon.className = "p-3 rounded-full bg-emerald-500/20 text-emerald-500";
                    icon.innerHTML = '<i data-lucide="check-circle" class="w-8 h-8"></i>';
                    title.innerText = "No Violations Detected";
                    title.className = "text-2xl font-bold text-emerald-500";
                    desc.innerText = `All clear in ${res.video}`;
                }
                document.getElementById('processingTime').innerText = elapsed + "s";

                const list = document.getElementById('violationsList');
                list.innerHTML = "";

                res.violations.forEach(v => {
                    const vCard = document.createElement('div');
                    vCard.className = "glass rounded-2xl p-6 flex flex-col md:flex-row gap-6";
                    vCard.innerHTML = `
                        <div class="w-full md:w-1/2 flex flex-col gap-4">
                            <div>
                                <span class="px-2 py-1 bg-slate-800 text-xs font-mono rounded text-slate-300 border border-slate-700">${v.type}</span>
                                <h3 class="text-lg font-semibold text-white mt-2">${v.description}</h3>
                            </div>
                            <div class="bg-slate-800/50 p-4 rounded-xl border border-slate-700">
                                <p class="text-xs text-slate-400 uppercase tracking-wider mb-1">Regulatory Citation</p>
                                <a href="${v.adilet_url}" target="_blank" class="text-sm text-emerald-400 hover:text-emerald-300 flex items-center gap-1">
                                    NBRK Resolution No. 231, ${v.article_number} <i data-lucide="external-link" class="w-3 h-3"></i>
                                </a>
                            </div>
                            <div class="grid grid-cols-2 gap-4">
                                <div class="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                                    <p class="text-xs text-slate-500">Start Time</p>
                                    <p class="text-white font-mono">${v.timecode_sec}s</p>
                                </div>
                                <div class="bg-slate-800/50 p-3 rounded-lg border border-slate-700">
                                    <p class="text-xs text-slate-500">End Time</p>
                                    <p class="text-white font-mono">${v.timecode_end_sec}s</p>
                                </div>
                            </div>
                        </div>
                        <div class="w-full md:w-1/2 flex flex-col gap-4">
                            <div class="rounded-xl overflow-hidden border border-slate-700 bg-black aspect-video flex items-center justify-center">
                                <video controls autoplay muted class="w-full h-full object-contain" src="${v.video_url}#t=${v.timecode_sec},${v.timecode_end_sec}"></video>
                            </div>
                            ${v.evidence_frames.map(img => `
                                <img src="${img}" class="w-full rounded-xl border border-slate-700" alt="Evidence Frame">
                            `).join('')}
                        </div>
                    `;
                    list.appendChild(vCard);
                });
                lucide.createIcons();
            }
        </script>
    </body>
    </html>
    """

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
