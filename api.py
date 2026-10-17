import os, uuid, shutil, json, time
from pathlib import Path
from typing import Dict, Any, List
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from engine import CashSecurityEngine

API_KEY = "forte-hack-2026"
api_key_header = APIKeyHeader(name="X-API-Key")
UPLOAD_DIR = Path("tmp_uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR = Path("baseline/evidence"); EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ForteBank Vision API", version="2.0.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"]
)

print("Initializing engine...")
engine = CashSecurityEngine(rules_csv="violation_rules_verified.csv")

# In-memory job store
jobs: Dict[str, Dict[str, Any]] = {}

def run_engine_task(job_id: str, video_path: Path, original_name: str):
    try:
        jobs[job_id]["status"] = "processing"
        result = engine.analyze_video(video_path, sample_fps=5, evidence_dir=str(EVIDENCE_DIR))
        
        playback_path = EVIDENCE_DIR / f"{job_id}_{original_name}"
        if playback_path.exists(): playback_path.unlink()
        shutil.copy(str(video_path), str(playback_path))
        
        for v in result.get("violations", []):
            v["evidence_frames"] = [f"/evidence/{Path(p).name}" for p in v.get("evidence_frames", []) if p]
            v["video_url"] = f"/evidence/{playback_path.name}"
            
        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = result
        jobs[job_id]["video_name"] = original_name
    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        if video_path.exists(): video_path.unlink()

def verify_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY: raise HTTPException(status_code=403, detail="Invalid X-API-Key")
    return api_key

@app.post("/api/v1/analyze")
async def analyze(background_tasks: BackgroundTasks, file: UploadFile = File(...), auth: str = Depends(verify_key)):
    if not file.filename.endswith(".mov"):
        raise HTTPException(status_code=400, detail="Only .mov files are supported")
    
    job_id = str(uuid.uuid4())
    tmp_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = {"status": "processing", "result": None, "error": None, "video_name": file.filename, "timestamp": time.time()}
    background_tasks.add_task(run_engine_task, job_id, tmp_path, file.filename)
    return {"job_id": job_id, "status": "processing"}

@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str, auth: str = Depends(verify_key)):
    if job_id not in jobs: raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

@app.get("/api/v1/history")
async def get_history(auth: str = Depends(verify_key)):
    # Return last 10 jobs sorted by newest
    sorted_jobs = sorted(jobs.items(), key=lambda x: x[1].get("timestamp", 0), reverse=True)[:10]
    return [{"job_id": k, **v} for k, v in sorted_jobs]

@app.get("/api/v1/health")
async def health():
    return {"status": "healthy", "device": engine.device, "active_jobs": sum(1 for j in jobs.values() if j["status"]=="processing"), "total_jobs": len(jobs)}

@app.get("/evidence/{filename}")
async def get_evidence(filename: str):
    p = EVIDENCE_DIR / filename
    if not p.exists(): raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(p)

@app.get("/", response_class=HTMLResponse)
async def ui():
    return """
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ForteBank Vision | Enterprise Compliance</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0B1121; color: #f8fafc; }
        .glass { background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); }
        .sidebar-item.active { background: rgba(16, 185, 129, 0.1); color: #10b981; border-right: 3px solid #10b981; }
        .sidebar-item:hover:not(.active) { background: rgba(255,255,255,0.05); }
        .toast { animation: slideIn 0.3s ease-out forwards; }
        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
        .fade-in { animation: fadeIn 0.4s ease-out forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0B1121; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
</head>
<body class="flex h-screen overflow-hidden">

    <!-- Sidebar -->
    <aside class="w-64 glass border-r border-slate-800 flex flex-col z-20">
        <div class="p-6 flex items-center gap-3 border-b border-slate-800">
            <div class="bg-emerald-500 p-2 rounded-lg shadow-lg shadow-emerald-500/20">
                <i data-lucide="shield-check" class="text-white w-6 h-6"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold tracking-tight">ForteBank</h1>
                <p class="text-xs text-emerald-400 font-medium tracking-wider">VISION AI</p>
            </div>
        </div>
        
        <nav class="flex-1 py-6 px-4 space-y-2">
            <a href="#" class="sidebar-item active flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors">
                <i data-lucide="layout-dashboard" class="w-5 h-5"></i> Dashboard
            </a>
            <a href="#" class="sidebar-item flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 transition-colors">
                <i data-lucide="folder-open" class="w-5 h-5"></i> Investigations
            </a>
            <a href="#" class="sidebar-item flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 transition-colors">
                <i data-lucide="file-text" class="w-5 h-5"></i> Regulatory Logs
            </a>
            <a href="#" class="sidebar-item flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 transition-colors">
                <i data-lucide="settings" class="w-5 h-5"></i> Settings
            </a>
        </nav>

        <div class="p-4 border-t border-slate-800">
            <div class="glass rounded-xl p-4 flex items-center gap-3">
                <div class="w-10 h-10 rounded-full bg-slate-700 flex items-center justify-center font-bold text-sm">AK</div>
                <div>
                    <p class="text-sm font-semibold">Admin User</p>
                    <p class="text-xs text-slate-400">Compliance Officer</p>
                </div>
            </div>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col relative overflow-y-auto">
        <!-- Header -->
        <header class="glass h-16 border-b border-slate-800 flex items-center justify-between px-8 sticky top-0 z-10">
            <h2 class="text-lg font-semibold text-slate-200">Operational Dashboard</h2>
            <div class="flex items-center gap-4 text-xs font-medium">
                <div class="flex items-center gap-2 bg-slate-800/50 px-3 py-1.5 rounded-full border border-slate-700">
                    <span class="relative flex h-2 w-2">
                      <span class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                      <span class="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    <span class="text-slate-300" id="sysStatus">Connecting...</span>
                </div>
            </div>
        </header>

        <div class="p-8 max-w-7xl mx-auto w-full space-y-8">
            <!-- Stats Grid -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div class="glass rounded-xl p-5 relative overflow-hidden">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="video" class="w-16 h-16 text-emerald-400"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Total Analyzed</p>
                    <p class="text-3xl font-bold mt-2 text-white" id="statTotal">0</p>
                </div>
                <div class="glass rounded-xl p-5 relative overflow-hidden">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="alert-triangle" class="w-16 h-16 text-red-500"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Violations Flagged</p>
                    <p class="text-3xl font-bold mt-2 text-red-500" id="statViolations">0</p>
                </div>
                <div class="glass rounded-xl p-5 relative overflow-hidden">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="check-circle" class="w-16 h-16 text-emerald-500"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Cleared Videos</p>
                    <p class="text-3xl font-bold mt-2 text-emerald-400" id="statCleared">0</p>
                </div>
                <div class="glass rounded-xl p-5 relative overflow-hidden">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="cpu" class="w-16 h-16 text-blue-400"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Inference Engine</p>
                    <p class="text-lg font-bold mt-2 text-blue-400" id="statDevice">YOLOv8n</p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <!-- Left Column: Upload & Results -->
                <div class="lg:col-span-2 space-y-6">
                    
                    <!-- Upload Zone -->
                    <div id="uploadSection" class="glass rounded-2xl border-2 border-dashed border-slate-700 hover:border-emerald-500 transition-all cursor-pointer p-10 text-center group" onclick="document.getElementById('fileInput').click()">
                        <input type="file" id="fileInput" class="hidden" accept=".mov" onchange="handleFileSelect(this)">
                        <div class="flex flex-col items-center gap-4">
                            <div class="p-4 bg-slate-800/80 rounded-full text-slate-400 group-hover:text-emerald-400 transition-colors shadow-lg">
                                <i data-lucide="cloud-upload" class="w-8 h-8"></i>
                            </div>
                            <div>
                                <h3 class="text-lg font-semibold text-white">Upload New Footage</h3>
                                <p class="text-sm text-slate-400 mt-1">Drag & drop .mov files here for instant compliance analysis</p>
                            </div>
                        </div>
                    </div>

                    <!-- Processing State -->
                    <div id="processingSection" class="hidden glass rounded-2xl p-8 flex flex-col items-center justify-center gap-4">
                        <div class="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div id="progressBar" class="absolute top-0 left-0 h-full bg-emerald-500 transition-all duration-300" style="width: 0%"></div>
                        </div>
                        <p class="text-sm font-medium text-slate-300" id="progressText">Initializing pipeline...</p>
                        <p class="text-xs text-slate-500">Running YOLOv8 inference on Apple MPS...</p>
                    </div>

                    <!-- Result View -->
                    <div id="resultSection" class="hidden flex-col gap-6 fade-in">
                        <div id="verdictBanner" class="glass rounded-2xl p-6 flex items-center justify-between shadow-lg">
                            <div class="flex items-center gap-4">
                                <div id="verdictIcon" class="p-3 rounded-full"></div>
                                <div>
                                    <h2 id="verdictTitle" class="text-2xl font-bold"></h2>
                                    <p id="verdictDesc" class="text-slate-400 text-sm mt-1"></p>
                                </div>
                            </div>
                        </div>
                        <div id="violationsList" class="flex flex-col gap-6"></div>
                    </div>
                </div>

                <!-- Right Column: Recent Jobs Table -->
                <div class="space-y-6">
                    <div class="glass rounded-2xl p-6">
                        <h3 class="text-sm font-bold text-white mb-4 flex items-center gap-2">
                            <i data-lucide="history" class="w-4 h-4 text-emerald-400"></i> Recent Investigations
                        </h3>
                        <div class="space-y-3" id="historyList">
                            <p class="text-xs text-slate-500 italic">No recent jobs.</p>
                        </div>
                    </div>
                    
                    <div class="glass rounded-2xl p-6 border-l-4 border-l-blue-500">
                        <h3 class="text-sm font-bold text-white mb-2 flex items-center gap-2">
                            <i data-lucide="shield" class="w-4 h-4 text-blue-400"></i> Legal Compliance Base
                        </h3>
                        <p class="text-xs text-slate-400 leading-relaxed">
                            Rules mapped strictly to <br>
                            <a href="https://adilet.zan.kz/rus/docs/V1900019680" target="_blank" class="text-blue-400 hover:underline">NBRK Resolution No. 231</a>.<br>
                            Automated citations for Article 22 (Unattended Workstation).
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </main>

    <!-- Toast Container -->
    <div id="toastContainer" class="fixed bottom-6 right-6 flex flex-col gap-3 z-50"></div>

    <script>
        const API_KEY = "forte-hack-2026";
        lucide.createIcons();
        
        // Initialize dashboard
        updateHealth();
        fetchHistory();
        setInterval(updateHealth, 10000);

        function showToast(msg, type="error") {
            const toast = document.createElement('div');
            const bg = type === "error" ? "bg-red-500/10 border-red-500/50 text-red-400" : "bg-emerald-500/10 border-emerald-500/50 text-emerald-400";
            toast.className = `toast glass border ${bg} px-4 py-3 rounded-lg shadow-xl flex items-center gap-3`;
            toast.innerHTML = `<i data-lucide="${type === 'error' ? 'alert-circle' : 'check-circle'}" class="w-5 h-5"></i> <span class="text-sm font-medium">${msg}</span>`;
            document.getElementById('toastContainer').appendChild(toast);
            lucide.createIcons();
            setTimeout(() => { toast.style.opacity = '0'; setTimeout(()=>toast.remove(), 300); }, 4000);
        }

        async function updateHealth() {
            try {
                const res = await fetch('/api/v1/health');
                const data = await res.json();
                document.getElementById('sysStatus').innerText = `System Online (${data.device.toUpperCase()})`;
                document.getElementById('statDevice').innerText = data.device.toUpperCase();
            } catch(e) {}
        }

        async function fetchHistory() {
            try {
                const res = await fetch('/api/v1/history', {headers: {"X-API-Key": API_KEY}});
                const jobs = await res.json();
                
                const list = document.getElementById('historyList');
                if (jobs.length === 0) return;
                
                list.innerHTML = "";
                let violations = 0;
                let cleared = 0;
                
                jobs.forEach(j => {
                    if (j.status === 'done') {
                        if (j.result.verdict === 'нарушение') violations++;
                        else cleared++;
                    }
                    
                    const icon = j.status === 'processing' ? 'loader-2' : (j.result?.verdict === 'нарушение' ? 'alert-triangle' : 'check-circle');
                    const color = j.status === 'processing' ? 'text-blue-400 animate-spin' : (j.result?.verdict === 'нарушение' ? 'text-red-400' : 'text-emerald-400');
                    
                    list.innerHTML += `
                        <div class="flex items-center justify-between p-3 rounded-lg bg-slate-800/50 border border-slate-700/50 hover:bg-slate-800 transition-colors">
                            <div class="flex items-center gap-3 overflow-hidden">
                                <i data-lucide="${icon}" class="w-4 h-4 ${color} shrink-0"></i>
                                <div class="truncate">
                                    <p class="text-xs font-medium text-slate-200 truncate">${j.video_name}</p>
                                    <p class="text-[10px] text-slate-500">${new Date(j.timestamp*1000).toLocaleTimeString()}</p>
                                </div>
                            </div>
                        </div>
                    `;
                });
                document.getElementById('statTotal').innerText = jobs.length;
                document.getElementById('statViolations').innerText = violations;
                document.getElementById('statCleared').innerText = cleared;
                lucide.createIcons();
            } catch(e) {}
        }

        function handleFileSelect(input) {
            if (input.files.length > 0) uploadVideo(input.files[0]);
        }

        const dropZone = document.getElementById('uploadSection');
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-emerald-500'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-emerald-500'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-emerald-500');
            if (e.dataTransfer.files.length) uploadVideo(e.dataTransfer.files[0]);
        });

        async function uploadVideo(file) {
            document.getElementById('uploadSection').classList.add('hidden');
            document.getElementById('resultSection').classList.add('hidden');
            document.getElementById('processingSection').classList.remove('hidden');
            document.getElementById('processingSection').classList.add('flex');
            
            let progress = 0;
            const bar = document.getElementById('progressBar');
            const text = document.getElementById('progressText');
            
            const phases = [
                { p: 20, t: "Uploading video..." },
                { p: 40, t: "Extracting frames via FFmpeg..." },
                { p: 70, t: "Running YOLOv8 inference..." },
                { p: 90, t: "Applying NBRK compliance rules..." },
                { p: 100, t: "Generating evidence package..." }
            ];
            let phaseIdx = 0;
            const phaseInt = setInterval(() => {
                if (phaseIdx < phases.length) {
                    bar.style.width = phases[phaseIdx].p + "%";
                    text.innerText = phases[phaseIdx].t;
                    phaseIdx++;
                }
            }, 1500);

            const formData = new FormData();
            formData.append("file", file);

            try {
                const res = await fetch("/api/v1/analyze", {
                    method: "POST", headers: {"X-API-Key": API_KEY}, body: formData
                });
                const data = await res.json();
                if (res.ok) pollStatus(data.job_id, phaseInt, bar, text);
                else throw new Error(data.detail);
            } catch (e) {
                clearInterval(phaseInt);
                showToast(e.message, 'error');
                resetUI();
            }
        }

        async function pollStatus(jobId, phaseInt, bar, text) {
            const interval = setInterval(async () => {
                const res = await fetch(`/api/v1/jobs/${jobId}`, {headers: {"X-API-Key": API_KEY}});
                const data = await res.json();
                
                if (data.status === "done") {
                    clearInterval(interval);
                    clearInterval(phaseInt);
                    bar.style.width = "100%";
                    text.innerText = "Analysis Complete";
                    setTimeout(() => renderResult(data.result), 400);
                } else if (data.status === "failed") {
                    clearInterval(interval);
                    clearInterval(phaseInt);
                    showToast("Analysis failed: " + data.error, 'error');
                    resetUI();
                }
            }, 1000);
        }

        function resetUI() {
            document.getElementById('uploadSection').classList.remove('hidden');
            document.getElementById('processingSection').classList.add('hidden');
            document.getElementById('processingSection').classList.remove('flex');
            document.getElementById('resultSection').classList.add('hidden');
            document.getElementById('resultSection').classList.remove('flex');
            document.getElementById('fileInput').value = "";
            document.getElementById('progressBar').style.width = "0%";
        }

        function renderResult(res) {
            document.getElementById('processingSection').classList.add('hidden');
            document.getElementById('processingSection').classList.remove('flex');
            document.getElementById('resultSection').classList.remove('hidden');
            document.getElementById('resultSection').classList.add('flex');
            
            const banner = document.getElementById('verdictBanner');
            const icon = document.getElementById('verdictIcon');
            const title = document.getElementById('verdictTitle');
            const desc = document.getElementById('verdictDesc');

            if (res.verdict === "нарушение") {
                banner.className = "glass rounded-2xl p-6 flex items-center justify-between shadow-lg border-l-4 border-red-500 bg-red-500/5";
                icon.className = "p-3 rounded-full bg-red-500/20 text-red-500";
                icon.innerHTML = '<i data-lucide="alert-triangle" class="w-8 h-8"></i>';
                title.innerText = "Violation Detected";
                title.className = "text-2xl font-bold text-red-500";
                desc.innerText = `Found ${res.violations.length} violation(s) in ${res.video}`;
            } else {
                banner.className = "glass rounded-2xl p-6 flex items-center justify-between shadow-lg border-l-4 border-emerald-500 bg-emerald-500/5";
                icon.className = "p-3 rounded-full bg-emerald-500/20 text-emerald-500";
                icon.innerHTML = '<i data-lucide="check-circle" class="w-8 h-8"></i>';
                title.innerText = "No Violations Detected";
                title.className = "text-2xl font-bold text-emerald-500";
                desc.innerText = `All clear in ${res.video}`;
            }

            const list = document.getElementById('violationsList');
            list.innerHTML = "";

            res.violations.forEach(v => {
                const vCard = document.createElement('div');
                vCard.className = "glass rounded-2xl p-6 flex flex-col md:flex-row gap-6 fade-in border border-slate-700/50";
                vCard.innerHTML = `
                    <div class="w-full md:w-1/2 flex flex-col gap-4">
                        <div>
                            <span class="px-2 py-1 bg-slate-800 text-[10px] uppercase font-bold tracking-wider rounded text-slate-400 border border-slate-700">${v.type}</span>
                            <h3 class="text-lg font-semibold text-white mt-3">${v.description}</h3>
                        </div>
                        <div class="bg-slate-800/80 p-4 rounded-xl border border-slate-700">
                            <p class="text-[10px] text-slate-400 uppercase tracking-wider mb-1 font-bold">Regulatory Citation</p>
                            <a href="${v.adilet_url}" target="_blank" class="text-sm text-emerald-400 hover:text-emerald-300 flex items-center gap-1 font-medium">
                                NBRK Resolution No. 231, ${v.article_number} <i data-lucide="external-link" class="w-3 h-3"></i>
                            </a>
                        </div>
                        <div class="grid grid-cols-2 gap-4">
                            <div class="bg-slate-800/80 p-3 rounded-lg border border-slate-700">
                                <p class="text-[10px] text-slate-500 uppercase font-bold">Start Time</p>
                                <p class="text-white font-mono text-lg">${v.timecode_sec}s</p>
                            </div>
                            <div class="bg-slate-800/80 p-3 rounded-lg border border-slate-700">
                                <p class="text-[10px] text-slate-500 uppercase font-bold">End Time</p>
                                <p class="text-white font-mono text-lg">${v.timecode_end_sec}s</p>
                            </div>
                        </div>
                    </div>
                    <div class="w-full md:w-1/2 flex flex-col gap-4">
                        <div class="rounded-xl overflow-hidden border border-slate-700 bg-black aspect-video flex items-center justify-center shadow-inner relative group">
                            <video controls class="w-full h-full object-contain" src="${v.video_url}#t=${v.timecode_sec},${v.timecode_end_sec}"></video>
                            <div class="absolute top-2 left-2 bg-black/60 text-white text-[10px] px-2 py-1 rounded backdrop-blur-sm border border-white/10 pointer-events-none">
                                Evidence Playback
                            </div>
                        </div>
                        ${v.evidence_frames.map(img => `
                            <div class="relative">
                                <img src="${img}" class="w-full rounded-xl border border-slate-700 shadow-lg" alt="Evidence Frame">
                                <div class="absolute bottom-2 right-2 bg-black/60 text-white text-[10px] px-2 py-1 rounded backdrop-blur-sm border border-white/10">YOLO Annotations</div>
                            </div>
                        `).join('')}
                    </div>
                `;
                list.appendChild(vCard);
            });
            lucide.createIcons();
            fetchHistory(); // Refresh table stats
        }
    </script>
</body>
</html>
"""
