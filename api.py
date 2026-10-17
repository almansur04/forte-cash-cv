import os, uuid, shutil, json, time, sqlite3
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Depends, BackgroundTasks, Request, Header, HTTPException
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram, Gauge
import uvicorn
from engine import CashSecurityEngine
from dotenv import load_dotenv
from production import (log, new_trace_id, current_trace_id, audit, slo, timer,
                        CircuitBreaker, retry)

load_dotenv()

# --- CONFIG ---
API_KEY = os.getenv("API_KEY", "forte-hack-2026")
UPLOAD_DIR = Path("tmp_uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
EVIDENCE_DIR = Path("baseline/evidence"); EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = "jobs.db"
MAX_QUEUE_DEPTH = 50

# ==== BUSINESS METRICS ====
JOBS_TOTAL = Counter("forte_jobs_total", "Total jobs", ["status"])
VIOLATIONS_TOTAL = Counter("forte_violations_total", "Violations detected", ["violation_type"])
INFERENCE_DURATION = Histogram("forte_inference_seconds", "Inference duration",
                               buckets=[1, 2, 5, 10, 20, 30, 60])
QUEUE_DEPTH = Gauge("forte_queue_depth", "Current processing queue depth")
CIRCUIT_STATE = Gauge("forte_circuit_breaker_open", "1 = breaker open")


# --- DB INIT (with idempotency + review columns) ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY, status TEXT, video_name TEXT, timestamp REAL,
        result TEXT, error TEXT, idempotency_key TEXT UNIQUE,
        review_status TEXT DEFAULT 'pending', reviewed_by TEXT,
        review_notes TEXT, reviewed_at REAL)""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_ts ON jobs(timestamp DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_idem ON jobs(idempotency_key)")
    conn.commit(); conn.close()
init_db()

# --- ENGINE + CIRCUIT BREAKER ---
print("Initializing engine...")
engine = CashSecurityEngine(rules_csv="violation_rules_verified.csv")
engine_breaker = CircuitBreaker("yolo-inference", failure_threshold=3, recovery_timeout=60.0)

# --- OBSERVABILITY & RATE LIMITING ---
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="ForteBank Vision API", version="3.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
Instrumentator().instrument(app).expose(app)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.middleware("http")
async def correlation_middleware(request: Request, call_next):
    tid = request.headers.get("X-Trace-Id") or new_trace_id()
    response = await call_next(request)
    response.headers["X-Trace-Id"] = tid
    return response

def err(status: int, code: str, msg: str):
    return JSONResponse(status_code=status, content={"error": {
        "code": code, "message": msg, "trace_id": current_trace_id()}})

def verify_key(api_key: str = Depends(APIKeyHeader(name="X-API-Key"))):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid X-API-Key")
    return api_key


# --- BACKGROUND WORKER ---
def run_engine_task(job_id: str, video_path: Path, original_name: str):
    new_trace_id()
    try:
        with timer(f"engine:{job_id}") as t:
            result = engine_breaker.call(
                engine.analyze_video, video_path,
                sample_fps=5, evidence_dir=str(EVIDENCE_DIR))
        INFERENCE_DURATION.observe(t.elapsed_ms / 1000)
        slo.record(t.elapsed_ms, is_error=False)
        CIRCUIT_STATE.set(0 if engine_breaker.state == "closed" else 1)

        playback_path = EVIDENCE_DIR / f"{job_id}_{original_name}"
        if playback_path.exists():
            playback_path.unlink()
        shutil.copy(str(video_path), str(playback_path))

        for v in result.get("violations", []):
            v["evidence_frames"] = [
                f"/evidence/{Path(p).name}"
                for p in v.get("evidence_frames", []) if p
            ]
            v["video_url"] = f"/evidence/{playback_path.name}"
            VIOLATIONS_TOTAL.labels(violation_type=v["type"]).inc()

        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE jobs SET status=?, result=? WHERE job_id=?",
                     ("done", json.dumps(result, ensure_ascii=False), job_id))
        conn.commit(); conn.close()

        JOBS_TOTAL.labels(status="done").inc()
        audit("system", "job_completed", resource=job_id,
              payload={"verdict": result["verdict"],
                       "count": len(result["violations"])})
    except Exception as e:
        CIRCUIT_STATE.set(1 if engine_breaker.state == "open" else 0)
        slo.record(0, is_error=True)
        log.error("engine_failed", extra={"job_id": job_id, "error": str(e)})
        conn = sqlite3.connect(DB_PATH)
        conn.execute("UPDATE jobs SET status=?, error=? WHERE job_id=?",
                     ("failed", str(e), job_id))
        conn.commit(); conn.close()
        JOBS_TOTAL.labels(status="failed").inc()
        audit("system", "job_failed", resource=job_id, payload={"error": str(e)})
    finally:
        if video_path.exists():
            video_path.unlink()


# --- API ENDPOINTS ---
@app.post("/api/v1/analyze")
@limiter.limit("10/minute")  # DDoS protection
async def analyze(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    idempotency_key: Optional[str] = Header(None, alias="Idempotency-Key"),
    auth: str = Depends(verify_key),
):
    if not file.filename.endswith(".mov"):
        return err(400, "INVALID_FILE_TYPE", "Only .mov files supported")

    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    depth = conn.execute("SELECT COUNT(*) c FROM jobs WHERE status='processing'").fetchone()["c"]
    QUEUE_DEPTH.set(depth)
    if depth >= MAX_QUEUE_DEPTH:
        conn.close()
        audit("api", "load_shed", payload={"depth": depth}, ip=request.client.host)
        return err(503, "QUEUE_OVERLOADED", f"Queue at {depth}/{MAX_QUEUE_DEPTH}")

    if idempotency_key:
        existing = conn.execute(
            "SELECT * FROM jobs WHERE idempotency_key=?", (idempotency_key,)
        ).fetchone()
        if existing:
            conn.close()
            res = dict(existing)
            if res["result"]:
                res["result"] = json.loads(res["result"])
            log.info("idempotent_replay", extra={"job_id": res["job_id"]})
            return res
    conn.close()

    job_id = str(uuid.uuid4())
    tmp_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    with open(tmp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO jobs (job_id, status, video_name, timestamp, idempotency_key) "
        "VALUES (?,?,?,?,?)",
        (job_id, "processing", file.filename, time.time(), idempotency_key),
    )
    conn.commit(); conn.close()

    audit("api", "job_submitted", resource=job_id,
          payload={"file": file.filename}, ip=request.client.host)
    background_tasks.add_task(run_engine_task, job_id, tmp_path, file.filename)
    return {"job_id": job_id, "status": "processing", "trace_id": current_trace_id()}


@app.get("/api/v1/jobs/{job_id}")
async def get_job(job_id: str, auth: str = Depends(verify_key)):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
    conn.close()
    if not row:
        return err(404, "JOB_NOT_FOUND", "Job not found")
    res = dict(row)
    if res["result"]:
        res["result"] = json.loads(res["result"])
    return res


@app.get("/api/v1/history")
async def get_history(auth: str = Depends(verify_key)):
    conn = sqlite3.connect(DB_PATH); conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM jobs ORDER BY timestamp DESC LIMIT 50").fetchall()
    conn.close()
    out = []
    for r in rows:
        d = dict(r)
        if d["result"]:
            d["result"] = json.loads(d["result"])
        out.append(d)
    return out


@app.post("/api/v1/jobs/{job_id}/review")
async def review_job(job_id: str, request: Request, auth: str = Depends(verify_key)):
    body = await request.json()
    decision = body.get("decision")
    notes = body.get("notes", "")
    if decision not in ("accept", "override"):
        return err(400, "INVALID_DECISION", "decision must be 'accept' or 'override'")
    conn = sqlite3.connect(DB_PATH)
    cur = conn.execute(
        "UPDATE jobs SET review_status=?, reviewed_by=?, review_notes=?, reviewed_at=? "
        "WHERE job_id=?",
        (decision, "compliance_officer", notes, time.time(), job_id),
    )
    if cur.rowcount == 0:
        conn.close()
        return err(404, "JOB_NOT_FOUND", "Job not found")
    conn.commit(); conn.close()
    audit("compliance_officer", f"review_{decision}", resource=job_id,
          payload={"notes": notes}, ip=request.client.host)
    return {"status": "ok", "job_id": job_id, "review_status": decision}


# --- HEALTH & OBSERVABILITY ENDPOINTS ---
@app.get("/health/live")
async def health_live():
    return {"status": "alive"}

@app.get("/health/ready")
async def health_ready():
    try:
        conn = sqlite3.connect(DB_PATH); conn.execute("SELECT 1"); conn.close()
    except Exception:
        return err(503, "DB_UNAVAILABLE", "Database not ready")
    if engine_breaker.state == "open":
        return err(503, "ENGINE_UNAVAILABLE", "Circuit breaker open")
    return {"status": "ready", "device": engine.device}

@app.get("/health/startup")
async def health_startup():
    return {"status": "started", "engine_loaded": engine is not None}

@app.get("/api/v1/health")
async def health_legacy():
    return {"status": "healthy", "device": engine.device}

@app.get("/slo")
async def get_slo(auth: str = Depends(verify_key)):
    snap = slo.snapshot()
    snap["circuit_breaker"] = engine_breaker.snapshot()
    return snap

@app.get("/audit")
async def get_audit(limit: int = 100, auth: str = Depends(verify_key)):
    conn = sqlite3.connect("audit.db"); conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/evidence/{filename}")
async def get_evidence(filename: str):
    p = EVIDENCE_DIR / filename
    if not p.exists():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(p)


# --- UI ---
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
    <script>
        tailwind.config = {
            darkMode: 'class',
            theme: {
                extend: {
                    colors: {
                        forte: {
                            magenta: '#A31551',
                            purple: '#4A148C',
                            green: '#00573F',
                            darkblue: '#001E3C',
                            blue: '#005EB8',
                            bg: '#0B1121'
                        }
                    },
                    fontFamily: { sans: ['Inter', 'sans-serif'] }
                }
            }
        }
    </script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; background-color: #0B1121; color: #f8fafc; }
        .glass { background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(12px); border: 1px solid rgba(255,255,255,0.05); }
        .sidebar-item.active { background: rgba(163, 21, 81, 0.15); color: #A31551; border-right: 3px solid #A31551; font-weight: 600; }
        .sidebar-item:hover:not(.active) { background: rgba(255,255,255,0.05); }
        .toast { animation: slideIn 0.3s ease-out forwards; }
        @keyframes slideIn { from { transform: translateX(100%); opacity: 0; } to { transform: translateX(0); opacity: 1; } }
        .fade-in { animation: fadeIn 0.4s ease-out forwards; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
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
            <div class="bg-forte-magenta p-2 rounded-lg shadow-lg shadow-forte-magenta/20">
                <i data-lucide="shield-check" class="text-white w-6 h-6"></i>
            </div>
            <div>
                <h1 class="text-lg font-bold tracking-tight">ForteBank</h1>
                <p class="text-xs text-forte-magenta font-medium tracking-wider">VISION AI</p>
            </div>
        </div>

        <nav class="flex-1 py-6 px-4 space-y-2">
            <a href="#" onclick="switchView('dashboard', this)" class="sidebar-item active flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors">
                <i data-lucide="layout-dashboard" class="w-5 h-5"></i> Dashboard
            </a>
            <a href="#" onclick="switchView('investigations', this)" class="sidebar-item flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 transition-colors">
                <i data-lucide="folder-open" class="w-5 h-5"></i> Investigations
            </a>
            <a href="#" onclick="switchView('regulatory', this)" class="sidebar-item flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 transition-colors">
                <i data-lucide="file-text" class="w-5 h-5"></i> Regulatory Logs
            </a>
            <a href="#" onclick="switchView('settings', this)" class="sidebar-item flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-slate-400 transition-colors">
                <i data-lucide="settings" class="w-5 h-5"></i> Settings
            </a>
        </nav>

        <div class="p-4 border-t border-slate-800">
            <div class="glass rounded-xl p-4 flex items-center gap-3 border-l-2 border-l-forte-magenta">
                <div class="w-10 h-10 rounded-full bg-forte-darkblue flex items-center justify-center font-bold text-sm">AK</div>
                <div>
                    <p class="text-sm font-semibold">Admin</p>
                    <p class="text-[10px] text-forte-magenta uppercase tracking-widest font-bold mt-0.5">Forte Corporate</p>
                </div>
            </div>
        </div>
    </aside>

    <!-- Main Content -->
    <main class="flex-1 flex flex-col relative overflow-y-auto">
        <header class="glass h-16 border-b border-slate-800 flex items-center justify-between px-8 sticky top-0 z-10">
            <h2 class="text-lg font-semibold text-slate-200" id="pageTitle">Operational Dashboard</h2>
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

        <!-- ================= VIEW: DASHBOARD ================= -->
        <div id="view-dashboard" class="view-section p-8 max-w-7xl mx-auto w-full space-y-8">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                <div class="glass rounded-xl p-5 relative overflow-hidden border-t-2 border-t-forte-blue">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="video" class="w-16 h-16 text-forte-blue"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Total Analyzed</p>
                    <p class="text-3xl font-bold mt-2 text-white" id="statTotal">0</p>
                </div>
                <div class="glass rounded-xl p-5 relative overflow-hidden border-t-2 border-t-forte-magenta">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="alert-triangle" class="w-16 h-16 text-forte-magenta"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Violations</p>
                    <p class="text-3xl font-bold mt-2 text-forte-magenta" id="statViolations">0</p>
                </div>
                <div class="glass rounded-xl p-5 relative overflow-hidden border-t-2 border-t-forte-green">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="check-circle" class="w-16 h-16 text-forte-green"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Cleared</p>
                    <p class="text-3xl font-bold mt-2 text-forte-green" id="statCleared">0</p>
                </div>
                <div class="glass rounded-xl p-5 relative overflow-hidden border-t-2 border-t-forte-purple">
                    <div class="absolute right-0 top-0 p-4 opacity-10"><i data-lucide="cpu" class="w-16 h-16 text-forte-purple"></i></div>
                    <p class="text-xs text-slate-400 font-medium uppercase tracking-wider">Inference Engine</p>
                    <p class="text-lg font-bold mt-2 text-forte-purple" id="statDevice">YOLOv8n</p>
                </div>
            </div>

            <div class="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div class="lg:col-span-2 space-y-6">
                    <div id="uploadSection" class="glass rounded-2xl border-2 border-dashed border-slate-700 hover:border-forte-magenta transition-all cursor-pointer p-10 text-center group" onclick="document.getElementById('fileInput').click()">
                        <input type="file" id="fileInput" class="hidden" accept=".mov" onchange="handleFileSelect(this)">
                        <div class="flex flex-col items-center gap-4">
                            <div class="p-4 bg-slate-800/80 rounded-full text-slate-400 group-hover:text-forte-magenta transition-colors shadow-lg">
                                <i data-lucide="cloud-upload" class="w-8 h-8"></i>
                            </div>
                            <div>
                                <h3 class="text-lg font-semibold text-white">Upload New Footage</h3>
                                <p class="text-sm text-slate-400 mt-1">Drag &amp; drop .mov files here for instant compliance analysis</p>
                            </div>
                        </div>
                    </div>

                    <div id="processingSection" class="hidden glass rounded-2xl p-8 flex flex-col items-center justify-center gap-4">
                        <div class="relative w-full h-2 bg-slate-800 rounded-full overflow-hidden">
                            <div id="progressBar" class="absolute top-0 left-0 h-full bg-forte-magenta transition-all duration-300" style="width: 0%"></div>
                        </div>
                        <p class="text-sm font-medium text-slate-300" id="progressText">Initializing pipeline...</p>
                    </div>

                    <div id="resultSection" class="hidden flex-col gap-6 fade-in"></div>
                </div>

                <div class="space-y-6">
                    <div class="glass rounded-2xl p-6 border-t-2 border-t-forte-blue">
                        <h3 class="text-sm font-bold text-white mb-4 flex items-center gap-2">
                            <i data-lucide="history" class="w-4 h-4 text-forte-blue"></i> Recent
                        </h3>
                        <div class="space-y-3" id="historyList">
                            <p class="text-xs text-slate-500 italic">No recent jobs.</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- ================= VIEW: INVESTIGATIONS ================= -->
        <div id="view-investigations" class="view-section hidden p-8 max-w-7xl mx-auto w-full space-y-6 fade-in">
            <div class="glass rounded-2xl p-6">
                <h3 class="text-lg font-bold text-white mb-4">All Investigations</h3>
                <div class="overflow-x-auto">
                    <table class="w-full text-sm text-left text-slate-400">
                        <thead class="text-xs text-slate-300 uppercase bg-slate-800/50 border-b border-slate-700">
                            <tr>
                                <th class="px-6 py-3">Video Name</th>
                                <th class="px-6 py-3">Date/Time</th>
                                <th class="px-6 py-3">Verdict</th>
                                <th class="px-6 py-3">Violations</th>
                            </tr>
                        </thead>
                        <tbody id="investigationsTableBody"></tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- ================= VIEW: REGULATORY ================= -->
        <div id="view-regulatory" class="view-section hidden p-8 max-w-7xl mx-auto w-full space-y-6 fade-in">
            <div class="glass rounded-2xl p-8 border-l-4 border-l-forte-magenta">
                <h2 class="text-2xl font-bold text-white mb-2">NBRK Resolution No. 231</h2>
                <p class="text-slate-400 mb-6">Dated 29.11.2019. Registered with the Ministry of Justice of the Republic of Kazakhstan on 03.12.2019 No. 19680.</p>

                <div class="bg-slate-800/50 rounded-xl p-6 border border-slate-700 mb-6">
                    <div class="flex justify-between items-start mb-4">
                        <h3 class="text-lg font-bold text-forte-magenta">Article 22. Cash Operations Procedure</h3>
                        <span class="px-3 py-1 bg-forte-green/20 text-forte-green text-xs font-bold rounded-full border border-forte-green/30">VERIFIED</span>
                    </div>
                    <p class="text-slate-200 text-lg italic mb-4">"Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента."</p>
                    <p class="text-slate-400 text-sm mb-4">(Cashier is not allowed to leave the workplace before the end of customer service.)</p>
                    <a href="https://adilet.zan.kz/rus/docs/V1900019680" target="_blank" class="text-forte-blue hover:underline text-sm flex items-center gap-1">
                        View full document on Adilet.zan.kz <i data-lucide="external-link" class="w-4 h-4"></i>
                    </a>
                </div>
            </div>
        </div>

        <!-- ================= VIEW: SETTINGS ================= -->
        <div id="view-settings" class="view-section hidden p-8 max-w-7xl mx-auto w-full space-y-6 fade-in">
            <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div class="glass rounded-2xl p-6">
                    <h3 class="text-lg font-bold text-white mb-4">System Configuration</h3>
                    <div class="space-y-4">
                        <div>
                            <label class="text-xs text-slate-400 uppercase">API Key</label>
                            <input type="text" value="forte-hack-2026" readonly class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-slate-300 mt-1">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 uppercase">Device</label>
                            <input type="text" id="settingsDevice" value="Detecting..." readonly class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-slate-300 mt-1">
                        </div>
                        <div>
                            <label class="text-xs text-slate-400 uppercase">Model</label>
                            <input type="text" value="YOLOv8n (Ultralytics)" readonly class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2 text-slate-300 mt-1">
                        </div>
                    </div>
                </div>

                <div class="glass rounded-2xl p-6">
                    <h3 class="text-lg font-bold text-white mb-4">Forte Brand Architecture</h3>
                    <div class="space-y-3">
                        <div class="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                            <span class="text-sm font-medium text-slate-300">Forte (Main/Retail)</span>
                            <span class="w-6 h-6 rounded-full bg-[#A31551] shadow-lg shadow-[#A31551]/50"></span>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                            <span class="text-sm font-medium text-slate-300">Forte Solo</span>
                            <span class="w-6 h-6 rounded-full bg-[#4A148C] shadow-lg shadow-[#4A148C]/50"></span>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                            <span class="text-sm font-medium text-slate-300">Forte Premier</span>
                            <span class="w-6 h-6 rounded-full bg-[#00573F] shadow-lg shadow-[#00573F]/50"></span>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                            <span class="text-sm font-medium text-slate-300">Forte Business</span>
                            <span class="w-6 h-6 rounded-full bg-[#001E3C] border border-slate-600"></span>
                        </div>
                        <div class="flex items-center justify-between p-3 bg-slate-800/50 rounded-lg border border-slate-700">
                            <span class="text-sm font-medium text-slate-300">Forte Corporate</span>
                            <span class="w-6 h-6 rounded-full bg-[#005EB8] shadow-lg shadow-[#005EB8]/50"></span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

    </main>

    <div id="toastContainer" class="fixed bottom-6 right-6 flex flex-col gap-3 z-50"></div>

    <script>
        const API_KEY = "forte-hack-2026";
        lucide.createIcons();

        function switchView(viewName, element) {
            document.querySelectorAll('.view-section').forEach(el => el.classList.add('hidden'));
            document.getElementById('view-' + viewName).classList.remove('hidden');
            document.querySelectorAll('.sidebar-item').forEach(el => el.classList.remove('active'));
            element.classList.add('active');
            const titles = {
                dashboard: 'Operational Dashboard',
                investigations: 'Investigation Logs',
                regulatory: 'Regulatory Base',
                settings: 'System Settings'
            };
            document.getElementById('pageTitle').innerText = titles[viewName];
            if (viewName === 'investigations') renderInvestigationsTable();
        }

        updateHealth();
        fetchHistory();
        setInterval(updateHealth, 10000);

        function showToast(msg, type = "error") {
            const toast = document.createElement('div');
            const bg = type === "error"
                ? "bg-forte-magenta/10 border-forte-magenta/50 text-forte-magenta"
                : "bg-forte-green/10 border-forte-green/50 text-forte-green";
            toast.className = `toast glass border ${bg} px-4 py-3 rounded-lg shadow-xl flex items-center gap-3`;
            toast.innerHTML = `<i data-lucide="${type === 'error' ? 'alert-circle' : 'check-circle'}" class="w-5 h-5"></i> <span class="text-sm font-medium">${msg}</span>`;
            document.getElementById('toastContainer').appendChild(toast);
            lucide.createIcons();
            setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 4000);
        }

        let globalJobs = [];

        async function updateHealth() {
            try {
                const res = await fetch('/api/v1/health');
                const data = await res.json();
                document.getElementById('sysStatus').innerText = `System Online (${data.device.toUpperCase()})`;
                document.getElementById('statDevice').innerText = data.device.toUpperCase();
                document.getElementById('settingsDevice').value = data.device.toUpperCase();
            } catch (e) {}
        }

        async function fetchHistory() {
            try {
                const res = await fetch('/api/v1/history', { headers: { "X-API-Key": API_KEY } });
                globalJobs = await res.json();

                const list = document.getElementById('historyList');
                if (globalJobs.length === 0) return;

                list.innerHTML = "";
                let violations = 0;
                let cleared = 0;

                globalJobs.slice(0, 5).forEach(j => {
                    if (j.status === 'done') {
                        if (j.result.verdict === 'нарушение') violations++;
                        else cleared++;
                    }
                    const icon = j.status === 'processing' ? 'loader-2' : (j.result?.verdict === 'нарушение' ? 'alert-triangle' : 'check-circle');
                    const color = j.status === 'processing' ? 'text-forte-blue animate-spin' : (j.result?.verdict === 'нарушение' ? 'text-forte-magenta' : 'text-forte-green');

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
                document.getElementById('statTotal').innerText = globalJobs.length;
                document.getElementById('statViolations').innerText = violations;
                document.getElementById('statCleared').innerText = cleared;
                lucide.createIcons();
            } catch (e) {}
        }

        function renderInvestigationsTable() {
            const tbody = document.getElementById('investigationsTableBody');
            tbody.innerHTML = "";
            globalJobs.forEach(j => {
                const verdict = j.status === 'processing' ? 'Processing...' : (j.result?.verdict === 'нарушение' ? '<span class="text-forte-magenta font-bold">Violation</span>' : '<span class="text-forte-green font-bold">Clean</span>');
                const count = j.status === 'done' && j.result.violations ? j.result.violations.length : 0;
                tbody.innerHTML += `
                    <tr class="border-b border-slate-700/50 hover:bg-slate-800/30">
                        <td class="px-6 py-4 font-medium text-slate-200">${j.video_name}</td>
                        <td class="px-6 py-4">${new Date(j.timestamp*1000).toLocaleString()}</td>
                        <td class="px-6 py-4">${verdict}</td>
                        <td class="px-6 py-4">${count}</td>
                    </tr>
                `;
            });
        }

        function handleFileSelect(input) {
            if (input.files.length > 0) uploadVideo(input.files[0]);
        }

        const dropZone = document.getElementById('uploadSection');
        dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('border-forte-magenta'); });
        dropZone.addEventListener('dragleave', () => dropZone.classList.remove('border-forte-magenta'));
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('border-forte-magenta');
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
                    method: "POST",
                    headers: { "X-API-Key": API_KEY },
                    body: formData
                });
                const data = await res.json();
                if (res.ok) pollStatus(data.job_id, phaseInt, bar, text);
                else throw new Error(data.detail || data.error?.message || "Unknown error");
            } catch (e) {
                clearInterval(phaseInt);
                showToast(e.message, 'error');
                resetUI();
            }
        }

        async function pollStatus(jobId, phaseInt, bar, text) {
            const interval = setInterval(async () => {
                const res = await fetch(`/api/v1/jobs/${jobId}`, { headers: { "X-API-Key": API_KEY } });
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

            const isViolation = res.verdict === "нарушение";
            const borderCol = isViolation ? "border-forte-magenta" : "border-forte-green";
            const bgCol = isViolation ? "bg-forte-magenta/5" : "bg-forte-green/5";
            const iconBg = isViolation ? "bg-forte-magenta/20 text-forte-magenta" : "bg-forte-green/20 text-forte-green";
            const iconName = isViolation ? "alert-triangle" : "check-circle";

            let html = `
                <div class="glass rounded-2xl p-6 flex items-center justify-between shadow-lg border-l-4 ${borderCol} ${bgCol}">
                    <div class="flex items-center gap-4">
                        <div class="p-3 rounded-full ${iconBg}">
                            <i data-lucide="${iconName}" class="w-8 h-8"></i>
                        </div>
                        <div>
                            <h2 class="text-2xl font-bold ${isViolation ? 'text-forte-magenta' : 'text-forte-green'}">${isViolation ? 'Violation Detected' : 'No Violations'}</h2>
                            <p class="text-slate-400 text-sm mt-1">${res.video}</p>
                        </div>
                    </div>
                </div>
                <div class="flex flex-col gap-6">
            `;

            res.violations.forEach(v => {
                html += `
                    <div class="glass rounded-2xl p-6 flex flex-col md:flex-row gap-6 border border-slate-700/50">
                        <div class="w-full md:w-1/2 flex flex-col gap-4">
                            <div>
                                <span class="px-2 py-1 bg-slate-800 text-[10px] uppercase font-bold tracking-wider rounded text-slate-400 border border-slate-700">${v.type}</span>
                                <h3 class="text-lg font-semibold text-white mt-3">${v.description}</h3>
                            </div>
                            <div class="bg-slate-800/80 p-4 rounded-xl border border-slate-700">
                                <p class="text-[10px] text-slate-400 uppercase tracking-wider mb-1 font-bold">Regulatory Citation</p>
                                <a href="${v.adilet_url}" target="_blank" class="text-sm text-forte-blue hover:underline flex items-center gap-1 font-medium">
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
                            <div class="rounded-xl overflow-hidden border border-slate-700 bg-black aspect-video relative">
                                <video controls class="w-full h-full object-contain" src="${v.video_url}#t=${v.timecode_sec},${v.timecode_end_sec}"></video>
                            </div>
                            ${v.evidence_frames.map(img => `<img src="${img}" class="w-full rounded-xl border border-slate-700 shadow-lg">`).join('')}
                        </div>
                    </div>
                `;
            });

            html += `</div>`;
            document.getElementById('resultSection').innerHTML = html;
            lucide.createIcons();
            fetchHistory();
        }
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)