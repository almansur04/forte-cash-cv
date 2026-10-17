from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, BackgroundTasks
from fastapi.security.api_key import APIKeyHeader
import uvicorn
import uuid
import shutil
import cv2
import os
import pandas as pd
from engine import CashSecurityEngine

app = FastAPI(
    title="ForteBank Cash Discipline API",
    description="Offline CV analysis of cashier workspaces.",
    version="1.0.0"
)

# Security
API_KEY = "forte-hack-2026"
api_key_header = APIKeyHeader(name="X-API-Key")

# In-Memory Database for Hackathon Prototype
jobs = {}

# Initialize our ML Engine (Singleton)
engine = CashSecurityEngine()

# Load Regulatory Rules mapping
try:
    rules_df = pd.read_csv("violation_rules.csv")
    rules_dict = rules_df.set_index("violation_type").to_dict(orient="index")
except FileNotFoundError:
    print("Warning: violation_rules.csv not found. Regulatory mapping will be empty.")
    rules_dict = {}

def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

def process_video_task(job_id: str, file_path: str):
    jobs[job_id]["status"] = "processing"
    
    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None: fps = 30
    frame_interval = int(fps)
    
    frame_idx = 0
    seconds_processed = 0
    violations_found = []
    
    # Process only first 5 seconds to keep the test fast
    while cap.isOpened() and seconds_processed < 5:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            _, frame_violations = engine.analyze_frame(frame_rgb)
            
            for v in frame_violations:
                # Map raw engine types to our CSV keys
                v_map = {
                    "unauthorized_person": "unauthorized_person_in_zone",
                    "unattended_open_safe": "unattended_cash_drawer"
                }
                csv_key = v_map.get(v["type"], v["type"])
                
                rule_info = rules_dict.get(csv_key, {"clause_number": "Глава 2", "presumed_violation": "Unknown Anomaly"})
                
                violations_found.append({
                    "type": csv_key,
                    "timecode_sec": seconds_processed + 1,
                    "description": f"{rule_info['presumed_violation']} (VLM Evidence: {v['evidence'].strip()})",
                    "clause": rule_info.get("clause_number", ""),
                    "regulatory_link": rule_info.get("adilet_link", "")
                })
            
            seconds_processed += 1
        frame_idx += 1
        
    cap.release()
    
    if os.path.exists(file_path):
        os.remove(file_path)
    
    # Verdict logic
    verdict = "нарушение" if len(violations_found) > 0 else "без нарушения"
    
    jobs[job_id]["status"] = "done"
    jobs[job_id]["result"] = {
        "verdict": verdict,
        "violations": violations_found
    }

@app.post("/api/v1/analyze", summary="Upload video for analysis")
async def analyze_video(background_tasks: BackgroundTasks, file: UploadFile = File(...), api_key: str = Depends(verify_api_key)):
    job_id = str(uuid.uuid4())
    file_path = f"temp_{job_id}.mov"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    jobs[job_id] = {"status": "pending", "result": None}
    
    # Send to background worker
    background_tasks.add_task(process_video_task, job_id, file_path)
    
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/v1/jobs/{job_id}", summary="Get job status and results")
def get_job_status(job_id: str, api_key: str = Depends(verify_api_key)):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]

if __name__ == "__main__":
    print("\n🚀 Starting ForteBank Cash Security API on http://127.0.0.1:8000")
    print("👉 View the Interactive UI at: http://127.0.0.1:8000/docs\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")