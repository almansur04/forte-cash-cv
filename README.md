# ForteBank Cash Discipline Monitoring

An offline, high-precision computer vision pipeline for detecting cash desk violations from top-down CCTV footage.

Built for the ForteBank Hackathon 2026. Fully local, zero cloud dependencies, CPU-deployable.

## The Problem
Banks have thousands of hours of cash desk footage. Cloud AI is banned due to data privacy. Humans cannot watch it all.

## The Solution
A deterministic rule engine that tracks the cashier's physical position and flags when they leave the workstation unattended. Mapped to **NBRK Resolution No. 231, Article 22**. 

## Architecture
1. **Detection:** YOLOv8n (Ultralytics) on Apple MPS (dev) / CPU (deploy).
2. **Tracking:** ByteTrack (persistent IDs).
3. **Rule Engine:** Temporal overlap logic (if cashier is outside desk zone for > 1.2s, flag).
4. **API:** FastAPI with asynchronous background tasks.
5. **UI:** Built-in HTML/JS dashboard for uploading and reviewing violations.

*(For a deep dive into why we rejected Vision-Language Models and Pose Estimation, see [`ARCHITECTURE.md`](ARCHITECTURE.md).)*

## Benchmarks (M1 Pro, 32GB, MPS)
- **Avg Processing Time:** 1.71s per video (well under the 5-minute limit).
- **Peak RAM:** 584 MB.
- **F1-Score:** 61.22% (Precision: 55.5%, Recall: 68.2%).
- See [`docs/final_benchmark.txt`](docs/final_benchmark.txt) for the full log.

## Quick Start (Local Dev on M1)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 8000
```
Open `http://127.0.0.1:8000` in your browser. Use the API Key `forte-hack-2026`.

## Docker Deployment (Linux/CPU)
```bash
docker build -f Dockerfile.base -t forte-cash-base:1.0 .
docker build -t forte-cash-api .
docker run -p 8000:8000 forte-cash-api
```

## Known Limitations (Phase 2 Roadmap)
- **Sitting violations** (e.g., mobile phone use at the desk) are not currently flagged. We tested VLM (Moondream2) and Pose estimation, but both produced unacceptable false positives on top-down angles. This requires a custom-trained action classifier.
- Rules are tuned to the provided camera angle. Different angles require adjusting `DESK_RECT` in `engine.py`.