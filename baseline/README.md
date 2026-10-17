# ForteBank Cash Discipline Monitoring

Offline computer-vision system for detecting cash desk violations from top-down video.
Built for the ForteBank Hackathon 2026.

## Architecture
- **Detection:** YOLOv8n (Ultralytics) on Apple MPS (dev) / CPU (deploy).
- **Rule Engine:** Deterministic temporal logic.
  - `cashier_away_from_workstation`: Center-point leaves the desk zone for > 1.6s.
- **Regulatory Link:** NBRK Resolution No. 231, Article 22.
- **API:** FastAPI with asynchronous background tasks.
- **UI:** Built-in HTML/JS dashboard for uploading and reviewing violations.

## RK Regulatory Base
| Rule | Article | Clause Text |
|---|---|---|
| `cashier_away_from_workstation` | п. 22 | Кассовому работнику не допускается покидать рабочее место до окончания обслуживания клиента. |

## Quick Start (Local Dev on M1)
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python api.py
```
Open `http://127.0.0.1:8000` in your browser. Use the API Key `forte-hack-2026`.

## Docker Deployment (Linux/CPU)
```bash
docker build -t forte-cash-cv .
docker run -p 8000:8000 forte-cash-cv
```

## Benchmarks (M1 Pro, 32GB, MPS)
- **Avg Processing Time:** 1.7s per video (well under the 5-minute limit).
- **Peak RAM:** 580 MB.

## Known Limitations
- **Sitting violations** (e.g., mobile phone use at the desk) are not currently flagged. The VLM (Moondream2) was tested but hallucinated desk phones. YOLO COCO class 67 (cell phone) fails on top-down resolution.
- Rules are tuned to the provided camera angle. Different angles require adjusting `DESK_RECT` in `engine.py`.