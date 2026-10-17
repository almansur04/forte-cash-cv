# ForteBank Cash Discipline Monitoring

An offline, production-grade computer vision pipeline for detecting cash desk violations from top-down CCTV footage.

Built for the ForteBank Hackathon 2026. Fully local, zero cloud dependencies, CPU-deployable, **1-6s per video on M1**.

---

## The Problem

Banks have thousands of hours of cash desk footage. Cloud AI is banned for privacy. Human reviewers cannot watch it all.

## The Solution

A deterministic YOLOv8n → ONNX Runtime rule engine that tracks the cashier's workstation position and flags abandonment. Every violation is mapped to verified RK law (NBRK Resolution No. 231, Article 22).

## Architecture

1. **Detection:** YOLOv8n, exported to ONNX, quantized to INT8, input 320×320
2. **Frame skipping:** `cv2.grab()` (advances decoder without decoding pixels) - 10× faster than naive read
3. **Rule engine:** Center-point-in-desk-rect with temporal streak (1.2s minimum)
4. **Early-stop:** Scan halts at first violation - a 25s video with violation at 11.5s scans only 23 frames
5. **API:** FastAPI + SQLite + SlowAPI + Prometheus metrics
6. **UI:** Single-page HTML/JS dashboard with ForteBank brand colors

## Benchmarks (M1 Pro, native ARM64)

| Video | Length | Violation | Time |
|---|---|---|---|
| Х-11.mov | 25s | at 11.5s | **1s** (early-stop) |
| Х-48.mov | 72s | clean | **6s** (full scan) |
| Peak RAM | - | - | **~179 MB** |

Baseline detection benchmark on 42 training clips (416×416 FP32): **F1 61.22%**. The INT8 320×320 production variant trades ~2-4% F1 for a 10× speedup - same detections, faster.

## Live Demo

- **Primary (fast):** hosted from developer M1 via Cloudflare Tunnel - 1-6s per video
- **Backup (slow):** https://forte-cash-api.onrender.com - Render free tier, 15-90s per video, 30-50s cold start
- Both run the same GitHub source (`Dockerfile.render`)

## Quick Start (Local)

```bash
docker build -f Dockerfile.base -t forte-cash-base:1.0 .
docker build -f Dockerfile -t forte-cash-api:latest .
docker run -p 8000:8000 forte-cash-api:latest
```

Open http://localhost:8000. API key: `forte-hack-2026`.

> **Note:** the Docker Hub image `almansur04/forte-cash-api:latest` is the earlier FP32 PyTorch baseline (253 MB). To get the fast INT8 ONNX build, build from source with the commands above - that's what both live deployments run.

## Deployment

- **Cloudflare Tunnel (dev, fast):** `cloudflared tunnel --protocol http2 --url http://localhost:8000` on an ARM64 machine
- **Render (public, slow):** uses `Dockerfile.render`, builds from GitHub on commit, free tier 0.1 CPU

## Regulatory Base

| Rule | Article | Adilet |
|---|---|---|
| Cashier leaves workstation zone | NBRK 231, п. 22 | [link](https://adilet.zan.kz/rus/docs/V1900019680) |

## Known Limitations

- Detects only "cashier away from workstation" (п. 22). Phones / reaching over / unattended valuables are not detected - those require a custom-trained action classifier (Phase 2).
- Rules are tuned for the given camera angle. Different angles require updating `DESK_RECT` in `engine.py`.
- Free-tier hosting sleeps after 15 min idle (30-50s wake).

See [`ARCHITECTURE.md`](ARCHITECTURE.md), [`COMPLIANCE.md`](COMPLIANCE.md), [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md), [`SCALING_ARCHITECTURE.md`](SCALING_ARCHITECTURE.md).