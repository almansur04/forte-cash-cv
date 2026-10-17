# Architecture & Engineering Decisions

This document explains why we chose a deterministic rule engine over modern Vision-Language Models (VLMs) or Pose Estimation.

## 1. Why not Vision-Language Models (VLMs)?
We initially tested Moondream2 (via Hugging Face) and LLaVA to detect mobile phone use.
- **The Hallucination Problem:** On top-down angles, VLMs consistently confused the bank's black IP desk phone for a personal mobile phone, causing a ~50% false positive rate.
- **The Latency Problem:** A single VLM query on M1 MPS took ~1.5 seconds. Querying multiple frames per video breached the 5-minute limit.
- **The Memory Problem:** Loading Moondream2 consumed 3.2GB of unified memory.

**Decision:** We dropped VLMs entirely. For a compliance tool, deterministic and predictable > smart and hallucinating.

## 2. Why not Pose Estimation?
We tested YOLOv8s-Pose to track the cashier's head and wrist.
- **The Keypoint Dropout Problem:** When the cashier leaned forward or turned their head, the model lost the keypoints. The engine interpreted this as "cashier left the desk," causing false positives.
- **The Chin-Rest Problem:** Pose estimation cannot distinguish a hand resting on a chin from a hand holding a phone to an ear.

**Decision:** We reverted to standard YOLOv8n bounding-box tracking with a highly tuned geometric overlap logic.

## 3. The Winning Logic (V9)
- **Spatial Geometry:** We define a `DESK_RECT` that covers the bottom 60% of the screen. 
- **The Leaning Fix:** By making the box tall (y=0.40 to 0.95), the cashier can lean forward to write without triggering the "away" rule.
- **The Dropout Fix:** If YOLO fails to detect the person for a single frame, we do not increment the "away" streak.
- **The Temporal Rule:** The cashier must be outside the desk zone for 1.2 seconds (6 consecutive frames at 5 FPS) to trigger a violation.

## 4. Deployment
We split the Docker build into `Dockerfile.base` (heavy ML dependencies) and `Dockerfile` (app code). This reduces rebuild time from 6 minutes to 2.7 seconds.