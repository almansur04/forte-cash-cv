# Architecture & Engineering Decisions

This document explains the engineering trade-offs behind the pipeline.

## 1. Why not Vision-Language Models (VLMs)?

We tested Moondream2 (HF) and LLaVA to detect phone use.
- **Hallucination:** on top-down angles, VLMs confused the bank's black IP desk phone for a personal mobile phone - ~50% false positive rate.
- **Latency:** one VLM query on M1 MPS took ~1.5s. Multiple queries per video breach the 5-minute budget.
- **Memory:** loading Moondream2 costs ~3.2 GB.

**Decision:** dropped VLMs entirely. For compliance, deterministic > smart.

## 2. Why not Pose Estimation?

We tested YOLOv8s-Pose for head/wrist tracking.
- **Keypoint dropout:** when the cashier leaned or turned, keypoints vanished, causing false "away" signals.
- **Chin-rest problem:** pose cannot distinguish a hand on a chin from a hand holding a phone to an ear.

**Decision:** reverted to YOLOv8n bounding-box tracking with geometric overlap logic.

## 3. The Winning Logic

- **Desk rectangle** covers the bottom of the frame where the workstation sits.
- **Torso center** (bbox midpoint) must be inside the rectangle for the cashier to count as "at desk."
- **Temporal rule:** absent for ≥ 1.2s (6 consecutive frames at 5 FPS) → violation.
- **Dropout tolerance:** if YOLO fails to detect any person, the away-streak does NOT increment.

## 4. Deployment Architecture

- Base image (`Dockerfile.base`): python 3.11 + ONNX Runtime + OpenCV + FastAPI stack + static ffmpeg. Cached and reused.
- App image (`Dockerfile`): copies `api.py`, `engine.py`, `production.py`, `violation_rules_verified.csv`, `yolov8n.onnx`, `yolov8n_int8.onnx`.
- Rebuilds take ~3s because the base layer is cached.

## 5. Why ONNX Runtime + INT8 (not PyTorch at inference)

The initial PyTorch pipeline (`yolov8n.pt`) worked but consumed **~600 MB RAM** and required torchvision + ultralytics (~250 MB) just to load. On CPU this made the app undeployable on 512 MB free tiers.

**Migration:**
1. Export YOLOv8n to ONNX (opset 12): `yolov8n.onnx` - 12 MB
2. Quantize dynamic INT8: `yolov8n_int8.onnx` - 3.3 MB
3. Replace `ultralytics.YOLO.track()` with `onnxruntime.InferenceSession`
4. Implement minimal NMS with `cv2.dnn.NMSBoxes` (no extra dependency)

**Result:** peak RAM drops to **~179 MB**, image size drops ~70%, no observable detection regression for our single-class use case.

## 6. Frame skipping with `cv2.grab()`

A naive loop calls `cap.read()` on every frame, discarding 29 of 30 for 2 FPS sampling. On CPU each decode is ~20 ms → **43 seconds wasted** on an 71-second video.

**Fix:** use `cap.grab()` for intermediate frames (decoder advances, no YUV→BGR conversion, no numpy allocation), then `cap.read()` only for the sampled frame. 10-50× faster skip.

Combined with early-stop (return at first violation), demo runtime drops:
- Clean 72s video: 60s → **6s**
- Violation 25s video: 3s → **1s**

## 7. Why we did NOT ship pre-seeded results

Pre-seeded demo entries would make the dashboard instantly populated for judges, but they obscure the fact that the pipeline actually works on new footage. We ship the real engine and accept that clean (unviolated) videos still take 6s on the fastest machine. Judges see genuine results.