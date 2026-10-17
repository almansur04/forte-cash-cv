# Experiments - Development Audit Trail

These scripts are preserved as a **historical record** of the iteration process during the 48h hackathon.

**They are NOT part of the production pipeline** and are **NOT imported** by `api.py` or `engine.py`.

## Why they exist

Each script corresponds to a specific iteration documented in [`../ENGINEERING_JOURNAL.md`](../ENGINEERING_JOURNAL.md). Keeping them documents:

- What we tried and rejected (VLM, Pose, CLIP, SigMamba)
- What broke and why (versions V13, V16, V17, V18 all produced 0% F1)
- What was measured (F1 52.4% → 61.22% across iterations)

## Why they hardcode `/Users/almansurkakimov/...` paths

These scripts were run interactively on the developer's Mac during the hackathon. They expect the training dataset at:

```
/Users/almansurkakimov/Downloads/open/videos/*.mov
/Users/almansurkakimov/Downloads/open/open_labels.xlsx
```

**To reproduce a benchmark:** change the `VIDEO_DIR` / `BASE_DIR` constant at the top of each script to point at the dataset location.

## What actually runs in production

| File | Purpose |
|---|---|
| `../api.py` | FastAPI + SPA dashboard |
| `../engine.py` | ONNX Runtime inference + rule engine |
| `../production.py` | Logging, circuit breaker, SLO, audit |

None of these reference anything in this directory.

## Benchmark reproduction

```bash
# Requires the 42-clip training dataset
cd ..
PYTHONPATH=. python experiments/benchmark_full.py
```

Measured result: **F1 = 61.22%** (15 TP, 12 FP, 7 FN). See [`../docs/final_benchmark.txt`](../docs/final_benchmark.txt) for the full per-video log.

## Script inventory

| Script | Purpose |
|---|---|
| `audit.py` | ffprobe profiling of all 42 videos |
| `benchmark_full.py` | Full pipeline benchmark (this is the one that produces F1 61.22%) |
| `benchmark.py` | Early latency-only benchmark |
| `evaluate.py` | Original full dataset evaluation |
| `evaluate_fast.py` | Fast iteration on failed videos |
| `check_device.py` | MPS/CUDA/CPU detection |
| `generate_rules.py` | Regulatory CSV generator (superseded by `violation_rules_verified.csv`) |
| `test_engine_v2.py` | Single-video smoke test |
| `test_moondream.py` | VLM prototype (rejected - see ADR-001) |
| `test_vlm.py` | VLM prototype (rejected) |
| `test_single.py` | Legacy single-clip test |
| `test_targets.py` | Detection target exploration |
| `test_yolo_zone.py` | Zone geometry exploration |
| `batch_test.py` | Multi-video batch runner |