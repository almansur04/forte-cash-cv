# Production Readiness & Engineering Philosophy

A production system is the whole engineered artifact that stands between a product and its users. It is not merely code that runs without errors on a developer's machine, but an architecture that operates under real traffic, real failures, and real consequences. Structural decisions about how components are organized, how they communicate, and how they evolve must satisfy both functional requirements and non-functional demands such as response time, availability, security, and cost.

At the scale of millions of users, every layer-architecture, frontend and UI, design, backend, machine learning, computer vision, legal and banking controls, reliability, performance, and optimization-must be treated as a first-class engineering product in its own right.

Below is how **ForteBank Vision** maps to these production principles.

## 1. Frontend & UI (Performance and Trust-Critical Surfaces)
The frontend is not a cosmetic wrapper; it is a performance-critical, accessibility-critical, and trust-critical surface.
- **Design System:** Built with a custom-tailored Tailwind CSS configuration implementing the official ForteBank Brand Architecture (Forte Magenta `#A31551`, Premier Green, Corporate Blue).
- **Error States & Recovery:** Implements toast notifications, graceful failure fallbacks, and multi-step loading indicators (`Extracting frames → YOLOv8 → Compliance Rules`) so users are never stranded when a dependency falters.
- **Usability:** Drag-and-drop uploads, auto-seeking HTML5 video players for evidence review, and a Single Page Application (SPA) router for seamless Dashboard/Investigations/Regulatory navigation.

## 2. Backend (The Operational Core)
The backend is the system's operational core. At millions of users, even a small inefficiency or unprotected endpoint becomes a systemic risk.
- **Persistence & State:** Replaced in-memory prototypes with a persistent SQLite database (`jobs.db`), implementing idempotent API job tracking.
- **Rate Limiting & Abuse Prevention:** Implemented SlowAPI to enforce 10 requests/minute per IP. This protects the ML inference queue from DDoS attacks, infinite frontend loops, and scrapers.
- **Asynchronous Queues:** Heavy video processing is offloaded to `BackgroundTasks`, preventing server timeouts during high concurrency.
- **Security:** Enforced API-Key authentication for all `/api/v1` endpoints.

## 3. Machine Learning & Computer Vision in the Production Path
ML and CV increasingly sit *inside* the production path. They require model serving infrastructure, fallback rules, and human-in-the-loop review.
- **Fallback Rules (Zero Hallucinations):** We explicitly rejected heavy Vision-Language Models (VLMs) and Pose Estimation in favor of deterministic YOLOv8n inference. This guarantees that the system never hallucinates a violation on a clean video.
- **Evidence Generation:** The system generates annotated JPEG evidence frames (Green Desk Zone, Red Cashier Bounding Box) and preserves video timestamps for immediate human-in-the-loop compliance review.
- **Performance:** Processes a 2-minute video in 1.7 seconds on Apple M1 (9 seconds on CPU), well within the 5-minute SLA.

## 4. Legal, Regulatory & Banking Controls
Legal and regulatory concerns are first-class production requirements, not afterthoughts.
- **Regulatory Encoding:** Compliance rules are mapped directly to **NBRK Resolution No. 231, Article 22**. No generic or hallucinated citations are used.
- **Auditability:** Every violation includes exact `timecode_sec`, the regulatory article number, a direct URL to Adilet.zan.kz, and visual evidence.

## 5. Observability & Reliability
Reliability must be designed in rather than bolted on. Observability functions as a system contract that makes latency and dependency chains legible.
- **Metrics:** Integrated **Prometheus FastAPI Instrumentator**, exposing a `/metrics` endpoint for Grafana dashboards. Tracks HTTP latency, request counts, and error rates.
- **Health Checks:** Exposes `/api/v1/health` for Kubernetes liveness and readiness probes.
- **Logs:** Structured uvicorn logging with clear error boundaries and exception handling.

## 6. Deployment & CI/CD
- **Containerization:** Split into a cached `Dockerfile.base` (heavy ML dependencies) and a lightweight `Dockerfile` (app code). This reduces build times from 6 minutes to 2.7 seconds, enabling rapid blue-green deployments.
- **CPU-Optimized:** Fully deployable to standard Linux CPU instances, eliminating the requirement for costly NVIDIA GPU infrastructure in every bank branch.