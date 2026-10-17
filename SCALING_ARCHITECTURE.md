# Scaling to 1 Million Users: From MVP to Production

The current submission is a fully functional **hackathon MVP**. However, the architecture was designed with production constraints in mind. Below is the roadmap for scaling this exact system to serve 1,000,000+ concurrent users across a national banking network.

## 1. Compute & Inference Scaling
- **Current:** YOLOv8n running on CPU/MPS inside a Docker container.
- **Production:** 
  - Separate the FastAPI web server from the YOLO inference workers.
  - Deploy inference workers on Kubernetes (EKS/GKE) with NVIDIA T4 GPUs.
  - Use **NVIDIA Triton Inference Server** for dynamic batching. This allows us to process 32 videos simultaneously on a single GPU, reducing cost per inference by ~85%.
  - **Queue System:** Use **Celery + Redis** (or AWS SQS). When a user uploads a video, it drops into the queue. Workers pull from the queue. This prevents server timeouts during traffic spikes.

## 2. Database & State Management
- **Current:** SQLite (persisted to disk).
- **Production:**
  - **PostgreSQL (RDS Aurora):** A managed, multi-AZ database for storing job metadata and compliance reports.
  - **S3 Object Storage:** Videos and evidence frames are uploaded directly to AWS S3 via presigned URLs. We never store media on the application server.
  - **Redis:** For caching the `/history` endpoint and managing rate limits globally across all API instances.

## 3. Security & Abuse Prevention
- **Current:** API Key auth + SlowAPI rate limiting (5/min).
- **Production:**
  - **OAuth2 / OIDC:** Integrate with ForteBank's Active Directory (SSO) for employees.
  - **RBAC (Role-Based Access Control):** Cashiers can only see their own desk. Branch managers see their branch. Regional managers see the region.
  - **WAF & Cloudflare:** Deploy behind Cloudflare to absorb DDoS attacks and block malicious bot traffic.
  - **Secrets Management:** Use AWS Secrets Manager or HashiCorp Vault for API keys and DB credentials.

## 4. Observability & Error Handling
- **Current:** Prometheus `/metrics` endpoint exposed.
- **Production:**
  - **Prometheus + Grafana:** Monitor HTTP latency, error rates, queue depth, and GPU utilization.
  - **OpenTelemetry:** Distributed tracing. Track a single video's journey from upload -> queue -> inference -> DB write.
  - **Sentry:** Real-time exception tracking.
  - **Alerting:** PagerDuty integration. If the inference queue depth exceeds 1,000 items for >5 minutes, page the on-call engineer.

## 5. CI/CD & Deployment
- **Current:** Manual Docker build.
- **Production:**
  - **GitHub Actions:** Automated testing (pytest) on every PR.
  - **Terraform:** Infrastructure-as-code for VPC, EKS cluster, and RDS.
  - **ArgoCD:** GitOps-based blue-green deployments with automated rollback.
