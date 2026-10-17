"""Production hardening: logging, correlation, circuit breaker, retry, SLOs, audit."""
import json, time, uuid, logging, threading, functools, sqlite3
from contextvars import ContextVar
from typing import Callable
from datetime import datetime, timezone

# ============ CORRELATION ID ============
trace_id_var: ContextVar[str] = ContextVar("trace_id", default="")

def new_trace_id() -> str:
    tid = str(uuid.uuid4()); trace_id_var.set(tid); return tid

def current_trace_id() -> str:
    return trace_id_var.get()

# ============ STRUCTURED JSON LOGGING ============
class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": current_trace_id(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            payload.update(record.extra)
        return json.dumps(payload, ensure_ascii=False)

handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
root = logging.getLogger(); root.handlers = [handler]; root.setLevel(logging.INFO)
log = logging.getLogger("forte")

# ============ SLO TRACKING ============
SLO_TARGETS = {"p99_latency_ms": 30000, "error_rate": 0.01, "availability": 0.999}

class SLOTracker:
    def __init__(self):
        self.lock = threading.Lock(); self.total = 0; self.errors = 0; self.latencies_ms = []
    def record(self, latency_ms: float, is_error: bool):
        with self.lock:
            self.total += 1
            if is_error: self.errors += 1
            self.latencies_ms.append(latency_ms)
            if len(self.latencies_ms) > 1000: self.latencies_ms = self.latencies_ms[-1000:]
    def snapshot(self):
        with self.lock:
            if not self.latencies_ms:
                return {"total": self.total, "errors": self.errors, "p99_ms": 0, "error_rate": 0, "slo_targets": SLO_TARGETS}
            s = sorted(self.latencies_ms)
            p99 = s[int(len(s) * 0.99)] if len(s) > 100 else s[-1]
            er = self.errors / max(self.total, 1)
            return {
                "total": self.total, "errors": self.errors,
                "p99_ms": round(p99, 1), "error_rate": round(er, 4),
                "slo_targets": SLO_TARGETS,
                "p99_ok": p99 <= SLO_TARGETS["p99_latency_ms"],
                "error_rate_ok": er <= SLO_TARGETS["error_rate"],
            }

slo = SLOTracker()

# ============ CIRCUIT BREAKER ============
class CircuitBreaker:
    def __init__(self, name: str, failure_threshold: int = 3, recovery_timeout: float = 60.0):
        self.name = name; self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0; self.state = "closed"; self.opened_at = 0.0
        self.lock = threading.Lock()
    def call(self, fn: Callable, *args, **kwargs):
        with self.lock:
            if self.state == "open":
                if time.time() - self.opened_at > self.recovery_timeout:
                    self.state = "half_open"
                    log.warning("circuit_half_open", extra={"breaker": self.name})
                else:
                    raise RuntimeError(f"CircuitBreaker[{self.name}] OPEN - degraded")
        try:
            result = fn(*args, **kwargs)
            with self.lock:
                if self.state == "half_open":
                    self.state = "closed"; self.failures = 0
                    log.info("circuit_closed", extra={"breaker": self.name})
            return result
        except Exception:
            with self.lock:
                self.failures += 1
                if self.failures >= self.failure_threshold:
                    self.state = "open"; self.opened_at = time.time()
                    log.error("circuit_opened", extra={"breaker": self.name, "failures": self.failures})
            raise
    def snapshot(self):
        return {"name": self.name, "state": self.state, "failures": self.failures}

# ============ RETRY ============
def retry(max_attempts=3, base_delay=1.0, exceptions=(Exception,)):
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            delay = base_delay
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        log.error("retry_exhausted", extra={"fn": fn.__name__, "attempts": attempt})
                        raise
                    log.warning("retry_attempt", extra={"fn": fn.__name__, "attempt": attempt, "delay_s": delay})
                    time.sleep(delay); delay *= 2
        return wrapper
    return decorator

# ============ AUDIT LOG ============
AUDIT_DB = "audit.db"

def init_audit():
    conn = sqlite3.connect(AUDIT_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL, trace_id TEXT, actor TEXT, action TEXT NOT NULL,
        resource TEXT, payload TEXT, ip TEXT
    )""")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_ts ON audit_log(ts DESC)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)")
    conn.commit(); conn.close()

def audit(actor: str, action: str, resource: str = "", payload: dict = None, ip: str = ""):
    conn = sqlite3.connect(AUDIT_DB)
    conn.execute(
        "INSERT INTO audit_log (ts, trace_id, actor, action, resource, payload, ip) VALUES (?,?,?,?,?,?,?)",
        (datetime.now(timezone.utc).isoformat(), current_trace_id(), actor, action,
         resource, json.dumps(payload or {}, ensure_ascii=False), ip))
    conn.commit(); conn.close()

init_audit()

# ============ TIMER ============
class timer:
    def __init__(self, label: str):
        self.label = label; self.start = 0; self.elapsed_ms = 0
    def __enter__(self):
        self.start = time.perf_counter(); return self
    def __exit__(self, *a):
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000
        log.info("timing", extra={"label": self.label, "ms": round(self.elapsed_ms, 1)})
