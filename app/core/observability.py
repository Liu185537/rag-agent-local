from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from threading import Lock


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MetricsCollector:
    total_requests: int = 0
    total_errors: int = 0
    total_latency_ms: float = 0.0
    by_path: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_status: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    last_updated_at: str = field(default_factory=utc_now)

    def __post_init__(self) -> None:
        self._lock = Lock()

    def record(self, path: str, status_code: int, latency_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            if status_code >= 500:
                self.total_errors += 1
            self.total_latency_ms += latency_ms
            self.by_path[path] += 1
            self.by_status[str(status_code)] += 1
            self.last_updated_at = utc_now()

    def snapshot(self) -> dict[str, object]:
        with self._lock:
            avg_latency = (
                self.total_latency_ms / self.total_requests if self.total_requests else 0.0
            )
            error_rate = self.total_errors / self.total_requests if self.total_requests else 0.0
            top_paths = sorted(self.by_path.items(), key=lambda x: x[1], reverse=True)[:10]
            return {
                "total_requests": self.total_requests,
                "total_errors": self.total_errors,
                "error_rate": round(error_rate, 6),
                "avg_latency_ms": round(avg_latency, 3),
                "by_status": dict(self.by_status),
                "top_paths": [{"path": p, "count": c} for p, c in top_paths],
                "last_updated_at": self.last_updated_at,
            }

