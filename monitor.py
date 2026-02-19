"""WaWi Chatbot Monitor — Python SDK.

Zero-dependency, fire-and-forget monitoring for your FastAPI chatbot.
All calls are non-blocking (background thread) and never crash the host app.

Usage:
    from chatbot_monitor import monitor
    monitor.init("http://localhost:7779")
    monitor.log_chat(user_id="u123", question="Was kostet X?", answer="...", duration_ms=340)
    monitor.log_query(sql="SELECT ...", results=[...], duration_ms=12, user_id="u123")
    monitor.log_error("Ollama timeout", error_type="llm_timeout")
    monitor.log_system("startup", "Chatbot v2.1 started")
"""

import json
import time
import uuid
import threading
from urllib.request import Request, urlopen
from urllib.error import URLError

_server_url: str = ""
_enabled: bool = False
_max_result_rows: int = 50


def init(server_url: str = "http://localhost:7779", max_result_rows: int = 50):
    """Initialize the monitor with the admin server URL."""
    global _server_url, _enabled, _max_result_rows
    _server_url = server_url.rstrip("/")
    _max_result_rows = max_result_rows
    _enabled = True


def _send(event: dict):
    """Send event to the monitor server in a background thread (fire-and-forget)."""
    if not _enabled:
        return

    def _post():
        try:
            data = json.dumps(event, ensure_ascii=False, default=str).encode("utf-8")
            req = Request(
                f"{_server_url}/event",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urlopen(req, timeout=2)
        except Exception:
            pass  # Never crash the host app

    threading.Thread(target=_post, daemon=True).start()


def log_chat(
    user_id: str,
    question: str,
    answer: str,
    duration_ms: float = 0,
    model: str = "",
    metadata: dict | None = None,
):
    """Log a chat interaction (user question + bot answer)."""
    evt = {
        "id": str(uuid.uuid4()),
        "type": "chat",
        "timestamp": time.time(),
        "user_id": user_id,
        "question": question,
        "answer": answer,
        "duration_ms": round(duration_ms, 1),
    }
    if model:
        evt["model"] = model
    if metadata:
        evt["metadata"] = metadata
    _send(evt)


def log_query(
    sql: str,
    results: list | None = None,
    duration_ms: float = 0,
    user_id: str = "",
    error: str = "",
    metadata: dict | None = None,
):
    """Log a database query executed by the chatbot."""
    evt = {
        "id": str(uuid.uuid4()),
        "type": "query",
        "timestamp": time.time(),
        "sql": sql,
        "duration_ms": round(duration_ms, 1),
    }
    if user_id:
        evt["user_id"] = user_id
    if error:
        evt["error"] = error
    if results is not None:
        # Truncate large result sets
        if len(results) > _max_result_rows:
            evt["results"] = results[:_max_result_rows]
            evt["results_truncated"] = True
            evt["results_total"] = len(results)
        else:
            evt["results"] = results
    if metadata:
        evt["metadata"] = metadata
    _send(evt)


def log_error(
    message: str,
    error_type: str = "error",
    user_id: str = "",
    details: str = "",
    metadata: dict | None = None,
):
    """Log an error that occurred in the chatbot."""
    evt = {
        "id": str(uuid.uuid4()),
        "type": "error",
        "timestamp": time.time(),
        "message": message,
        "error_type": error_type,
    }
    if user_id:
        evt["user_id"] = user_id
    if details:
        evt["details"] = details
    if metadata:
        evt["metadata"] = metadata
    _send(evt)


def log_system(
    action: str,
    message: str = "",
    metadata: dict | None = None,
):
    """Log a system event (startup, shutdown, config change, etc.)."""
    evt = {
        "id": str(uuid.uuid4()),
        "type": "system",
        "timestamp": time.time(),
        "action": action,
    }
    if message:
        evt["message"] = message
    if metadata:
        evt["metadata"] = metadata
    _send(evt)
