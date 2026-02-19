"""WaWi Chatbot Monitor — Python SDK.

Zero-dependency, fire-and-forget monitoring for your FastAPI chatbot.
All calls are non-blocking (background thread) and never crash the host app.
Auto-connects to localhost:7779 — no configuration needed for local setups.

=== QUICKSTART (1 line) ===

    import monitor                       # auto-connects to localhost:7779

=== FASTAPI MIDDLEWARE (2 lines) ===

    import monitor
    app.add_middleware(monitor.JsonMiddleware)  # auto-logs all POST /chat requests

=== MANUAL LOGGING ===

    monitor.chat(user_id, question, answer, duration_ms, model="llama3:8b")
    monitor.query(sql, results, duration_ms, user_id=user_id)
    monitor.error("Ollama timeout", error_type="llm_timeout")

=== OLLAMA WRAPPER ===

    answer, ms = monitor.ollama_chat(prompt, model="llama3:8b")

=== DB WRAPPER ===

    rows, ms = monitor.db_query(cursor, "SELECT * FROM articles WHERE id=?", (4711,))
"""

import json
import time
import uuid
import threading
import functools
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Auto-connect to localhost:7779 on import
# ---------------------------------------------------------------------------
_server_url: str = "http://localhost:7779"
_enabled: bool = True
_max_result_rows: int = 50


def init(server_url: str = "http://localhost:7779", max_result_rows: int = 50):
    """Override the default server URL. Usually not needed for local setups."""
    global _server_url, _enabled, _max_result_rows
    _server_url = server_url.rstrip("/")
    _max_result_rows = max_result_rows
    _enabled = True


def disable():
    """Disable monitoring (e.g. in tests)."""
    global _enabled
    _enabled = False


def enable():
    """Re-enable monitoring."""
    global _enabled
    _enabled = True


# ---------------------------------------------------------------------------
# Internal: fire-and-forget sender
# ---------------------------------------------------------------------------
def _send(event: dict):
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


# ---------------------------------------------------------------------------
# Core logging — short names for convenience
# ---------------------------------------------------------------------------
def chat(
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


def query(
    sql: str,
    results: list | None = None,
    duration_ms: float = 0,
    user_id: str = "",
    error: str = "",
    metadata: dict | None = None,
):
    """Log a database query."""
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
        if len(results) > _max_result_rows:
            evt["results"] = results[:_max_result_rows]
            evt["results_truncated"] = True
            evt["results_total"] = len(results)
        else:
            evt["results"] = results
    if metadata:
        evt["metadata"] = metadata
    _send(evt)


def error(
    message: str,
    error_type: str = "error",
    user_id: str = "",
    details: str = "",
    metadata: dict | None = None,
):
    """Log an error."""
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


def system(
    action: str,
    message: str = "",
    metadata: dict | None = None,
):
    """Log a system event (startup, shutdown, config change)."""
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


# Keep old names as aliases for backwards compat
log_chat = chat
log_query = query
log_error = error
log_system = system


# ---------------------------------------------------------------------------
# Ollama wrapper — call Ollama and auto-log the chat
# ---------------------------------------------------------------------------
def ollama_chat(
    prompt: str,
    model: str = "llama3:8b",
    user_id: str = "",
    ollama_url: str = "http://localhost:11434",
    system_prompt: str = "",
    temperature: float | None = None,
) -> tuple[str, float]:
    """Call Ollama and auto-log to monitor. Returns (answer, duration_ms).

    Usage:
        answer, ms = monitor.ollama_chat("Was kostet Artikel 4711?", user_id="u1")
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    body: dict = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        body["options"] = {"temperature": temperature}

    start = time.time()
    try:
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{ollama_url}/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=120)
        result = json.loads(resp.read())
        answer = result.get("message", {}).get("content", "")
        duration_ms = (time.time() - start) * 1000

        chat(user_id=user_id, question=prompt, answer=answer,
             duration_ms=duration_ms, model=model)
        return answer, duration_ms

    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        error(str(e), error_type="ollama_error", user_id=user_id,
              details=f"model={model}, prompt_len={len(prompt)}")
        raise


def ollama_generate(
    prompt: str,
    model: str = "llama3:8b",
    user_id: str = "",
    ollama_url: str = "http://localhost:11434",
    system_prompt: str = "",
) -> tuple[str, float]:
    """Call Ollama /api/generate and auto-log. Returns (response, duration_ms)."""
    body: dict = {"model": model, "prompt": prompt, "stream": False}
    if system_prompt:
        body["system"] = system_prompt

    start = time.time()
    try:
        data = json.dumps(body).encode("utf-8")
        req = Request(
            f"{ollama_url}/api/generate",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=120)
        result = json.loads(resp.read())
        answer = result.get("response", "")
        duration_ms = (time.time() - start) * 1000

        chat(user_id=user_id, question=prompt, answer=answer,
             duration_ms=duration_ms, model=model)
        return answer, duration_ms

    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        error(str(e), error_type="ollama_error", user_id=user_id)
        raise


# ---------------------------------------------------------------------------
# DB query wrapper — execute SQL and auto-log
# ---------------------------------------------------------------------------
def db_query(
    cursor,
    sql: str,
    params: tuple = (),
    user_id: str = "",
) -> tuple[list[dict], float]:
    """Execute a DB query and auto-log to monitor. Returns (rows_as_dicts, duration_ms).

    Usage:
        rows, ms = monitor.db_query(cursor, "SELECT * FROM articles WHERE id=?", (4711,))
    """
    start = time.time()
    try:
        cursor.execute(sql, params)
        columns = [desc[0] for desc in cursor.description] if cursor.description else []
        raw_rows = cursor.fetchall()
        rows = [dict(zip(columns, row)) for row in raw_rows]
        duration_ms = (time.time() - start) * 1000

        query(sql=sql, results=rows, duration_ms=duration_ms, user_id=user_id)
        return rows, duration_ms

    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        query(sql=sql, error=str(e), duration_ms=duration_ms, user_id=user_id)
        raise


def db_execute(
    cursor,
    sql: str,
    params: tuple = (),
    user_id: str = "",
) -> float:
    """Execute a non-SELECT SQL statement and auto-log. Returns duration_ms.

    Usage:
        ms = monitor.db_execute(cursor, "UPDATE articles SET price=? WHERE id=?", (19.99, 4711))
    """
    start = time.time()
    try:
        cursor.execute(sql, params)
        duration_ms = (time.time() - start) * 1000
        query(sql=sql, duration_ms=duration_ms, user_id=user_id,
              metadata={"rowcount": cursor.rowcount})
        return duration_ms
    except Exception as e:
        duration_ms = (time.time() - start) * 1000
        query(sql=sql, error=str(e), duration_ms=duration_ms, user_id=user_id)
        raise


# ---------------------------------------------------------------------------
# FastAPI middleware — auto-log JSON chat endpoints
# ---------------------------------------------------------------------------
class ChatMiddleware:
    """ASGI middleware that auto-logs chat requests to the monitor.

    Intercepts POST requests to configurable paths and logs the
    question/answer + timing automatically. Zero config needed.

    Usage:
        from fastapi import FastAPI
        import monitor

        app = FastAPI()
        app.add_middleware(monitor.ChatMiddleware)

    Or with custom config:
        app.add_middleware(monitor.ChatMiddleware,
            paths=["/chat", "/api/ask"],
            question_field="query",
            answer_field="response",
            user_id_field="session_id",
        )
    """

    def __init__(
        self,
        app,
        paths: list[str] | None = None,
        question_field: str = "question",
        answer_field: str = "answer",
        user_id_field: str = "user_id",
        model_field: str = "model",
    ):
        self.app = app
        self.paths = set(paths or ["/chat", "/api/chat", "/ask", "/api/ask", "/message", "/api/message"])
        self.question_field = question_field
        self.answer_field = answer_field
        self.user_id_field = user_id_field
        self.model_field = model_field

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "")

        if method != "POST" or path not in self.paths:
            await self.app(scope, receive, send)
            return

        # Capture request body
        request_body = b""
        request_complete = False

        async def receive_wrapper():
            nonlocal request_body, request_complete
            message = await receive()
            if message["type"] == "http.request":
                request_body += message.get("body", b"")
                if not message.get("more_body", False):
                    request_complete = True
            return message

        # Capture response body
        response_body = b""
        response_status = 200

        async def send_wrapper(message):
            nonlocal response_body, response_status
            if message["type"] == "http.response.start":
                response_status = message.get("status", 200)
            elif message["type"] == "http.response.body":
                response_body += message.get("body", b"")
            await send(message)

        start = time.time()
        await self.app(scope, receive_wrapper, send_wrapper)
        duration_ms = (time.time() - start) * 1000

        # Parse and log
        try:
            req = json.loads(request_body) if request_body else {}
            res = json.loads(response_body) if response_body else {}

            question = req.get(self.question_field, "")
            answer = res.get(self.answer_field, "")
            user_id = req.get(self.user_id_field, "") or res.get(self.user_id_field, "")
            model_name = res.get(self.model_field, "") or req.get(self.model_field, "")

            if question:
                if response_status >= 400:
                    error(
                        f"HTTP {response_status} on {path}",
                        error_type="http_error",
                        user_id=user_id,
                        details=response_body.decode("utf-8", errors="replace")[:500],
                    )
                else:
                    chat(
                        user_id=user_id or "anonymous",
                        question=question,
                        answer=answer,
                        duration_ms=duration_ms,
                        model=model_name,
                    )
        except Exception:
            pass  # Never crash the host app


# ---------------------------------------------------------------------------
# Decorator — wrap any function to auto-log as chat
# ---------------------------------------------------------------------------
def track_chat(
    model: str = "",
    user_id_arg: str = "user_id",
    question_arg: str = "question",
):
    """Decorator that auto-logs the function's input/output as a chat event.

    Usage:
        @monitor.track_chat(model="llama3:8b")
        def ask_bot(question: str, user_id: str = "anon") -> str:
            return ollama.chat(question)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                # Try to extract question and user_id from kwargs or args
                q = kwargs.get(question_arg, args[0] if args else "")
                uid = kwargs.get(user_id_arg, "")
                answer_text = result if isinstance(result, str) else str(result)
                chat(user_id=uid or "anonymous", question=str(q),
                     answer=answer_text, duration_ms=duration_ms, model=model)
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                uid = kwargs.get(user_id_arg, "")
                error(str(e), error_type="function_error", user_id=uid,
                      details=f"function={func.__name__}")
                raise
        return wrapper
    return decorator


def track_query(user_id_arg: str = "user_id"):
    """Decorator that auto-logs a function's SQL query execution.

    Usage:
        @monitor.track_query()
        def get_article_price(article_id: int, user_id: str = "") -> list:
            cursor.execute("SELECT price FROM articles WHERE id=?", (article_id,))
            return cursor.fetchall()
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start) * 1000
                uid = kwargs.get(user_id_arg, "")
                query(sql=f"[{func.__name__}]", results=result if isinstance(result, list) else None,
                      duration_ms=duration_ms, user_id=uid)
                return result
            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                uid = kwargs.get(user_id_arg, "")
                query(sql=f"[{func.__name__}]", error=str(e),
                      duration_ms=duration_ms, user_id=uid)
                raise
        return wrapper
    return decorator
