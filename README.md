# WaWi Chatbot Admin Monitor

Live admin dashboard for monitoring your WaWi chatbot — conversations, SQL queries, errors, and moderation.

## Architecture

```
Users → Web Chat → FastAPI Chatbot → Ollama (localhost:11434)
                        ↓ (import monitor)
              Admin Dashboard (localhost:7779)
```

Everything runs on localhost — zero config, zero dependencies, just import and go.

## Quick Start

### 1. Start the monitor server

```bash
python chatbot-monitor/server.py
```

Dashboard: **http://localhost:7779**

### 2. Connect your chatbot (pick your style)

#### Option A: One-liner auto-logging with Ollama wrapper
```python
import monitor

# Calls Ollama AND logs to dashboard automatically
answer, ms = monitor.ollama_chat("Was kostet Artikel 4711?", user_id="user1")
```
That's it. The question, answer, timing, and model all appear on the dashboard.

#### Option B: FastAPI middleware (zero-touch)
```python
from fastapi import FastAPI
import monitor

app = FastAPI()
app.add_middleware(monitor.ChatMiddleware)   # <-- this one line

@app.post("/chat")
def chat(question: str, user_id: str = "anon"):
    answer = your_llm_call(question)
    return {"answer": answer}
# Every POST to /chat is auto-logged — question, answer, timing, errors
```

#### Option C: Manual logging (full control)
```python
import monitor

monitor.chat("user1", "Was kostet X?", "X kostet 29,90 EUR.", duration_ms=1200, model="llama3:8b")
monitor.query("SELECT price FROM articles WHERE id=4711", results=[{"price": 29.90}], duration_ms=8)
monitor.error("Ollama timeout", error_type="llm_timeout")
```

#### Option D: Decorator
```python
import monitor

@monitor.track_chat(model="llama3:8b")
def ask_bot(question: str, user_id: str = "anon") -> str:
    return call_ollama(question)
# Return value is auto-logged as the answer
```

## Wrappers (auto-log everything)

### Ollama
```python
# Chat API (messages format)
answer, ms = monitor.ollama_chat(
    "Wie viele Schrauben M8 sind auf Lager?",
    model="llama3:8b",
    user_id="user1",
    system_prompt="Du bist ein WaWi-Assistent.",
)

# Generate API (simple prompt)
answer, ms = monitor.ollama_generate("Beschreibe Artikel 4711", user_id="user1")
```

### Database
```python
# SELECT — returns rows as list of dicts
rows, ms = monitor.db_query(cursor, "SELECT * FROM articles WHERE id=?", (4711,))
# rows = [{"id": 4711, "name": "Schraube M6", "price": 2.49}]

# INSERT/UPDATE/DELETE
ms = monitor.db_execute(cursor, "UPDATE articles SET price=? WHERE id=?", (19.99, 4711))
```

Both wrappers auto-log SQL, results, timing, and errors to the dashboard.

### FastAPI Middleware

```python
app.add_middleware(monitor.ChatMiddleware)
```

Auto-detects POST requests to `/chat`, `/api/chat`, `/ask`, `/api/ask`, `/message`, `/api/message`.

Custom field mapping:
```python
app.add_middleware(monitor.ChatMiddleware,
    paths=["/chat", "/api/v2/ask"],     # which endpoints to monitor
    question_field="query",              # JSON field name for question
    answer_field="response",             # JSON field name for answer
    user_id_field="session_id",          # JSON field name for user ID
)
```

## Full FastAPI Example

```python
from fastapi import FastAPI
import monitor
import sqlite3

app = FastAPI()
app.add_middleware(monitor.ChatMiddleware)
db = sqlite3.connect("wawi.db")

@app.on_event("startup")
def startup():
    monitor.system("startup", "WaWi Chatbot started")

@app.post("/chat")
def chat_endpoint(body: dict):
    user_id = body.get("user_id", "anon")
    question = body.get("question", "")

    # Query DB — auto-logged
    rows, _ = monitor.db_query(
        db.cursor(),
        "SELECT * FROM articles WHERE name LIKE ?",
        (f"%{question}%",),
        user_id=user_id,
    )

    # Call Ollama — auto-logged
    context = f"DB results: {rows}"
    answer, _ = monitor.ollama_chat(
        f"{question}\n\nContext: {context}",
        user_id=user_id,
        system_prompt="Du bist ein WaWi-Assistent. Antworte auf Deutsch.",
    )

    return {"answer": answer, "user_id": user_id}
```

## Dashboard

4-panel layout:

| Live Feed | Chat Detail | SQL Inspector | Stats & Moderation |
|-----------|------------|---------------|-------------------|
| All events, filterable | Q&A bubbles + moderation | Syntax-highlighted SQL + result tables | Users, response times, errors, topics |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/event` | Ingest an event |
| GET | `/api/stream` | SSE event stream |
| GET | `/api/events?type=chat&user_id=x&limit=100` | Query events |
| GET | `/api/conversations` | Group chats by user |
| GET | `/api/stats` | Dashboard statistics |
| POST | `/api/moderate/<id>` | Flag/review/note an event |
| GET | `/api/health` | Health check |

## Configuration

Default: everything on localhost, zero config needed.

| Setting | Default | Override |
|---------|---------|----------|
| Monitor server | `localhost:7779` | `monitor.init("http://other:7779")` |
| Ollama | `localhost:11434` | `monitor.ollama_chat(..., ollama_url="http://other:11434")` |
| Max events | 5000 | `MAX_EVENTS` in `server.py` |
| Max result rows | 50 | `monitor.init(max_result_rows=100)` |
| Disable in tests | — | `monitor.disable()` |

## Zero Dependencies

Both `server.py` and `monitor.py` use only Python stdlib. No pip install needed.
