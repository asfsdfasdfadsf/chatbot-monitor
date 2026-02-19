# WaWi Chatbot Admin Monitor

Live admin dashboard for monitoring your WaWi chatbot — conversations, SQL queries, errors, and moderation.

## Architecture

```
Users → Web Chat → FastAPI Chatbot → Ollama
                        ↓ (monitor SDK)
                   POST /event → server.py (port 7779)
                        ↓ (SSE)
              Admin Dashboard http://localhost:7779
```

## Quick Start

### 1. Start the monitor server

```bash
cd chatbot-monitor
python server.py
```

Server runs on **http://localhost:7779** — open in browser for the dashboard.

### 2. Integrate the SDK in your FastAPI chatbot

```python
from chatbot_monitor import monitor

# On startup
monitor.init("http://localhost:7779")

# Log a chat interaction
monitor.log_chat(
    user_id="user123",
    question="Was kostet Artikel 4711?",
    answer="Artikel 4711 kostet 29,90 EUR.",
    duration_ms=1240,
    model="llama3:8b"
)

# Log a database query
monitor.log_query(
    sql="SELECT price FROM articles WHERE id = 4711",
    results=[{"price": 29.90}],
    duration_ms=12,
    user_id="user123"
)

# Log an error
monitor.log_error("Ollama timeout after 30s", error_type="llm_timeout", user_id="user123")

# Log a system event
monitor.log_system("startup", "Chatbot v2.1 started")
```

## Dashboard

4-panel layout:

| Live Feed | Chat Detail | SQL Inspector | Stats & Moderation |
|-----------|------------|---------------|-------------------|
| All events, filterable | Q&A bubbles + moderation | Syntax-highlighted SQL + result tables | Users, response times, errors, topics |

### Features

- **Real-time SSE** — events appear instantly
- **Filtering** — by type (chat/query/error) or flagged items
- **Moderation** — flag, review, and add notes to any event
- **SQL Inspector** — syntax highlighting, result tables, timing badges
- **Stats** — active users, response times, error rates, hourly volume, top topics
- **Persistence** — events saved to `data/events.jsonl`, survive restarts

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/event` | Ingest an event |
| GET | `/api/stream` | SSE event stream |
| GET | `/api/events?type=chat&user_id=x&limit=100` | Query events |
| GET | `/api/conversations` | Group chats by user |
| GET | `/api/stats` | Dashboard statistics |
| POST | `/api/moderate/<id>` | Flag/review/note an event |
| GET | `/api/moderation` | All moderation state |
| GET | `/api/health` | Health check |

## Event Types

### chat
```json
{"type": "chat", "user_id": "u1", "question": "...", "answer": "...", "duration_ms": 1200, "model": "llama3:8b"}
```

### query
```json
{"type": "query", "sql": "SELECT ...", "results": [...], "duration_ms": 15, "user_id": "u1"}
```

### error
```json
{"type": "error", "message": "Ollama timeout", "error_type": "llm_timeout", "user_id": "u1"}
```

### system
```json
{"type": "system", "action": "startup", "message": "Chatbot started"}
```

## Configuration

- **Port**: 7779 (change `PORT` in `server.py`)
- **Max events**: 5000 ring buffer
- **Max result rows**: 50 (SDK truncates larger result sets)
- **Persistence**: `data/events.jsonl` (auto-created)

## Zero Dependencies

Both `server.py` and `monitor.py` use only Python stdlib. No pip install needed.
