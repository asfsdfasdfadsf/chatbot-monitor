# WaWi Chatbot Admin Monitor

Live admin dashboard for monitoring your WaWi chatbot — conversations, SQL queries, errors, moderation, user management, and authentication.

## Architecture

```
Users → Web Chat → FastAPI Chatbot → LLM Provider
                        ↓ (import monitor)       ↑
              Admin Dashboard (localhost:7779) ───┘
                  ↓ (cookie auth, roles)
            Admin: full control        Providers:
            User: chat + own events    • Ollama (local)
                                       • OpenAI (ChatGPT)
                                       • Anthropic (Claude)
```

Everything runs on localhost — zero config, zero dependencies, just import and go.

## Features

- **Multi-provider LLM** — switch between Ollama (local), OpenAI (ChatGPT), and Anthropic (Claude)
- **Live event feed** — chat, query, error, and system events in real-time via SSE
- **Built-in chat** — proxy questions to any configured LLM provider from the dashboard
- **SQL inspector** — syntax-highlighted queries with result tables
- **Moderation** — flag, review, and annotate chat events
- **User management** — block/unblock chatbot end-users, set priorities
- **Authentication** — cookie-based login/register with session management
- **Role-based access** — admin (full control) vs user (chat + own events)
- **Account management** — create accounts, change roles and priorities, delete accounts
- **Bot kill switch** — admin can globally disable the bot with one toggle
- **User priorities** — high/normal/low priority on both dashboard accounts and chatbot end-users
- **Python SDK** — fire-and-forget logging, Ollama wrapper, DB wrapper, FastAPI middleware
- **Zero dependencies** — pure Python stdlib, no pip install needed

## Quick Start

### 1. Start the monitor server

```bash
python chatbot-monitor/server.py
```

Dashboard: **http://localhost:7779**

On first startup, a default admin account is created:
- **Username:** `admin`
- **Password:** `admin`

Change this password or create a new admin account after first login.

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

### 3. Check user status from your chatbot

```python
import monitor

# Simple block check
if not monitor.is_user_allowed("user123"):
    return "You have been blocked by an administrator."

# Full status (allowed, priority, bot_enabled)
status = monitor.get_user_status("user123")
if not status["bot_enabled"]:
    return "The bot is currently disabled for maintenance."
if not status["allowed"]:
    return "You have been blocked."
print(f"User priority: {status['priority']}")  # high, normal, or low
```

## Authentication & Roles

### Login/Register

The dashboard requires authentication. On page load, users see a login screen with login and register tabs.

- **Admin** — full control: all events, moderation, user management, settings, accounts, bot toggle
- **User** — limited view: chat tab, own events in the feed, no access to admin features

Sessions use HTTP cookies (24h expiry). No HTTPS required for LAN deployments.

### Default Admin

On first startup (when no admin account exists), the server creates:
- Username: `admin`, Password: `admin`

### Account Management (Admin)

Admins can:
- View all dashboard accounts
- Change account roles (admin/user)
- Set account priorities (high/normal/low)
- Delete accounts (cannot delete self or last admin)

### Bot Kill Switch

Admins can globally disable the bot via the dashboard or config API. When disabled:
- `POST /event` returns 503
- `POST /api/chat` returns 503
- `GET /api/users/<id>/check` reports `bot_enabled: false`
- SDK `get_user_status()` reflects the disabled state

## LLM Providers

The dashboard supports three LLM providers, configurable in Settings:

| Provider | Models | Auth |
|----------|--------|------|
| **Ollama** (default) | Any local model (llama3, mistral, etc.) | No key needed |
| **OpenAI** | GPT-4o, GPT-4, GPT-3.5-turbo, o3-mini, etc. | API key required |
| **Anthropic** | Claude Sonnet 4, Claude Haiku 4.5, Claude 3.5 Sonnet, etc. | API key required |

**Setup**: Go to Settings (admin only) → select provider → enter API key → select model → Save.

**OpenAI-compatible providers**: The OpenAI Base URL can be changed to use compatible APIs like Groq (`https://api.groq.com/openai/v1`), Together.ai, Azure OpenAI, or any OpenAI-compatible endpoint.

**API key security**: Keys are stored in `data/config.json` on disk. The `GET /api/config` endpoint returns masked keys (e.g. `...abcd`) — full keys are never sent to the browser.

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

    # Check if user is allowed
    if not monitor.is_user_allowed(user_id):
        return {"error": "You have been blocked."}

    # Check full status (including bot toggle)
    status = monitor.get_user_status(user_id)
    if not status["bot_enabled"]:
        return {"error": "Bot is currently disabled."}

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

4-panel layout with role-based visibility:

| Live Feed | Chat Detail | SQL Inspector | Stats & Moderation |
|-----------|------------|---------------|-------------------|
| All events, filterable | Q&A bubbles + moderation | Syntax-highlighted SQL + result tables | Users, response times, errors, topics |

**Admin sees:** All tabs, settings, user management, accounts panel, bot toggle, moderation controls, clear button

**User sees:** Chat tab, own events in feed

## API Endpoints

### Public (no auth required)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Health check (status, event/user counts, uptime) |
| POST | `/event` | Ingest an event from SDK |
| GET | `/api/users/<id>/check` | Check if user is allowed + priority + bot status |
| POST | `/api/auth/login` | Login with `{username, password}` |
| POST | `/api/auth/register` | Register with `{username, password}` |
| POST | `/api/auth/logout` | Logout (clears session cookie) |
| GET | `/api/auth/me` | Get current session user info |

### User-level (requires login)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/stream` | SSE event stream (filtered by role) |
| POST | `/api/chat` | Send question to Ollama |
| GET | `/api/models` | List available Ollama models |

### Admin-level (requires admin role)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events` | Query events (`?type=chat&user_id=x&limit=100&flagged=true`) |
| GET | `/api/conversations` | Group chats by user |
| GET | `/api/stats` | Dashboard statistics |
| POST | `/api/moderate/<id>` | Flag/review/note an event |
| GET | `/api/moderation` | Get all moderation data |
| GET/POST | `/api/config` | Get/update server config (including `bot_enabled`) |
| GET | `/api/users` | List all chatbot end-users |
| GET | `/api/users/<id>` | Get user detail with recent events |
| POST | `/api/users/<id>/block` | Block a chatbot end-user |
| POST | `/api/users/<id>/unblock` | Unblock a chatbot end-user |
| POST | `/api/users/<id>/update` | Update user note, status, or priority |
| GET | `/api/accounts` | List all dashboard accounts |
| POST | `/api/accounts/<name>/role` | Change account role (`{role: "admin"\|"user"}`) |
| POST | `/api/accounts/<name>/priority` | Change account priority (`{priority: "high"\|"normal"\|"low"}`) |
| POST | `/api/accounts/<name>/delete` | Delete a dashboard account |
| POST | `/api/clear` | Clear all events |

## SDK Reference

### Functions

| Function | Description |
|----------|-------------|
| `monitor.chat(user_id, question, answer, ...)` | Log a chat event |
| `monitor.query(sql, results, ...)` | Log a database query |
| `monitor.error(message, error_type, ...)` | Log an error |
| `monitor.system(action, message, ...)` | Log a system event |
| `monitor.ollama_chat(prompt, ...)` | Call Ollama + auto-log |
| `monitor.ollama_generate(prompt, ...)` | Call Ollama generate + auto-log |
| `monitor.db_query(cursor, sql, params, ...)` | Execute SELECT + auto-log |
| `monitor.db_execute(cursor, sql, params, ...)` | Execute INSERT/UPDATE/DELETE + auto-log |
| `monitor.is_user_allowed(user_id)` | Check if user is blocked (fail-open) |
| `monitor.get_user_status(user_id)` | Get `{allowed, priority, bot_enabled}` (fail-open) |
| `monitor.init(server_url)` | Override server URL |
| `monitor.disable()` / `monitor.enable()` | Toggle monitoring |

### Decorators

| Decorator | Description |
|-----------|-------------|
| `@monitor.track_chat(model="...")` | Auto-log function input/output as chat |
| `@monitor.track_query()` | Auto-log function as query |

### Aliases (backward compat)

`monitor.log_chat`, `monitor.log_query`, `monitor.log_error`, `monitor.log_system`

## Configuration

Default: everything on localhost, zero config needed.

| Setting | Default | Override |
|---------|---------|----------|
| LLM Provider | `ollama` | Dashboard settings (`"ollama"`, `"openai"`, `"anthropic"`) |
| Monitor server | `localhost:7779` | `monitor.init("http://other:7779")` |
| Ollama URL | `localhost:11434` | Dashboard settings or `monitor.ollama_chat(..., ollama_url="...")` |
| OpenAI API key | (empty) | Dashboard settings |
| OpenAI Base URL | `https://api.openai.com/v1` | Dashboard settings (change for Groq, Azure, etc.) |
| Anthropic API key | (empty) | Dashboard settings |
| Default model | `llama3:8b` | Dashboard settings |
| System prompt | (empty) | Dashboard settings |
| Temperature | `0.7` | Dashboard settings |
| Bot enabled | `true` | Dashboard settings or `POST /api/config` |
| Max events | 5000 | `MAX_EVENTS` in `server.py` |
| Max result rows | 50 | `monitor.init(max_result_rows=100)` |
| Session TTL | 24 hours | `SESSION_TTL` in `server.py` |
| Disable in tests | -- | `monitor.disable()` |

## Data Storage

All data stored in `data/` directory (git-ignored):

| File | Contents |
|------|----------|
| `data/events.jsonl` | All events (append-only JSONL) |
| `data/config.json` | Server configuration |
| `data/users.json` | Chatbot end-user profiles |
| `data/accounts.json` | Dashboard account credentials |

## Testing

```bash
# Start the server
python server.py

# Run the full test suite (in another terminal)
python test_all.py
```

The test suite covers 180 tests across 24 sections: server connectivity, authentication, role-based access control, event ingestion, event queries, conversations, stats, moderation, config, Ollama chat, SDK, user management, user permissions, user status, priorities, bot toggle, account management, persistence, SSE streaming, and static file serving.

## Zero Dependencies

Both `server.py` and `monitor.py` use only Python stdlib. No pip install needed.

## Files

| File | Description |
|------|-------------|
| `server.py` | HTTP server: auth, events, SSE, Ollama proxy, user/account management |
| `monitor.py` | Python SDK: fire-and-forget logging, wrappers, middleware |
| `public/index.html` | Dashboard UI: login, chat, feed, stats, admin panels |
| `test_all.py` | Comprehensive test suite (180 tests) |
