# WaWi Chatbot Admin Monitor

Live admin dashboard for monitoring your WaWi chatbot — conversations, SQL queries, errors, moderation, user management, authentication, database connectivity, and RAG knowledge base.

## Architecture

```
Users → Web Chat → FastAPI Chatbot → LLM Provider
                        ↓ (import monitor)       ↑
              Admin Dashboard (localhost:7779) ───┘
                  ↓ (cookie auth, roles)
            Admin: full control          Providers:
            Mitarbeiter: knowledge+chat  • Ollama (local)
            User: chat only              • OpenAI (ChatGPT OAuth / API key)
                                         • Anthropic (Claude)
                                         • OpenRouter (OAuth, 200+ models)
                  ↓
            Database (MSSQL / SQLite)
            Knowledge Base (RAG embeddings)
```

Everything runs on localhost — zero config, zero external dependencies for the core server.

## Features

- **Multi-provider LLM** — switch between Ollama (local), OpenAI (ChatGPT), Anthropic (Claude), and OpenRouter (200+ models via OAuth)
- **Agent Mode** — LLM autonomously calls tools (DB queries, RAG search, web search, calculator, datetime, schema listing) in a loop until it has enough info to answer
- **OpenAI OAuth PKCE** — login with your ChatGPT account (Plus/Pro) instead of API keys
- **Chat memory** — session-based conversation history (max 20 messages per session)
- **Chat file upload** — attach .txt, .md, .csv, .json, .xlsx files directly in chat for analysis
- **Database connectivity** — connect to MSSQL (JTL-WaWi) or SQLite databases for NL-to-SQL queries
- **Knowledge Base (RAG)** — upload documents (.txt, .md, .csv, .xlsx), auto-chunk and embed, cosine similarity search injected into chat context
- **Live event feed** — chat, query, error, and system events in real-time via SSE
- **SQL inspector** — syntax-highlighted queries with result tables, admin can run raw SQL
- **Moderation** — flag, review, and annotate chat events
- **User management** — block/unblock chatbot end-users, set priorities
- **Authentication** — cookie-based login/register with session management
- **Three-tier roles** — admin (full control) / mitarbeiter (knowledge base + chat) / user (chat only)
- **Account management** — create accounts, change roles and priorities, delete accounts
- **Bot kill switch** — admin can globally disable the bot with one toggle
- **German by default** — AI responds in German unless the user writes in another language
- **Python SDK** — fire-and-forget logging, Ollama wrapper, DB wrapper, FastAPI middleware
- **Zero core dependencies** — pure Python stdlib server, optional `pyodbc` for MSSQL, `openpyxl` for Excel

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

### 2. Optional dependencies

```bash
pip install pyodbc     # For MSSQL database connections
pip install openpyxl   # For Excel file uploads (.xlsx)
```

Both are optional — the server runs without them but the respective features will be unavailable.

### 3. Connect your chatbot (pick your style)

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

### 4. Check user status from your chatbot

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

### Three-Tier Roles

| Role | Access |
|------|--------|
| **Admin** | Full control: all events, moderation, user management, settings, accounts, bot toggle, knowledge base, database config, raw SQL |
| **Mitarbeiter** | Chat, knowledge base (upload/manage documents), embedding stats, own events |
| **User** | Chat tab, own events in feed |

Sessions use HTTP cookies (24h expiry). No HTTPS required for LAN deployments.

### Default Admin

On first startup (when no admin account exists), the server creates:
- Username: `admin`, Password: `admin`

### Account Management (Admin)

Admins can:
- View all dashboard accounts
- Change account roles (admin/mitarbeiter/user)
- Set account priorities (high/normal/low)
- Delete accounts (cannot delete self or last admin)

### Bot Kill Switch

Admins can globally disable the bot via the dashboard or config API. When disabled:
- `POST /event` returns 503
- `POST /api/chat` returns 503
- `GET /api/users/<id>/check` reports `bot_enabled: false`
- SDK `get_user_status()` reflects the disabled state

## LLM Providers

The dashboard supports four LLM providers, configurable in Settings:

| Provider | Models | Auth |
|----------|--------|------|
| **Ollama** (default) | Any local model (llama3, mistral, etc.) | No key needed |
| **OpenAI** | GPT-4o, GPT-4, o3-mini, etc. | API key or ChatGPT OAuth login |
| **Anthropic** | Claude Sonnet 4, Claude Haiku 4.5, etc. | API key required |
| **OpenRouter** | GPT-4, Claude, Llama, 200+ models | OAuth login (no key needed) |

### OpenAI OAuth (Login with ChatGPT)

Instead of providing an API key, you can login with your ChatGPT account (Plus/Pro required):

1. Select "OpenAI" as provider in Settings
2. Click "Login with ChatGPT"
3. Authorize in your browser at `auth.openai.com`
4. You're connected — no API key needed

Uses the same OAuth PKCE flow as Codex CLI. The OAuth token is exchanged for an API key when possible; otherwise, calls are routed through the ChatGPT backend API.

### OpenRouter (OAuth)

Select "OpenRouter (OAuth Login)" as provider → click "Login with OpenRouter" → authorize → connected. Access to 200+ models from OpenAI, Anthropic, Meta, Google, and more.

### OpenAI-compatible providers

The OpenAI Base URL can be changed to use compatible APIs like Groq (`https://api.groq.com/openai/v1`), Together.ai, Azure OpenAI, or any OpenAI-compatible endpoint.

**API key security**: Keys are stored in `data/config.json` on disk. The `GET /api/config` endpoint returns masked keys — full keys are never sent to the browser.

## Database Connectivity

Connect to MSSQL (e.g. JTL-WaWi) or SQLite databases. The LLM generates SQL from natural language questions, the server executes it safely, and the LLM summarizes the results.

### Setup

1. Go to Settings → Database section
2. Select type (MSSQL or SQLite)
3. Enter connection details
4. Click "Test Connection"
5. Enable the database checkbox

### NL-to-SQL Flow

```
User question → LLM generates SQL → Server executes (read-only) → LLM summarizes results
```

Safety features:
- Read-only: INSERT/UPDATE/DELETE/DROP/ALTER are rejected
- 30-second timeout
- 100-row limit
- All queries logged to the SQL tab

### MSSQL (JTL-WaWi)

Requires `pyodbc` and an ODBC driver (e.g. "ODBC Driver 17 for SQL Server"). Configure server, database, username, password, and driver in Settings.

### SQLite

Built-in, no extra dependencies. Just provide the file path.

## Knowledge Base (RAG)

Upload documents that get chunked, embedded, and used as context for all conversations.

### Supported Formats

| Format | Details |
|--------|---------|
| `.txt` | Plain text |
| `.md` | Markdown |
| `.csv` | Comma-separated values |
| `.json` | JSON data |
| `.xlsx` / `.xls` | Excel spreadsheets (requires `openpyxl`) |

### How It Works

1. Upload a document via the Knowledge tab (admin/mitarbeiter)
2. Text is split into chunks (~500 tokens each)
3. Each chunk is embedded using OpenAI or Ollama embeddings
4. When a user chats, the question is embedded and matched against stored chunks via cosine similarity
5. Top-K relevant chunks are injected into the system prompt as context

### Excel Support

Excel files are parsed with smart tabular chunking — column headers are preserved in every chunk so the LLM always has context for what the data means.

### Embedding Providers

- **Auto** (default): Uses OpenAI if API key is available, otherwise Ollama
- **OpenAI**: `text-embedding-3-small` (recommended)
- **Ollama**: `nomic-embed-text` or any embedding model

### Management

The Knowledge tab (right panel) shows:
- Document list with filename, chunk count, upload date, uploader
- Upload area for files or paste text
- Search test to verify RAG retrieval
- Stats (total documents, chunks, DB size)

## Agent Mode

When enabled, the LLM can autonomously call tools in a loop (up to N steps) to gather information before answering. This replaces the simple NL-to-SQL two-pass approach with a proper tool-calling agent.

### Available Tools

| Tool | Description | Privilege | Requires |
|------|-------------|-----------|----------|
| `query_database` | Execute read-only SQL SELECT queries | Admin/Mitarbeiter | DB enabled |
| `search_knowledge` | Search the RAG knowledge base | Admin/Mitarbeiter | RAG enabled |
| `web_search` | Search the web via DuckDuckGo | All users | Web Search enabled |
| `calculate` | Safe math expression evaluator | All users | Always available |
| `get_datetime` | Get current date and time | All users | Always available |
| `list_tables` | List database tables and columns | Admin/Mitarbeiter | DB enabled |

### How It Works

```
User question → Agent loop (up to max_steps):
  → LLM decides which tool(s) to call
  → Server executes tool, returns result
  → LLM reasons about results, calls more tools or produces final answer
→ Final answer displayed with collapsible tool log
```

The agent supports **native function calling** for providers that support it (OpenAI API key, Anthropic, Ollama, OpenRouter) and falls back to **text-based tool calling** (XML `<tool_call>` parsing) for providers without native support (e.g. ChatGPT OAuth backend).

### Setup

1. Go to **Settings** → **Agent Mode** section
2. Check **Enable Agent Mode**
3. Set **Max Steps** (default: 8, max: 20)
4. Optionally enable/disable **Web Search**
5. Save

### DSGVO / Permission Model

- **Admin/Mitarbeiter**: Access to all tools including database and knowledge base
- **Regular users**: Only non-privileged tools (calculator, datetime, web search)
- Tools are filtered based on the user's role and which features are enabled in config

### Tool Log in Chat

When agent mode is used, bot messages display:
- A purple **Agent** badge next to "Bot"
- A collapsible **"Tools used"** section showing each tool call with input/output, color-coded by tool type

### Fallback

When agent mode is OFF, the existing NL-to-SQL two-pass approach still works as before. Agent mode is a strict superset — it can do everything the old approach did, plus more.

## Chat Features

### Memory

Per-session message history (max 20 messages). Conversations persist across page reloads within the same session. "New Chat" button clears history.

### File Upload

Click the paperclip icon to attach a file directly in the chat. The file content is extracted and sent alongside your question for analysis. Supports .txt, .md, .csv, .json, .log, .xlsx, .xls.

### German Default

When no custom system prompt is set, the AI defaults to responding in German. It will switch languages if the user writes in another language.

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

**Admin sees:** All tabs, settings, user management, accounts panel, bot toggle, moderation controls, knowledge base, database config, raw SQL execution, clear button

**Mitarbeiter sees:** Chat tab, knowledge base tab (upload/manage documents), own events in feed

**User sees:** Chat tab, own events in feed

Right panel tabs: **Users | Stats | Accounts | Knowledge**

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
| POST | `/api/chat` | Send question to LLM provider |
| POST | `/api/chat/upload` | Upload file for chat analysis (multipart) |
| GET | `/api/models` | List available models |

### Mitarbeiter-level (requires mitarbeiter or admin role)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/embeddings/documents` | List all knowledge base documents |
| POST | `/api/embeddings/upload` | Upload document for RAG (multipart) |
| POST | `/api/embeddings/text` | Upload raw text with title |
| DELETE | `/api/embeddings/documents/<id>` | Delete a document |
| GET | `/api/embeddings/search` | Test RAG search (`?q=...`) |
| GET | `/api/embeddings/stats` | Knowledge base statistics |

### Admin-level (requires admin role)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/events` | Query events (`?type=chat&user_id=x&limit=100&flagged=true`) |
| GET | `/api/conversations` | Group chats by user |
| GET | `/api/stats` | Dashboard statistics |
| POST | `/api/moderate/<id>` | Flag/review/note an event |
| GET | `/api/moderation` | Get all moderation data |
| GET/POST | `/api/config` | Get/update server config |
| GET | `/api/users` | List all chatbot end-users |
| GET | `/api/users/<id>` | Get user detail with recent events |
| POST | `/api/users/<id>/block` | Block a chatbot end-user |
| POST | `/api/users/<id>/unblock` | Unblock a chatbot end-user |
| POST | `/api/users/<id>/update` | Update user note, status, or priority |
| GET | `/api/accounts` | List all dashboard accounts |
| POST | `/api/accounts/<name>/role` | Change account role |
| POST | `/api/accounts/<name>/priority` | Change account priority |
| POST | `/api/accounts/<name>/delete` | Delete a dashboard account |
| POST | `/api/db/test` | Test database connection |
| GET | `/api/db/schema` | Get database schema |
| POST | `/api/db/query` | Execute raw SQL query |
| POST | `/api/clear` | Clear all events |
| GET | `/api/openai/auth/start` | Start OpenAI OAuth flow |
| POST | `/api/openai/auth/callback` | Complete OpenAI OAuth exchange |
| POST | `/api/openrouter/exchange` | Exchange OpenRouter OAuth code |

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
| LLM Provider | `ollama` | Dashboard settings (`"ollama"`, `"openai"`, `"anthropic"`, `"openrouter"`) |
| Monitor server | `localhost:7779` | `monitor.init("http://other:7779")` |
| Ollama URL | `localhost:11434` | Dashboard settings |
| OpenAI API key | (empty) | Dashboard settings or OAuth login |
| OpenAI Base URL | `https://api.openai.com/v1` | Dashboard settings |
| Anthropic API key | (empty) | Dashboard settings |
| Default model | `llama3:8b` | Dashboard settings |
| System prompt | (empty, German default) | Dashboard settings |
| Temperature | `0.7` | Dashboard settings |
| Bot enabled | `true` | Dashboard settings or `POST /api/config` |
| DB type | (empty) | Dashboard settings (`"mssql"`, `"sqlite"`) |
| RAG enabled | `false` | Dashboard settings |
| RAG top-K | `5` | Dashboard settings |
| Embedding provider | `auto` | Dashboard settings (`"auto"`, `"openai"`, `"ollama"`) |
| Agent mode | `false` | Dashboard settings |
| Agent max steps | `8` | Dashboard settings (1–20) |
| Agent web search | `true` | Dashboard settings |
| Max events | 5000 | `MAX_EVENTS` in `server.py` |
| Session TTL | 24 hours | `SESSION_TTL` in `server.py` |

## Data Storage

All data stored in `data/` directory (git-ignored):

| File | Contents |
|------|----------|
| `data/events.jsonl` | All events (append-only JSONL) |
| `data/config.json` | Server configuration (API keys, DB credentials) |
| `data/users.json` | Chatbot end-user profiles |
| `data/accounts.json` | Dashboard account credentials |
| `data/embeddings.db` | Knowledge base vectors (SQLite) |

## Testing

```bash
# Start the server
python server.py

# Run the full test suite (in another terminal)
python test_all.py
```

The test suite covers 221 tests across authentication, role-based access, event ingestion, conversations, stats, moderation, config, multi-provider LLM, OpenRouter, OAuth, database connectivity, RAG embeddings, persistence, SSE streaming, and static file serving.

## Files

| File | Description |
|------|-------------|
| `server.py` | HTTP server: auth, events, SSE, multi-provider LLM, database, RAG, user/account management |
| `monitor.py` | Python SDK: fire-and-forget logging, wrappers, middleware |
| `public/index.html` | Dashboard UI: login, chat, feed, stats, admin panels, knowledge base |
| `test_all.py` | Comprehensive test suite (221 tests) |
