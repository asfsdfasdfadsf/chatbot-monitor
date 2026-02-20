#!/usr/bin/env python3
"""WaWi Chatbot Admin Monitor — zero-dependency server on port 7779.

Receives events from the chatbot SDK, proxies chat requests to Ollama,
manages users (block/allow), serves the admin dashboard, and pushes
live updates via SSE.

Features: cookie-based auth, admin/user roles, bot kill switch, user priorities.
"""

import base64
import json
import hashlib
import os
import re
import secrets
import struct
import sys
import time
import uuid
import io
import threading
import queue
from collections import defaultdict
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs, urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError
from pathlib import Path
import sqlite3  # always available

try:
    import pyodbc
    HAS_PYODBC = True
except ImportError:
    HAS_PYODBC = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    import openpyxl
    HAS_OPENPYXL = True
except ImportError:
    HAS_OPENPYXL = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

# ---------------------------------------------------------------------------
# OpenAI OAuth PKCE constants (from Codex CLI)
# ---------------------------------------------------------------------------
OPENAI_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
OPENAI_AUTH_URL = "https://auth.openai.com/oauth/authorize"
OPENAI_TOKEN_URL = "https://auth.openai.com/oauth/token"
OPENAI_SCOPE = "openid profile email offline_access"
OPENAI_CALLBACK_PORT = 1455  # Must match Codex CLI's registered redirect URI

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = 7779
MAX_EVENTS = 5000
DATA_DIR = Path(__file__).resolve().parent / "data"
EVENTS_FILE = DATA_DIR / "events.jsonl"
CONFIG_FILE = DATA_DIR / "config.json"
USERS_FILE = DATA_DIR / "users.json"
ACCOUNTS_FILE = DATA_DIR / "accounts.json"
PUBLIC_DIR = Path(__file__).resolve().parent / "public"

config = {
    "provider": "ollama",  # "ollama" | "openai" | "anthropic" | "openrouter"
    "ollama_url": "http://localhost:11434",
    "default_model": "llama3:8b",
    "system_prompt": "",
    "temperature": 0.7,
    "bot_enabled": True,
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "anthropic_api_key": "",
    "openrouter_api_key": "",
    "openai_oauth_token": "",
    "openai_refresh_token": "",
    "openai_token_expires": 0,
    "db_type": "",              # "" | "mssql" | "sqlite"
    "db_mssql_server": "",
    "db_mssql_database": "",
    "db_mssql_user": "",
    "db_mssql_password": "",
    "db_mssql_driver": "",
    "db_sqlite_path": "",
    "db_enabled": False,
    "db_schema_cache": "",
    "rag_enabled": False,
    "rag_top_k": 5,
    "rag_chunk_size": 500,
    "embedding_provider": "auto",  # "auto"|"openai"|"ollama"
    "embedding_model": "",
}
config_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
events: list[dict] = []
events_lock = threading.Lock()
moderation: dict[str, dict] = {}
moderation_lock = threading.Lock()
sse_clients: list[tuple[queue.Queue, dict]] = []  # (queue, session_info)
sse_lock = threading.Lock()

# User management: user_id -> { status, first_seen, last_seen, ... }
# status: "active" | "blocked" | "restricted"
users: dict[str, dict] = {}
users_lock = threading.Lock()

# Account management: username -> { username, password_hash, salt, role, priority, created_at }
accounts: dict[str, dict] = {}
accounts_lock = threading.Lock()

# Session management: session_id -> { username, role, created_at, expires_at }
sessions: dict[str, dict] = {}
sessions_lock = threading.Lock()
SESSION_TTL = 86400  # 24 hours

STOP_WORDS = frozenset(
    "der die das ein eine einer eines einem einen und oder aber wenn dann "
    "als auch noch schon doch nur mal so da hier dort wie was wer wo wann "
    "warum ist sind war waren hat hatte haben wird werden kann können muss "
    "müssen soll sollen darf dürfen möchte ich du er sie es wir ihr man "
    "mich mir dich dir sich uns euch ihm ihn ihr ihnen mein dein sein "
    "unser euer nicht kein keine mehr viel sehr gut auf aus bei mit von zu "
    "für über unter nach vor zwischen durch gegen ohne um an in im am zum "
    "zur the a an is are was were has have had will would can could shall "
    "should may might must do does did not no and or but if then than that "
    "this these those it its he she they we you i me my his her our your "
    "their what which who whom how when where why all some any much many "
    "more most other another such each every both few several than too very "
    "just also still already yet again once never always often sometimes "
    "about after before between from into through during with without of "
    "to for on at by up down in out off over under above below".split()
)


def extract_topics(text: str, top_n: int = 5) -> list[str]:
    words = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", text.lower())
    freq: dict[str, int] = defaultdict(int)
    for w in words:
        if w not in STOP_WORDS:
            freq[w] += 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------
def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _verify_password(password: str, password_hash: str, salt: str) -> bool:
    return secrets.compare_digest(_hash_password(password, salt), password_hash)


# ---------------------------------------------------------------------------
# Account persistence
# ---------------------------------------------------------------------------
def load_accounts():
    global accounts
    if ACCOUNTS_FILE.exists():
        try:
            with open(ACCOUNTS_FILE, "r", encoding="utf-8") as f:
                accounts = json.load(f)
        except Exception:
            pass


def save_accounts():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with accounts_lock:
            with open(ACCOUNTS_FILE, "w", encoding="utf-8") as f:
                json.dump(accounts, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[accounts] Error saving: {e}", file=sys.stderr)


def ensure_default_admin():
    with accounts_lock:
        has_admin = any(a.get("role") == "admin" for a in accounts.values())
    if not has_admin:
        salt = secrets.token_hex(16)
        with accounts_lock:
            accounts["admin"] = {
                "username": "admin",
                "password_hash": _hash_password("admin", salt),
                "salt": salt,
                "role": "admin",
                "priority": "normal",
                "created_at": time.time(),
            }
        save_accounts()
        print("[chatbot-monitor] Created default admin account (admin/admin)")


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------
def create_session(username: str, role: str) -> str:
    sid = secrets.token_hex(32)
    now = time.time()
    with sessions_lock:
        sessions[sid] = {
            "username": username,
            "role": role,
            "created_at": now,
            "expires_at": now + SESSION_TTL,
        }
    return sid


def get_session_by_id(sid: str) -> dict | None:
    if not sid:
        return None
    with sessions_lock:
        s = sessions.get(sid)
    if s and s["expires_at"] > time.time():
        return s
    if s:
        with sessions_lock:
            sessions.pop(sid, None)
    return None


def delete_session(sid: str):
    with sessions_lock:
        sessions.pop(sid, None)


def _cleanup_sessions_loop():
    while True:
        time.sleep(300)  # every 5 minutes
        now = time.time()
        with sessions_lock:
            expired = [k for k, v in sessions.items() if v["expires_at"] <= now]
            for k in expired:
                del sessions[k]
        # Clean up chat history for expired sessions
        for k in expired:
            clear_chat_history(k)


# Start background cleanup thread
threading.Thread(target=_cleanup_sessions_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# Chat memory — per-session conversation history
# ---------------------------------------------------------------------------
chat_sessions: dict[str, list[dict]] = {}  # session_id -> messages
chat_session_meta: dict[str, dict] = {}  # session_id -> {username, created_at, last_activity, model}
chat_sessions_lock = threading.Lock()
MAX_CHAT_HISTORY = 20


def get_chat_history(session_id: str) -> list[dict]:
    with chat_sessions_lock:
        return list(chat_sessions.get(session_id, []))


def append_chat_message(session_id: str, role: str, content: str, username: str = "", model: str = ""):
    now = time.time()
    with chat_sessions_lock:
        if session_id not in chat_sessions:
            chat_sessions[session_id] = []
        chat_sessions[session_id].append({"role": role, "content": content, "timestamp": now})
        # Trim to max history (keep last MAX_CHAT_HISTORY messages)
        if len(chat_sessions[session_id]) > MAX_CHAT_HISTORY:
            chat_sessions[session_id] = chat_sessions[session_id][-MAX_CHAT_HISTORY:]
        # Update session metadata
        if session_id not in chat_session_meta:
            chat_session_meta[session_id] = {
                "username": username,
                "created_at": now,
                "last_activity": now,
                "model": model,
            }
        else:
            chat_session_meta[session_id]["last_activity"] = now
            if username:
                chat_session_meta[session_id]["username"] = username
            if model:
                chat_session_meta[session_id]["model"] = model


def clear_chat_history(session_id: str):
    with chat_sessions_lock:
        chat_sessions.pop(session_id, None)
        chat_session_meta.pop(session_id, None)


# ---------------------------------------------------------------------------
# OpenAI OAuth PKCE helpers
# ---------------------------------------------------------------------------
_openai_pkce_pending: dict[str, dict] = {}  # state -> {verifier, created_at}
_pkce_lock = threading.Lock()


def _pkce_code_verifier() -> str:
    """Generate a PKCE code verifier (64 bytes, base64url-encoded, matching Codex CLI)."""
    return secrets.token_urlsafe(64)


def _pkce_code_challenge(verifier: str) -> str:
    """SHA256 + base64url-encode the verifier."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


_oauth_callback_server = None
_oauth_callback_lock = threading.Lock()


def _start_oauth_callback_server(state: str):
    """Spin up a temporary HTTP server on port 1455 to catch the OAuth callback.

    When OpenAI redirects to http://localhost:1455/auth/callback?code=X&state=Y,
    this server captures it and redirects the browser to our main dashboard
    with the code+state params so the JS can complete the exchange.
    """
    global _oauth_callback_server

    with _oauth_callback_lock:
        # If already running, don't start another
        if _oauth_callback_server is not None:
            try:
                _oauth_callback_server.shutdown()
            except Exception:
                pass
            _oauth_callback_server = None

    main_port = PORT

    class CallbackHandler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def do_GET(self):
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)
            code = (qs.get("code") or [""])[0]
            cb_state = (qs.get("state") or [""])[0]
            error = (qs.get("error") or [""])[0]

            if error:
                html = f"""<!DOCTYPE html><html><body>
                <h2>OpenAI Login Failed</h2><p>Error: {error}</p>
                <p><a href="http://localhost:{main_port}/">Back to Dashboard</a></p>
                </body></html>"""
            else:
                # Redirect to main dashboard with code+state
                redirect_url = f"http://localhost:{main_port}/?code={code}&state={cb_state}"
                html = f"""<!DOCTYPE html><html><head>
                <meta http-equiv="refresh" content="0;url={redirect_url}">
                </head><body>Redirecting...</body></html>"""

            body = html.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

            # Shut down after handling the callback (in a thread to not deadlock)
            threading.Thread(target=self._shutdown, daemon=True).start()

        def _shutdown(self):
            global _oauth_callback_server
            time.sleep(1)
            with _oauth_callback_lock:
                if _oauth_callback_server is not None:
                    try:
                        _oauth_callback_server.shutdown()
                    except Exception:
                        pass
                    _oauth_callback_server = None

    def _run():
        global _oauth_callback_server
        try:
            from http.server import HTTPServer
            srv = HTTPServer(("127.0.0.1", OPENAI_CALLBACK_PORT), CallbackHandler)
            srv.timeout = 300  # 5 min max wait
            with _oauth_callback_lock:
                _oauth_callback_server = srv
            srv.serve_forever()
        except OSError:
            pass  # Port already in use

    threading.Thread(target=_run, daemon=True).start()


def refresh_openai_token() -> str | None:
    """Use refresh_token to get a new access token. Also attempts API key exchange. Returns new token or None."""
    with config_lock:
        refresh_tok = config.get("openai_refresh_token", "")
    if not refresh_tok:
        return None

    try:
        body = json.dumps({
            "grant_type": "refresh_token",
            "client_id": OPENAI_CLIENT_ID,
            "refresh_token": refresh_tok,
            "scope": "openid profile email",
        }).encode("utf-8")
        req = Request(
            OPENAI_TOKEN_URL, data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=10)
        result = json.loads(resp.read())
        access_token = result.get("access_token", "")
        id_token = result.get("id_token", "")
        new_refresh = result.get("refresh_token", refresh_tok)
        expires_in = result.get("expires_in", 3600)

        # Try to exchange id_token for API key
        api_key = ""
        if id_token:
            try:
                exchange_body = urlencode({
                    "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                    "client_id": OPENAI_CLIENT_ID,
                    "requested_token": "openai-api-key",
                    "subject_token": id_token,
                    "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
                }).encode("utf-8")
                req2 = Request(
                    OPENAI_TOKEN_URL, data=exchange_body,
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                    method="POST",
                )
                resp2 = urlopen(req2, timeout=10)
                result2 = json.loads(resp2.read())
                api_key = result2.get("access_token", "")
                print(f"[oauth] Refresh token exchange succeeded, got API key: {bool(api_key)}", file=sys.stderr)
            except Exception as exc:
                print(f"[oauth] Refresh token exchange for API key failed: {exc}", file=sys.stderr)

        with config_lock:
            config["openai_oauth_token"] = access_token
            config["openai_refresh_token"] = new_refresh
            config["openai_token_expires"] = time.time() + expires_in - 60
            if api_key:
                config["openai_api_key"] = api_key
        save_config()
        return api_key if api_key else access_token
    except Exception as e:
        print(f"[oauth] Token refresh failed: {e}", file=sys.stderr)
        return None


def get_openai_bearer_token() -> str:
    """Return a valid OpenAI bearer token. Prefers API key from token exchange, then OAuth, then manual key."""
    with config_lock:
        api_key = config.get("openai_api_key", "")
        oauth_token = config.get("openai_oauth_token", "")
        expires = config.get("openai_token_expires", 0)

    # Prefer API key (obtained via OAuth token exchange) — works with all endpoints
    if api_key:
        return api_key

    # If we have a valid OAuth token, use it
    if oauth_token and time.time() < expires:
        return oauth_token

    # Try to refresh
    if config.get("openai_refresh_token"):
        new_token = refresh_openai_token()
        if new_token:
            return new_token

    return ""


# ---------------------------------------------------------------------------
# Config persistence
# ---------------------------------------------------------------------------
def load_config():
    global config
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            config.update(saved)
        except Exception:
            pass


def save_config():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[config] Error saving: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# User management persistence
# ---------------------------------------------------------------------------
def load_users():
    global users
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                users = json.load(f)
        except Exception:
            pass


def save_users():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with users_lock:
            with open(USERS_FILE, "w", encoding="utf-8") as f:
                json.dump(users, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[users] Error saving: {e}", file=sys.stderr)


def touch_user(user_id: str):
    """Auto-register / update last_seen for a user on every event."""
    if not user_id or user_id == "admin":
        return
    now = time.time()
    with users_lock:
        if user_id not in users:
            users[user_id] = {
                "user_id": user_id,
                "status": "active",
                "first_seen": now,
                "last_seen": now,
                "message_count": 0,
                "error_count": 0,
                "note": "",
                "priority": "normal",
            }
        users[user_id]["last_seen"] = now


def increment_user_stat(user_id: str, field: str, amount: int = 1):
    if not user_id or user_id == "admin":
        return
    with users_lock:
        if user_id in users:
            users[user_id][field] = users[user_id].get(field, 0) + amount


def is_user_allowed(user_id: str) -> bool:
    """Check if a user is allowed to use the bot."""
    if not user_id or user_id == "admin":
        return True
    with users_lock:
        u = users.get(user_id)
        if u and u.get("status") == "blocked":
            return False
    return True


def get_user_summary(user_id: str) -> dict:
    """Get user info + recent activity from events."""
    with users_lock:
        u = dict(users.get(user_id, {}))
    if not u:
        u = {"user_id": user_id, "status": "unknown"}

    # Compute live stats from events
    with events_lock:
        user_events = [e for e in events if e.get("user_id") == user_id]

    u["total_events"] = len(user_events)
    u["total_chats"] = sum(1 for e in user_events if e["type"] == "chat")
    u["total_queries"] = sum(1 for e in user_events if e["type"] == "query")
    u["total_errors"] = sum(1 for e in user_events if e["type"] == "error")
    u["recent_events"] = user_events[-20:]  # last 20
    return u


# ---------------------------------------------------------------------------
# Event persistence
# ---------------------------------------------------------------------------
def load_events():
    global events
    if not EVENTS_FILE.exists():
        return
    loaded = []
    with open(EVENTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    evt = json.loads(line)
                    loaded.append(evt)
                    mod = evt.get("_moderation")
                    if mod:
                        moderation[evt["id"]] = mod
                except json.JSONDecodeError:
                    continue
    events = loaded[-MAX_EVENTS:]


def rebuild_users_from_events():
    """Rebuild user stats from loaded events (on startup)."""
    for evt in events:
        uid = evt.get("user_id")
        if uid and uid != "admin":
            touch_user(uid)
            if evt.get("type") == "chat":
                increment_user_stat(uid, "message_count")
            elif evt.get("type") == "error":
                increment_user_stat(uid, "error_count")


def append_event_to_file(evt: dict):
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[persist] Error writing event: {e}", file=sys.stderr)


def persist_moderation(event_id: str, mod: dict):
    try:
        with events_lock:
            for evt in events:
                if evt["id"] == event_id:
                    evt["_moderation"] = mod
                    break
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            with events_lock:
                for evt in events:
                    f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[persist] Error writing moderation: {e}", file=sys.stderr)


def clear_all_events():
    global events, moderation
    with events_lock:
        events = []
    with moderation_lock:
        moderation = {}
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            pass
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Ingest + broadcast
# ---------------------------------------------------------------------------
def ingest_event(evt: dict) -> dict:
    if "id" not in evt:
        evt["id"] = str(uuid.uuid4())
    if "timestamp" not in evt:
        evt["timestamp"] = time.time()
    if "type" not in evt:
        evt["type"] = "system"

    # Track user
    uid = evt.get("user_id")
    if uid:
        touch_user(uid)
        if evt["type"] == "chat":
            increment_user_stat(uid, "message_count")
        elif evt["type"] == "error":
            increment_user_stat(uid, "error_count")

    with events_lock:
        events.append(evt)
        while len(events) > MAX_EVENTS:
            events.pop(0)

    append_event_to_file(evt)
    broadcast_event(evt)

    # Save users periodically (every 10th event to avoid thrashing)
    if len(events) % 10 == 0:
        save_users()

    return evt


# ---------------------------------------------------------------------------
# LLM providers
# ---------------------------------------------------------------------------
def call_ollama(messages: list[dict], model: str, temperature: float) -> tuple[str, float]:
    with config_lock:
        ollama_url = config["ollama_url"]

    body: dict = {"model": model, "messages": messages, "stream": False}
    if temperature is not None:
        body["options"] = {"temperature": temperature}

    start = time.time()
    data = json.dumps(body).encode("utf-8")
    req = Request(
        f"{ollama_url}/api/chat", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    resp = urlopen(req, timeout=120)
    result = json.loads(resp.read())
    answer = result.get("message", {}).get("content", "")
    duration_ms = (time.time() - start) * 1000
    return answer, round(duration_ms, 1)


CHATGPT_BACKEND_URL = "https://chatgpt.com/backend-api/codex"


def _using_oauth_token() -> bool:
    """Check if we're using a raw OAuth token (no API key from token exchange)."""
    with config_lock:
        has_api_key = bool(config.get("openai_api_key"))
        has_oauth = bool(config.get("openai_oauth_token"))
    # Use ChatGPT backend if we have OAuth but no proper API key
    return has_oauth and not has_api_key


def _call_openai_chat_completions(base_url: str, bearer: str, messages: list[dict], model: str, temperature: float) -> str:
    """Call OpenAI Chat Completions API."""
    if not model:
        model = "gpt-4o-mini"
    body = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature
    data = json.dumps(body).encode("utf-8")
    print(f"[openai] Chat Completions request: model={model}, messages={len(messages)}", file=sys.stderr)
    req = Request(
        f"{base_url}/chat/completions", data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {bearer}",
        },
        method="POST",
    )
    try:
        resp = urlopen(req, timeout=120)
    except Exception as e:
        if hasattr(e, 'read'):
            try:
                err_body = e.read().decode()
                print(f"[openai] Chat Completions error body: {err_body}", file=sys.stderr)
            except Exception:
                pass
        raise
    result = json.loads(resp.read())
    return result["choices"][0]["message"]["content"]


def _call_openai_responses(base_url: str, bearer: str, messages: list[dict], model: str, temperature: float) -> str:
    """Call OpenAI Responses API (works with both api.openai.com and chatgpt.com backend)."""
    if not model:
        model = "gpt-4o-mini"
    system_text = ""
    input_parts = []
    for m in messages:
        if m["role"] == "system":
            system_text = m["content"]
        else:
            input_parts.append(m)

    body = {"model": model}
    if system_text:
        body["instructions"] = system_text
    body["input"] = [{"role": m["role"], "content": m["content"]} for m in input_parts]
    if temperature is not None:
        body["temperature"] = temperature

    data = json.dumps(body).encode("utf-8")
    url = f"{base_url}/responses"
    print(f"[openai] Responses API request: url={url}, model={model}, input_parts={len(input_parts)}", file=sys.stderr)
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {bearer}",
    }
    req = Request(url, data=data, headers=headers, method="POST")
    try:
        resp = urlopen(req, timeout=120)
    except Exception as e:
        if hasattr(e, 'read'):
            try:
                err_body = e.read().decode()
                print(f"[openai] Responses API error body: {err_body}", file=sys.stderr)
            except Exception:
                pass
        raise
    result = json.loads(resp.read())
    answer = ""
    for item in result.get("output", []):
        if item.get("type") == "message":
            for c in item.get("content", []):
                if c.get("type") == "output_text":
                    answer += c.get("text", "")
    if not answer:
        answer = result.get("output_text", "") or str(result)
    return answer


def call_openai(messages: list[dict], model: str, temperature: float) -> tuple[str, float]:
    with config_lock:
        base_url = config.get("openai_base_url", "https://api.openai.com/v1").rstrip("/")

    bearer = get_openai_bearer_token()
    if not bearer:
        raise ValueError("OpenAI API key not configured — use Login with OpenAI or set an API key in Settings")

    use_oauth = _using_oauth_token()

    start = time.time()

    if use_oauth:
        # OAuth token (ChatGPT login): must use chatgpt.com backend, NOT api.openai.com
        # The raw OAuth access_token is only valid for the ChatGPT backend API
        print(f"[openai] Using ChatGPT backend (OAuth token)", file=sys.stderr)
        try:
            answer = _call_openai_responses(CHATGPT_BACKEND_URL, bearer, messages, model, temperature)
        except Exception as e1:
            print(f"[openai] ChatGPT backend failed: {e1}", file=sys.stderr)
            if "429" in str(e1):
                raise ValueError("OpenAI rate limit reached — your account usage limit may be exceeded. Please wait or check your OpenAI plan.")
            raise ValueError(f"OpenAI API call failed. Error: {e1}")
    else:
        # API key: use standard api.openai.com with Chat Completions
        answer = _call_openai_chat_completions(base_url, bearer, messages, model, temperature)

    duration_ms = (time.time() - start) * 1000
    return answer, round(duration_ms, 1)


def call_anthropic(messages: list[dict], model: str, temperature: float) -> tuple[str, float]:
    with config_lock:
        api_key = config["anthropic_api_key"]

    if not api_key:
        raise ValueError("Anthropic API key not configured")

    # Anthropic API requires system prompt in a separate field
    system_prompt = ""
    chat_messages = []
    for m in messages:
        if m["role"] == "system":
            system_prompt = m["content"]
        else:
            chat_messages.append(m)

    body: dict = {
        "model": model,
        "max_tokens": 4096,
        "messages": chat_messages,
    }
    if system_prompt:
        body["system"] = system_prompt
    if temperature is not None:
        body["temperature"] = temperature

    start = time.time()
    data = json.dumps(body).encode("utf-8")
    req = Request(
        "https://api.anthropic.com/v1/messages", data=data,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    resp = urlopen(req, timeout=120)
    result = json.loads(resp.read())
    answer = result["content"][0]["text"]
    duration_ms = (time.time() - start) * 1000
    return answer, round(duration_ms, 1)


def call_openrouter(messages: list[dict], model: str, temperature: float) -> tuple[str, float]:
    with config_lock:
        api_key = config["openrouter_api_key"]

    if not api_key:
        raise ValueError("OpenRouter API key not configured — use Login with OpenRouter in Settings")

    body = {"model": model, "messages": messages}
    if temperature is not None:
        body["temperature"] = temperature

    start = time.time()
    data = json.dumps(body).encode("utf-8")
    req = Request(
        "https://openrouter.ai/api/v1/chat/completions", data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "http://localhost:7779",
            "X-Title": "WaWi Chatbot Monitor",
        },
        method="POST",
    )
    resp = urlopen(req, timeout=120)
    result = json.loads(resp.read())
    answer = result["choices"][0]["message"]["content"]
    duration_ms = (time.time() - start) * 1000
    return answer, round(duration_ms, 1)


def call_llm(messages: list[dict], model: str, temperature: float) -> tuple[str, float]:
    """Dispatch to the configured LLM provider."""
    with config_lock:
        provider = config.get("provider", "ollama")
    if provider == "openai":
        return call_openai(messages, model, temperature)
    elif provider == "anthropic":
        return call_anthropic(messages, model, temperature)
    elif provider == "openrouter":
        return call_openrouter(messages, model, temperature)
    else:
        return call_ollama(messages, model, temperature)


OPENAI_MODELS = [
    {"name": "gpt-5.2", "size": 0},
    {"name": "gpt-5.2-chat-latest", "size": 0},
    {"name": "gpt-5.2-pro", "size": 0},
    {"name": "gpt-5.2-codex", "size": 0},
    {"name": "gpt-4.1", "size": 0},
    {"name": "gpt-4.1-mini", "size": 0},
    {"name": "gpt-4.1-nano", "size": 0},
    {"name": "gpt-4o", "size": 0},
    {"name": "gpt-4o-mini", "size": 0},
    {"name": "o3", "size": 0},
    {"name": "o3-mini", "size": 0},
    {"name": "o4-mini", "size": 0},
]

ANTHROPIC_MODELS = [
    {"name": "claude-sonnet-4-20250514", "size": 0},
    {"name": "claude-haiku-4-5-20251001", "size": 0},
    {"name": "claude-3-5-sonnet-20241022", "size": 0},
    {"name": "claude-3-5-haiku-20241022", "size": 0},
    {"name": "claude-3-opus-20240229", "size": 0},
]


def list_models() -> list[dict]:
    with config_lock:
        provider = config.get("provider", "ollama")

    if provider == "openai":
        bearer = get_openai_bearer_token()
        if not bearer:
            return OPENAI_MODELS  # fallback list
        use_oauth = _using_oauth_token()
        if use_oauth:
            # OAuth: fetch models from ChatGPT backend
            models_url = f"{CHATGPT_BACKEND_URL}/models"
        else:
            with config_lock:
                base_url = config.get("openai_base_url", "https://api.openai.com/v1").rstrip("/")
            models_url = f"{base_url}/models"
        try:
            req = Request(
                models_url, method="GET",
                headers={"Authorization": f"Bearer {bearer}"},
            )
            resp = urlopen(req, timeout=5)
            data = json.loads(resp.read())
            models = data.get("data", [])
            # Filter to chat models
            chat_models = [
                {"name": m["id"], "size": 0}
                for m in models
                if any(m["id"].startswith(p) for p in ("gpt-", "o1", "o3", "o4", "chatgpt-"))
            ]
            if chat_models:
                return sorted(chat_models, key=lambda x: x["name"])
            return OPENAI_MODELS
        except Exception:
            return OPENAI_MODELS

    elif provider == "anthropic":
        return ANTHROPIC_MODELS

    elif provider == "openrouter":
        with config_lock:
            api_key = config["openrouter_api_key"]
        if not api_key:
            return OPENAI_MODELS + ANTHROPIC_MODELS  # show common models as fallback
        try:
            req = Request(
                "https://openrouter.ai/api/v1/models", method="GET",
                headers={"Authorization": f"Bearer {api_key}"},
            )
            resp = urlopen(req, timeout=5)
            data = json.loads(resp.read())
            models = data.get("data", [])
            return [{"name": m["id"], "size": 0} for m in models[:100]]  # cap at 100
        except Exception:
            return OPENAI_MODELS + ANTHROPIC_MODELS

    else:  # ollama
        with config_lock:
            ollama_url = config["ollama_url"]
        try:
            req = Request(f"{ollama_url}/api/tags", method="GET")
            resp = urlopen(req, timeout=5)
            data = json.loads(resp.read())
            models = data.get("models", [])
            return [{"name": m["name"], "size": m.get("size", 0)} for m in models]
        except Exception:
            return []


# ---------------------------------------------------------------------------
# Database connection module
# ---------------------------------------------------------------------------
_db_thread_local = threading.local()


def get_db_connection():
    """Return a DB connection or None, cached per-thread."""
    with config_lock:
        db_type = config.get("db_type", "")
        db_enabled = config.get("db_enabled", False)
    if not db_enabled or not db_type:
        return None

    # Check if we already have a connection for this thread
    existing = getattr(_db_thread_local, "conn", None)
    existing_type = getattr(_db_thread_local, "conn_type", "")
    if existing and existing_type == db_type:
        try:
            # Test if connection is still alive
            if db_type == "sqlite":
                existing.execute("SELECT 1")
            else:
                existing.execute("SELECT 1")
            return existing
        except Exception:
            try:
                existing.close()
            except Exception:
                pass
            _db_thread_local.conn = None

    try:
        if db_type == "mssql":
            if not HAS_PYODBC:
                return None
            with config_lock:
                server = config.get("db_mssql_server", "")
                database = config.get("db_mssql_database", "")
                user = config.get("db_mssql_user", "")
                password = config.get("db_mssql_password", "")
                driver = config.get("db_mssql_driver", "ODBC Driver 17 for SQL Server")
            conn_str = f"DRIVER={{{driver}}};SERVER={server};DATABASE={database};UID={user};PWD={password}"
            conn = pyodbc.connect(conn_str, timeout=10)
            _db_thread_local.conn = conn
            _db_thread_local.conn_type = db_type
            return conn
        elif db_type == "sqlite":
            with config_lock:
                db_path = config.get("db_sqlite_path", "")
            if not db_path:
                return None
            # Resolve relative paths against data dir
            p = Path(db_path)
            if not p.is_absolute():
                p = DATA_DIR / db_path
            conn = sqlite3.connect(str(p), timeout=10)
            conn.row_factory = sqlite3.Row
            _db_thread_local.conn = conn
            _db_thread_local.conn_type = db_type
            return conn
    except Exception as e:
        print(f"[db] Connection error: {e}", file=sys.stderr)
    return None


def get_db_schema(force_refresh=False) -> str:
    """Discover tables + columns. Returns formatted string for LLM."""
    with config_lock:
        cached = config.get("db_schema_cache", "")
    if cached and not force_refresh:
        return cached

    conn = get_db_connection()
    if not conn:
        return ""

    with config_lock:
        db_type = config.get("db_type", "")

    schema_lines = []
    try:
        if db_type == "mssql":
            cursor = conn.cursor()
            cursor.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE "
                "FROM INFORMATION_SCHEMA.COLUMNS ORDER BY TABLE_NAME, ORDINAL_POSITION"
            )
            current_table = None
            for row in cursor.fetchall():
                tbl, col, dtype = row[0], row[1], row[2]
                if tbl != current_table:
                    current_table = tbl
                    schema_lines.append(f"\nTable: {tbl}")
                schema_lines.append(f"  - {col} ({dtype})")
            cursor.close()
        elif db_type == "sqlite":
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [r[0] if not isinstance(r, sqlite3.Row) else r["name"] for r in cursor.fetchall()]
            for tbl in tables:
                schema_lines.append(f"\nTable: {tbl}")
                cursor.execute(f"PRAGMA table_info(`{tbl}`)")
                for col_info in cursor.fetchall():
                    col_name = col_info[1] if not isinstance(col_info, sqlite3.Row) else col_info["name"]
                    col_type = col_info[2] if not isinstance(col_info, sqlite3.Row) else col_info["type"]
                    schema_lines.append(f"  - {col_name} ({col_type})")
            cursor.close()
    except Exception as e:
        print(f"[db] Schema error: {e}", file=sys.stderr)
        return f"Error reading schema: {e}"

    schema = "\n".join(schema_lines).strip()
    with config_lock:
        config["db_schema_cache"] = schema
    save_config()
    return schema


_WRITE_SQL_RE = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|MERGE|EXEC|EXECUTE|GRANT|REVOKE)\b",
    re.IGNORECASE,
)


def execute_db_query(sql: str, params=None) -> dict:
    """Execute a SELECT query safely. Returns result dict."""
    # Read-only check
    if _WRITE_SQL_RE.search(sql):
        return {"error": "Only SELECT queries are allowed", "columns": [], "rows": [], "row_count": 0, "duration_ms": 0}

    conn = get_db_connection()
    if not conn:
        return {"error": "No database connection", "columns": [], "rows": [], "row_count": 0, "duration_ms": 0}

    with config_lock:
        db_type = config.get("db_type", "")

    start = time.time()
    try:
        cursor = conn.cursor()
        if db_type == "mssql":
            cursor.execute(sql, params or ())
        else:
            cursor.execute(sql, params or ())

        if cursor.description is None:
            cursor.close()
            return {"error": "Query returned no results", "columns": [], "rows": [], "row_count": 0,
                    "duration_ms": round((time.time() - start) * 1000, 1)}

        columns = [desc[0] for desc in cursor.description]
        rows_raw = cursor.fetchmany(100)  # limit to 100 rows
        rows = [dict(zip(columns, row)) for row in rows_raw]
        row_count = len(rows)
        cursor.close()
        duration_ms = round((time.time() - start) * 1000, 1)

        # Log as event
        ingest_event({
            "type": "query",
            "sql": sql,
            "row_count": row_count,
            "duration_ms": duration_ms,
            "results": rows[:10],  # only store first 10 in event for feed
        })

        return {"columns": columns, "rows": rows, "row_count": row_count, "duration_ms": duration_ms}
    except Exception as e:
        duration_ms = round((time.time() - start) * 1000, 1)
        return {"error": str(e), "columns": [], "rows": [], "row_count": 0, "duration_ms": duration_ms}


# ---------------------------------------------------------------------------
# Embedding store
# ---------------------------------------------------------------------------
EMBEDDINGS_DB = DATA_DIR / "embeddings.db"


def init_embedding_db():
    """Create embedding tables if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(EMBEDDINGS_DB))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            filename TEXT,
            content TEXT,
            chunk_count INTEGER,
            uploaded_by TEXT,
            uploaded_at REAL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT REFERENCES documents(id),
            content TEXT,
            chunk_index INTEGER,
            embedding BLOB,
            token_count INTEGER
        )
    """)
    conn.commit()
    conn.close()


def _get_embedding_conn():
    """Get a connection to the embeddings database."""
    return sqlite3.connect(str(EMBEDDINGS_DB))


def embed_text(text: str) -> list:
    """Get embedding vector for text. Returns list of floats."""
    with config_lock:
        provider = config.get("embedding_provider", "auto")
        model = config.get("embedding_model", "")
        openai_key = config.get("openai_api_key", "")
        ollama_url = config.get("ollama_url", "http://localhost:11434")

    # Auto-detect provider
    if provider == "auto":
        if openai_key:
            provider = "openai"
        else:
            provider = "ollama"

    if provider == "openai":
        if not model:
            model = "text-embedding-3-small"
        bearer = get_openai_bearer_token()
        if not bearer:
            raise ValueError("No OpenAI API key available for embeddings")
        with config_lock:
            base_url = config.get("openai_base_url", "https://api.openai.com/v1").rstrip("/")
        body = json.dumps({"model": model, "input": text}).encode("utf-8")
        req = Request(
            f"{base_url}/embeddings", data=body,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {bearer}"},
            method="POST",
        )
        resp = urlopen(req, timeout=30)
        result = json.loads(resp.read())
        return result["data"][0]["embedding"]

    elif provider == "ollama":
        if not model:
            model = "nomic-embed-text"
        body = json.dumps({"model": model, "prompt": text}).encode("utf-8")
        req = Request(
            f"{ollama_url}/api/embeddings", data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        resp = urlopen(req, timeout=60)
        result = json.loads(resp.read())
        return result.get("embedding", [])

    raise ValueError(f"Unknown embedding provider: {provider}")


def parse_excel(file_bytes: bytes) -> str:
    """Parse an Excel (.xlsx) file and return its content as text.

    Each sheet becomes a section with a header row repeated for context.
    Format: 'Header1 | Header2 | ...' on the first row, then data rows.
    """
    if not HAS_OPENPYXL:
        raise ImportError("openpyxl is not installed. Run: pip install openpyxl")
    wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
    parts = []
    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        all_rows = []
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) if c is not None else "" for c in row]
            if any(cells):
                all_rows.append(cells)
        if all_rows:
            header = all_rows[0]
            header_line = " | ".join(header)
            lines = [f"## {sheet_name}", f"Spalten: {header_line}"]
            for row_cells in all_rows[1:]:
                # Format as "Header1: Value1, Header2: Value2, ..."
                pairs = []
                for h, v in zip(header, row_cells):
                    if v.strip():
                        pairs.append(f"{h}: {v}")
                if pairs:
                    lines.append(", ".join(pairs))
            parts.append("\n".join(lines))
    wb.close()
    return "\n\n".join(parts)


def chunk_tabular_text(text: str, chunk_size: int = 500) -> list:
    """Chunk tabular text (from Excel) keeping section headers with each chunk."""
    sections = re.split(r"(?=^## )", text, flags=re.MULTILINE)
    chunks = []

    def _count_tokens(t):
        if HAS_TIKTOKEN:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(t))
            except Exception:
                pass
        return len(t) // 4

    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.split("\n")
        # First two lines are "## SheetName" and "Spalten: ..." — keep as header
        header_lines = []
        data_lines = []
        for i, line in enumerate(lines):
            if i < 2 or line.startswith("## ") or line.startswith("Spalten:"):
                header_lines.append(line)
            else:
                data_lines.append(line)
        header = "\n".join(header_lines)

        current_chunk_lines = []
        current_tokens = _count_tokens(header)
        for dline in data_lines:
            line_tokens = _count_tokens(dline)
            if current_tokens + line_tokens > chunk_size and current_chunk_lines:
                chunks.append(header + "\n" + "\n".join(current_chunk_lines))
                current_chunk_lines = []
                current_tokens = _count_tokens(header)
            current_chunk_lines.append(dline)
            current_tokens += line_tokens
        if current_chunk_lines:
            chunks.append(header + "\n" + "\n".join(current_chunk_lines))

    return chunks if chunks else [text]


def chunk_text(text: str, chunk_size: int = 500) -> list:
    """Split text into chunks by paragraphs, respecting token limits."""
    paragraphs = re.split(r"\n\s*\n", text)
    chunks = []
    current_chunk = ""

    def _count_tokens(t):
        if HAS_TIKTOKEN:
            try:
                enc = tiktoken.get_encoding("cl100k_base")
                return len(enc.encode(t))
            except Exception:
                pass
        return len(t) // 4  # rough fallback

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = _count_tokens(para)
        if para_tokens > chunk_size:
            # Split long paragraph by sentences
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
            sentences = re.split(r"(?<=[.!?])\s+", para)
            for sent in sentences:
                if _count_tokens(current_chunk + " " + sent) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = sent
                else:
                    current_chunk = (current_chunk + " " + sent).strip()
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = ""
        elif _count_tokens(current_chunk + "\n\n" + para) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para
        else:
            current_chunk = (current_chunk + "\n\n" + para).strip()

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def store_document(filename: str, content: str, username: str) -> dict:
    """Chunk, embed, and store a document. Returns doc info."""
    with config_lock:
        cs = config.get("rag_chunk_size", 500)

    doc_id = str(uuid.uuid4())
    # Use tabular chunker for Excel files (keeps headers with every chunk)
    is_tabular = filename.lower().endswith((".xlsx", ".xls")) or content.lstrip().startswith("## ") and "Spalten:" in content[:500]
    if is_tabular:
        chunks = chunk_tabular_text(content, cs)
    else:
        chunks = chunk_text(content, cs)

    conn = _get_embedding_conn()
    try:
        conn.execute(
            "INSERT INTO documents (id, filename, content, chunk_count, uploaded_by, uploaded_at) VALUES (?,?,?,?,?,?)",
            (doc_id, filename, content, len(chunks), username, time.time()),
        )

        for i, chunk_content in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            try:
                embedding = embed_text(chunk_content)
                if HAS_NUMPY:
                    emb_blob = np.array(embedding, dtype=np.float32).tobytes()
                else:
                    emb_blob = struct.pack(f"{len(embedding)}f", *embedding)
            except Exception as e:
                print(f"[embed] Error embedding chunk {i}: {e}", file=sys.stderr)
                emb_blob = b""

            token_count = len(chunk_content) // 4
            if HAS_TIKTOKEN:
                try:
                    enc = tiktoken.get_encoding("cl100k_base")
                    token_count = len(enc.encode(chunk_content))
                except Exception:
                    pass

            conn.execute(
                "INSERT INTO chunks (id, doc_id, content, chunk_index, embedding, token_count) VALUES (?,?,?,?,?,?)",
                (chunk_id, doc_id, chunk_content, i, emb_blob, token_count),
            )

        conn.commit()
    finally:
        conn.close()

    return {"doc_id": doc_id, "filename": filename, "chunk_count": len(chunks)}


def search_embeddings(query: str, top_k: int = 5) -> list:
    """Embed query and find top-K similar chunks using cosine similarity."""
    if not HAS_NUMPY:
        return []

    try:
        query_emb = np.array(embed_text(query), dtype=np.float32)
    except Exception as e:
        print(f"[rag] Error embedding query: {e}", file=sys.stderr)
        return []

    conn = _get_embedding_conn()
    try:
        cursor = conn.execute("SELECT id, doc_id, content, embedding, token_count FROM chunks WHERE embedding != ''")
        results = []
        for row in cursor.fetchall():
            chunk_id, doc_id, content, emb_blob, token_count = row
            if not emb_blob:
                continue
            try:
                emb = np.frombuffer(emb_blob, dtype=np.float32)
                if len(emb) != len(query_emb):
                    continue
                # Cosine similarity
                dot = np.dot(query_emb, emb)
                norm = np.linalg.norm(query_emb) * np.linalg.norm(emb)
                if norm == 0:
                    continue
                similarity = float(dot / norm)
                results.append({
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "content": content,
                    "similarity": round(similarity, 4),
                    "token_count": token_count,
                })
            except Exception:
                continue

        results.sort(key=lambda x: -x["similarity"])
        return results[:top_k]
    finally:
        conn.close()


def delete_document(doc_id: str) -> bool:
    """Remove document and all its chunks."""
    conn = _get_embedding_conn()
    try:
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
        conn.commit()
        return True
    except Exception as e:
        print(f"[embed] Error deleting doc {doc_id}: {e}", file=sys.stderr)
        return False
    finally:
        conn.close()


def get_embedding_stats() -> dict:
    """Return stats about the embedding store."""
    conn = _get_embedding_conn()
    try:
        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        db_size = EMBEDDINGS_DB.stat().st_size if EMBEDDINGS_DB.exists() else 0
        return {"documents": doc_count, "chunks": chunk_count, "db_size_bytes": db_size}
    except Exception:
        return {"documents": 0, "chunks": 0, "db_size_bytes": 0}
    finally:
        conn.close()


def list_documents() -> list:
    """List all documents with metadata."""
    conn = _get_embedding_conn()
    try:
        cursor = conn.execute("SELECT id, filename, chunk_count, uploaded_by, uploaded_at FROM documents ORDER BY uploaded_at DESC")
        return [
            {"id": r[0], "filename": r[1], "chunk_count": r[2], "uploaded_by": r[3], "uploaded_at": r[4]}
            for r in cursor.fetchall()
        ]
    except Exception:
        return []
    finally:
        conn.close()


def get_document_detail(doc_id: str) -> dict | None:
    """Get full document with content and all chunks."""
    conn = _get_embedding_conn()
    try:
        row = conn.execute("SELECT id, filename, content, chunk_count, uploaded_by, uploaded_at FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return None
        doc = {"id": row[0], "filename": row[1], "content": row[2], "chunk_count": row[3], "uploaded_by": row[4], "uploaded_at": row[5]}
        chunks_cursor = conn.execute("SELECT id, content, chunk_index, token_count FROM chunks WHERE doc_id=? ORDER BY chunk_index", (doc_id,))
        doc["chunks"] = [
            {"id": r[0], "content": r[1], "chunk_index": r[2], "token_count": r[3]}
            for r in chunks_cursor.fetchall()
        ]
        return doc
    except Exception:
        return None
    finally:
        conn.close()


def update_document(doc_id: str, filename: str | None, content: str | None, username: str) -> dict:
    """Update a document. If content changed, re-chunk and re-embed."""
    conn = _get_embedding_conn()
    try:
        row = conn.execute("SELECT id, filename, content, chunk_count FROM documents WHERE id=?", (doc_id,)).fetchone()
        if not row:
            return {"error": "Document not found"}

        old_filename, old_content, old_chunk_count = row[1], row[2], row[3]
        new_filename = filename if filename is not None else old_filename
        new_content = content if content is not None else old_content

        content_changed = (new_content != old_content)

        if content_changed:
            # Re-chunk and re-embed
            with config_lock:
                cs = config.get("rag_chunk_size", 500)
            chunks = chunk_text(new_content, cs)

            # Delete old chunks
            conn.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))

            # Insert new chunks
            for i, chunk_content in enumerate(chunks):
                chunk_id = str(uuid.uuid4())
                try:
                    embedding = embed_text(chunk_content)
                    if HAS_NUMPY:
                        emb_blob = np.array(embedding, dtype=np.float32).tobytes()
                    else:
                        emb_blob = struct.pack(f"{len(embedding)}f", *embedding)
                except Exception as e:
                    print(f"[embed] Error embedding chunk {i}: {e}", file=sys.stderr)
                    emb_blob = b""

                token_count = len(chunk_content) // 4
                if HAS_TIKTOKEN:
                    try:
                        enc = tiktoken.get_encoding("cl100k_base")
                        token_count = len(enc.encode(chunk_content))
                    except Exception:
                        pass

                conn.execute(
                    "INSERT INTO chunks (id, doc_id, content, chunk_index, embedding, token_count) VALUES (?,?,?,?,?,?)",
                    (chunk_id, doc_id, chunk_content, i, emb_blob, token_count),
                )

            conn.execute("UPDATE documents SET filename=?, content=?, chunk_count=?, uploaded_by=?, uploaded_at=? WHERE id=?",
                         (new_filename, new_content, len(chunks), username, time.time(), doc_id))
        else:
            # Only rename
            conn.execute("UPDATE documents SET filename=? WHERE id=?", (new_filename, doc_id))

        conn.commit()
        return {"ok": True, "doc_id": doc_id, "filename": new_filename,
                "chunk_count": len(chunks) if content_changed else old_chunk_count,
                "re_embedded": content_changed}
    except Exception as e:
        return {"error": str(e)}
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------
def compute_stats() -> dict:
    with events_lock:
        evts = list(events)

    now = time.time()
    one_hour_ago = now - 3600

    total = len(evts)
    chats = [e for e in evts if e.get("type") == "chat"]
    queries = [e for e in evts if e.get("type") == "query"]
    errors = [e for e in evts if e.get("type") == "error"]

    recent_users = set()
    recent_msgs = 0
    for e in evts:
        ts = e.get("timestamp", 0)
        if ts > one_hour_ago:
            uid = e.get("user_id")
            if uid:
                recent_users.add(uid)
            if e.get("type") == "chat":
                recent_msgs += 1

    durations = [c.get("duration_ms", 0) for c in chats if c.get("duration_ms")]
    avg_response = (sum(durations) / len(durations)) if durations else 0

    query_durations = [q.get("duration_ms", 0) for q in queries if q.get("duration_ms")]
    avg_query = (sum(query_durations) / len(query_durations)) if query_durations else 0

    error_rate = (len(errors) / total * 100) if total > 0 else 0

    all_questions = " ".join(c.get("question", "") for c in chats)
    topics = extract_topics(all_questions, top_n=10)

    hourly = defaultdict(int)
    twenty_four_ago = now - 86400
    for e in evts:
        ts = e.get("timestamp", 0)
        if ts > twenty_four_ago and e.get("type") == "chat":
            hour = datetime.fromtimestamp(ts).strftime("%H:00")
            hourly[hour] += 1
    hourly_sorted = dict(sorted(hourly.items()))

    with moderation_lock:
        flagged_count = sum(1 for m in moderation.values() if m.get("flagged"))
        reviewed_count = sum(1 for m in moderation.values() if m.get("reviewed"))
        unreviewed = flagged_count - reviewed_count

    type_counts = defaultdict(int)
    for e in evts:
        type_counts[e.get("type", "unknown")] += 1

    with users_lock:
        blocked_count = sum(1 for u in users.values() if u.get("status") == "blocked")
        total_users = len(users)

    return {
        "total_events": total,
        "total_chats": len(chats),
        "total_queries": len(queries),
        "total_errors": len(errors),
        "active_users": len(recent_users),
        "total_users": total_users,
        "blocked_users": blocked_count,
        "messages_last_hour": recent_msgs,
        "avg_response_ms": round(avg_response, 1),
        "avg_query_ms": round(avg_query, 1),
        "error_rate": round(error_rate, 2),
        "topics": topics,
        "hourly": hourly_sorted,
        "flagged": flagged_count,
        "unreviewed": unreviewed,
        "reviewed": reviewed_count,
        "type_counts": dict(type_counts),
    }


# ---------------------------------------------------------------------------
# SSE
# ---------------------------------------------------------------------------
def broadcast_event(evt: dict):
    data = json.dumps(evt, ensure_ascii=False)
    dead = []
    with sse_lock:
        for client_entry in sse_clients:
            q = client_entry[0]
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(client_entry)
        for item in dead:
            sse_clients.remove(item)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class MonitorHandler(SimpleHTTPRequestHandler):

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "GET /api/stream" in msg:
            return
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)

    def _send_json(self, data, status=200, headers=None):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        if headers:
            for k, v in headers:
                self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    # --- Auth helpers ---

    def _get_session(self) -> tuple[dict | None, str]:
        """Parse Cookie header, return (session_dict_or_None, session_id)."""
        cookie = self.headers.get("Cookie", "")
        sid = ""
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith("session="):
                sid = part[8:]
                break
        return get_session_by_id(sid), sid

    # Role hierarchy: admin > mitarbeiter > user
    _ROLE_LEVEL = {"admin": 3, "mitarbeiter": 2, "user": 1}

    def _require_auth(self, role: str = "user") -> dict | None:
        """Returns session if authorized, sends 401/403 and returns None otherwise.
        Role check uses hierarchy: admin > mitarbeiter > user."""
        session, _ = self._get_session()
        if not session:
            self._send_json({"error": "not_authenticated"}, 401)
            return None
        required_level = self._ROLE_LEVEL.get(role, 1)
        user_level = self._ROLE_LEVEL.get(session.get("role", "user"), 1)
        if user_level < required_level:
            self._send_json({"error": "forbidden"}, 403)
            return None
        return session

    def _session_cookie(self, sid: str) -> tuple[str, str]:
        return ("Set-Cookie", f"session={sid}; Path=/; HttpOnly; SameSite=Lax; Max-Age={SESSION_TTL}")

    def _clear_cookie(self) -> tuple[str, str]:
        return ("Set-Cookie", "session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ----- POST -----

    def do_POST(self):
        path = urlparse(self.path).path

        # Public routes (no auth required)
        if path == "/event":
            return self._handle_post_event()
        if path == "/api/auth/login":
            return self._handle_login()
        if path == "/api/auth/register":
            return self._handle_register()
        if path == "/api/auth/logout":
            return self._handle_logout()

        # User-level routes
        if path == "/api/chat":
            session = self._require_auth("user")
            if not session:
                return
            return self._handle_chat(session)
        if path == "/api/chat/upload":
            session = self._require_auth("user")
            if not session:
                return
            return self._handle_chat_file_upload(session)
        if path == "/api/chat/clear":
            session = self._require_auth("user")
            if not session:
                return
            return self._handle_chat_clear(session)

        # Admin-level routes
        if path == "/api/openai/auth/start":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_openai_auth_start()
        if path == "/api/openai/auth/callback":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_openai_auth_callback()
        if path == "/api/openrouter/exchange":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_openrouter_exchange()
        if path == "/api/clear":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_clear()
        if path == "/api/config":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_post_config()
        if path.startswith("/api/moderate/"):
            session = self._require_auth("admin")
            if not session:
                return
            event_id = path.split("/api/moderate/", 1)[1]
            return self._handle_moderate(event_id)
        if path.startswith("/api/accounts/"):
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_account_action(path)
        if path.startswith("/api/users/"):
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_user_action(path)

        # Database endpoints (admin-only)
        if path == "/api/db/test":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_db_test()
        if path == "/api/db/query":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_db_query()

        # Embedding endpoints (admin-only)
        if path == "/api/embeddings/upload":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_embeddings_upload(session)
        if path == "/api/embeddings/text":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_embeddings_text(session)

        self._send_json({"error": "Not found"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/api/embeddings/documents/"):
            session = self._require_auth("admin")
            if not session:
                return
            doc_id = path.split("/api/embeddings/documents/", 1)[1].rstrip("/")
            return self._handle_embeddings_delete(doc_id)
        self._send_json({"error": "Not found"}, 404)

    def do_PUT(self):
        path = urlparse(self.path).path
        if path.startswith("/api/embeddings/documents/"):
            session = self._require_auth("admin")
            if not session:
                return
            doc_id = path.split("/api/embeddings/documents/", 1)[1].rstrip("/")
            return self._handle_embeddings_update(doc_id, session)
        self._send_json({"error": "Not found"}, 404)

    # --- Auth endpoint handlers ---

    def _handle_login(self):
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        username = data.get("username", "").strip()
        password = data.get("password", "")
        if not username or not password:
            self._send_json({"error": "username and password required"}, 400)
            return

        with accounts_lock:
            acct = accounts.get(username)
        if not acct or not _verify_password(password, acct["password_hash"], acct["salt"]):
            self._send_json({"error": "invalid_credentials"}, 401)
            return

        sid = create_session(username, acct["role"])
        self._send_json(
            {"ok": True, "user": {"username": username, "role": acct["role"]}},
            headers=[self._session_cookie(sid)],
        )

    def _handle_register(self):
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        username = data.get("username", "").strip()
        password = data.get("password", "")
        if not username or not password:
            self._send_json({"error": "username and password required"}, 400)
            return
        if len(username) < 2:
            self._send_json({"error": "username too short"}, 400)
            return
        if len(password) < 3:
            self._send_json({"error": "password too short"}, 400)
            return

        with accounts_lock:
            if username in accounts:
                self._send_json({"error": "username_taken"}, 409)
                return

        salt = secrets.token_hex(16)
        with accounts_lock:
            accounts[username] = {
                "username": username,
                "password_hash": _hash_password(password, salt),
                "salt": salt,
                "role": "user",
                "priority": "normal",
                "created_at": time.time(),
            }
        save_accounts()

        sid = create_session(username, "user")
        self._send_json(
            {"ok": True, "user": {"username": username, "role": "user"}},
            headers=[self._session_cookie(sid)],
        )

    def _handle_logout(self):
        session, sid = self._get_session()
        if sid:
            clear_chat_history(sid)
            delete_session(sid)
        self._send_json({"ok": True}, headers=[self._clear_cookie()])

    def _handle_openrouter_exchange(self):
        """Exchange an OpenRouter OAuth code for an API key."""
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        code = data.get("code", "").strip()
        if not code:
            self._send_json({"error": "code is required"}, 400)
            return

        try:
            req_body = json.dumps({"code": code}).encode("utf-8")
            req = Request(
                "https://openrouter.ai/api/v1/auth/keys",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            resp = urlopen(req, timeout=10)
            result = json.loads(resp.read())
            api_key = result.get("key", "")
            if not api_key:
                self._send_json({"error": "No key returned from OpenRouter"}, 502)
                return

            with config_lock:
                config["openrouter_api_key"] = api_key
                config["provider"] = "openrouter"
            save_config()

            self._send_json({"ok": True, "provider": "openrouter"})
        except Exception as e:
            self._send_json({"error": f"OpenRouter exchange failed: {e}"}, 502)

    def _handle_chat_file_upload(self, session: dict):
        """POST /api/chat/upload — Upload a file and return parsed text for chat context."""
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "Content-Type must be multipart/form-data"}, 400)
            return

        try:
            boundary = None
            for part in ctype.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[9:].strip('"')
                    break
            if not boundary:
                self._send_json({"error": "No boundary"}, 400)
                return

            body = self._read_body()
            parts = body.split(("--" + boundary).encode())
            filename = ""
            file_bytes = b""
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue
                headers_raw = part[:header_end].decode("utf-8", errors="replace")
                body_raw = part[header_end + 4:]
                if body_raw.endswith(b"\r\n"):
                    body_raw = body_raw[:-2]
                fn_match = re.search(r'filename="([^"]*)"', headers_raw)
                if fn_match and fn_match.group(1):
                    filename = fn_match.group(1)
                    file_bytes = body_raw

            if not file_bytes:
                self._send_json({"error": "No file content"}, 400)
                return
            if not filename:
                filename = "upload.txt"

            # Parse file content
            if filename.lower().endswith((".xlsx", ".xls")):
                if not HAS_OPENPYXL:
                    self._send_json({"error": "Excel not supported (openpyxl missing)"}, 400)
                    return
                text = parse_excel(file_bytes)
            else:
                text = file_bytes.decode("utf-8", errors="replace")

            # Truncate if too large (max ~8000 tokens ≈ 32000 chars)
            max_chars = 32000
            truncated = False
            if len(text) > max_chars:
                text = text[:max_chars]
                truncated = True

            self._send_json({
                "ok": True,
                "filename": filename,
                "content": text,
                "char_count": len(text),
                "truncated": truncated,
            })
        except Exception as e:
            self._send_json({"error": f"File parse failed: {e}"}, 500)

    def _handle_chat_clear(self, session: dict):
        """Clear chat history for the current session."""
        _, sid = self._get_session()
        if sid:
            clear_chat_history(sid)
        self._send_json({"ok": True})

    def _handle_openai_auth_start(self):
        """Generate PKCE verifier/challenge and return auth URL."""
        verifier = _pkce_code_verifier()
        challenge = _pkce_code_challenge(verifier)
        state = secrets.token_urlsafe(32)

        # Store pending PKCE state
        with _pkce_lock:
            # Clean old entries (> 10 min)
            now = time.time()
            expired = [k for k, v in _openai_pkce_pending.items()
                       if now - v["created_at"] > 600]
            for k in expired:
                del _openai_pkce_pending[k]
            _openai_pkce_pending[state] = {
                "verifier": verifier,
                "created_at": now,
            }

        # Must use port 1455 + /auth/callback — that's what's registered with the Codex client_id
        callback_url = f"http://localhost:{OPENAI_CALLBACK_PORT}/auth/callback"
        params = urlencode({
            "response_type": "code",
            "client_id": OPENAI_CLIENT_ID,
            "redirect_uri": callback_url,
            "scope": OPENAI_SCOPE,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": state,
            "codex_cli_simplified_flow": "true",
            "id_token_add_organizations": "true",
        })
        auth_url = f"{OPENAI_AUTH_URL}?{params}"

        # Start temporary callback server on port 1455 to catch the redirect
        _start_oauth_callback_server(state)

        self._send_json({"auth_url": auth_url, "state": state})

    def _handle_openai_auth_callback(self):
        """Exchange authorization code + PKCE verifier for tokens, then obtain API key."""
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        code = data.get("code", "").strip()
        state = data.get("state", "").strip()
        if not code or not state:
            self._send_json({"error": "code and state are required"}, 400)
            return

        # Look up PKCE verifier
        with _pkce_lock:
            pending = _openai_pkce_pending.pop(state, None)
        if not pending:
            self._send_json({"error": "invalid or expired state"}, 400)
            return

        verifier = pending["verifier"]
        callback_url = f"http://localhost:{OPENAI_CALLBACK_PORT}/auth/callback"

        try:
            # Step 1: Exchange auth code for tokens (access_token, id_token, refresh_token)
            token_body = urlencode({
                "grant_type": "authorization_code",
                "client_id": OPENAI_CLIENT_ID,
                "code": code,
                "redirect_uri": callback_url,
                "code_verifier": verifier,
            }).encode("utf-8")
            req = Request(
                OPENAI_TOKEN_URL, data=token_body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                method="POST",
            )
            resp = urlopen(req, timeout=10)
            result = json.loads(resp.read())

            access_token = result.get("access_token", "")
            id_token = result.get("id_token", "")
            refresh_token = result.get("refresh_token", "")
            expires_in = result.get("expires_in", 3600)

            if not access_token:
                self._send_json({"error": "No access_token returned from OpenAI"}, 502)
                return

            # Step 2: Exchange id_token for an API key (like Codex CLI does)
            # This converts the ChatGPT subscription OAuth token into a usable API key
            api_key = ""
            if id_token:
                try:
                    exchange_body = urlencode({
                        "grant_type": "urn:ietf:params:oauth:grant-type:token-exchange",
                        "client_id": OPENAI_CLIENT_ID,
                        "requested_token": "openai-api-key",
                        "subject_token": id_token,
                        "subject_token_type": "urn:ietf:params:oauth:token-type:id_token",
                    }).encode("utf-8")
                    req2 = Request(
                        OPENAI_TOKEN_URL, data=exchange_body,
                        headers={"Content-Type": "application/x-www-form-urlencoded"},
                        method="POST",
                    )
                    resp2 = urlopen(req2, timeout=10)
                    result2 = json.loads(resp2.read())
                    api_key = result2.get("access_token", "")
                    print(f"[oauth] Token exchange succeeded, got API key: {bool(api_key)}", file=sys.stderr)
                except Exception as exc:
                    print(f"[oauth] Token exchange for API key failed: {exc}", file=sys.stderr)
                    if hasattr(exc, 'read'):
                        try:
                            err_body = exc.read().decode()
                            print(f"[oauth] Exchange error body: {err_body}", file=sys.stderr)
                        except Exception:
                            pass

            with config_lock:
                config["openai_oauth_token"] = access_token
                config["openai_refresh_token"] = refresh_token
                config["openai_token_expires"] = time.time() + expires_in - 60
                if api_key:
                    config["openai_api_key"] = api_key
                config["provider"] = "openai"
            save_config()

            self._send_json({"ok": True, "provider": "openai", "has_api_key": bool(api_key)})
        except Exception as e:
            self._send_json({"error": f"OpenAI token exchange failed: {e}"}, 502)

    # --- Event handlers ---

    def _handle_post_event(self):
        try:
            body = self._read_body()
            evt = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        # Check bot_enabled
        with config_lock:
            bot_on = config.get("bot_enabled", True)
        if not bot_on:
            self._send_json({"error": "bot_disabled"}, 503)
            return

        # Check if user is blocked
        uid = evt.get("user_id", "")
        if uid and not is_user_allowed(uid):
            self._send_json({"error": "user_blocked", "user_id": uid}, 403)
            return

        evt = ingest_event(evt)
        self._send_json({"ok": True, "id": evt["id"]})

    def _handle_chat(self, session: dict):
        # Check bot_enabled
        with config_lock:
            bot_on = config.get("bot_enabled", True)
        if not bot_on:
            self._send_json({"error": "bot_disabled"}, 503)
            return

        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        question = data.get("question", "").strip()
        file_context = data.get("file_context", "").strip()
        file_name = data.get("file_name", "").strip()
        if not question:
            self._send_json({"error": "question is required"}, 400)
            return

        # If file content is attached, prepend it to the question
        if file_context:
            file_label = f"[Datei: {file_name}]" if file_name else "[Hochgeladene Datei]"
            question = f"{file_label}\n{file_context}\n\n{question}"

        # Admin can override user_id; regular users use their session username
        if session["role"] == "admin" and data.get("user_id"):
            user_id = data["user_id"]
        else:
            user_id = session["username"]

        # Check if user is blocked
        if not is_user_allowed(user_id):
            self._send_json({"error": "user_blocked", "user_id": user_id}, 403)
            return

        with config_lock:
            model = data.get("model") or config["default_model"]
            system_prompt = data.get("system_prompt", config["system_prompt"])
            temperature = data.get("temperature", config["temperature"])
            db_enabled = config.get("db_enabled", False)
            rag_enabled = config.get("rag_enabled", False)
            rag_top_k = config.get("rag_top_k", 5)

        # DSGVO: Only mitarbeiter and admin get DB/RAG access
        user_role = session.get("role", "user")
        if user_role not in ("admin", "mitarbeiter"):
            db_enabled = False
            rag_enabled = False

        # Get session ID for chat history
        _, sid = self._get_session()

        # Append user message to session history
        append_chat_message(sid, "user", question, username=user_id, model=model)

        # Build system prompt with optional DB schema and RAG context
        system_parts = []
        # Default to German if no custom system prompt is set
        if system_prompt:
            system_parts.append(system_prompt)
        else:
            system_parts.append("Antworte immer auf Deutsch, es sei denn der Benutzer schreibt in einer anderen Sprache.")

        # RAG: inject relevant knowledge base chunks
        rag_chunks = []
        has_rag_context = False
        if rag_enabled:
            try:
                rag_chunks = search_embeddings(question, rag_top_k)
                relevant = [c for c in rag_chunks if c["similarity"] > 0.3]
                if relevant:
                    has_rag_context = True
                    ctx = "\n---\n".join(c["content"] for c in relevant)
                    system_parts.append(
                        "KNOWLEDGE BASE (use this to answer factual/company questions directly — do NOT use SQL for information found here):\n---\n" + ctx + "\n---"
                    )
            except Exception as e:
                print(f"[rag] Search error: {e}", file=sys.stderr)

        # DB: inject schema and instruct LLM to generate SQL
        if db_enabled:
            schema = get_db_schema()
            if schema:
                db_instruction = (
                    "You also have access to a database. Here is the schema:\n" + schema +
                    "\n\nIMPORTANT: Only generate a SQL query (wrapped in ```sql ... ```) when the user asks about "
                    "data, records, statistics, or information that would be stored in these database tables. "
                    "If the question can be answered from the knowledge base context above or from general knowledge, "
                    "answer directly WITHOUT SQL. After I execute a SQL query, I'll give you the results to summarize."
                )
                system_parts.append(db_instruction)

        # Build messages: combined system prompt + history
        messages = []
        if system_parts:
            messages.append({"role": "system", "content": "\n\n".join(system_parts)})
        messages.extend(get_chat_history(sid))

        total_start = time.time()
        try:
            answer, duration_ms = call_llm(messages, model, temperature)
        except Exception as e:
            ingest_event({
                "type": "error", "message": str(e),
                "error_type": "ollama_error", "user_id": user_id,
            })
            self._send_json({"error": str(e)}, 502)
            return

        # NL-to-SQL: check if LLM returned a SQL block
        sql_executed = None
        if db_enabled and "```sql" in answer:
            sql_match = re.search(r"```sql\s*\n?(.*?)\n?\s*```", answer, re.DOTALL)
            if sql_match:
                sql_code = sql_match.group(1).strip()
                query_result = execute_db_query(sql_code)

                if query_result.get("error"):
                    # Error — ask LLM to explain
                    error_msg = f"SQL query failed: {query_result['error']}\nQuery: {sql_code}"
                    append_chat_message(sid, "assistant", answer, username=user_id, model=model)
                    append_chat_message(sid, "user", error_msg, username=user_id, model=model)
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content": error_msg})
                    try:
                        answer, extra_ms = call_llm(messages, model, temperature)
                        duration_ms += extra_ms
                    except Exception:
                        pass
                    sql_executed = {"sql": sql_code, "error": query_result["error"]}
                else:
                    # Success — ask LLM to summarize results
                    rows = query_result["rows"]
                    result_text = f"Query executed successfully. {query_result['row_count']} rows returned"
                    if rows:
                        # Format as markdown table for the LLM
                        cols = query_result["columns"]
                        header = " | ".join(cols)
                        sep = " | ".join("---" for _ in cols)
                        data_rows = "\n".join(" | ".join(str(r.get(c, "")) for c in cols) for r in rows[:50])
                        result_text += f":\n\n{header}\n{sep}\n{data_rows}"
                    result_text += "\n\nPlease summarize these results for the user."

                    append_chat_message(sid, "assistant", answer, username=user_id, model=model)
                    append_chat_message(sid, "user", result_text, username=user_id, model=model)
                    messages.append({"role": "assistant", "content": answer})
                    messages.append({"role": "user", "content": result_text})
                    try:
                        answer, extra_ms = call_llm(messages, model, temperature)
                        duration_ms += extra_ms
                    except Exception:
                        pass
                    sql_executed = {"sql": sql_code, "row_count": query_result["row_count"],
                                    "duration_ms": query_result["duration_ms"]}

        total_ms = round((time.time() - total_start) * 1000, 1)

        # Append assistant response to session history
        append_chat_message(sid, "assistant", answer, username=user_id, model=model)

        evt_data = {
            "type": "chat", "user_id": user_id, "question": question,
            "answer": answer, "duration_ms": total_ms, "model": model,
        }
        if sql_executed:
            evt_data["sql_executed"] = sql_executed
        evt = ingest_event(evt_data)

        resp = {
            "ok": True, "id": evt["id"], "answer": answer,
            "duration_ms": total_ms, "model": model,
        }
        if sql_executed:
            resp["sql_executed"] = sql_executed
        self._send_json(resp)

    def _handle_clear(self):
        clear_all_events()
        broadcast_event({"type": "_clear"})
        self._send_json({"ok": True})

    def _handle_post_config(self):
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        db_config_changed = False
        with config_lock:
            for key in ("ollama_url", "default_model", "system_prompt", "temperature",
                        "provider", "openai_base_url"):
                if key in data:
                    config[key] = data[key]
            # Only update API keys/tokens if a real value is sent (not masked); allow empty to clear
            for key in ("openai_api_key", "anthropic_api_key", "openrouter_api_key",
                         "openai_oauth_token", "openai_refresh_token", "db_mssql_password"):
                if key in data and not str(data[key]).startswith("..."):
                    config[key] = data[key]
            if "provider" in data and data["provider"] in ("ollama", "openai", "anthropic", "openrouter"):
                config["provider"] = data["provider"]
            if "bot_enabled" in data:
                config["bot_enabled"] = bool(data["bot_enabled"])
            # DB config keys
            for key in ("db_type", "db_mssql_server", "db_mssql_database", "db_mssql_user",
                        "db_mssql_driver", "db_sqlite_path"):
                if key in data:
                    if config.get(key) != data[key]:
                        db_config_changed = True
                    config[key] = data[key]
            if "db_enabled" in data:
                config["db_enabled"] = bool(data["db_enabled"])
            # RAG config keys
            if "rag_enabled" in data:
                config["rag_enabled"] = bool(data["rag_enabled"])
            for key in ("rag_top_k", "rag_chunk_size"):
                if key in data:
                    config[key] = int(data[key])
            for key in ("embedding_provider", "embedding_model"):
                if key in data:
                    config[key] = data[key]
        # Refresh schema cache if DB config changed
        if db_config_changed:
            with config_lock:
                config["db_schema_cache"] = ""
        save_config()
        with config_lock:
            self._send_json({"ok": True, "config": dict(config)})

    def _handle_moderate(self, event_id: str):
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        with moderation_lock:
            mod = moderation.get(event_id, {})
            if "flagged" in data:
                mod["flagged"] = bool(data["flagged"])
            if "reviewed" in data:
                mod["reviewed"] = bool(data["reviewed"])
            if "note" in data:
                mod["note"] = str(data["note"])
            mod["updated_at"] = time.time()
            moderation[event_id] = mod

        persist_moderation(event_id, mod)
        broadcast_event({
            "type": "_moderation_update", "event_id": event_id,
            "moderation": mod, "timestamp": time.time(),
        })
        self._send_json({"ok": True, "moderation": mod})

    def _handle_user_action(self, path: str):
        """POST /api/users/<id>/block | /api/users/<id>/unblock | /api/users/<id>/update"""
        parts = path.split("/api/users/", 1)[1].split("/")
        if len(parts) < 2:
            self._send_json({"error": "Invalid path"}, 400)
            return

        user_id = parts[0]
        action = parts[1]

        if action == "block":
            with users_lock:
                if user_id in users:
                    users[user_id]["status"] = "blocked"
                else:
                    users[user_id] = {
                        "user_id": user_id, "status": "blocked",
                        "first_seen": time.time(), "last_seen": time.time(),
                        "message_count": 0, "error_count": 0, "note": "",
                        "priority": "normal",
                    }
            save_users()
            broadcast_event({"type": "_user_update", "user_id": user_id, "status": "blocked"})
            self._send_json({"ok": True, "user_id": user_id, "status": "blocked"})

        elif action == "unblock":
            with users_lock:
                if user_id in users:
                    users[user_id]["status"] = "active"
            save_users()
            broadcast_event({"type": "_user_update", "user_id": user_id, "status": "active"})
            self._send_json({"ok": True, "user_id": user_id, "status": "active"})

        elif action == "update":
            try:
                body = self._read_body()
                data = json.loads(body)
            except (json.JSONDecodeError, Exception) as e:
                self._send_json({"error": str(e)}, 400)
                return
            with users_lock:
                if user_id in users:
                    if "note" in data:
                        users[user_id]["note"] = str(data["note"])
                    if "status" in data and data["status"] in ("active", "blocked", "restricted"):
                        users[user_id]["status"] = data["status"]
                    if "priority" in data and data["priority"] in ("high", "normal", "low"):
                        users[user_id]["priority"] = data["priority"]
            save_users()
            with users_lock:
                self._send_json({"ok": True, "user": users.get(user_id, {})})

        else:
            self._send_json({"error": "Unknown action"}, 400)

    def _handle_account_action(self, path: str):
        """POST /api/accounts/<name>/role | priority | delete"""
        parts = path.split("/api/accounts/", 1)[1].split("/")
        if len(parts) < 2:
            self._send_json({"error": "Invalid path"}, 400)
            return

        username = parts[0]
        action = parts[1]

        if action == "role":
            try:
                body = self._read_body()
                data = json.loads(body)
            except (json.JSONDecodeError, Exception) as e:
                self._send_json({"error": str(e)}, 400)
                return
            new_role = data.get("role")
            if new_role not in ("admin", "mitarbeiter", "user"):
                self._send_json({"error": "role must be admin, mitarbeiter, or user"}, 400)
                return
            with accounts_lock:
                if username not in accounts:
                    self._send_json({"error": "account not found"}, 404)
                    return
                accounts[username]["role"] = new_role
            save_accounts()
            # Update any active sessions for this user
            with sessions_lock:
                for s in sessions.values():
                    if s["username"] == username:
                        s["role"] = new_role
            self._send_json({"ok": True, "username": username, "role": new_role})

        elif action == "priority":
            try:
                body = self._read_body()
                data = json.loads(body)
            except (json.JSONDecodeError, Exception) as e:
                self._send_json({"error": str(e)}, 400)
                return
            priority = data.get("priority")
            if priority not in ("high", "normal", "low"):
                self._send_json({"error": "priority must be high, normal, or low"}, 400)
                return
            with accounts_lock:
                if username not in accounts:
                    self._send_json({"error": "account not found"}, 404)
                    return
                accounts[username]["priority"] = priority
            save_accounts()
            self._send_json({"ok": True, "username": username, "priority": priority})

        elif action == "delete":
            session, _ = self._get_session()
            if session and session["username"] == username:
                self._send_json({"error": "cannot delete own account"}, 400)
                return
            with accounts_lock:
                if username not in accounts:
                    self._send_json({"error": "account not found"}, 404)
                    return
                # Cannot delete last admin
                if accounts[username]["role"] == "admin":
                    admin_count = sum(1 for a in accounts.values() if a["role"] == "admin")
                    if admin_count <= 1:
                        self._send_json({"error": "cannot delete last admin"}, 400)
                        return
                del accounts[username]
            save_accounts()
            # Remove sessions for deleted user
            with sessions_lock:
                to_del = [k for k, v in sessions.items() if v["username"] == username]
                for k in to_del:
                    del sessions[k]
            self._send_json({"ok": True, "username": username})

        else:
            self._send_json({"error": "Unknown action"}, 400)

    # ----- GET -----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        # Public routes (no auth required)
        if path == "/auth/callback":
            return self._handle_openai_oauth_redirect(params)
        if path == "/api/auth/me":
            return self._handle_me()
        if path.startswith("/api/users/") and path.endswith("/check"):
            user_id = path.split("/api/users/", 1)[1].replace("/check", "")
            allowed = is_user_allowed(user_id)
            with users_lock:
                u = users.get(user_id, {})
                priority = u.get("priority", "normal")
            with config_lock:
                bot_enabled = config.get("bot_enabled", True)
            return self._send_json({
                "user_id": user_id, "allowed": allowed,
                "priority": priority, "bot_enabled": bot_enabled,
            })
        if path == "/api/health":
            return self._send_json({
                "status": "ok", "events": len(events),
                "users": len(users),
                "uptime_s": round(time.time() - SERVER_START),
            })

        # SSE stream (user-level auth)
        if path == "/api/stream":
            session = self._require_auth("user")
            if not session:
                return
            return self._handle_sse(session)

        # User-level endpoints
        if path == "/api/models":
            session = self._require_auth("user")
            if not session:
                return
            return self._handle_models()

        # Admin-level endpoints
        if path == "/api/events":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_get_events(params)
        if path == "/api/conversations":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_conversations(params)
        if path == "/api/stats":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_stats()
        if path == "/api/moderation":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_get_moderation()
        if path == "/api/config":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_get_config()
        if path == "/api/users":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_get_users()
        if path == "/api/accounts":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_get_accounts()
        if path == "/api/admin/chats":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_admin_chats()
        if path.startswith("/api/admin/chats/"):
            session = self._require_auth("admin")
            if not session:
                return
            chat_sid = path.split("/api/admin/chats/", 1)[1].rstrip("/")
            return self._handle_admin_chat_detail(chat_sid)
        if path.startswith("/api/users/") and path.count("/") == 3:
            session = self._require_auth("admin")
            if not session:
                return
            user_id = path.split("/api/users/", 1)[1].rstrip("/")
            return self._handle_get_user(user_id)

        # Database endpoints (admin-only)
        if path == "/api/db/schema":
            session = self._require_auth("admin")
            if not session:
                return
            return self._handle_db_schema()

        # Embedding endpoints (read: mitarbeiter+, write: admin only)
        if path == "/api/embeddings/documents":
            session = self._require_auth("mitarbeiter")
            if not session:
                return
            return self._handle_embeddings_list()
        if path.startswith("/api/embeddings/documents/"):
            session = self._require_auth("mitarbeiter")
            if not session:
                return
            doc_id = path.split("/api/embeddings/documents/", 1)[1].rstrip("/")
            return self._handle_embeddings_detail(doc_id)
        if path == "/api/embeddings/search":
            session = self._require_auth("mitarbeiter")
            if not session:
                return
            return self._handle_embeddings_search(params)
        if path == "/api/embeddings/stats":
            session = self._require_auth("mitarbeiter")
            if not session:
                return
            return self._handle_embeddings_stats()

        # Static files (public)
        self._serve_static(path)

    def _handle_me(self):
        session, _ = self._get_session()
        if not session:
            self._send_json({"error": "not_authenticated"}, 401)
            return
        self._send_json({"user": {"username": session["username"], "role": session["role"]}})

    def _handle_openai_oauth_redirect(self, params: dict):
        """Serve a small HTML page that relays the OAuth code+state to the main dashboard."""
        code = (params.get("code") or [""])[0]
        state = (params.get("state") or [""])[0]
        error = (params.get("error") or [""])[0]
        # Redirect to dashboard root with the params so the JS can pick them up
        if error:
            html = f"""<!DOCTYPE html><html><body>
            <h2>OpenAI Login Failed</h2><p>Error: {error}</p>
            <p><a href="/">Back to Dashboard</a></p></body></html>"""
        else:
            # Redirect to / with code+state so the existing JS handler picks it up
            html = f"""<!DOCTYPE html><html><head>
            <meta http-equiv="refresh" content="0;url=/?code={code}&state={state}">
            </head><body>Redirecting...</body></html>"""
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _serve_static(self, path: str):
        if path == "/" or path == "":
            path = "/index.html"
        filepath = PUBLIC_DIR / path.lstrip("/")
        if not filepath.exists() or not filepath.is_file():
            self._send_json({"error": "Not found"}, 404)
            return

        content = filepath.read_bytes()
        ct = "text/html"
        if path.endswith(".js"):
            ct = "application/javascript"
        elif path.endswith(".css"):
            ct = "text/css"
        elif path.endswith(".json"):
            ct = "application/json"
        elif path.endswith(".png"):
            ct = "image/png"
        elif path.endswith(".svg"):
            ct = "image/svg+xml"

        self.send_response(200)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _handle_sse(self, session: dict):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=500)
        session_info = {"username": session["username"], "role": session["role"]}
        client_entry = (q, session_info)
        with sse_lock:
            sse_clients.append(client_entry)

        try:
            # Send session info first
            session_data = json.dumps(session_info, ensure_ascii=False)
            self.wfile.write(f"event: session\ndata: {session_data}\n\n".encode())
            self.wfile.flush()

            if session["role"] == "admin":
                # Admin gets all events, moderation, and users
                with events_lock:
                    init = list(events)
                init_data = json.dumps(init, ensure_ascii=False)
                self.wfile.write(f"event: init\ndata: {init_data}\n\n".encode())
                self.wfile.flush()

                with moderation_lock:
                    mod_data = json.dumps(moderation, ensure_ascii=False)
                self.wfile.write(f"event: moderation\ndata: {mod_data}\n\n".encode())
                self.wfile.flush()

                with users_lock:
                    users_data = json.dumps(users, ensure_ascii=False)
                self.wfile.write(f"event: users\ndata: {users_data}\n\n".encode())
                self.wfile.flush()
            else:
                # Non-admin users get only their own events + system events
                username = session["username"]
                with events_lock:
                    user_events = [
                        e for e in events
                        if e.get("user_id") == username
                        or e.get("type") in ("system", "_clear")
                    ]
                init_data = json.dumps(user_events, ensure_ascii=False)
                self.wfile.write(f"event: init\ndata: {init_data}\n\n".encode())
                self.wfile.flush()

            while True:
                try:
                    data = q.get(timeout=15)
                    # Filter for non-admin users
                    if session["role"] != "admin":
                        try:
                            evt = json.loads(data)
                            evt_type = evt.get("type", "")
                            evt_uid = evt.get("user_id", "")
                            # Skip admin-only internal events
                            if evt_type.startswith("_") and evt_type != "_clear":
                                continue
                            # Skip other users' events
                            if evt_uid and evt_uid != session["username"]:
                                continue
                        except json.JSONDecodeError:
                            pass
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with sse_lock:
                try:
                    sse_clients.remove(client_entry)
                except ValueError:
                    pass

    def _handle_get_events(self, params: dict):
        type_filter = params.get("type", [None])[0]
        user_filter = params.get("user_id", [None])[0]
        flagged_only = params.get("flagged", [""])[0] == "true"
        limit = int(params.get("limit", [100])[0])
        offset = int(params.get("offset", [0])[0])

        with events_lock:
            filtered = list(events)

        if type_filter:
            filtered = [e for e in filtered if e.get("type") == type_filter]
        if user_filter:
            filtered = [e for e in filtered if e.get("user_id") == user_filter]
        if flagged_only:
            with moderation_lock:
                flagged_ids = {eid for eid, m in moderation.items() if m.get("flagged")}
            filtered = [e for e in filtered if e["id"] in flagged_ids]

        total = len(filtered)
        page = filtered[offset:offset + limit]

        self._send_json({"total": total, "events": page})

    def _handle_conversations(self, params: dict):
        with events_lock:
            chats = [e for e in events if e.get("type") == "chat"]

        convos: dict[str, dict] = {}
        for c in chats:
            uid = c.get("user_id", "anonymous")
            if uid not in convos:
                convos[uid] = {
                    "user_id": uid, "message_count": 0,
                    "first_message": c.get("timestamp", 0),
                    "last_message": c.get("timestamp", 0),
                    "messages": [],
                }
            convos[uid]["message_count"] += 1
            convos[uid]["last_message"] = max(convos[uid]["last_message"], c.get("timestamp", 0))
            convos[uid]["messages"].append(c)

        result = sorted(convos.values(), key=lambda x: -x["last_message"])
        self._send_json({"conversations": result})

    def _handle_stats(self):
        self._send_json(compute_stats())

    def _handle_get_moderation(self):
        with moderation_lock:
            self._send_json({"moderation": dict(moderation)})

    def _handle_models(self):
        models = list_models()
        self._send_json({"models": models})

    def _handle_get_config(self):
        with config_lock:
            cfg = dict(config)
        # Mask API keys and tokens — show only last 4 chars
        for key in ("openai_api_key", "anthropic_api_key", "openrouter_api_key",
                     "openai_oauth_token", "openai_refresh_token", "db_mssql_password"):
            val = cfg.get(key, "")
            if val:
                cfg[key] = "..." + val[-4:]
            else:
                cfg[key] = ""
        self._send_json({"config": cfg})

    def _handle_get_users(self):
        with users_lock:
            user_list = list(users.values())
        # Enrich with live event counts
        with events_lock:
            for u in user_list:
                uid = u["user_id"]
                user_evts = [e for e in events if e.get("user_id") == uid]
                u["total_events"] = len(user_evts)
                u["total_chats"] = sum(1 for e in user_evts if e["type"] == "chat")
                u["total_errors"] = sum(1 for e in user_evts if e["type"] == "error")

        user_list.sort(key=lambda x: -x.get("last_seen", 0))
        self._send_json({"users": user_list})

    def _handle_get_user(self, user_id: str):
        self._send_json({"user": get_user_summary(user_id)})

    def _handle_admin_chats(self):
        """GET /api/admin/chats — list all active chat sessions with metadata."""
        with chat_sessions_lock:
            result = []
            for sid, messages in chat_sessions.items():
                meta = chat_session_meta.get(sid, {})
                result.append({
                    "session_id": sid,
                    "username": meta.get("username", ""),
                    "message_count": len(messages),
                    "created_at": meta.get("created_at", 0),
                    "last_activity": meta.get("last_activity", 0),
                    "model": meta.get("model", ""),
                })
        result.sort(key=lambda x: -x["last_activity"])
        self._send_json({"sessions": result})

    def _handle_admin_chat_detail(self, session_id: str):
        """GET /api/admin/chats/<session_id> — return full message history + metadata."""
        with chat_sessions_lock:
            messages = list(chat_sessions.get(session_id, []))
            meta = dict(chat_session_meta.get(session_id, {}))
        if not messages and not meta:
            self._send_json({"error": "session not found"}, 404)
            return
        self._send_json({"session_id": session_id, "meta": meta, "messages": messages})

    def _handle_get_accounts(self):
        with accounts_lock:
            acct_list = []
            for a in accounts.values():
                acct_list.append({
                    "username": a["username"],
                    "role": a["role"],
                    "priority": a.get("priority", "normal"),
                    "created_at": a.get("created_at"),
                })
        self._send_json({"accounts": acct_list})

    # --- Database endpoint handlers ---

    def _handle_db_test(self):
        """POST /api/db/test — Test database connection."""
        conn = get_db_connection()
        if conn is None:
            with config_lock:
                db_type = config.get("db_type", "")
                db_enabled = config.get("db_enabled", False)
            if not db_type:
                self._send_json({"ok": False, "error": "No database type configured"})
                return
            if not db_enabled:
                self._send_json({"ok": False, "error": "Database is not enabled"})
                return
            if db_type == "mssql" and not HAS_PYODBC:
                self._send_json({"ok": False, "error": "pyodbc is not installed. Run: pip install pyodbc"})
                return
            self._send_json({"ok": False, "error": "Could not connect to database — check your settings"})
            return

        schema = get_db_schema(force_refresh=True)
        table_count = schema.count("Table:")
        self._send_json({"ok": True, "tables": table_count, "schema_preview": schema[:500]})

    def _handle_db_schema(self):
        """GET /api/db/schema — Return cached schema."""
        schema = get_db_schema()
        self._send_json({"schema": schema})

    def _handle_db_query(self):
        """POST /api/db/query — Execute raw SQL (admin only, for testing)."""
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        sql = data.get("sql", "").strip()
        if not sql:
            self._send_json({"error": "sql is required"}, 400)
            return

        result = execute_db_query(sql)
        self._send_json(result)

    # --- Embedding endpoint handlers ---

    def _handle_embeddings_list(self):
        """GET /api/embeddings/documents — List all documents."""
        docs = list_documents()
        self._send_json({"documents": docs})

    def _handle_embeddings_upload(self, session: dict):
        """POST /api/embeddings/upload — Upload a text file (multipart/form-data)."""
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            self._send_json({"error": "Content-Type must be multipart/form-data"}, 400)
            return

        try:
            # Parse boundary from content type
            boundary = None
            for part in ctype.split(";"):
                part = part.strip()
                if part.startswith("boundary="):
                    boundary = part[9:].strip('"')
                    break
            if not boundary:
                self._send_json({"error": "No boundary in Content-Type"}, 400)
                return

            body = self._read_body()
            # Parse multipart manually (avoid cgi.FieldStorage deprecation)
            parts = body.split(("--" + boundary).encode())
            filename = ""
            file_bytes = b""
            for part in parts:
                if b"Content-Disposition" not in part:
                    continue
                # Extract headers and body
                header_end = part.find(b"\r\n\r\n")
                if header_end == -1:
                    continue
                headers_raw = part[:header_end].decode("utf-8", errors="replace")
                body_raw = part[header_end + 4:]
                # Strip trailing \r\n--
                if body_raw.endswith(b"\r\n"):
                    body_raw = body_raw[:-2]

                # Get filename if present
                fn_match = re.search(r'filename="([^"]*)"', headers_raw)
                name_match = re.search(r'name="([^"]*)"', headers_raw)

                if fn_match and fn_match.group(1):
                    filename = fn_match.group(1)
                    file_bytes = body_raw
                elif name_match and name_match.group(1) == "filename":
                    filename = body_raw.decode("utf-8", errors="replace").strip()

            if not file_bytes:
                self._send_json({"error": "No file content found"}, 400)
                return
            if not filename:
                filename = "upload.txt"

            # Handle Excel files
            if filename.lower().endswith((".xlsx", ".xls")):
                if not HAS_OPENPYXL:
                    self._send_json({"error": "Excel support requires openpyxl. Run: pip install openpyxl"}, 400)
                    return
                file_content = parse_excel(file_bytes)
            else:
                file_content = file_bytes.decode("utf-8", errors="replace")

            result = store_document(filename, file_content, session["username"])
            self._send_json({"ok": True, **result})
        except Exception as e:
            self._send_json({"error": f"Upload failed: {e}"}, 500)

    def _handle_embeddings_text(self, session: dict):
        """POST /api/embeddings/text — Upload raw text with title."""
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        title = data.get("title", "").strip() or "Untitled"
        content = data.get("content", "").strip()
        if not content:
            self._send_json({"error": "content is required"}, 400)
            return

        try:
            result = store_document(title, content, session["username"])
            self._send_json({"ok": True, **result})
        except Exception as e:
            self._send_json({"error": f"Embedding failed: {e}"}, 500)

    def _handle_embeddings_delete(self, doc_id: str):
        """DELETE /api/embeddings/documents/<id> — Delete a document."""
        ok = delete_document(doc_id)
        if ok:
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "Failed to delete document"}, 500)

    def _handle_embeddings_detail(self, doc_id: str):
        """GET /api/embeddings/documents/<id> — Full document with content and chunks."""
        doc = get_document_detail(doc_id)
        if not doc:
            self._send_json({"error": "Document not found"}, 404)
            return
        self._send_json({"document": doc})

    def _handle_embeddings_update(self, doc_id: str, session: dict):
        """PUT /api/embeddings/documents/<id> — Update document name/content, re-embed if changed."""
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        filename = data.get("filename")
        content = data.get("content")
        if filename is None and content is None:
            self._send_json({"error": "Nothing to update"}, 400)
            return

        try:
            result = update_document(doc_id, filename, content, session["username"])
            if "error" in result:
                self._send_json(result, 400)
            else:
                self._send_json(result)
        except Exception as e:
            self._send_json({"error": f"Update failed: {e}"}, 500)

    def _handle_embeddings_search(self, params: dict):
        """GET /api/embeddings/search?q=... — Test search."""
        query = (params.get("q") or [""])[0]
        if not query:
            self._send_json({"error": "q parameter required"}, 400)
            return
        with config_lock:
            top_k = config.get("rag_top_k", 5)
        try:
            results = search_embeddings(query, top_k)
            self._send_json({"results": results})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_embeddings_stats(self):
        """GET /api/embeddings/stats — Total docs, chunks, DB size."""
        self._send_json(get_embedding_stats())


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


SERVER_START = time.time()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_config()
    load_users()
    load_events()
    load_accounts()
    ensure_default_admin()
    rebuild_users_from_events()
    save_users()
    init_embedding_db()
    print(f"[chatbot-monitor] Loaded {len(events)} events, {len(users)} users, {len(accounts)} accounts")
    print(f"[chatbot-monitor] Ollama: {config['ollama_url']}  Model: {config['default_model']}")
    print(f"[chatbot-monitor] DB: type={config.get('db_type','none')} enabled={config.get('db_enabled',False)}")
    print(f"[chatbot-monitor] RAG: enabled={config.get('rag_enabled',False)} embeddings_db={EMBEDDINGS_DB}")

    server = ThreadedHTTPServer(("0.0.0.0", PORT), MonitorHandler)
    print(f"[chatbot-monitor] Dashboard: http://localhost:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[chatbot-monitor] Shutting down...")
        save_users()
        save_accounts()
        server.shutdown()


if __name__ == "__main__":
    main()
