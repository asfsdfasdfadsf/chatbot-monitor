#!/usr/bin/env python3
"""WaWi Chatbot Admin Monitor — zero-dependency server on port 7779.

Receives events from the chatbot SDK, proxies chat requests to Ollama,
manages users (block/allow), serves the admin dashboard, and pushes
live updates via SSE.
"""

import json
import os
import re
import sys
import time
import uuid
import threading
import queue
from collections import defaultdict
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from urllib.request import Request, urlopen
from urllib.error import URLError
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = 7779
MAX_EVENTS = 5000
DATA_DIR = Path(__file__).resolve().parent / "data"
EVENTS_FILE = DATA_DIR / "events.jsonl"
CONFIG_FILE = DATA_DIR / "config.json"
USERS_FILE = DATA_DIR / "users.json"
PUBLIC_DIR = Path(__file__).resolve().parent / "public"

config = {
    "ollama_url": "http://localhost:11434",
    "default_model": "llama3:8b",
    "system_prompt": "",
    "temperature": 0.7,
}
config_lock = threading.Lock()

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
events: list[dict] = []
events_lock = threading.Lock()
moderation: dict[str, dict] = {}
moderation_lock = threading.Lock()
sse_clients: list[queue.Queue] = []
sse_lock = threading.Lock()

# User management: user_id -> { status, first_seen, last_seen, ... }
# status: "active" | "blocked" | "restricted"
users: dict[str, dict] = {}
users_lock = threading.Lock()

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
# Ollama proxy
# ---------------------------------------------------------------------------
def call_ollama(question: str, model: str, system_prompt: str, temperature: float) -> tuple[str, float]:
    with config_lock:
        ollama_url = config["ollama_url"]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})

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


def list_ollama_models() -> list[dict]:
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
        for q in sse_clients:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------
class MonitorHandler(SimpleHTTPRequestHandler):

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "GET /api/stream" in msg:
            return
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", file=sys.stderr)

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length)

    def _cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ----- POST -----

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/event":
            return self._handle_post_event()
        if path == "/api/chat":
            return self._handle_chat()
        if path == "/api/clear":
            return self._handle_clear()
        if path == "/api/config":
            return self._handle_post_config()
        if path.startswith("/api/moderate/"):
            event_id = path.split("/api/moderate/", 1)[1]
            return self._handle_moderate(event_id)
        if path.startswith("/api/users/"):
            return self._handle_user_action(path)

        self._send_json({"error": "Not found"}, 404)

    def _handle_post_event(self):
        try:
            body = self._read_body()
            evt = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        # Check if user is blocked
        uid = evt.get("user_id", "")
        if uid and not is_user_allowed(uid):
            self._send_json({"error": "user_blocked", "user_id": uid}, 403)
            return

        evt = ingest_event(evt)
        self._send_json({"ok": True, "id": evt["id"]})

    def _handle_chat(self):
        try:
            body = self._read_body()
            data = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        question = data.get("question", "").strip()
        if not question:
            self._send_json({"error": "question is required"}, 400)
            return

        user_id = data.get("user_id", "admin")

        # Check if user is blocked
        if not is_user_allowed(user_id):
            self._send_json({"error": "user_blocked", "user_id": user_id}, 403)
            return

        with config_lock:
            model = data.get("model") or config["default_model"]
            system_prompt = data.get("system_prompt", config["system_prompt"])
            temperature = data.get("temperature", config["temperature"])

        try:
            answer, duration_ms = call_ollama(question, model, system_prompt, temperature)
        except Exception as e:
            ingest_event({
                "type": "error", "message": str(e),
                "error_type": "ollama_error", "user_id": user_id,
            })
            self._send_json({"error": str(e)}, 502)
            return

        evt = ingest_event({
            "type": "chat", "user_id": user_id, "question": question,
            "answer": answer, "duration_ms": duration_ms, "model": model,
        })

        self._send_json({
            "ok": True, "id": evt["id"], "answer": answer,
            "duration_ms": duration_ms, "model": model,
        })

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

        with config_lock:
            for key in ("ollama_url", "default_model", "system_prompt", "temperature"):
                if key in data:
                    config[key] = data[key]
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
            save_users()
            with users_lock:
                self._send_json({"ok": True, "user": users.get(user_id, {})})

        else:
            self._send_json({"error": "Unknown action"}, 400)

    # ----- GET -----

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/api/stream":
            return self._handle_sse()
        if path == "/api/events":
            return self._handle_get_events(params)
        if path == "/api/conversations":
            return self._handle_conversations(params)
        if path == "/api/stats":
            return self._handle_stats()
        if path == "/api/moderation":
            return self._handle_get_moderation()
        if path == "/api/models":
            return self._handle_models()
        if path == "/api/config":
            return self._handle_get_config()
        if path == "/api/users":
            return self._handle_get_users()
        if path.startswith("/api/users/") and path.count("/") == 3:
            # GET /api/users/<id>
            user_id = path.split("/api/users/", 1)[1].rstrip("/")
            return self._handle_get_user(user_id)
        if path.startswith("/api/users/") and path.endswith("/check"):
            user_id = path.split("/api/users/", 1)[1].replace("/check", "")
            allowed = is_user_allowed(user_id)
            return self._send_json({"user_id": user_id, "allowed": allowed})
        if path == "/api/health":
            return self._send_json({
                "status": "ok", "events": len(events),
                "users": len(users),
                "uptime_s": round(time.time() - SERVER_START),
            })

        self._serve_static(path)

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

    def _handle_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        q: queue.Queue = queue.Queue(maxsize=500)
        with sse_lock:
            sse_clients.append(q)

        try:
            with events_lock:
                init = list(events)
            init_data = json.dumps(init, ensure_ascii=False)
            self.wfile.write(f"event: init\ndata: {init_data}\n\n".encode())
            self.wfile.flush()

            with moderation_lock:
                mod_data = json.dumps(moderation, ensure_ascii=False)
            self.wfile.write(f"event: moderation\ndata: {mod_data}\n\n".encode())
            self.wfile.flush()

            # Send users state
            with users_lock:
                users_data = json.dumps(users, ensure_ascii=False)
            self.wfile.write(f"event: users\ndata: {users_data}\n\n".encode())
            self.wfile.flush()

            while True:
                try:
                    data = q.get(timeout=15)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

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
        models = list_ollama_models()
        self._send_json({"models": models})

    def _handle_get_config(self):
        with config_lock:
            self._send_json({"config": dict(config)})

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
    rebuild_users_from_events()
    save_users()
    print(f"[chatbot-monitor] Loaded {len(events)} events, {len(users)} users")
    print(f"[chatbot-monitor] Ollama: {config['ollama_url']}  Model: {config['default_model']}")

    server = ThreadedHTTPServer(("0.0.0.0", PORT), MonitorHandler)
    print(f"[chatbot-monitor] Dashboard: http://localhost:{PORT}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[chatbot-monitor] Shutting down...")
        save_users()
        server.shutdown()


if __name__ == "__main__":
    main()
