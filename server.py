#!/usr/bin/env python3
"""WaWi Chatbot Admin Monitor — zero-dependency server on port 7779.

Receives events from the chatbot SDK (monitor.py), persists them to JSONL,
serves the admin dashboard, and pushes live updates via SSE.
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
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PORT = 7779
MAX_EVENTS = 5000
DATA_DIR = Path(__file__).resolve().parent / "data"
EVENTS_FILE = DATA_DIR / "events.jsonl"
PUBLIC_DIR = Path(__file__).resolve().parent / "public"

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
events: list[dict] = []
events_lock = threading.Lock()
moderation: dict[str, dict] = {}  # event_id -> {flagged, reviewed, note}
moderation_lock = threading.Lock()
sse_clients: list[queue.Queue] = []
sse_lock = threading.Lock()

# German + English stop words for topic extraction
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
    """Extract top-N topic words from text."""
    words = re.findall(r"\b[a-zA-ZäöüÄÖÜß]{3,}\b", text.lower())
    freq: dict[str, int] = defaultdict(int)
    for w in words:
        if w not in STOP_WORDS:
            freq[w] += 1
    return [w for w, _ in sorted(freq.items(), key=lambda x: -x[1])[:top_n]]


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_events():
    """Load events from JSONL on startup."""
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
                    # Restore moderation state
                    mod = evt.get("_moderation")
                    if mod:
                        moderation[evt["id"]] = mod
                except json.JSONDecodeError:
                    continue
    # Keep only the last MAX_EVENTS
    events = loaded[-MAX_EVENTS:]


def append_event_to_file(evt: dict):
    """Append a single event to the JSONL file."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[persist] Error writing event: {e}", file=sys.stderr)


def persist_moderation(event_id: str, mod: dict):
    """Update the moderation state in the JSONL file (rewrite approach)."""
    # For simplicity, we just append a moderation-update marker.
    # On load, the last moderation state wins via the _moderation field.
    try:
        with events_lock:
            for evt in events:
                if evt["id"] == event_id:
                    evt["_moderation"] = mod
                    # Re-persist the event line
                    break
        # Rewrite the whole file (safe for <5000 events)
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVENTS_FILE, "w", encoding="utf-8") as f:
            with events_lock:
                for evt in events:
                    f.write(json.dumps(evt, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[persist] Error writing moderation: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Stats engine
# ---------------------------------------------------------------------------
def compute_stats() -> dict:
    """Compute dashboard statistics from current events."""
    with events_lock:
        evts = list(events)

    now = time.time()
    one_hour_ago = now - 3600

    total = len(evts)
    chats = [e for e in evts if e.get("type") == "chat"]
    queries = [e for e in evts if e.get("type") == "query"]
    errors = [e for e in evts if e.get("type") == "error"]

    # Active users (last hour)
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

    # Average response time
    durations = [c.get("duration_ms", 0) for c in chats if c.get("duration_ms")]
    avg_response = (sum(durations) / len(durations)) if durations else 0

    # Query times
    query_durations = [q.get("duration_ms", 0) for q in queries if q.get("duration_ms")]
    avg_query = (sum(query_durations) / len(query_durations)) if query_durations else 0

    # Error rate
    error_rate = (len(errors) / total * 100) if total > 0 else 0

    # Top topics from chat questions
    all_questions = " ".join(c.get("question", "") for c in chats)
    topics = extract_topics(all_questions, top_n=10)

    # Hourly distribution (last 24h)
    hourly = defaultdict(int)
    twenty_four_ago = now - 86400
    for e in evts:
        ts = e.get("timestamp", 0)
        if ts > twenty_four_ago and e.get("type") == "chat":
            hour = datetime.fromtimestamp(ts).strftime("%H:00")
            hourly[hour] += 1
    hourly_sorted = dict(sorted(hourly.items()))

    # Moderation stats
    with moderation_lock:
        flagged_count = sum(1 for m in moderation.values() if m.get("flagged"))
        reviewed_count = sum(1 for m in moderation.values() if m.get("reviewed"))
        unreviewed = flagged_count - reviewed_count

    # Type breakdown
    type_counts = defaultdict(int)
    for e in evts:
        type_counts[e.get("type", "unknown")] += 1

    return {
        "total_events": total,
        "total_chats": len(chats),
        "total_queries": len(queries),
        "total_errors": len(errors),
        "active_users": len(recent_users),
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
    """Push event to all connected SSE clients."""
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
    """Handles API requests and serves static files."""

    def log_message(self, fmt, *args):
        # Suppress noisy SSE keep-alive logs
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
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    # ----- POST endpoints -----

    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/event":
            return self._handle_post_event()
        if path.startswith("/api/moderate/"):
            event_id = path.split("/api/moderate/", 1)[1]
            return self._handle_moderate(event_id)

        self._send_json({"error": "Not found"}, 404)

    def _handle_post_event(self):
        """Ingest an event from the chatbot SDK."""
        try:
            body = self._read_body()
            evt = json.loads(body)
        except (json.JSONDecodeError, Exception) as e:
            self._send_json({"error": str(e)}, 400)
            return

        # Ensure required fields
        if "id" not in evt:
            evt["id"] = str(uuid.uuid4())
        if "timestamp" not in evt:
            evt["timestamp"] = time.time()
        if "type" not in evt:
            evt["type"] = "system"

        with events_lock:
            events.append(evt)
            # Ring buffer
            while len(events) > MAX_EVENTS:
                events.pop(0)

        append_event_to_file(evt)
        broadcast_event(evt)
        self._send_json({"ok": True, "id": evt["id"]})

    def _handle_moderate(self, event_id: str):
        """Flag / review / note an event."""
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

        # Broadcast moderation update
        broadcast_event({
            "type": "_moderation_update",
            "event_id": event_id,
            "moderation": mod,
            "timestamp": time.time(),
        })

        self._send_json({"ok": True, "moderation": mod})

    # ----- GET endpoints -----

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
        if path == "/api/health":
            return self._send_json({"status": "ok", "events": len(events), "uptime_s": round(time.time() - SERVER_START)})

        # Static files
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
        """Server-Sent Events stream."""
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
            # Send existing events as initial batch
            with events_lock:
                init = list(events)
            init_data = json.dumps(init, ensure_ascii=False)
            self.wfile.write(f"event: init\ndata: {init_data}\n\n".encode())
            self.wfile.flush()

            # Send moderation state
            with moderation_lock:
                mod_data = json.dumps(moderation, ensure_ascii=False)
            self.wfile.write(f"event: moderation\ndata: {mod_data}\n\n".encode())
            self.wfile.flush()

            while True:
                try:
                    data = q.get(timeout=15)
                    self.wfile.write(f"data: {data}\n\n".encode())
                    self.wfile.flush()
                except queue.Empty:
                    # Keep-alive
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    def _handle_get_events(self, params: dict):
        """GET /api/events?type=chat&user_id=xxx&limit=100&offset=0"""
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

        # Attach moderation info
        with moderation_lock:
            for e in page:
                mod = moderation.get(e["id"])
                if mod:
                    e = {**e, "_moderation": mod}

        self._send_json({"total": total, "events": page})

    def _handle_conversations(self, params: dict):
        """GET /api/conversations — group chats by user_id."""
        with events_lock:
            chats = [e for e in events if e.get("type") == "chat"]

        convos: dict[str, dict] = {}
        for c in chats:
            uid = c.get("user_id", "anonymous")
            if uid not in convos:
                convos[uid] = {
                    "user_id": uid,
                    "message_count": 0,
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


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


SERVER_START = time.time()


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    load_events()
    print(f"[chatbot-monitor] Loaded {len(events)} events from {EVENTS_FILE}")

    server = ThreadedHTTPServer(("0.0.0.0", PORT), MonitorHandler)
    print(f"[chatbot-monitor] Server running on http://localhost:{PORT}")
    print(f"[chatbot-monitor] Dashboard: http://localhost:{PORT}/")
    print(f"[chatbot-monitor] Health:    http://localhost:{PORT}/api/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[chatbot-monitor] Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
