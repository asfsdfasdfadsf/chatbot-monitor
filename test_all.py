#!/usr/bin/env python3
"""Comprehensive test suite for chatbot-monitor.

Tests every endpoint, the SSE stream, the SDK, moderation, config,
user management (block/unblock/permissions), and the full Ollama chat
flow via a fake Ollama server.
"""

import json
import sys
import time
import threading
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

BASE = "http://localhost:7779"
PASS = 0
FAIL = 0


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def get(path):
    r = urlopen(f"{BASE}{path}", timeout=5)
    return json.loads(r.read())


def post(path, data):
    body = json.dumps(data).encode()
    req = Request(f"{BASE}{path}", data=body,
                  headers={"Content-Type": "application/json"}, method="POST")
    r = urlopen(req, timeout=10)
    return json.loads(r.read())


# =========================================================================
# Fake Ollama server (mimics /api/tags and /api/chat)
# =========================================================================
class FakeOllamaHandler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        if self.path == "/api/tags":
            resp = {"models": [
                {"name": "llama3:8b", "size": 4700000000},
                {"name": "mistral:7b", "size": 4100000000},
            ]}
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        if self.path == "/api/chat":
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length))
            question = data["messages"][-1]["content"]
            model = data.get("model", "llama3:8b")

            # Simulate a small delay
            time.sleep(0.1)

            resp = {
                "message": {
                    "role": "assistant",
                    "content": f"Fake response to: {question}"
                },
                "model": model,
                "done": True,
            }
            body = json.dumps(resp).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)


def start_fake_ollama():
    server = HTTPServer(("127.0.0.1", 11434), FakeOllamaHandler)
    server.daemon_threads = True
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


# =========================================================================
# Tests
# =========================================================================
def main():
    print("=" * 60)
    print("  CHATBOT MONITOR — FULL TEST SUITE")
    print("=" * 60)

    # ---- Prerequisite: server must be running ----
    print("\n[1] Server connectivity")
    try:
        h = get("/api/health")
        test("Health endpoint reachable", h["status"] == "ok")
        test("Health returns event count", "events" in h)
        test("Health returns user count", "users" in h)
        test("Health returns uptime", "uptime_s" in h)
    except Exception as e:
        print(f"  FATAL: Cannot reach server at {BASE}: {e}")
        print("  Start the server first: python server.py")
        sys.exit(1)

    # ---- Clear any leftover data ----
    print("\n[2] Clear events")
    r = post("/api/clear", {})
    test("Clear returns ok", r.get("ok") is True)
    h = get("/api/health")
    test("Events are 0 after clear", h["events"] == 0)

    # ---- Event ingestion ----
    print("\n[3] Event ingestion (POST /event)")
    evt1 = post("/event", {
        "type": "chat", "user_id": "user-A",
        "question": "Was kostet Artikel 4711?",
        "answer": "Artikel 4711 kostet 29,90 EUR.",
        "duration_ms": 1200, "model": "llama3:8b",
    })
    test("Chat event accepted", evt1.get("ok") is True)
    test("Chat event has ID", len(evt1.get("id", "")) > 0)

    evt2 = post("/event", {
        "type": "query", "sql": "SELECT price FROM articles WHERE id=4711",
        "results": [{"price": 29.90}], "duration_ms": 8, "user_id": "user-A",
    })
    test("Query event accepted", evt2.get("ok") is True)

    evt3 = post("/event", {
        "type": "error", "message": "Connection timeout",
        "error_type": "llm_timeout", "user_id": "user-B",
        "details": "requests.exceptions.ConnectTimeout",
    })
    test("Error event accepted", evt3.get("ok") is True)

    evt4 = post("/event", {
        "type": "system", "action": "startup",
        "message": "Chatbot v2.1 started",
    })
    test("System event accepted", evt4.get("ok") is True)

    # Auto-assigned fields
    evt5 = post("/event", {"type": "chat", "question": "No ID test", "answer": "ok"})
    test("Auto-assigns ID", len(evt5.get("id", "")) > 0)

    evt6 = post("/event", {})
    test("Missing type defaults to system", evt6.get("ok") is True)

    h = get("/api/health")
    test("All 6 events stored", h["events"] == 6, f"got {h['events']}")

    # ---- GET /api/events with filters ----
    print("\n[4] Event queries (GET /api/events)")
    all_evts = get("/api/events?limit=100")
    test("Returns total count", all_evts["total"] == 6, f"got {all_evts['total']}")
    test("Returns events array", len(all_evts["events"]) == 6)

    chats_only = get("/api/events?type=chat")
    test("Filter by type=chat", chats_only["total"] == 2, f"got {chats_only['total']}")

    user_a = get("/api/events?user_id=user-A")
    test("Filter by user_id", user_a["total"] == 2, f"got {user_a['total']}")

    limited = get("/api/events?limit=2&offset=1")
    test("Pagination (limit+offset)", len(limited["events"]) == 2)

    # ---- Conversations ----
    print("\n[5] Conversations (GET /api/conversations)")
    convos = get("/api/conversations")
    test("Returns conversations", len(convos["conversations"]) > 0)
    user_a_convo = next((c for c in convos["conversations"] if c["user_id"] == "user-A"), None)
    test("User-A conversation found", user_a_convo is not None)
    test("User-A has 1 chat message", user_a_convo["message_count"] == 1 if user_a_convo else False)

    # ---- Stats ----
    print("\n[6] Stats (GET /api/stats)")
    s = get("/api/stats")
    test("Returns total_events", s["total_events"] == 6, f"got {s['total_events']}")
    test("Returns total_chats", s["total_chats"] == 2)
    test("Returns total_queries", s["total_queries"] == 1)
    test("Returns total_errors", s["total_errors"] == 1)
    test("Returns active_users", s["active_users"] >= 1)
    test("Returns avg_response_ms", s["avg_response_ms"] > 0)
    test("Returns avg_query_ms", s["avg_query_ms"] > 0)
    test("Returns error_rate", s["error_rate"] > 0)
    test("Returns topics list", isinstance(s["topics"], list))
    test("Topics extracted", len(s["topics"]) > 0, f"got {s['topics']}")
    test("Returns hourly dict", isinstance(s["hourly"], dict))
    test("Returns type_counts", "chat" in s["type_counts"])

    # ---- Moderation ----
    print("\n[7] Moderation")
    chat_id = evt1["id"]

    # Flag
    m1 = post(f"/api/moderate/{chat_id}", {"flagged": True})
    test("Flag event returns ok", m1.get("ok") is True)
    test("Moderation shows flagged", m1["moderation"]["flagged"] is True)

    # Add note
    m2 = post(f"/api/moderate/{chat_id}", {"note": "Check this answer"})
    test("Add note returns ok", m2.get("ok") is True)
    test("Note is saved", m2["moderation"]["note"] == "Check this answer")

    # Review
    m3 = post(f"/api/moderate/{chat_id}", {"reviewed": True})
    test("Review event returns ok", m3.get("ok") is True)
    test("Moderation shows reviewed", m3["moderation"]["reviewed"] is True)

    # Get all moderation
    mod_all = get("/api/moderation")
    test("GET /api/moderation returns data", chat_id in mod_all["moderation"])
    test("Moderation state is complete",
         mod_all["moderation"][chat_id]["flagged"] is True and
         mod_all["moderation"][chat_id]["reviewed"] is True and
         mod_all["moderation"][chat_id]["note"] == "Check this answer")

    # Filter flagged events
    flagged = get("/api/events?flagged=true")
    test("Filter flagged events", flagged["total"] >= 1)

    # Unflag
    m4 = post(f"/api/moderate/{chat_id}", {"flagged": False})
    test("Unflag works", m4["moderation"]["flagged"] is False)

    # ---- Config ----
    print("\n[8] Config (GET/POST /api/config)")
    cfg = get("/api/config")
    test("Config returns ollama_url", "ollama_url" in cfg["config"])
    test("Config returns default_model", "default_model" in cfg["config"])
    test("Config returns system_prompt", "system_prompt" in cfg["config"])
    test("Config returns temperature", "temperature" in cfg["config"])

    # Save new config
    new_cfg = post("/api/config", {
        "ollama_url": "http://localhost:11434",
        "default_model": "llama3:8b",
        "system_prompt": "Du bist ein Test-Assistent.",
        "temperature": 0.3,
    })
    test("Config save returns ok", new_cfg.get("ok") is True)
    test("Config saved correctly", new_cfg["config"]["system_prompt"] == "Du bist ein Test-Assistent.")
    test("Temperature saved", new_cfg["config"]["temperature"] == 0.3)

    # Reload and verify persistence
    cfg2 = get("/api/config")
    test("Config persists across reads", cfg2["config"]["system_prompt"] == "Du bist ein Test-Assistent.")

    # ---- Fake Ollama + Chat ----
    print("\n[9] Chat via Ollama (POST /api/chat)")
    print("     Starting fake Ollama on :11434...")
    fake_server = start_fake_ollama()
    time.sleep(0.3)

    # Test models endpoint with fake Ollama
    models = get("/api/models")
    test("Models endpoint returns list", len(models["models"]) == 2, f"got {models['models']}")
    test("llama3:8b in model list", any(m["name"] == "llama3:8b" for m in models["models"]))
    test("mistral:7b in model list", any(m["name"] == "mistral:7b" for m in models["models"]))

    # Test actual chat
    pre_count = get("/api/health")["events"]
    chat_r = post("/api/chat", {
        "question": "Wie viele Schrauben M8 sind auf Lager?",
        "model": "llama3:8b",
        "user_id": "admin-test",
    })
    test("Chat returns ok", chat_r.get("ok") is True)
    test("Chat returns answer", "Fake response to:" in chat_r.get("answer", ""), f"got: {chat_r.get('answer', '')[:50]}")
    test("Chat returns duration_ms", chat_r.get("duration_ms", 0) > 0, f"got {chat_r.get('duration_ms')}")
    test("Chat returns model", chat_r.get("model") == "llama3:8b")

    post_count = get("/api/health")["events"]
    test("Chat event was logged", post_count == pre_count + 1, f"pre={pre_count} post={post_count}")

    # Test chat with different model
    chat_r2 = post("/api/chat", {
        "question": "Test with mistral",
        "model": "mistral:7b",
        "user_id": "admin-test",
    })
    test("Chat with mistral works", chat_r2.get("ok") is True)
    test("Model parameter respected", chat_r2.get("model") == "mistral:7b")

    # Test chat with system prompt from config
    chat_r3 = post("/api/chat", {
        "question": "System prompt test",
        "user_id": "admin-test",
    })
    test("Chat uses default model from config", chat_r3.get("ok") is True)

    # Test chat without question (should fail)
    try:
        req = Request(f"{BASE}/api/chat",
                      data=json.dumps({"user_id": "x"}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        r = urlopen(req, timeout=5)
        data = json.loads(r.read())
        test("Empty question rejected", False, "should have returned 400")
    except Exception as e:
        test("Empty question rejected", "400" in str(e) or "HTTP Error" in str(e))

    # Test blocked user cannot chat via /api/chat
    # First block a user, then try to chat
    post("/api/users/blocked-chatter/block", {})
    try:
        req = Request(f"{BASE}/api/chat",
                      data=json.dumps({"question": "Hello", "user_id": "blocked-chatter"}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        r = urlopen(req, timeout=5)
        data = json.loads(r.read())
        test("Blocked user rejected from /api/chat", False, "should have returned 403")
    except HTTPError as e:
        test("Blocked user rejected from /api/chat", e.code == 403, f"got {e.code}")
    except Exception as e:
        test("Blocked user rejected from /api/chat", "403" in str(e), str(e))
    # Unblock for cleanup
    post("/api/users/blocked-chatter/unblock", {})

    fake_server.shutdown()

    # ---- SDK test ----
    print("\n[10] SDK (monitor.py)")
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    import monitor
    monitor.init(BASE)

    count_pre = get("/api/health")["events"]

    monitor.chat("sdk-user", "SDK question?", "SDK answer!", duration_ms=100, model="test")
    monitor.query("SELECT 1", results=[{"x": 1}], duration_ms=5, user_id="sdk-user")
    monitor.error("SDK test error", error_type="test_error")
    monitor.system("test", "SDK system event")
    time.sleep(1)  # Wait for background threads

    count_post = get("/api/health")["events"]
    test("SDK sent 4 events", count_post == count_pre + 4, f"pre={count_pre} post={count_post}")

    # Test SDK short names
    count_pre2 = count_post
    monitor.log_chat("alias-test", "q", "a", duration_ms=1)
    monitor.log_query("SELECT 2", duration_ms=1)
    monitor.log_error("alias error")
    monitor.log_system("alias action")
    time.sleep(1)

    count_post2 = get("/api/health")["events"]
    test("SDK aliases work", count_post2 == count_pre2 + 4, f"pre={count_pre2} post={count_post2}")

    # ---- User Management ----
    print("\n[11] User management")

    # Users should have been auto-registered from events above
    users_r = get("/api/users")
    test("GET /api/users returns list", "users" in users_r)
    test("Users auto-registered", len(users_r["users"]) >= 2,
         f"got {len(users_r['users'])} users")

    # Check specific user exists (user-A from event ingestion)
    user_a_found = any(u["user_id"] == "user-A" for u in users_r["users"])
    test("user-A was auto-registered", user_a_found)

    # Get single user detail
    user_detail = get("/api/users/user-A")
    test("GET /api/users/<id> returns user", user_detail["user"]["user_id"] == "user-A")
    test("User detail has total_chats", user_detail["user"]["total_chats"] >= 1,
         f"got {user_detail['user'].get('total_chats')}")
    test("User detail has recent_events", isinstance(user_detail["user"]["recent_events"], list))

    # Check user permission (active user)
    check_a = get("/api/users/user-A/check")
    test("Active user is allowed", check_a["allowed"] is True)

    # Block a user
    block_r = post("/api/users/user-B/block", {})
    test("Block user returns ok", block_r.get("ok") is True)
    test("Block returns status=blocked", block_r.get("status") == "blocked")

    # Check blocked user permission
    check_b = get("/api/users/user-B/check")
    test("Blocked user is not allowed", check_b["allowed"] is False)

    # Blocked user's event should be rejected (403)
    try:
        req = Request(f"{BASE}/event",
                      data=json.dumps({"type": "chat", "user_id": "user-B",
                                       "question": "blocked", "answer": "nope"}).encode(),
                      headers={"Content-Type": "application/json"}, method="POST")
        r = urlopen(req, timeout=5)
        data = json.loads(r.read())
        test("Blocked user event rejected", False, "should have returned 403")
    except HTTPError as e:
        body = json.loads(e.read())
        test("Blocked user event rejected", e.code == 403, f"got {e.code}")
        test("Rejection says user_blocked", body.get("error") == "user_blocked")
    except Exception as e:
        test("Blocked user event rejected", "403" in str(e), str(e))
        test("Rejection says user_blocked", False, "could not parse")

    # Unblock the user
    unblock_r = post("/api/users/user-B/unblock", {})
    test("Unblock user returns ok", unblock_r.get("ok") is True)
    test("Unblock returns status=active", unblock_r.get("status") == "active")

    # Check unblocked user is now allowed
    check_b2 = get("/api/users/user-B/check")
    test("Unblocked user is allowed again", check_b2["allowed"] is True)

    # Unblocked user's event should now be accepted
    evt_unblocked = post("/event", {
        "type": "chat", "user_id": "user-B",
        "question": "I'm back!", "answer": "Welcome back!",
    })
    test("Unblocked user event accepted", evt_unblocked.get("ok") is True)

    # Update user note
    update_r = post("/api/users/user-A/update", {"note": "VIP customer"})
    test("Update user returns ok", update_r.get("ok") is True)
    test("User note saved", update_r.get("user", {}).get("note") == "VIP customer")

    # Block a new unknown user (auto-creates)
    block_new = post("/api/users/brand-new-user/block", {})
    test("Block new user auto-creates", block_new.get("ok") is True)
    check_new = get("/api/users/brand-new-user/check")
    test("Newly blocked user is not allowed", check_new["allowed"] is False)
    # Unblock for cleanup
    post("/api/users/brand-new-user/unblock", {})

    # Stats should include user counts
    s2 = get("/api/stats")
    test("Stats has total_users", s2["total_users"] >= 2)
    test("Stats has blocked_users", "blocked_users" in s2)

    # SDK is_user_allowed
    print("\n[12] SDK is_user_allowed")
    test("SDK: active user allowed", monitor.is_user_allowed("user-A") is True)
    post("/api/users/sdk-block-test/block", {})
    test("SDK: blocked user not allowed", monitor.is_user_allowed("sdk-block-test") is False)
    post("/api/users/sdk-block-test/unblock", {})
    test("SDK: unblocked user allowed", monitor.is_user_allowed("sdk-block-test") is True)

    # ---- Users persistence ----
    print("\n[13] Users persistence")
    from pathlib import Path
    users_file = Path(__file__).parent / "data" / "users.json"
    test("Users file exists", users_file.exists())
    if users_file.exists():
        users_data = json.loads(users_file.read_text())
        test("Users file has data", len(users_data) >= 2, f"got {len(users_data)} users")

    # ---- Persistence ----
    print("\n[14] JSONL persistence")
    from pathlib import Path
    jsonl = Path(__file__).parent / "data" / "events.jsonl"
    lines = jsonl.read_text(encoding="utf-8").strip().split("\n")
    test("JSONL file has events", len(lines) >= 10, f"got {len(lines)} lines")

    # Verify each line is valid JSON
    valid = all(json.loads(l) for l in lines if l.strip())
    test("All JSONL lines are valid JSON", valid)

    # ---- Config persistence ----
    cfg_file = Path(__file__).parent / "data" / "config.json"
    test("Config file exists", cfg_file.exists())
    if cfg_file.exists():
        cfg_data = json.loads(cfg_file.read_text())
        test("Config file has correct data", cfg_data.get("system_prompt") == "Du bist ein Test-Assistent.")

    # ---- SSE ----
    print("\n[15] SSE stream (GET /api/stream)")
    import urllib.request
    req = urllib.request.Request(f"{BASE}/api/stream")
    resp = urllib.request.urlopen(req, timeout=5)
    chunk = resp.read(200).decode()
    test("SSE returns event: init", "event: init" in chunk)
    test("SSE returns data payload", "data:" in chunk)
    resp.close()

    # ---- Static files ----
    print("\n[16] Static file serving")
    req = urllib.request.Request(f"{BASE}/")
    resp = urllib.request.urlopen(req, timeout=5)
    html = resp.read().decode()
    test("Dashboard HTML served", "WaWi Chatbot Monitor" in html)
    test("Dashboard has chat input", "chatInput" in html)
    test("Dashboard has model select", "modelSel" in html)
    test("Dashboard has settings modal", "settingsModal" in html)
    test("Dashboard has clear button", "clearEvents" in html)
    test("Dashboard has users panel", "usersBody" in html)
    test("Dashboard has user management", "userAction" in html)
    resp.close()

    # ---- Final clear ----
    print("\n[17] Final cleanup")
    r = post("/api/clear", {})
    test("Final clear works", r.get("ok") is True)
    h = get("/api/health")
    test("Events back to 0", h["events"] == 0)

    # ---- Summary ----
    print("\n" + "=" * 60)
    total = PASS + FAIL
    if FAIL == 0:
        print(f"  ALL {total} TESTS PASSED")
    else:
        print(f"  {PASS}/{total} passed, {FAIL} FAILED")
    print("=" * 60)
    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
