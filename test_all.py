#!/usr/bin/env python3
"""Comprehensive test suite for chatbot-monitor.

Tests every endpoint, the SSE stream, the SDK, moderation, config,
user management (block/unblock/permissions), authentication, roles,
bot toggle, priorities, and the full Ollama chat flow via a fake Ollama server.
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
ADMIN_COOKIE = ""  # Set after login


def test(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}  {detail}")


def login(username, password):
    """Login and return the session cookie string, or empty string on failure."""
    body = json.dumps({"username": username, "password": password}).encode()
    req = Request(f"{BASE}/api/auth/login", data=body,
                  headers={"Content-Type": "application/json"}, method="POST")
    try:
        r = urlopen(req, timeout=5)
        # Extract Set-Cookie header
        cookie_header = r.headers.get("Set-Cookie", "")
        # Parse "session=xxx; Path=/; ..."
        for part in cookie_header.split(";"):
            part = part.strip()
            if part.startswith("session="):
                return part  # "session=xxx"
        return ""
    except Exception:
        return ""


def get(path, cookie=None):
    if cookie is None:
        cookie = ADMIN_COOKIE
    req = Request(f"{BASE}{path}")
    if cookie:
        req.add_header("Cookie", cookie)
    r = urlopen(req, timeout=5)
    return json.loads(r.read())


def post(path, data, cookie=None):
    if cookie is None:
        cookie = ADMIN_COOKIE
    body = json.dumps(data).encode()
    req = Request(f"{BASE}{path}", data=body,
                  headers={"Content-Type": "application/json"}, method="POST")
    if cookie:
        req.add_header("Cookie", cookie)
    r = urlopen(req, timeout=10)
    return json.loads(r.read())


def post_raw(path, data, cookie=None):
    """Like post() but returns the response object for header inspection."""
    if cookie is None:
        cookie = ADMIN_COOKIE
    body = json.dumps(data).encode()
    req = Request(f"{BASE}{path}", data=body,
                  headers={"Content-Type": "application/json"}, method="POST")
    if cookie:
        req.add_header("Cookie", cookie)
    return urlopen(req, timeout=10)


def get_raw(path, cookie=None):
    """Like get() but returns the response object."""
    if cookie is None:
        cookie = ADMIN_COOKIE
    req = Request(f"{BASE}{path}")
    if cookie:
        req.add_header("Cookie", cookie)
    return urlopen(req, timeout=5)


def expect_error(method, path, data=None, expected_code=None, cookie=None):
    """Make a request expecting an HTTP error. Returns (status_code, body_dict)."""
    if cookie is None:
        cookie = ADMIN_COOKIE
    try:
        if method == "GET":
            req = Request(f"{BASE}{path}")
        else:
            body = json.dumps(data or {}).encode()
            req = Request(f"{BASE}{path}", data=body,
                          headers={"Content-Type": "application/json"}, method="POST")
        if cookie:
            req.add_header("Cookie", cookie)
        r = urlopen(req, timeout=5)
        return (r.status, json.loads(r.read()))
    except HTTPError as e:
        try:
            body = json.loads(e.read())
        except Exception:
            body = {}
        return (e.code, body)
    except Exception as e:
        return (0, {"error": str(e)})


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
    global ADMIN_COOKIE

    print("=" * 60)
    print("  CHATBOT MONITOR — FULL TEST SUITE")
    print("=" * 60)

    # ---- Prerequisite: server must be running ----
    print("\n[1] Server connectivity")
    try:
        h = get("/api/health", cookie="")
        test("Health endpoint reachable", h["status"] == "ok")
        test("Health returns event count", "events" in h)
        test("Health returns user count", "users" in h)
        test("Health returns uptime", "uptime_s" in h)
    except Exception as e:
        print(f"  FATAL: Cannot reach server at {BASE}: {e}")
        print("  Start the server first: python server.py")
        sys.exit(1)

    # ---- Authentication ----
    print("\n[2] Authentication (login/register/logout/me)")

    # Login as default admin
    ADMIN_COOKIE = login("admin", "admin")
    test("Admin login returns session cookie", len(ADMIN_COOKIE) > 10, f"got '{ADMIN_COOKIE}'")

    # /api/auth/me with valid session
    me = get("/api/auth/me")
    test("GET /me returns username", me.get("user", {}).get("username") == "admin")
    test("GET /me returns role=admin", me.get("user", {}).get("role") == "admin")

    # /api/auth/me without cookie -> 401
    code, _ = expect_error("GET", "/api/auth/me", cookie="")
    test("GET /me without cookie returns 401", code == 401, f"got {code}")

    # Invalid login
    bad_cookie = login("admin", "wrongpassword")
    test("Bad password returns empty cookie", bad_cookie == "")

    bad_cookie2 = login("nonexistent", "pass")
    test("Nonexistent user returns empty cookie", bad_cookie2 == "")

    # Register a new user
    resp = post_raw("/api/auth/register", {"username": "testuser", "password": "testpass"}, cookie="")
    reg_data = json.loads(resp.read())
    test("Register returns ok", reg_data.get("ok") is True)
    test("Register returns user role", reg_data.get("user", {}).get("role") == "user")
    reg_cookie_header = resp.headers.get("Set-Cookie", "")
    user_cookie = ""
    for part in reg_cookie_header.split(";"):
        part = part.strip()
        if part.startswith("session="):
            user_cookie = part
            break
    test("Register returns session cookie", len(user_cookie) > 10)

    # /me as new user
    me2 = get("/api/auth/me", cookie=user_cookie)
    test("Registered user /me works", me2.get("user", {}).get("username") == "testuser")
    test("Registered user role is user", me2.get("user", {}).get("role") == "user")

    # Duplicate registration
    code, body = expect_error("POST", "/api/auth/register",
                              {"username": "testuser", "password": "other"}, cookie="")
    test("Duplicate username rejected", code == 409, f"got {code}")

    # Logout
    post("/api/auth/logout", {}, cookie=user_cookie)
    code, _ = expect_error("GET", "/api/auth/me", cookie=user_cookie)
    test("After logout /me returns 401", code == 401, f"got {code}")

    # Re-login as testuser for role tests
    user_cookie = login("testuser", "testpass")
    test("Re-login as testuser works", len(user_cookie) > 10)

    # ---- Role-based access control ----
    print("\n[3] Role-based access control")

    # User can access /api/models (user-level)
    models_r = get("/api/models", cookie=user_cookie)
    test("User can access /api/models", "models" in models_r)

    # User cannot access admin endpoints
    code, _ = expect_error("GET", "/api/events?limit=10", cookie=user_cookie)
    test("User cannot access /api/events", code == 403, f"got {code}")

    code, _ = expect_error("GET", "/api/stats", cookie=user_cookie)
    test("User cannot access /api/stats", code == 403, f"got {code}")

    code, _ = expect_error("GET", "/api/users", cookie=user_cookie)
    test("User cannot access /api/users", code == 403, f"got {code}")

    code, _ = expect_error("POST", "/api/clear", {}, cookie=user_cookie)
    test("User cannot access /api/clear", code == 403, f"got {code}")

    code, _ = expect_error("GET", "/api/accounts", cookie=user_cookie)
    test("User cannot access /api/accounts", code == 403, f"got {code}")

    code, _ = expect_error("GET", "/api/config", cookie=user_cookie)
    test("User cannot access /api/config", code == 403, f"got {code}")

    code, _ = expect_error("GET", "/api/conversations", cookie=user_cookie)
    test("User cannot access /api/conversations", code == 403, f"got {code}")

    # No cookie at all -> 401
    code, _ = expect_error("GET", "/api/events?limit=10", cookie="")
    test("No cookie -> 401 on protected endpoints", code == 401, f"got {code}")

    # ---- Clear events (as admin) ----
    print("\n[4] Clear events")
    r = post("/api/clear", {})
    test("Clear returns ok", r.get("ok") is True)
    h = get("/api/health", cookie="")
    test("Events are 0 after clear", h["events"] == 0)

    # ---- Event ingestion ----
    print("\n[5] Event ingestion (POST /event)")
    evt1 = post("/event", {
        "type": "chat", "user_id": "user-A",
        "question": "Was kostet Artikel 4711?",
        "answer": "Artikel 4711 kostet 29,90 EUR.",
        "duration_ms": 1200, "model": "llama3:8b",
    }, cookie="")  # SDK endpoint, no auth needed
    test("Chat event accepted", evt1.get("ok") is True)
    test("Chat event has ID", len(evt1.get("id", "")) > 0)

    evt2 = post("/event", {
        "type": "query", "sql": "SELECT price FROM articles WHERE id=4711",
        "results": [{"price": 29.90}], "duration_ms": 8, "user_id": "user-A",
    }, cookie="")
    test("Query event accepted", evt2.get("ok") is True)

    evt3 = post("/event", {
        "type": "error", "message": "Connection timeout",
        "error_type": "llm_timeout", "user_id": "user-B",
        "details": "requests.exceptions.ConnectTimeout",
    }, cookie="")
    test("Error event accepted", evt3.get("ok") is True)

    evt4 = post("/event", {
        "type": "system", "action": "startup",
        "message": "Chatbot v2.1 started",
    }, cookie="")
    test("System event accepted", evt4.get("ok") is True)

    # Auto-assigned fields
    evt5 = post("/event", {"type": "chat", "question": "No ID test", "answer": "ok"}, cookie="")
    test("Auto-assigns ID", len(evt5.get("id", "")) > 0)

    evt6 = post("/event", {}, cookie="")
    test("Missing type defaults to system", evt6.get("ok") is True)

    h = get("/api/health", cookie="")
    test("All 6 events stored", h["events"] == 6, f"got {h['events']}")

    # ---- GET /api/events with filters ----
    print("\n[6] Event queries (GET /api/events)")
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
    print("\n[7] Conversations (GET /api/conversations)")
    convos = get("/api/conversations")
    test("Returns conversations", len(convos["conversations"]) > 0)
    user_a_convo = next((c for c in convos["conversations"] if c["user_id"] == "user-A"), None)
    test("User-A conversation found", user_a_convo is not None)
    test("User-A has 1 chat message", user_a_convo["message_count"] == 1 if user_a_convo else False)

    # ---- Stats ----
    print("\n[8] Stats (GET /api/stats)")
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
    print("\n[9] Moderation")
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
    print("\n[10] Config (GET/POST /api/config)")
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
    print("\n[11] Chat via Ollama (POST /api/chat)")
    print("     Starting fake Ollama on :11434...")
    fake_server = start_fake_ollama()
    time.sleep(0.3)

    # Test models endpoint with fake Ollama
    models = get("/api/models")
    test("Models endpoint returns list", len(models["models"]) == 2, f"got {models['models']}")
    test("llama3:8b in model list", any(m["name"] == "llama3:8b" for m in models["models"]))
    test("mistral:7b in model list", any(m["name"] == "mistral:7b" for m in models["models"]))

    # Test actual chat (as admin, can set user_id)
    pre_count = get("/api/health", cookie="")["events"]
    chat_r = post("/api/chat", {
        "question": "Wie viele Schrauben M8 sind auf Lager?",
        "model": "llama3:8b",
        "user_id": "admin-test",
    })
    test("Chat returns ok", chat_r.get("ok") is True)
    test("Chat returns answer", "Fake response to:" in chat_r.get("answer", ""), f"got: {chat_r.get('answer', '')[:50]}")
    test("Chat returns duration_ms", chat_r.get("duration_ms", 0) > 0, f"got {chat_r.get('duration_ms')}")
    test("Chat returns model", chat_r.get("model") == "llama3:8b")

    post_count = get("/api/health", cookie="")["events"]
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
    code, _ = expect_error("POST", "/api/chat", {"user_id": "x"})
    test("Empty question rejected", code == 400, f"got {code}")

    # Test user-role chat (user_id comes from session)
    user_cookie = login("testuser", "testpass")
    chat_user = post("/api/chat", {
        "question": "User role chat test",
    }, cookie=user_cookie)
    test("User-role chat works", chat_user.get("ok") is True)

    # Test blocked user cannot chat via /api/chat
    post("/api/users/blocked-chatter/block", {})
    # Need to create a dashboard account for blocked-chatter to test chat blocking
    # Actually, the chat endpoint uses session username, so let's test by blocking the
    # chatbot end-user that gets created when admin chats as "blocked-chatter"
    code, _ = expect_error("POST", "/api/chat",
                           {"question": "Hello", "user_id": "blocked-chatter"})
    test("Blocked user rejected from /api/chat", code == 403, f"got {code}")
    # Unblock for cleanup
    post("/api/users/blocked-chatter/unblock", {})

    fake_server.shutdown()

    # ---- SDK test ----
    print("\n[12] SDK (monitor.py)")
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    import monitor
    monitor.init(BASE)

    count_pre = get("/api/health", cookie="")["events"]

    monitor.chat("sdk-user", "SDK question?", "SDK answer!", duration_ms=100, model="test")
    monitor.query("SELECT 1", results=[{"x": 1}], duration_ms=5, user_id="sdk-user")
    monitor.error("SDK test error", error_type="test_error")
    monitor.system("test", "SDK system event")
    time.sleep(1)  # Wait for background threads

    count_post = get("/api/health", cookie="")["events"]
    test("SDK sent 4 events", count_post == count_pre + 4, f"pre={count_pre} post={count_post}")

    # Test SDK short names
    count_pre2 = count_post
    monitor.log_chat("alias-test", "q", "a", duration_ms=1)
    monitor.log_query("SELECT 2", duration_ms=1)
    monitor.log_error("alias error")
    monitor.log_system("alias action")
    time.sleep(1)

    count_post2 = get("/api/health", cookie="")["events"]
    test("SDK aliases work", count_post2 == count_pre2 + 4, f"pre={count_pre2} post={count_post2}")

    # ---- User Management ----
    print("\n[13] User management")

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

    # Check user permission (active user) — no auth needed (SDK endpoint)
    check_a = get("/api/users/user-A/check", cookie="")
    test("Active user is allowed", check_a["allowed"] is True)

    # Block a user
    block_r = post("/api/users/user-B/block", {})
    test("Block user returns ok", block_r.get("ok") is True)
    test("Block returns status=blocked", block_r.get("status") == "blocked")

    # Check blocked user permission
    check_b = get("/api/users/user-B/check", cookie="")
    test("Blocked user is not allowed", check_b["allowed"] is False)

    # Blocked user's event should be rejected (403)
    code, body = expect_error("POST", "/event",
                              {"type": "chat", "user_id": "user-B",
                               "question": "blocked", "answer": "nope"},
                              cookie="")
    test("Blocked user event rejected", code == 403, f"got {code}")
    test("Rejection says user_blocked", body.get("error") == "user_blocked")

    # Unblock the user
    unblock_r = post("/api/users/user-B/unblock", {})
    test("Unblock user returns ok", unblock_r.get("ok") is True)
    test("Unblock returns status=active", unblock_r.get("status") == "active")

    # Check unblocked user is now allowed
    check_b2 = get("/api/users/user-B/check", cookie="")
    test("Unblocked user is allowed again", check_b2["allowed"] is True)

    # Unblocked user's event should now be accepted
    evt_unblocked = post("/event", {
        "type": "chat", "user_id": "user-B",
        "question": "I'm back!", "answer": "Welcome back!",
    }, cookie="")
    test("Unblocked user event accepted", evt_unblocked.get("ok") is True)

    # Update user note
    update_r = post("/api/users/user-A/update", {"note": "VIP customer"})
    test("Update user returns ok", update_r.get("ok") is True)
    test("User note saved", update_r.get("user", {}).get("note") == "VIP customer")

    # Block a new unknown user (auto-creates)
    block_new = post("/api/users/brand-new-user/block", {})
    test("Block new user auto-creates", block_new.get("ok") is True)
    check_new = get("/api/users/brand-new-user/check", cookie="")
    test("Newly blocked user is not allowed", check_new["allowed"] is False)
    # Unblock for cleanup
    post("/api/users/brand-new-user/unblock", {})

    # Stats should include user counts
    s2 = get("/api/stats")
    test("Stats has total_users", s2["total_users"] >= 2)
    test("Stats has blocked_users", "blocked_users" in s2)

    # ---- SDK is_user_allowed ----
    print("\n[14] SDK is_user_allowed")
    test("SDK: active user allowed", monitor.is_user_allowed("user-A") is True)
    post("/api/users/sdk-block-test/block", {})
    test("SDK: blocked user not allowed", monitor.is_user_allowed("sdk-block-test") is False)
    post("/api/users/sdk-block-test/unblock", {})
    test("SDK: unblocked user allowed", monitor.is_user_allowed("sdk-block-test") is True)

    # ---- SDK get_user_status ----
    print("\n[15] SDK get_user_status")
    status = monitor.get_user_status("user-A")
    test("get_user_status returns allowed", status["allowed"] is True)
    test("get_user_status returns priority", status["priority"] in ("high", "normal", "low"))
    test("get_user_status returns bot_enabled", isinstance(status["bot_enabled"], bool))

    # ---- Priority ----
    print("\n[16] User priority")
    # Set priority on chatbot end-user
    pr = post("/api/users/user-A/update", {"priority": "high"})
    test("Set user priority returns ok", pr.get("ok") is True)
    test("Priority saved", pr.get("user", {}).get("priority") == "high")

    # Verify via /check endpoint
    check_pr = get("/api/users/user-A/check", cookie="")
    test("/check returns priority", check_pr.get("priority") == "high")
    test("/check returns bot_enabled", "bot_enabled" in check_pr)

    # Reset priority
    post("/api/users/user-A/update", {"priority": "normal"})

    # ---- Bot toggle ----
    print("\n[17] Bot toggle (bot_enabled)")
    # Disable bot
    cfg_off = post("/api/config", {"bot_enabled": False})
    test("Bot disabled via config", cfg_off.get("ok") is True)
    test("Config shows bot_enabled=false", cfg_off["config"].get("bot_enabled") is False)

    # SDK /event should return 503
    code, body = expect_error("POST", "/event",
                              {"type": "chat", "user_id": "user-A",
                               "question": "hello", "answer": "world"},
                              cookie="")
    test("POST /event returns 503 when bot disabled", code == 503, f"got {code}")

    # /api/chat should return 503
    # Need fake ollama for this — just check the error code
    code, _ = expect_error("POST", "/api/chat",
                           {"question": "test", "user_id": "admin-test"})
    test("POST /api/chat returns 503 when bot disabled", code == 503, f"got {code}")

    # /check should report bot_enabled=false
    check_bot = get("/api/users/user-A/check", cookie="")
    test("/check shows bot_enabled=false", check_bot.get("bot_enabled") is False)

    # SDK get_user_status should reflect it
    status_off = monitor.get_user_status("user-A")
    test("SDK get_user_status shows bot_enabled=false", status_off["bot_enabled"] is False)

    # Re-enable bot
    cfg_on = post("/api/config", {"bot_enabled": True})
    test("Bot re-enabled", cfg_on["config"].get("bot_enabled") is True)

    # Verify event ingestion works again
    evt_after = post("/event", {
        "type": "system", "action": "bot_re_enabled",
    }, cookie="")
    test("Events accepted after bot re-enabled", evt_after.get("ok") is True)

    # ---- Account management ----
    print("\n[18] Account management")

    # List accounts
    accts = get("/api/accounts")
    test("GET /api/accounts returns list", "accounts" in accts)
    test("Admin account exists", any(a["username"] == "admin" for a in accts["accounts"]))
    test("testuser account exists", any(a["username"] == "testuser" for a in accts["accounts"]))
    # Passwords should NOT be in the response
    admin_acct = next(a for a in accts["accounts"] if a["username"] == "admin")
    test("No password_hash in response", "password_hash" not in admin_acct)
    test("No salt in response", "salt" not in admin_acct)

    # Change role
    role_r = post("/api/accounts/testuser/role", {"role": "admin"})
    test("Change role returns ok", role_r.get("ok") is True)

    # Verify role changed
    accts2 = get("/api/accounts")
    tu = next(a for a in accts2["accounts"] if a["username"] == "testuser")
    test("Role changed to admin", tu["role"] == "admin")

    # Change back to user
    post("/api/accounts/testuser/role", {"role": "user"})

    # Set account priority
    pri_r = post("/api/accounts/testuser/priority", {"priority": "high"})
    test("Set account priority returns ok", pri_r.get("ok") is True)

    accts3 = get("/api/accounts")
    tu2 = next(a for a in accts3["accounts"] if a["username"] == "testuser")
    test("Account priority set to high", tu2["priority"] == "high")

    # Reset
    post("/api/accounts/testuser/priority", {"priority": "normal"})

    # Cannot delete self
    code, body = expect_error("POST", "/api/accounts/admin/delete", {})
    test("Cannot delete own account", code in (400, 403), f"got {code}")

    # Register another user to test deletion
    post("/api/auth/register", {"username": "deleteme", "password": "deleteme"}, cookie="")
    del_r = post("/api/accounts/deleteme/delete", {})
    test("Delete account returns ok", del_r.get("ok") is True)

    # Verify deleted
    accts4 = get("/api/accounts")
    test("Deleted account gone", not any(a["username"] == "deleteme" for a in accts4["accounts"]))

    # Deleted user cannot login
    dead_cookie = login("deleteme", "deleteme")
    test("Deleted user cannot login", dead_cookie == "")

    # ---- Provider config ----
    print("\n[19] Provider config (multi-LLM)")

    # Default provider is ollama
    cfg = get("/api/config")
    test("Default provider is ollama", cfg["config"].get("provider") == "ollama")
    test("Config has openai_api_key field", "openai_api_key" in cfg["config"])
    test("Config has anthropic_api_key field", "anthropic_api_key" in cfg["config"])
    test("Config has openai_base_url field", "openai_base_url" in cfg["config"])

    # Set provider to openai with API key
    post("/api/config", {
        "provider": "openai",
        "openai_api_key": "sk-test-1234567890abcdef",
        "openai_base_url": "https://api.openai.com/v1",
    })
    cfg2 = get("/api/config")
    test("Provider changed to openai", cfg2["config"]["provider"] == "openai")
    test("OpenAI API key is masked", cfg2["config"]["openai_api_key"].startswith("..."))
    test("OpenAI key shows last 4 chars", cfg2["config"]["openai_api_key"] == "...cdef")
    test("OpenAI base URL saved", cfg2["config"]["openai_base_url"] == "https://api.openai.com/v1")

    # Sending masked key back should NOT overwrite the real key
    post("/api/config", {"openai_api_key": "...cdef"})
    # Verify by reading config file directly
    from pathlib import Path
    cfg_raw = json.loads((Path(__file__).parent / "data" / "config.json").read_text())
    test("Masked key not saved over real key", cfg_raw["openai_api_key"] == "sk-test-1234567890abcdef")

    # Set provider to anthropic
    post("/api/config", {
        "provider": "anthropic",
        "anthropic_api_key": "sk-ant-testkey9876",
    })
    cfg3 = get("/api/config")
    test("Provider changed to anthropic", cfg3["config"]["provider"] == "anthropic")
    test("Anthropic key masked", cfg3["config"]["anthropic_api_key"] == "...9876")

    # Model list changes with provider
    # Anthropic should return hardcoded models
    models_ant = get("/api/models")
    test("Anthropic models returned", len(models_ant["models"]) > 0)
    test("Claude model in list", any("claude" in m["name"] for m in models_ant["models"]))

    # Switch back to ollama for remaining tests
    post("/api/config", {
        "provider": "ollama",
        "default_model": "llama3:8b",
        "openai_api_key": "",
        "anthropic_api_key": "",
    })
    cfg4 = get("/api/config")
    test("Provider back to ollama", cfg4["config"]["provider"] == "ollama")

    # ---- OpenRouter provider config ----
    print("\n[20] OpenRouter provider config")

    # Set provider to openrouter with API key
    post("/api/config", {
        "provider": "openrouter",
        "openrouter_api_key": "sk-or-v1-testkey1234abcd",
    })
    cfg_or = get("/api/config")
    test("Provider changed to openrouter", cfg_or["config"]["provider"] == "openrouter")
    test("OpenRouter key is masked", cfg_or["config"]["openrouter_api_key"].startswith("..."))
    test("OpenRouter key shows last 4 chars", cfg_or["config"]["openrouter_api_key"] == "...abcd")
    test("Config has openrouter_api_key field", "openrouter_api_key" in cfg_or["config"])

    # Masked key should not overwrite real key
    post("/api/config", {"openrouter_api_key": "...abcd"})
    cfg_raw_or = json.loads((Path(__file__).parent / "data" / "config.json").read_text())
    test("OpenRouter masked key not saved over real key", cfg_raw_or["openrouter_api_key"] == "sk-or-v1-testkey1234abcd")

    # OpenRouter exchange endpoint — requires admin auth
    code_noauth, _ = expect_error("POST", "/api/openrouter/exchange",
                                  {"code": "testcode"}, cookie="")
    test("OpenRouter exchange requires auth", code_noauth == 401, f"got {code_noauth}")

    # OpenRouter exchange — user role should be denied
    code_user, _ = expect_error("POST", "/api/openrouter/exchange",
                                {"code": "testcode"}, cookie=user_cookie)
    test("OpenRouter exchange requires admin", code_user == 403, f"got {code_user}")

    # OpenRouter exchange — missing code
    code_nocode, body_nocode = expect_error("POST", "/api/openrouter/exchange",
                                            {}, cookie=ADMIN_COOKIE)
    test("OpenRouter exchange requires code", code_nocode == 400, f"got {code_nocode}")

    # OpenRouter exchange — invalid code (will fail against real API = 502)
    code_bad, body_bad = expect_error("POST", "/api/openrouter/exchange",
                                      {"code": "invalid-code-xyz"}, cookie=ADMIN_COOKIE)
    test("OpenRouter exchange with invalid code returns 502", code_bad == 502, f"got {code_bad}")

    # Restore to ollama
    post("/api/config", {
        "provider": "ollama",
        "openrouter_api_key": "",
    })
    cfg_restore = get("/api/config")
    test("Provider restored to ollama", cfg_restore["config"]["provider"] == "ollama")

    # ---- Users persistence ----
    print("\n[21] Users persistence")
    from pathlib import Path
    users_file = Path(__file__).parent / "data" / "users.json"
    test("Users file exists", users_file.exists())
    if users_file.exists():
        users_data = json.loads(users_file.read_text())
        test("Users file has data", len(users_data) >= 2, f"got {len(users_data)} users")

    # ---- Persistence ----
    print("\n[22] JSONL persistence")
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

    # ---- Accounts persistence ----
    accounts_file = Path(__file__).parent / "data" / "accounts.json"
    test("Accounts file exists", accounts_file.exists())
    if accounts_file.exists():
        acct_data = json.loads(accounts_file.read_text())
        test("Accounts file has admin", "admin" in acct_data)

    # ---- SSE ----
    print("\n[23] SSE stream (GET /api/stream)")
    import urllib.request
    req = urllib.request.Request(f"{BASE}/api/stream")
    req.add_header("Cookie", ADMIN_COOKIE)
    resp = urllib.request.urlopen(req, timeout=5)
    chunk = resp.read(500).decode()
    test("SSE returns event: session", "event: session" in chunk)
    test("SSE returns event: init", "event: init" in chunk)
    test("SSE returns data payload", "data:" in chunk)
    resp.close()

    # ---- Static files ----
    print("\n[24] Static file serving")
    req = urllib.request.Request(f"{BASE}/")
    resp = urllib.request.urlopen(req, timeout=5)
    html = resp.read().decode()
    test("Dashboard HTML served", "WaWi Chatbot Monitor" in html)
    test("Dashboard has login screen", "loginScreen" in html)
    test("Dashboard has chat input", "chatInput" in html)
    test("Dashboard has model select", "modelSel" in html)
    test("Dashboard has settings modal", "settingsModal" in html)
    test("Dashboard has clear button", "clearEvents" in html)
    test("Dashboard has users panel", "usersBody" in html)
    test("Dashboard has user management", "userAction" in html)
    test("Dashboard has accounts tab", "tAccounts" in html)
    resp.close()

    # ---- Final clear ----
    print("\n[25] Final cleanup")
    r = post("/api/clear", {})
    test("Final clear works", r.get("ok") is True)
    h = get("/api/health", cookie="")
    test("Events back to 0", h["events"] == 0)

    # Clean up testuser account
    post("/api/accounts/testuser/delete", {})

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
