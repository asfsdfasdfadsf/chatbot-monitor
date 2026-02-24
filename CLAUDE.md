# Agent Monitor Project

## What it is
A live web dashboard at **http://localhost:7778** that shows what Claude Code is doing in real-time — like Cursor/AutoGPT agent view. Built in `C:/Users/User/Desktop/claude/agent-monitor/`.

## How it works
```
Claude Code hooks → bash hook.sh → POST /event → SSE → Browser dashboard
```
- Claude Code's PostToolUse hook fires on every tool call
- `hook.sh` reads the JSON from stdin and curls it to the local server
- `server.py` (pure Python stdlib, zero dependencies) receives events and pushes them via Server-Sent Events
- `public/index.html` renders a live dashboard with contextual viewers

## Dashboard features
- **Timeline** (left) — every tool call, color-coded by type (Read=blue, Edit=orange, Bash=red, etc.)
- **Live Viewer** (center) — auto-switches based on tool type:
  - **Code editor** view for Read/Edit/Write (with syntax highlighting, diff view for edits)
  - **Terminal** view for Bash (black terminal with $ prompt and blinking cursor)
  - **Google browser** view for WebSearch (fake Chrome with search results)
  - **Browser** view for WebFetch (with address bar and page content)
  - **Search panel** for Grep/Glob (with pattern highlighting)
  - **Agent orb** for Task (pulsing orb with task description)
- **Stats** (right) — read/write/command/search counts, tool usage bar chart, file tree
- Auto-follow mode tracks latest event; click timeline items to pin

## Files
- `agent-monitor/server.py` — ThreadingHTTPServer + SSE, port 7778
- `agent-monitor/public/index.html` — Full dashboard UI (single file, ~815 lines)
- `agent-monitor/hook.sh` — Bash hook that curls event JSON to server
- `agent-monitor/hook.ps1` — PowerShell version (BROKEN on Windows, don't use)
- `agent-monitor/setup.py` — Auto-configures hooks in settings.json

## Hook configuration
Settings are in `~/.claude/settings.json`. Current working config uses **bash** (not PowerShell).

**IMPORTANT: Hook format changed.** The new format uses nested matcher groups:
- Each event contains an array of **matcher groups** (objects with optional `matcher` string + `hooks` array)
- `matcher` is a **regex string** (e.g. `"Bash"`, `"Edit|Write"`, `".*"`), NOT an object
- Omit `matcher` entirely to match all tools (equivalent to `"*"`)
- The inner `hooks` array contains the actual handler objects (`type`, `command`, etc.)

```json
{
  "hooks": {
    "PostToolUse": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "bash C:/Users/User/Desktop/claude/agent-monitor/hook.sh"
          }
        ]
      }
    ]
  }
}
```
PowerShell hook was broken (stdin reading with `$input` and `[Console]::In.ReadToEnd()` both failed). Bash + curl works.

## Running
1. Start server: `cd agent-monitor && nohup python server.py > /tmp/agent-monitor.log 2>&1 &`
2. Open: http://localhost:7778
3. Use Claude Code — events appear live automatically via hooks

## Known issues / notes
- Server uses `ThreadingHTTPServer` (required for SSE to work alongside POST)
- SSE uses per-client queues for reliable delivery
- `MAX_EVENTS = 1000` (ring buffer)
- Debug indicator in top-right of dashboard shows connection status (orange text)
- If server dies, restart with the command above
- **Hooks only reload on Claude Code session restart** — this is the #1 gotcha. If hooks aren't firing, the session was likely started before hooks were configured. Must exit and start a new session.
- The `websockets` pip package is installed but NOT used (was removed in favor of pure stdlib)
- `hook.sh` currently has debug logging to `/tmp/hook-debug.log` — can remove once confirmed working
- Hook was tested manually (`echo JSON | bash hook.sh`) and works fine — the issue was always session reload

## Current status (2026-02-19)
- Server: **running** on port 7778, serves dashboard and accepts POST /event ✓
- Hook config: **fixed** in `~/.claude/settings.json` — was broken due to hook format change (matcher was `{}` object instead of string; fixed by omitting matcher to match all) ✓
- Hook script: **works** when called manually ✓
- **NOT YET VERIFIED end-to-end** — need to start a fresh Claude Code session so hooks actually load, then confirm events appear on dashboard
- Next step: **restart Claude Code session** and verify the full pipeline works live

## What the user wants
The user wants to **see everything Claude does live on the dashboard** — the whole point is a clean real-time view of agent activity. Every Read, Edit, Bash, Search, WebSearch etc. should show up automatically with the appropriate contextual viewer.

---

# Chatbot Monitor Project

## What it is
A multi-provider LLM chatbot admin dashboard at **http://localhost:7779**. Built in `C:/Users/User/Desktop/claude/chatbot-monitor/`. GitHub repo: `asfsdfasdfadsf/chatbot-monitor`.

## Features
- **Multi-provider support**: Ollama, OpenAI, Anthropic, OpenRouter
- **OpenAI OAuth PKCE login**: Login with ChatGPT account (Plus/Pro) instead of API keys — uses Codex CLI client_id
- **OpenRouter OAuth**: One-click login for 200+ models
- **Chat memory**: Session-based conversation history (max 20 messages per session)
- **User management**: Admin/user roles, blocking, priorities
- **Bot kill switch**: Admin can disable/enable the chatbot
- **Cookie-based auth**: Session management with auto-cleanup

## Architecture
- `server.py` — Pure Python stdlib server (zero dependencies), port 7779
- `public/index.html` — Single-file dashboard UI
- `test_all.py` — Comprehensive test suite (221 tests)
- `data/config.json` — Runtime config (gitignored, contains tokens)
- `data/accounts.json` — User accounts (gitignored)

## OpenAI OAuth PKCE flow
The OAuth flow matches what Codex CLI does:
1. Generate PKCE verifier/challenge, build auth URL with `codex_cli_simplified_flow=true`
2. Spin up temporary callback server on **port 1455** (registered redirect URI for the Codex client_id)
3. User logs in at `auth.openai.com`, gets redirected to `localhost:1455/auth/callback`
4. Callback server captures code+state, redirects browser to main dashboard on port 7779
5. Dashboard JS calls `/api/openai/auth/callback` with code+state
6. Server exchanges code for tokens (access_token, id_token, refresh_token)
7. **Token exchange**: Exchanges `id_token` for an API key via `urn:ietf:params:oauth:grant-type:token-exchange`
8. API key is stored and used for all subsequent API calls

**Key constants:**
- Client ID: `app_EMoamEEZ73f0CkXaXp7hrann`
- Callback port: 1455 (hardcoded, must match Codex CLI's registered redirect URI)
- Auth URL: `https://auth.openai.com/oauth/authorize`
- Token URL: `https://auth.openai.com/oauth/token`
- Scopes: `openid profile email offline_access`

**Known issue:** ChatGPT "Go" (free) plan may not support API access. Plus/Pro plans work. If the token exchange for API key fails, falls back to Responses API with raw OAuth token.

## Chat memory
- Per-session message history stored in `chat_sessions` dict
- Max 20 messages per session (oldest trimmed)
- "New Chat" button clears history
- History cleared on logout and session expiry
- All `call_*` functions take `messages` array (not single question)
- Anthropic: system messages extracted into separate `system` field (API requirement)

## LLM call flow
`get_openai_bearer_token()` priority: API key (from token exchange) > OAuth token > manual API key

When using raw OAuth token (no API key), `call_openai` uses the Responses API (`/v1/responses`). When using an API key, it uses Chat Completions (`/v1/chat/completions`).

## Running
```bash
cd chatbot-monitor && nohup python server.py > /tmp/chatbot-monitor.log 2>&1 &
```

## Model list
Fallback model list includes: gpt-5.2, gpt-5.2-chat-latest, gpt-5.2-pro, gpt-5.2-codex, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, gpt-4o, gpt-4o-mini, o3, o3-mini, o4-mini. Live model list fetched from `/v1/models` when API key is available.

## Agent Viewer (live tool call panel)
When Agent Mode is enabled, a **live side panel** slides in from the right showing each tool call as it happens with rich contextual viewers — inspired by the agent-monitor dashboard.

### How it works
```
Agent loop (server.py)          SSE (existing)         Browser (index.html)
  _agent_start  ──broadcast──▶  SSE queue  ──stream──▶  Panel opens
  _agent_thinking ──broadcast──▶            ──stream──▶  Orb pulses
  _execute_tool ──broadcast──▶             ──stream──▶  Viewer renders
  _agent_done   ──broadcast──▶             ──stream──▶  "Done" state
```

### SSE events emitted during agent loop
- **`_agent_start`** — fired before the loop; contains `run_id`, `user_id`, `question`, `available_tools`, `model`, `max_steps`
- **`_agent_thinking`** — fired at the top of each loop iteration (before LLM call); contains `step`
- **`_agent_step`** — fired after each `_execute_tool()` call; contains `tool`, `input`, `output` (truncated to 2000 chars), `duration_ms`
- **`_agent_done`** — fired when the agent finishes (normal, max-steps, or error); contains `answer`, `total_steps`, `duration_ms`

Per-user scoping is free — existing SSE filter delivers events only to the user who made the request (admins see all). The SSE filter was updated to allow `_agent_*` events through for non-admin users.

### Frontend panel
- **Fixed overlay** (420px, right side, z-index 100, slides in with CSS transform)
- **Timeline** (left 140px strip) — color-coded step items, click to pin (disables auto-follow)
- **Viewer** (main area) — auto-switches based on tool type:
  - **DB terminal** (`query_database`, `list_tables`) — macOS titlebar dots, dark body, purple SQL with keyword highlighting, result table
  - **DuckDuckGo browser** (`web_search`) — Chrome-style address bar, search results with titles/URLs/snippets
  - **RAG search panel** (`search_knowledge`) — search icon + query bar, chunk cards with similarity scores
  - **Calculator** (`calculate`) — centered large expression, equals sign, green result
  - **Clock** (`get_datetime`) — large monospace time display with date
  - **Generic fallback** (any other tool) — JSON input/output blocks

### JS state
`agentRunId`, `agentSteps[]`, `agentAutoFollow`, `agentActiveStep`, `agentPanelOpen`

### Key implementation details
- All 4 provider code paths (Anthropic, OpenAI/OpenRouter, Ollama, text fallback) broadcast `_agent_step` with timing
- `run_agent_loop()` takes `user_id` and `run_id` parameters; `_handle_chat()` generates `run_id = uuid4()`
- Error path in `_handle_chat()` also broadcasts `_agent_done` with `error: true`
- **Windows gotcha**: `pkill` does NOT kill Python processes on Windows. Must use `powershell.exe -Command "Get-Process python | Stop-Process -Force"` to restart the server with new code

---

# EAR WEEE Lookup Project

## What it is
A Python script that queries the German EAR SOAP API to look up WEEE registration numbers (WEEE-Reg.-Nr.) for manufacturers. Used to generate JTL-Wawi workflow templates that automatically insert the correct WEEE number based on manufacturer name.

## Files
- `ear_weee_lookup.py` — Main lookup script, queries EAR SOAP API (`soap.ear-system.de/ear-soap/v2/VerzeichnisService?wsdl`)
- `workflow_komplett.txt` — Original JTL-Wawi workflow (717 entries, Liquid template format). Many WEEE numbers are wrong due to the original script bug.
- `workflow_komplett_fixed.txt` — Corrected workflow (717 entries). 179 suspicious shared numbers cleared, 434 entries with verified WEEE numbers, 283 without.

## Bug that was fixed (2026-02-20)
The original `suche_marke()` and `suche_hersteller()` functions took the **first API result** without verifying it matched the searched manufacturer. The EAR API returns multiple results for partial matches, so searching for "Kingston" might return Sharp's result first, assigning Sharp's WEEE number `28737525` to Kingston.

### Fix: Name matching/scoring
Added `normalize_name()` (strips company suffixes like GmbH, AG, Inc) and `score_name_match()` (0-100 scoring: exact=100, contains=80, word overlap=60). Both functions now iterate all API results, score each against the search name, and only return the best match if score >= 50. If no good match → returns None instead of guessing.

## Workflow cleanup approach
1. Parsed all 717 entries from `workflow_komplett.txt`
2. Counted how many different manufacturers share each WEEE number
3. Numbers appearing 3+ times across unrelated companies → cleared (these are API errors)
4. Numbers appearing 1-2 times → kept (likely correct)
5. **Known brand families** preserved (companies that legitimately share one WEEE number):
   - `84860160`: Krups/Tefal/Rowenta/Moulinex (Groupe SEB)
   - `32322754`: Braun/Oral-B (P&G)
   - `93935395`: Corsair/Elgato
   - `31310269`: SanDisk/Western Digital
   - `12593135`: Barbie/MATTEL/Minions
   - `48428293`: HP/HyperX
   - `14281402`: Seagate/LaCie
   - `92899867`: Robert Bosch/DREMEL
   - `44682106`: Conceptronic/Equip/DDC
   - `84054365`: Hikoki/Metabo
   - `35469319`: Crucial/Micron
   - Plus same-name variants: DrayTek, Wagner, BaByliss, Denon, Fellowes, NVIDIA, IVS/swisstone, Acer/AOpen

## Important findings
- **Amazon `89633988`**: Confirmed correct via amazon.de — belongs to Amazon EU S.a.r.l. Was initially cleared by the frequency algorithm but manually restored.
- **`28737525`**: Belongs to Sharp (confirmed via testsieger.de). Was wrongly assigned to Kingston, be quiet!, Palit, etc.
- **`10737648`**: Wrong for Siemens — BSH's correct number is `57986696`.
- Many cleared entries could not be verified online — would need direct EAR database access at `ear-system.de/ear-verzeichnis/hersteller` for full verification.

## Workflow format
Single-line Liquid template for JTL-Wawi:
```
{% if Vorgang.Allgemein.Stammdaten.Hersteller.Name contains "NAME" %}WEEE_NUMBER{% elsif ... %}
```

## WEEE number format
8-digit number, displayed as `DE 12345678` on products. Each manufacturer registers per product category, so one company can have multiple numbers.
