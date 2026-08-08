# CLAUDE.md - JARVIS Build Contract

You are the sole engineer building **JARVIS**, a fully local, free, voice-first AI assistant for Windows that monitors the machine and answers spoken questions about it.

This file is the single source of truth. The task ledger below is the build plan. `/loop` executes it.

---

## 0. NON-NEGOTIABLE CONSTRAINTS

Violating any of these is a build failure. Do not "temporarily" break them.

1. **Fully local.** No cloud LLM, no hosted STT/TTS, no API keys, no telemetry. Network access is allowed only for: package installs, model downloads, and explicitly user-enabled tools (SearXNG, Home Assistant). If a library phones home by default, disable it.
2. **Zero cost.** No paid services, no trials, no accounts. If a component requires signup, it is disqualified.
3. **Permissive licenses only.** Apache-2.0 / MIT / BSD preferred. **Reject on sight:** Coqui XTTS-v2 (CPML non-commercial), F5-TTS weights (CC-BY-NC), Fish Speech (CC-BY-NC-SA), Picovoice Porcupine (free tier is non-commercial), any GPL component linked in-process (Piper's `piper1-gpl` fork may only be used as a separate HTTP server subprocess, never imported).
4. **Windows native.** Target Windows 11, native Python. No WSL2 dependency in the critical path.
5. **Read-only by default.** Every system tool is read-only unless explicitly listed in the mutating-tool allowlist. No LLM-generated command is ever auto-executed.
6. **No secrets in the repo.** No tokens, no absolute user paths hardcoded. Everything user-specific goes in `config/config.yaml` (gitignored) with `config/config.example.yaml` committed.

---

## 0b. DEVELOPMENT HOST VS TARGET HOST

The **target host** is Windows 11. The **development host** may be Linux or macOS. This does not relax
any constraint in §0; it constrains how the code is written:

- Every Windows-only import (`win32*`, `wmi`, `clr`/pythonnet, `pywinauto`, `pygetwindow`,
  `windows_toasts`) and every optional engine (`sounddevice`, `openwakeword`, `faster_whisper`,
  `kokoro`, `pynvml`, `mss`) must be imported **lazily inside the function that needs it**, never at
  module import time.
- Every module in `src/jarvis/` must import cleanly on Linux with only the core dependency set
  installed. This is enforced by `tests/test_imports.py`.
- Capability detection lives in `jarvis.util.platform`. Code asks `is_windows()` / `has_nvml()` and
  degrades to a structured, speakable error rather than raising.
- Anything that genuinely needs Windows, a GPU, a microphone, or a human is marked
  `@pytest.mark.manual` and recorded in `PROGRESS.md` as `NEEDS_MANUAL_VERIFY` with the exact check
  to perform on the Windows host.

---

## 1. LOCKED STACK

Do not substitute components. If a component genuinely cannot be made to work after 3 documented attempts, mark the task `BLOCKED`, write the failure into `PROGRESS.md`, and move on. Do not silently swap in an alternative.

| Layer | Choice | Package | License |
|---|---|---|---|
| Wake word | openWakeWord, `hey_jarvis` model | `openwakeword` | Apache-2.0 |
| VAD / endpointing | Silero VAD | `silero-vad` (via torch hub or onnx) | MIT |
| STT | faster-whisper, `distil-large-v3` int8 on GPU / `small.en` fallback | `faster-whisper` | MIT |
| STT CPU fallback | whisper.cpp | `pywhispercpp` | MIT |
| LLM runtime | Ollama, native Windows | HTTP `localhost:11434` | MIT |
| LLM model | Qwen3, size by VRAM (see §2) | via `ollama pull` | Apache-2.0 |
| TTS | Kokoro-82M, voice `bm_george` | `kokoro` + `soundfile` | Apache-2.0 |
| Audio I/O | sounddevice (bundles PortAudio) | `sounddevice` | MIT |
| System metrics | psutil | `psutil` | BSD-3 |
| GPU metrics | NVML | `pynvml` | BSD-3 |
| Temps / fans | LibreHardwareMonitorLib via pythonnet | `pythonnet` + bundled DLL | MPL-2.0 |
| Windows APIs | pywin32 / WMI | `pywin32`, `wmi` | PSF / MIT |
| Vision (Phase 4) | Qwen3-VL 8B or Moondream 3 via Ollama | Ollama | Apache-2.0 |
| UI shell | Tauri, frameless transparent always-on-top | Rust + `tauri` | MIT |
| UI render | Three.js particle orb, WebSocket-driven | `three` | MIT |
| Orchestration | Custom thin Python loop, Ollama native tool calling | none | - |

**Explicitly rejected, do not introduce:** LangChain, LlamaIndex, CrewAI, Electron, Coqui TTS, Porcupine, edge-tts (requires internet), Dipeshpal/Jarvis_AI (routes through a remote server).

Reference implementations you may read and adapt patterns from (do not vendor wholesale): `KoljaB/RealtimeVoiceChat` for the barge-in loop, `dnhkng/GLaDOS` for the circular-buffer low-latency pattern, `jincocodev/openclaw-jarvis-ui` for the HUD.

---

## 2. HARDWARE TIERING

Read `hardware.tier` from config. When it is `auto`, `scripts/verify_cuda.py` detects the tier at
first run and writes the resolved value back into `config/config.yaml`.

| Tier | Condition | STT | LLM |
|---|---|---|---|
| `cpu` | no CUDA GPU, or under 6 GB usable VRAM | whisper.cpp `base.en` | `qwen3:4b` Q4_K_M |
| `gpu-6` | 6-7 GB VRAM | faster-whisper `small.en` int8 | `qwen3:4b` Q4_K_M |
| `gpu-8` | 8-11 GB VRAM | faster-whisper `small.en` int8 | `qwen3:8b` Q4_K_M |
| `gpu-12` | 12-15 GB VRAM | faster-whisper `distil-large-v3` int8 | `qwen3:8b` Q5_K_M |
| `gpu-16` | 16-23 GB VRAM | faster-whisper `distil-large-v3` int8 | `qwen3:14b` Q4_K_M |
| `gpu-24` | 24 GB+ VRAM | faster-whisper `distil-large-v3` float16 | `qwen3:32b` Q4_K_M |

`gpu-6` closes the gap in the original table, where a 6 GB or 7 GB card matched no tier at all. A
card below 6 GB is treated as `cpu` for the LLM but may still run faster-whisper if CUDA is present.

The vision tool requires roughly 6 GB of free VRAM for the VLM. On `cpu` and `gpu-6` it is disabled
automatically with a spoken explanation rather than an error.

---

## 3. ARCHITECTURE

```
                    +------------------------------+
   mic --> ring --> | wake (openWakeWord)          |
          buffer    |   +-> VAD (Silero) endpoint  |
                    |        +-> STT (faster-whisper)
                    +------------+-----------------+
                                 v
                    +------------------------------+
                    | Orchestrator                 |
                    |  memory (rolling + summary)  |
                    |  Ollama chat + tools[]       |
                    |  tool dispatch --> registry  |
                    |  confirmation gate (mutating)|
                    +------------+-----------------+
                                 v token stream
                    +------------------------------+
                    | TTS (Kokoro) -> sentence     |
                    |  chunker -> sounddevice out  |
                    |  barge-in: VAD kills playback|
                    +------------+-----------------+
                                 v state events
                    +------------------------------+
                    | WebSocket :8765 -> Tauri HUD |
                    +------------------------------+

  +------------------------------------------------+
  | jarvis-helper.exe (elevated sidecar, JSON-RPC   |
  | over named pipe): LibreHardwareMonitor temps,   |
  | fans, Windows Update query. Nothing else.       |
  +------------------------------------------------+
```

**Latency budget (gpu-12 target, end of user speech to first audio out):**

| Stage | Budget |
|---|---|
| VAD endpoint decision | <= 250 ms |
| STT finalize | <= 200 ms |
| LLM time-to-first-token | <= 400 ms |
| TTS time-to-first-audio | <= 300 ms |
| **Total p95** | **<= 1200 ms** |

A tool call adds one LLM round trip. Budget <= 2000 ms p95 for tool-answering turns. Every stage must stream. `src/jarvis/util/latency.py` instruments each stage and logs a per-turn breakdown. If a phase's acceptance test shows a stage over budget, fix it before marking the task DONE.

---

## 4. REPO LAYOUT

Create exactly this. Do not invent parallel structures.

```
jarvis/
|- CLAUDE.md
|- PROGRESS.md                    # loop state ledger, machine-maintained
|- README.md
|- pyproject.toml                 # uv-managed
|- .gitignore
|- .python-version                # 3.12
|- config/
|  |- config.example.yaml         # committed
|  \- config.yaml                 # gitignored
|- models/                        # gitignored, downloaded artifacts
|- vendor/
|  \- LibreHardwareMonitorLib.dll
|- src/jarvis/
|  |- __init__.py
|  |- main.py                     # entrypoint
|  |- config.py                   # pydantic-settings loader
|  |- state.py                    # AssistantState enum + event bus
|  |- audio/
|  |  |- devices.py
|  |  |- ring.py                  # continuous circular capture buffer
|  |  |- wake.py
|  |  |- vad.py
|  |  |- stt.py
|  |  |- tts.py
|  |  \- player.py                # streaming playback + interrupt
|  |- brain/
|  |  |- llm.py                   # Ollama client, streaming, tool calls
|  |  |- memory.py
|  |  |- persona.py               # system prompt
|  |  \- orchestrator.py          # the turn loop
|  |- tools/
|  |  |- registry.py              # decorator -> JSON schema
|  |  |- gate.py                  # confirmation + allowlist
|  |  |- sys_cpu.py
|  |  |- sys_memory.py
|  |  |- sys_disk.py
|  |  |- sys_gpu.py
|  |  |- sys_network.py
|  |  |- sys_process.py
|  |  |- sys_thermal.py           # via helper
|  |  |- sys_updates.py           # via helper
|  |  |- sys_events.py
|  |  |- apps.py
|  |  |- media.py
|  |  |- files.py
|  |  |- vision.py
|  |  |- websearch.py
|  |  |- reminders.py
|  |  \- shell.py                 # gated, Phase 4
|  |- helper/
|  |  |- __main__.py              # elevated sidecar entrypoint
|  |  |- rpc.py                   # named pipe JSON-RPC
|  |  \- lhm.py                   # pythonnet + LibreHardwareMonitor
|  |- ui/
|  |  |- server.py                # websocket state broadcaster
|  |  \- app/                     # Tauri project
|  |     |- src-tauri/
|  |     \- src/                  # Three.js orb + HUD
|  \- util/
|     |- logging.py
|     |- latency.py
|     |- errors.py
|     \- platform.py              # capability detection (added, see §0b)
|- tests/
|  |- test_tools_*.py
|  |- test_registry.py
|  |- test_gate.py
|  \- bench_latency.py
\- scripts/
   |- setup_env.ps1
   |- pull_models.ps1
   |- verify_cuda.py
   \- build_helper.ps1
```

---

## 5. ENGINEERING RULES

- **Python 3.12**, managed with `uv`. All deps pinned in `pyproject.toml`. Never `pip install` ad hoc without adding it to the manifest.
- **Type hints everywhere.** `pydantic` models for all tool inputs and outputs. `ruff` + `mypy --strict` on `src/`. Both must pass before any task is DONE.
- **No blocking calls in the audio path.** Audio capture, STT, LLM, and TTS run on separate threads/async tasks communicating over queues. A slow LLM must never stall mic capture.
- **Every tool is a pure function** taking a pydantic input model and returning a pydantic output model. Registration via a `@tool` decorator that auto-generates the JSON schema Ollama receives. Tool descriptions must be precise and include units, since schema quality drives tool-calling reliability more than model size.
- **Tool output is spoken.** Return values must be phrasable. Return `{"cpu_percent": 43.2, "top_process": "chrome.exe"}`, not a 200-row table. Cap list returns at 5 items.
- **Fail loud in logs, fail soft in voice.** A tool exception must never crash the loop. Catch, log with traceback, return a structured error the LLM can verbalize as "I couldn't read that sensor."
- **Every task gets a commit.** Conventional format: `T-1.3: implement Silero VAD endpointing`.
- **No em dashes in any generated prose, docs, comments, or spoken output.** Enforced by `tests/test_style.py`.

---

## 6. SAFETY CONTRACT

This is the section you are most likely to get wrong. Read it twice.

- **Read-only default.** Tools in `sys_*`, `files.search`, `vision.*`, `websearch.*` are read-only. They perform no writes, no process kills, no config changes.
- **Mutating-tool allowlist.** Only these may change system state, and only in Phase 4: `apps.launch`, `apps.focus`, `media.control`, `reminders.create`, `reminders.delete`, `shell.run`.
- **Confirmation gate.** Every mutating tool call routes through `tools/gate.py`, which returns a `ConfirmationRequired` result. The orchestrator speaks the intended action back and waits for an explicit affirmative ("yes", "confirm", "do it") within 15 seconds. Timeout equals denial. No exceptions, no "obviously safe" bypasses.
- **`shell.run` hardening** (all mandatory):
  - Command allowlist, not a denylist. Denylist is a secondary layer only.
  - Blocked commands: `rm`, `del`, `rmdir`, `format`, `diskpart`, `shutdown`, `restart`, `reg`, `regedit`, `net`, `netsh`, `takeown`, `icacls`, `sc`, `bcdedit`, `vssadmin`, `cipher`.
  - Blocked shell operators: `&`, `&&`, `|`, `;`, backtick, `$(`.
  - Blocked PowerShell args: `-enc`, `-EncodedCommand`, `-Command`, `-e`, `-nop` combined with download cmdlets.
  - Always invoke as `powershell.exe -NoProfile -NonInteractive -File` or with a fixed argv. Never build a command string from LLM output.
  - Max command length 512 chars. Working directory restricted to a configured allowlist.
  - Hard 30 second timeout. Output truncated to 4 KB.
  - Every invocation logged to `logs/shell_audit.jsonl` with timestamp, command, exit code, and whether the user confirmed.
- **Never auto-execute, ever:** registry edits, user or permission changes, disk formatting, bulk file deletion, firewall or network config, service start/stop, driver installs, download-then-run.
- **Privilege separation.** The main process runs **non-elevated**. Only `jarvis-helper.exe` runs elevated, exposes exactly three RPC methods (`get_thermals`, `get_fans`, `get_pending_updates`), and accepts no arbitrary commands. The helper never takes an LLM-authored string as input.

---

## 7. TASK LEDGER

`/loop` works this list top to bottom. Each task has an ID, dependencies, and a verification command that must exit 0.

### Phase 0 - Scaffold

- **T-0.1** Init repo, `uv` project, Python 3.12, `.gitignore`, `pyproject.toml` with ruff+mypy+pytest config. *Verify:* `uv run ruff check . && uv run mypy src/`
- **T-0.2** Write `config.example.yaml` covering every setting referenced in this file. Implement `config.py` with pydantic-settings, sensible defaults, clear errors on missing required keys. *Verify:* `uv run pytest tests/test_config.py`
- **T-0.3** `util/logging.py` (structured JSON to file + pretty to console), `util/latency.py` (context-manager stage timer, per-turn breakdown), `util/errors.py`, `util/platform.py`. *Verify:* `uv run pytest tests/test_util.py`
- **T-0.4** `scripts/verify_cuda.py`: detect GPU, VRAM, CUDA/cuDNN availability, write the resolved `hardware.tier` into config. Must print a clear remediation message on the classic `cudnn_ops64_9.dll` failure. *Verify:* `uv run python scripts/verify_cuda.py`
- **T-0.5** `scripts/setup_env.ps1` and `scripts/pull_models.ps1`: install deps, download openWakeWord `hey_jarvis`, Kokoro weights, pull the tier-appropriate Ollama model. Idempotent. *Verify:* script runs twice cleanly.

### Phase 1 - Voice loop MVP

- **T-1.1** `audio/devices.py` + `audio/ring.py`: enumerate devices, continuous 16 kHz mono capture into a circular buffer with ~2 s of pre-roll so the wake word's trailing audio is not lost. *Verify:* `uv run pytest tests/test_ring.py`
- **T-1.2** `audio/wake.py`: openWakeWord on the ring buffer, configurable threshold, cooldown, emits `WakeDetected` with the pre-roll slice. *Verify:* manual - say the wake word 10 times, log >= 9 detections, <= 1 false accept over 10 minutes of ambient audio.
- **T-1.3** `audio/vad.py`: Silero VAD endpointing with configurable min-speech and trailing-silence thresholds. Target endpoint decision <= 250 ms. *Verify:* `uv run pytest tests/test_vad.py`
- **T-1.4** `audio/stt.py`: faster-whisper streaming transcription, tier-aware model selection, whisper.cpp fallback path on `cpu` tier. Log RTF per utterance. *Verify:* `uv run pytest tests/test_stt.py` (fixture WAV, WER sanity check)
- **T-1.5** `audio/tts.py` + `audio/player.py`: Kokoro synthesis with voice `bm_george`, sentence-level chunking so playback starts on the first complete sentence, streaming output via sounddevice, `stop()` that cuts playback within 100 ms. *Verify:* `uv run pytest tests/test_tts.py` and measure time-to-first-audio <= 300 ms.
- **T-1.6** `brain/llm.py`: Ollama client, streaming chat, native tool-call parsing, retry on malformed tool JSON (max 2 retries, then verbalize failure). *Verify:* `uv run pytest tests/test_llm.py` against a live Ollama.
- **T-1.7** `brain/memory.py`: rolling message window with token cap and a running summary that compacts when the cap is hit. Persists to SQLite so context survives restart. *Verify:* `uv run pytest tests/test_memory.py`
- **T-1.8** `brain/persona.py`: system prompt. Dry, precise, British-butler register. Answers in one or two short spoken sentences unless asked to elaborate. Never reads out raw JSON. Never invents metrics it did not get from a tool. *Verify:* `uv run pytest tests/test_persona.py`
- **T-1.9** `brain/orchestrator.py` + `main.py`: wire the full turn loop with threaded stages and queues. Implement **barge-in**: VAD stays live during playback, user speech kills TTS and starts a new turn. Idle timeout returns to wake-word mode after 30 s. *Verify:* `uv run python -m jarvis` - full spoken round trip, and `uv run python tests/bench_latency.py` reports p95 <= 1200 ms.

**Phase 1 gate:** say "Hey Jarvis, what time is it" and hear a spoken answer, then interrupt mid-sentence and have it stop. Do not proceed to Phase 2 until this passes.

### Phase 2 - System monitoring

- **T-2.1** `tools/registry.py`: `@tool` decorator producing Ollama-compatible JSON schemas from pydantic models, plus a registry with categories and a read-only flag. *Verify:* `uv run pytest tests/test_registry.py`
- **T-2.2** `tools/sys_cpu.py`: overall and per-core utilization, load trend, frequency, core/thread count, uptime. Handle psutil's zero-on-first-call behavior with an interval. *Verify:* `uv run pytest tests/test_tools_cpu.py`
- **T-2.3** `tools/sys_memory.py`: RAM used/free/percent, swap, top 5 memory consumers. *Verify:* `uv run pytest tests/test_tools_memory.py`
- **T-2.4** `tools/sys_disk.py`: per-volume usage, free space, read/write throughput, SMART health via `smartctl` if present (degrade gracefully if not). Never call `Win32_Product`. *Verify:* `uv run pytest tests/test_tools_disk.py`
- **T-2.5** `tools/sys_gpu.py`: NVML utilization, VRAM used/total, temperature, power draw, clocks, top GPU processes. Graceful "no NVIDIA GPU detected" path. *Verify:* `uv run pytest tests/test_tools_gpu.py`
- **T-2.6** `tools/sys_network.py`: throughput up/down over a sampling interval, active adapter, connection count, top network-consuming processes. *Verify:* `uv run pytest tests/test_tools_network.py`
- **T-2.7** `tools/sys_process.py`: top N by CPU or memory, lookup by name, "what's using my CPU" answer shape. Capped at 5 results. *Verify:* `uv run pytest tests/test_tools_process.py`
- **T-2.8** `helper/`: build the elevated sidecar. pythonnet loads `LibreHardwareMonitorLib.dll`, exposes exactly `get_thermals`, `get_fans`, `get_pending_updates` over a named pipe with JSON-RPC. `scripts/build_helper.ps1` produces `jarvis-helper.exe` with a UAC manifest. Main process auto-starts it once, caches results for 5 s, and degrades gracefully if the helper is unavailable or the user declines elevation. *Verify:* `uv run pytest tests/test_helper_rpc.py` (mocked) plus one manual elevated run.
- **T-2.9** `tools/sys_thermal.py` and `tools/sys_updates.py`: thin clients over the helper RPC. Windows Update query prefers `PSWindowsUpdate` `Get-WindowsUpdate`, falls back to the `Microsoft.Update.Session` COM API. Report count, names, and total size of pending updates. Never install anything. *Verify:* `uv run pytest tests/test_tools_thermal.py`
- **T-2.10** `tools/sys_events.py`: recent System and Application event-log errors and warnings, last 24 h, capped at 5, summarized. *Verify:* `uv run pytest tests/test_tools_events.py`
- **T-2.11** Register all Phase 2 tools with the LLM, tune descriptions, and add a multi-tool integration test covering "how's my system doing" (should chain CPU + memory + disk + GPU into one spoken summary). *Verify:* `uv run pytest tests/test_integration_monitoring.py`

**Phase 2 gate:** ask aloud, and get correct spoken answers for: overall system health, what is using the CPU, GPU temperature, free disk space, pending Windows updates, and uptime.

### Phase 3 - HUD

- **T-3.1** `ui/server.py`: WebSocket on `localhost:8765` broadcasting `{state, audio_level, transcript, response, metrics}` at 30 Hz. States: `idle`, `listening`, `thinking`, `speaking`, `error`. *Verify:* `uv run pytest tests/test_ui_server.py`
- **T-3.2** Scaffold the Tauri app: frameless, transparent, always-on-top, click-through except over interactive elements, no taskbar entry, system tray with show/hide/quit. *Verify:* `npm run tauri build` succeeds and the window renders transparent over the desktop.
- **T-3.3** Three.js orb: ~2000-point BufferGeometry with PointsMaterial and additive blending, per-particle sine/cosine organic drift, displacement and color driven by `audio_level`, distinct palette per state (idle dim cyan, listening bright cyan, thinking amber pulse, speaking white-hot, error red). *Verify:* visual check against each state, plus 60 fps sustained.
- **T-3.4** HUD panels: live CPU/RAM/GPU/network sparklines, rolling transcript, current state label. Arc-reactor framing. Draggable. *Verify:* visual.
- **T-3.5** Wire orb and HUD to the WebSocket, with reconnect-on-drop. Add a `--headless` flag so the core runs without the UI. *Verify:* `uv run python -m jarvis --headless` works, and the UI reconnects after a core restart.

### Phase 4 - Multipurpose and gated actions

- **T-4.1** `tools/gate.py`: confirmation flow, allowlist enforcement, audit logging. Ship this **before** any mutating tool. *Verify:* `uv run pytest tests/test_gate.py` including a test that an unconfirmed mutating call is refused.
- **T-4.2** `tools/apps.py`: launch, focus, list windows via `pywinauto`/`pygetwindow`. Gated. *Verify:* `uv run pytest tests/test_tools_apps.py`
- **T-4.3** `tools/media.py`: play/pause, next, previous, volume via media keys. Gated. *Verify:* `uv run pytest tests/test_tools_media.py`
- **T-4.4** `tools/files.py`: fast local search via the Everything CLI if installed, `os.scandir` fallback. Read-only. Results capped at 5. *Verify:* `uv run pytest tests/test_tools_files.py`
- **T-4.5** `tools/vision.py`: screenshot via `mss`, send to a local VLM through Ollama, answer "what's on my screen". Read-only. *Verify:* `uv run pytest tests/test_tools_vision.py`
- **T-4.6** `tools/reminders.py`: SQLite-backed timers and reminders with Windows toast notifications. Gated on create/delete. *Verify:* `uv run pytest tests/test_tools_reminders.py`
- **T-4.7** `tools/websearch.py`: SearXNG at the configured URL, disabled unless configured. Returns 3 results with snippets. *Verify:* `uv run pytest tests/test_tools_websearch.py`
- **T-4.8** `tools/shell.py`: implement **every** hardening rule in §6. Disabled by default in config. *Verify:* `uv run pytest tests/test_tools_shell.py` - must include red-team cases for operator injection, encoded commands, path escape, and each blocked command.

### Phase 5 - Hardening

- **T-5.1** Full latency benchmark suite, p50/p95/p99 per stage, written to `bench/results.json`. Fail CI if p95 regresses more than 15 percent. *Verify:* `uv run python tests/bench_latency.py`
- **T-5.2** Crash resilience: supervisor that restarts a dead audio or TTS thread, health check endpoint, graceful shutdown on SIGINT. *Verify:* `uv run pytest tests/test_resilience.py`
- **T-5.3** `README.md` with install, model download, hardware tiers, troubleshooting (CUDA/cuDNN DLL resolution, PyAudio/PortAudio, admin prompts, Defender false positives, Python version pinning), and the full tool catalog.
- **T-5.4** Autostart: Task Scheduler entry via `scripts/install_autostart.ps1`, opt-in, non-elevated. *Verify:* script installs and removes cleanly.

---

## 8. `/loop` PROTOCOL

The `/loop` command file lives at `.claude/commands/loop.md`. Its behavior:

1. Read `PROGRESS.md`. If it does not exist, generate it from the §7 ledger with every task set to `TODO`.
2. Select the first task whose status is `TODO` and whose dependencies are all `DONE`.
3. If the task depends on an unfilled `[[FILL: ...]]` placeholder from §9, **stop the loop** and print exactly what is needed. Do not guess. Do not use a placeholder value.
4. Implement the task. Keep the diff focused on that task only. No opportunistic refactors.
5. Run the task's verification command. On failure, diagnose and retry up to 3 times. If still failing, set status `BLOCKED`, record the failure and what was tried in `PROGRESS.md`, and continue to the next independent task.
6. Run `uv run ruff check . && uv run mypy src/` and the full test suite. All must pass before marking `DONE`.
7. Update `PROGRESS.md`: status, timestamp, one-line note, any follow-up discovered.
8. Commit: `git add -A && git commit -m "<task-id>: <summary>"`.
9. Repeat from step 2 until one of: the current phase is fully `DONE`, 8 tasks have completed this run, a phase gate requires manual verification, or the loop is blocked on user input.
10. Print a run summary: tasks completed, tasks blocked, next action required from the user.

**Loop guardrails.** Never run destructive shell commands against the developer's machine. Never force-push, never rewrite history. Never commit `config/config.yaml`, `models/`, or anything in `logs/`. Never elevate the Claude Code process itself. If a task requires the user to physically speak into a microphone or visually inspect the UI, mark it `NEEDS_MANUAL_VERIFY`, describe the exact check, and move on.

---

## 9. CONFIGURATION - RESOLVED

All values below are resolved. Hardware values are `auto`, meaning they are detected at first run by
`scripts/verify_cuda.py` and written into `config/config.yaml`. This satisfies §0.6, which forbids
hardcoding machine-specific absolute paths or specs into the committed tree.

### Machine

```yaml
gpu_model:            auto        # detected via NVML at first run
gpu_vram_gb:          auto        # detected via NVML at first run
cpu_model:            auto        # detected via platform/WMI at first run
ram_gb:               auto        # detected via psutil at first run
windows_version:      auto        # detected via platform.win32_ver at first run
cuda_installed:       auto        # probed: nvidia-smi + ctranslate2 CUDA init
cuda_version:         auto        # reported by nvidia-smi when present
python_installed:     "3.12"      # pinned by .python-version, uv provisions it
ollama_installed:     auto        # probed at startup via GET localhost:11434/api/tags
repo_path:            auto        # resolved from the package location at runtime
```

### Audio

```yaml
mic_device_name:      auto        # default input device, overridable in config.yaml
speaker_device_name:  auto        # default output device, overridable in config.yaml
```

### Persona

```yaml
wake_word:            hey_jarvis
assistant_name:       Jarvis
tts_voice:            am_onyx     # was bm_george; changed by the operator, see PROGRESS.md
user_address_form:    sir
response_style:       "terse, one or two spoken sentences unless asked to elaborate"
```

### Feature switches

```yaml
enable_elevated_helper:   yes     # built and shipped; the user still approves the UAC prompt
enable_shell_tool:        no      # off by default per §6, opt in via config.yaml
enable_vision_tool:       yes     # auto-disables on tiers with under 6 GB VRAM
enable_websearch:         no      # off until searxng_url is set
searxng_url:              null
enable_home_assistant:    no
home_assistant_url:       null
home_assistant_token:     null    # config.yaml only, never in this file
everything_cli_path:      auto    # probed on PATH and in Program Files, falls back to os.scandir
```

### UI

```yaml
build_hud:            yes
rust_installed:       auto        # checked by setup_env.ps1, HUD build skipped with a warning if absent
node_installed:       auto        # checked by setup_env.ps1
hud_position:         bottom-right
accent_color:         "#22d3ee"
```

### Scope

```yaml
target_phase:         5
extra_tools_wanted:   none
```

---

## 10. WHEN TO STOP AND ASK

Halt the loop and ask the user, rather than guessing, if:

- A locked-stack component fails after 3 documented attempts.
- A task would require breaking a §0 constraint or a §6 safety rule.
- A phase gate needs a human to speak, listen, or look at the screen.
- The hardware tier cannot support the task (for example vision on a 4 GB GPU).
- A dependency's license turns out to be more restrictive than §1 assumed.

Never work around a blocker by relaxing a constraint. Report it and wait.
