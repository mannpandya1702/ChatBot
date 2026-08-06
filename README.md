# JARVIS

A fully local, voice-first assistant for Windows that monitors the machine and answers spoken
questions about it. No cloud, no API keys, no accounts, no telemetry. Everything runs on your
own hardware.

Say "Hey Jarvis, how's my system doing" and get a spoken answer built from live readings.

`CLAUDE.md` is the build contract. `PROGRESS.md` is the build ledger.

---

## What it does

- Listens continuously for a wake word without sending audio anywhere.
- Transcribes locally with faster-whisper, thinks locally with Qwen3 through Ollama, and speaks
  locally with Kokoro.
- Answers questions about this machine using real tools: processor, memory, disks, GPU, network,
  processes, temperatures, fans, pending updates, and the event log.
- Shows a frameless transparent HUD with a particle orb and live sparklines.
- Lets you interrupt it mid-sentence. Playback stops within about a tenth of a second.
- Refuses to change anything on your machine without asking you out loud first.

---

## Requirements

| | |
|---|---|
| OS | Windows 11 (Windows 10 22H2 works, untested) |
| Python | 3.12, provisioned by `uv` |
| Package manager | [uv](https://docs.astral.sh/uv/) |
| LLM runtime | [Ollama](https://ollama.com/download) for Windows |
| GPU | Optional. NVIDIA with 6 GB or more unlocks the faster tiers |
| Disk | About 12 GB for models on the default tier |
| HUD | Optional. Needs Rust and Node |

Everything is Apache-2.0, MIT, BSD, or MPL-2.0. Nothing here costs money or requires an account.

---

## Install

```powershell
git clone <your-fork> jarvis
cd jarvis

# Installs dependencies, creates config.yaml, and detects your hardware tier.
.\scripts\setup_env.ps1

# Pulls the Ollama model and the wake word, VAD, and TTS weights for your tier.
.\scripts\pull_models.ps1

# Confirm everything is reachable before you start talking to it.
uv run python -m jarvis --check
```

Then:

```powershell
uv run python -m jarvis
```

Say "Hey Jarvis", wait for the orb to brighten, and ask your question.

### Try it without a microphone

```powershell
uv run python -m jarvis --say "how much memory is free"
```

This runs the full turn loop, tools included, and prints the reply. It is the quickest way to
check that Ollama and the tools are working.

---

## Hardware tiers

The tier is detected once by `scripts/verify_cuda.py` and written into `config/config.yaml`.
Override it there if the detection guesses wrong.

| Tier | VRAM | STT | LLM |
|---|---|---|---|
| `cpu` | none, or under 6 GB | whisper.cpp `base.en` | `qwen3:4b` |
| `gpu-6` | 6 to 7 GB | faster-whisper `small.en` | `qwen3:4b` |
| `gpu-8` | 8 to 11 GB | faster-whisper `small.en` | `qwen3:8b` |
| `gpu-12` | 12 to 15 GB | faster-whisper `distil-large-v3` | `qwen3:8b` |
| `gpu-16` | 16 to 23 GB | faster-whisper `distil-large-v3` | `qwen3:14b` |
| `gpu-24` | 24 GB or more | faster-whisper `distil-large-v3` float16 | `qwen3:32b` |

The vision tool needs roughly 6 GB of spare VRAM and disables itself on `cpu` and `gpu-6`.

**Latency target** on `gpu-12`, from the end of your sentence to the first sound of the reply:
1200 ms at p95. Measure yours with `uv run python tests/bench_latency.py`.

---

## Tool catalogue

Read-only unless marked otherwise. Mutating tools always require a spoken confirmation.

### System

| Tool | Answers | Units |
|---|---|---|
| `sys.cpu` | How busy is the CPU, how long has it been up | percent, MHz, seconds |
| `sys.memory` | How much RAM is free, what is using it | GB, MB, percent |
| `sys.disk` | How much disk space is left, is the drive healthy | GB, MB/s |
| `sys.gpu` | How hot is the GPU, how much VRAM is free | Celsius, MB, watts, MHz |
| `sys.network` | How fast is the connection, is anything downloading | Mb/s, GB |
| `sys.process.top` | What is using my CPU or memory | percent, MB |
| `sys.process.find` | Is Chrome running | percent, MB |
| `sys.thermal` | How hot is my CPU | Celsius. Needs the elevated helper |
| `sys.fans` | How fast are the fans spinning | RPM. Needs the elevated helper |
| `sys.updates` | Do I have updates pending | count, MB. Needs the elevated helper |
| `sys.events` | Has anything crashed | counts, last 24 hours |

### Everything else

| Tool | Answers | Notes |
|---|---|---|
| `files.search` | Where is my tax return | Uses Everything if installed |
| `reminders.list` | What reminders do I have | |
| `vision.screen` | What is on my screen | Needs about 6 GB VRAM |
| `websearch.search` | Outside-world questions | Off until SearXNG is configured |
| `apps.list_windows` | What do I have open | |
| **`apps.launch`** | Open Spotify | **Mutating.** Names only, never paths |
| **`apps.focus`** | Bring Chrome to the front | **Mutating** |
| **`media.control`** | Pause the music, turn it down | **Mutating** |
| **`reminders.create`** | Remind me in ten minutes | **Mutating** |
| **`reminders.delete`** | Cancel that reminder | **Mutating** |
| **`shell.run`** | Run an allowlisted command | **Mutating.** Off by default |

---

## Safety

The design assumption is that a language model will eventually ask for something you did not
intend, so the guardrails do not depend on the model behaving.

- **Read-only by default.** A tool must opt in to being able to change anything.
- **Six mutating tools, and no more.** The allowlist is in `config.yaml`. A tool that mutates but
  is not on it is refused outright, not prompted for.
- **Spoken confirmation.** JARVIS reads the action back and waits for a clear yes. Silence, an
  ambiguous answer, and a timeout are all refusals. "No, don't do it" is a refusal even though it
  contains the phrase "do it".
- **Single-use grants.** A yes authorises exactly the tool and arguments that were read out, once.
  It cannot be replayed or transferred to a different call.
- **Two independent gates.** The confirmation gate refuses, and the tool registry refuses again
  underneath it. Bypassing one does not get you through.
- **`shell.run` is allowlist-first**, off by default, and never builds a command string from model
  output. Commands are tokenised and passed as fixed argv with no shell involved, so there is
  nothing to inject into. Every invocation and every refusal is logged to
  `logs/shell_audit.jsonl`.
- **Privilege separation.** JARVIS runs non-elevated. Only `jarvis-helper.exe` is elevated, and it
  exposes exactly three parameterless sensor methods. A request carrying any parameter at all is
  rejected before dispatch, so there is no field an LLM-authored value could travel in.

---

## The elevated helper

CPU temperatures, fan speeds, and the pending update list are not readable without administrator
rights. Rather than elevate the whole assistant, a small sidecar does only that.

```powershell
.\scripts\build_helper.ps1     # produces jarvis-helper.exe with a UAC manifest
.\jarvis-helper.exe            # approve the prompt
```

You need `vendor\LibreHardwareMonitorLib.dll` first. It is MPL-2.0 and is not redistributed here;
download it from the
[LibreHardwareMonitor releases](https://github.com/LibreHardwareMonitor/LibreHardwareMonitor).

Declining the UAC prompt is fine. Those three tools then report that they are unavailable, and
everything else keeps working.

---

## The HUD

```powershell
cd src\jarvis\ui\app
npm install
npm run tauri build
```

Frameless, transparent, always on top, no taskbar entry, click-through except over the drag handle
and the transcript. Tray icon gives show, hide, and quit.

Run without it:

```powershell
uv run python -m jarvis --headless
```

---

## Configuration

`config/config.yaml` is yours and is gitignored. `config/config.example.yaml` documents every
setting with its default. An empty `config.yaml` is valid; every key is optional.

Environment variables override the file:

```powershell
$env:JARVIS_LLM__MODEL = "qwen3:14b"
$env:JARVIS_LOGGING__LEVEL = "DEBUG"
```

Worth knowing about:

| Setting | Default | Why you would change it |
|---|---|---|
| `wake.threshold` | 0.5 | Raise toward 0.7 if it triggers on the television |
| `vad.trailing_silence_ms` | 500 | Lower if it feels slow to reply, raise if it cuts you off |
| `orchestrator.idle_timeout_s` | 30 | How long follow-ups work without the wake word again |
| `orchestrator.always_listening` | false | Skip the wake word entirely, useful at a desk |
| `persona.user_address_form` | sir | Set to `""` for no form of address |
| `tools.enable_shell` | false | Both this and `shell.enabled` must be true |

---

## Autostart

```powershell
.\scripts\install_autostart.ps1            # non-elevated logon task
.\scripts\install_autostart.ps1 -Headless
.\scripts\install_autostart.ps1 -Remove
```

The task runs at your privilege level, never elevated. The script removes it and fails loudly if
it somehow registered otherwise.

---

## Troubleshooting

### `Could not locate cudnn_ops64_9.dll`

The classic one. faster-whisper needs the cuDNN 9 runtime for CUDA 12, which the CUDA toolkit
does not install.

```powershell
uv pip install nvidia-cudnn-cu12 nvidia-cublas-cu12
```

Then make sure the DLL directory is on `PATH`, or set `stt.device: cpu` in `config.yaml` to work
around it. `uv run python scripts/verify_cuda.py` detects this specifically and prints the fix.

### Ollama is not running

```powershell
ollama serve
ollama list
```

If the model is missing, run `.\scripts\pull_models.ps1`. JARVIS names the exact `ollama pull`
command when it hits this.

### No audio device, or PortAudio errors

```powershell
uv run python -c "from jarvis.audio.devices import describe_devices; print(describe_devices())"
```

Set `audio.input_device` and `audio.output_device` in `config.yaml` to the exact names it lists.
`sounddevice` bundles PortAudio, so a separate install is not needed; if it fails to load,
reinstall with `uv sync --extra audio --reinstall-package sounddevice`.

### The wake word never fires, or fires constantly

Raise or lower `wake.threshold`. Check the input device is the one you are speaking into, and
that Windows is not applying an aggressive noise suppression effect to it.

### Windows Defender flags jarvis-helper.exe

PyInstaller executables are frequently false-positived, and this one requests elevation, which
raises the score further. Add an exclusion for the repository folder, or build it yourself and
verify the source, which is `src/jarvis/helper/`.

### Temperatures and fans report as unavailable

The helper is not running, or you declined the UAC prompt. `.\jarvis-helper.exe --check` reports
its elevation state.

### It answers slowly

```powershell
uv run python tests/bench_latency.py
```

The report names the stage that is over budget. Common causes: the model is not resident, so raise
`llm.keep_alive`; thinking mode is on, so set `llm.think: false`; or the tier picked a model too
large for the card.

### Wrong Python version

`.python-version` pins 3.12 and `uv` provisions it. `uv run python --version` should print 3.12.x.
Do not run JARVIS with a system Python; `uv run` is the supported entrypoint.

---

## Development

```powershell
uv sync --group dev            # core plus test tooling
uv run ruff check .
uv run mypy src/
uv run pytest -q               # 1300+ tests, no hardware needed
uv run pytest -m manual -v     # the hardware and human checks
```

The suite runs on Linux or macOS with no GPU, no microphone, and no Windows. Every Windows-only
and engine-specific import is lazy, and `tests/test_imports.py` enforces that. Anything that
genuinely needs hardware is marked `manual` and deselected by default; `PROGRESS.md` lists each
one with the exact check to perform on the Windows host.

```
src/jarvis/
  audio/    devices, ring buffer, wake word, VAD, STT, TTS, player
  brain/    Ollama client, memory, persona, orchestrator
  tools/    registry, confirmation gate, and every tool
  helper/   the elevated sidecar
  ui/       websocket broadcaster and the Tauri HUD
  util/     logging, latency, errors, platform detection, resilience
```

---

## Licence

MIT. Component licences are listed in `CLAUDE.md` §1; all are permissive.
LibreHardwareMonitorLib.dll is MPL-2.0 and is loaded at runtime rather than redistributed.
