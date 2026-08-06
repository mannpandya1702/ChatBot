# PROGRESS

Machine-maintained build ledger for `/loop`. See `CLAUDE.md` §7 for task definitions.

Last run: 2026-08-06
Target phase: 5

Statuses: `TODO`, `IN_PROGRESS`, `DONE`, `BLOCKED`, `NEEDS_MANUAL_VERIFY`.

## Environment note

The build host is Linux; the target host is Windows 11. Per `CLAUDE.md` §0b every module
must import and unit-test on the build host, and anything that genuinely needs Windows, a
GPU, a microphone, or a human eye is marked `NEEDS_MANUAL_VERIFY` with the exact check to
run on the Windows machine. Those checks are collected at the bottom of this file.

## Phase 0 - Scaffold

- [x] T-0.1 `DONE` - repo, uv project, Python 3.12, .gitignore, pyproject with ruff+mypy+pytest.
      Note: uv needs `[tool.uv] environments` to fork the lock per platform, otherwise
      openWakeWord's Linux-only tflite-runtime makes the universal resolution unsatisfiable.
- [x] T-0.2 `DONE` - `config.py` plus `config/config.example.yaml` covering every section.
      Note: `gate.affirmatives` must be quoted in YAML; bare `yes`/`no` parse as booleans.
- [x] T-0.3 `DONE` - `util/logging.py`, `util/latency.py`, `util/errors.py`, `util/platform.py`.
- [ ] T-0.4 `TODO` - `scripts/verify_cuda.py`.
- [ ] T-0.5 `TODO` - `scripts/setup_env.ps1`, `scripts/pull_models.ps1`.

## Phase 1 - Voice loop MVP

- [ ] T-1.1 `TODO` - `audio/devices.py` + `audio/ring.py`.
- [ ] T-1.2 `TODO` - `audio/wake.py`.
- [ ] T-1.3 `TODO` - `audio/vad.py`.
- [ ] T-1.4 `TODO` - `audio/stt.py`.
- [ ] T-1.5 `TODO` - `audio/tts.py` + `audio/player.py`.
- [ ] T-1.6 `TODO` - `brain/llm.py`.
- [ ] T-1.7 `TODO` - `brain/memory.py`.
- [ ] T-1.8 `TODO` - `brain/persona.py`.
- [ ] T-1.9 `TODO` - `brain/orchestrator.py` + `main.py`.

## Phase 2 - System monitoring

- [x] T-2.1 `DONE` - `tools/registry.py`. Built ahead of Phase 1 because T-1.9's orchestrator
      depends on it; the dependency edge runs backwards against the ledger's phase order.
      Note: `ToolRegistry.__len__` makes an empty registry falsy, so `target or registry`
      silently wrote to the global registry. Fixed with an explicit `is None` check.
- [ ] T-2.2 `TODO` - `tools/sys_cpu.py`.
- [ ] T-2.3 `TODO` - `tools/sys_memory.py`.
- [ ] T-2.4 `TODO` - `tools/sys_disk.py`.
- [ ] T-2.5 `TODO` - `tools/sys_gpu.py`.
- [ ] T-2.6 `TODO` - `tools/sys_network.py`.
- [ ] T-2.7 `TODO` - `tools/sys_process.py`.
- [ ] T-2.8 `TODO` - `helper/` elevated sidecar.
- [ ] T-2.9 `TODO` - `tools/sys_thermal.py`, `tools/sys_updates.py`.
- [ ] T-2.10 `TODO` - `tools/sys_events.py`.
- [ ] T-2.11 `TODO` - register Phase 2 tools, multi-tool integration test.

## Phase 3 - HUD

- [ ] T-3.1 `TODO` - `ui/server.py`.
- [ ] T-3.2 `TODO` - Tauri app scaffold.
- [ ] T-3.3 `TODO` - Three.js particle orb.
- [ ] T-3.4 `TODO` - HUD panels.
- [ ] T-3.5 `TODO` - wire orb to WebSocket, `--headless` flag.

## Phase 4 - Multipurpose and gated actions

- [ ] T-4.1 `TODO` - `tools/gate.py`. Must ship before any mutating tool.
- [ ] T-4.2 `TODO` - `tools/apps.py`.
- [ ] T-4.3 `TODO` - `tools/media.py`.
- [ ] T-4.4 `TODO` - `tools/files.py`.
- [ ] T-4.5 `TODO` - `tools/vision.py`.
- [ ] T-4.6 `TODO` - `tools/reminders.py`.
- [ ] T-4.7 `TODO` - `tools/websearch.py`.
- [ ] T-4.8 `TODO` - `tools/shell.py`.

## Phase 5 - Hardening

- [ ] T-5.1 `TODO` - latency benchmark suite.
- [ ] T-5.2 `TODO` - crash resilience and supervisor.
- [ ] T-5.3 `TODO` - README.
- [ ] T-5.4 `TODO` - autostart script.

## Manual verification checklist (run on the Windows 11 host)

Filled in as tasks land.
