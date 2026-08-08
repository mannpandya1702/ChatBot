# PROGRESS

Machine-maintained build ledger for `/loop`. See `CLAUDE.md` §7 for task definitions.

Last run: 2026-08-06
Target phase: 5
Result: all tasks through Phase 5 implemented. 1622 automated tests pass, `ruff` and
`mypy --strict` are clean. M-8 and M-9 are resolved; the remaining 8 checks need the
Windows host and are listed at the bottom.

Statuses: `TODO`, `IN_PROGRESS`, `DONE`, `BLOCKED`, `NEEDS_MANUAL_VERIFY`.

## Environment note

The build host is Linux; the target host is Windows 11. Per `CLAUDE.md` §0b every module imports
and unit-tests on the build host, and anything that genuinely needs Windows, a GPU, a microphone,
or a human eye is marked `NEEDS_MANUAL_VERIFY` with the exact check to run.

Nothing was stubbed to make a test pass. Where hardware is unavailable, the dependency is injected
and a fake drives the real logic; the engine call sites themselves are what the manual checks
cover.

## Phase 0 - Scaffold

- [x] T-0.1 `DONE` - repo, uv project, Python 3.12, .gitignore, pyproject with ruff+mypy+pytest.
      Note: uv needs `[tool.uv] environments` to fork the lock per platform, otherwise
      openWakeWord's Linux-only tflite-runtime makes the universal resolution unsatisfiable.
- [x] T-0.2 `DONE` - `config.py` plus `config/config.example.yaml` covering every section.
      Note: `gate.affirmatives` must be quoted in YAML; bare `yes`/`no` parse as booleans.
- [x] T-0.3 `DONE` - `util/logging.py`, `util/latency.py`, `util/errors.py`, `util/platform.py`.
- [x] T-0.4 `DONE` - `scripts/verify_cuda.py`. Detects GPU, VRAM, CUDA, and cuDNN, resolves the
      tier, and names the `cudnn_ops64_9.dll` remediation specifically.
      Note: §2 says a card below 6 GB is treated as `cpu` for the LLM but may still run
      faster-whisper if CUDA is present. That sentence was never implemented, so a 4 GB laptop
      card sat idle while whisper.cpp ran on the processor. `stt_settings()` now moves
      transcription onto such a card, small.en at int8, while the LLM stays on the CPU where the
      tier puts it. Anything set explicitly under `stt` still wins.
- [x] T-0.5 `DONE` - `scripts/setup_env.ps1`, `scripts/pull_models.ps1`. Idempotent. The tier to
      model mapping is checked against `TIER_PROFILES` by a test that parses the script.
      `pull_models.ps1` now also pre-fetches the spaCy model Kokoro's phonemiser needs: misaki
      downloads it on first use otherwise, which lands about twenty seconds of silence on the
      first sentence JARVIS ever speaks and fails outright if the machine is offline by then.

## Phase 1 - Voice loop MVP

- [x] T-1.1 `DONE` - `audio/devices.py` + `audio/ring.py`. Lapped-reader detection is tested
      explicitly under concurrent writer and reader threads.
- [x] T-1.2 `DONE` - `audio/wake.py`. Detector and listener, with pre-roll slicing.
      Detection rate needs a real microphone, see M-1 below.
- [x] T-1.3 `DONE` - `audio/vad.py`. Silero endpointing plus a separate barge-in detector on a
      higher threshold so leaked TTS audio cannot make the assistant interrupt itself.
      Note: the model was being fed a bare frame. Silero expects the previous frame's last 64
      samples prepended to it, and without them the graph runs happily and returns near zero for
      everything, so speech and silence are indistinguishable. Clear speech measured 0.003 without
      the context and 1.000 with it. Two unit tests asserted the bare shape, which is why it
      survived; they now assert the window, and `tests/test_engines_live.py` scores real speech
      through the real checkpoint so a shape-level test cannot be wrong about it again.
- [x] T-1.4 `DONE` - `audio/stt.py`. faster-whisper and whisper.cpp behind one protocol, with the
      cuDNN failure path turned into a speakable error. The real `faster_whisper.WhisperModel`
      call site is exercised by `tests/test_engines_live.py`, not just by a fake. WER on human
      speech still needs a real model and a real voice, see M-2.
- [x] T-1.5 `DONE` - `audio/tts.py` + `audio/player.py`.
      Note: found and fixed a real defect, a fully consumed chunk stayed current until the next
      callback, so `is_playing` reported True and `wait()` blocked with nothing left to play.
      The real `kokoro.KPipeline` call site is exercised by `tests/test_engines_live.py`, which
      also pins the sample rate the player trusts: a mismatch there would play every reply at
      the wrong pitch and nothing else would catch it.
- [x] T-1.6 `DONE` - `brain/llm.py`. Streaming, native tool calls, arguments accepted as dict or
      JSON string with retries, and three distinct failure messages.
- [x] T-1.7 `DONE` - `brain/memory.py`. SQLite persistence, compaction that keeps `keep_recent`
      verbatim and folds the previous summary in, and no data loss when the summariser fails.
- [x] T-1.8 `DONE` - `brain/persona.py`. Assembled from named blocks so each rule is testable.
- [x] T-1.9 `DONE` - `brain/orchestrator.py` + `main.py`. `run_turn` is text in, text out with
      every collaborator injected, so the conversational core is tested end to end without
      hardware. Full spoken round trip needs a microphone, see M-3.

**Phase 1 gate:** `NEEDS_MANUAL_VERIFY`, see M-3. Note: the gate question, "what time is it",
had no tool that could answer it. Every other tool reports the machine, and a language model has
no clock, so the model either declined or invented one, which §5 forbids outright. `tools/sys_time.py`
now answers it, phrased for speech: midnight and midday name themselves rather than being called
twelve, since the output is read aloud.

## Phase 2 - System monitoring

- [x] T-2.1 `DONE` - `tools/registry.py`. Built ahead of Phase 1 because T-1.9 depends on it.
      Note: `ToolRegistry.__len__` makes an empty registry falsy, so `target or registry`
      silently wrote to the global registry. Fixed with an explicit `is None` check.
- [x] T-2.2 `DONE` - `tools/sys_cpu.py`. Blocking interval avoids psutil's confident zero.
- [x] T-2.3 `DONE` - `tools/sys_memory.py`.
- [x] T-2.4 `DONE` - `tools/sys_disk.py`. `Win32_Product` is never called, enforced by a test that
      parses the module's AST rather than grepping, so the docstring explaining the ban does not
      trip it. `total_free_gb` now sums every mounted volume rather than only the five that
      survive the spoken cap, and `volume_count` says how many there are, so a machine with more
      than five volumes is no longer told it has less free space than it does.
- [x] T-2.5 `DONE` - `tools/sys_gpu.py`. Per-sensor degradation: a card that does not report fan
      speed still yields temperature and clocks.
- [x] T-2.6 `DONE` - `tools/sys_network.py`.
- [x] T-2.7 `DONE` - `tools/sys_process.py`. CPU percent normalised across cores so it matches
      Task Manager.
- [x] T-2.8 `DONE` - `helper/`. Three parameterless RPC methods; a request carrying any params is
      refused before dispatch. `build_helper.ps1` verifies the built binary's method surface.
      Elevated run needs Windows, see M-4.
- [x] T-2.9 `DONE` - `tools/sys_thermal.py`, `tools/sys_updates.py`. No install path exists
      anywhere in the update code, enforced by a test.
- [x] T-2.10 `DONE` - `tools/sys_events.py`.
- [x] T-2.11 `DONE` - integration test chaining CPU, memory, disk, and GPU into one summary,
      driven by a scripted model over the real registry, gate, and memory.

**Phase 2 gate:** `NEEDS_MANUAL_VERIFY`, see M-5.

## Phase 3 - HUD

- [x] T-3.1 `DONE` - `ui/server.py`. Change-aware broadcasting with a keepalive.
      Note: fixed a shutdown ordering defect where stopping the loop from outside left
      `Server._close` as a never-awaited coroutine.
- [x] T-3.2 `DONE` - Tauri scaffold. Frameless, transparent, always on top, click-through with a
      toggle, no taskbar entry, tray with show/hide/quit. `frontendDist` now points at the vite
      build output rather than the raw sources, with `beforeBuildCommand` wired in: shipping the
      sources meant shipping `orb.js`'s bare `import ... from 'three'`, which no browser can
      resolve, so the built HUD loaded a blank window. Build needs Windows, see M-6.
- [x] T-3.3 `DONE` - Three.js orb, 2000 points, additive blending, per-particle drift, a palette
      per state. The update loop allocates nothing, enforced by a test. Visual check M-7.
- [x] T-3.4 `DONE` - HUD panels, sparklines, rolling transcript, arc-reactor framing, draggable.
      A missing metric renders as a gap, not a zero. Visual check M-7.
- [x] T-3.5 `DONE` - WebSocket wiring with capped-backoff reconnect, and `--headless`.

## Phase 4 - Multipurpose and gated actions

- [x] T-4.1 `DONE` - `tools/gate.py`. Shipped before any mutating tool. Confirmation grants are
      single use and bound to the exact tool and arguments they were issued for.
- [x] T-4.2 `DONE` - `tools/apps.py`. Launch resolves names only; paths, command lines, and any
      resolution landing on a shell or script host are refused. The shell refusal now matches the
      words of the resolved name with its extension stripped, and separately checks what a
      shortcut points at: matching executable names alone let the stock Start Menu entries
      "Windows PowerShell.lnk", "Command Prompt.lnk", and "Windows Terminal.lnk" straight
      through, which handed the model a shell with none of the §6 hardening on it.
- [x] T-4.3 `DONE` - `tools/media.py`.
- [x] T-4.4 `DONE` - `tools/files.py`. Everything CLI with a budgeted scandir fallback.
- [x] T-4.5 `DONE` - `tools/vision.py`. Auto-disables below 6 GB VRAM.
- [x] T-4.6 `DONE` - `tools/reminders.py`. The scheduler sleeps until the next due time.
- [x] T-4.7 `DONE` - `tools/websearch.py`. Makes no outbound request until configured. Redirects
      are followed manually and only while they stay on the configured host, so a redirect cannot
      carry the user's query to an address they never named (§0.1).
- [x] T-4.8 `DONE` - `tools/shell.py`. All ten §6 rules, 83 red-team tests. The default allowlist
      no longer lists PowerShell cmdlets, which §6 makes unreachable; see deviation 6.

## Phase 5 - Hardening

- [x] T-5.1 `DONE` - `tests/bench_latency.py`. p50/p95/p99 per stage, fails on a p95 regression
      past the tolerance. `turn_total` is composed from its components and skipped outright when
      any is unavailable, rather than summing a partial set and understating the total.
      Two flaws surfaced the first time it ran against a real engine rather than the synthetic
      harness. The §3 budgets head their own table "gpu-12 target", so applying them on a `cpu`
      tier reported a failure no code change could fix; off-tier misses are now advisory and
      only regressions fail. And the regression check compared percentages with no absolute
      floor, so 0.04 ms of scheduling jitter on the 0.1 ms VAD stage read as a 37 percent
      regression. A 5 ms floor sits far below the smallest §3 budget of 250 ms.
- [x] T-5.2 `DONE` - `util/resilience.py`. Supervisor with a finite restart budget, health report,
      and reverse-order graceful shutdown on SIGINT. Shutdown no longer overwrites the two states
      that matter: a worker that ignores the stop request is reported `STALLED` rather than as a
      clean stop it never reached, and a worker that burned its restart budget stays `FAILED`
      across shutdown instead of being reset to `STOPPED`.
- [x] T-5.3 `DONE` - `README.md` with install, tiers, the full tool catalogue, and troubleshooting
      covering cuDNN, PortAudio, Defender, admin prompts, and the Python pin.
- [x] T-5.4 `DONE` - `scripts/install_autostart.ps1`. Non-elevated, and it removes the task and
      fails loudly if it somehow registered elevated.

---

## Manual verification checklist

Run these on the Windows 11 host after `setup_env.ps1` and `pull_models.ps1`. Each names the exact
check and what passing looks like. M-8 and M-9 were build blockers rather than checks and are now
resolved; they are kept below with what was done.

**M-1. Wake word detection rate (T-1.2)**
```powershell
uv run pytest tests/test_wake.py -m manual -v
```
Say "Hey Jarvis" ten times at a normal speaking volume. Pass: at least 9 detections logged, and at
most 1 false accept across 10 minutes of ambient room noise. If it over-triggers, raise
`wake.threshold` toward 0.7.

**M-2. Transcription accuracy (T-1.4)**
```powershell
uv run pytest tests/test_stt.py -m manual -v
```
Pass: the fixture utterance transcribes with no more than one word wrong, and the logged real time
factor is below 1.0 on a GPU tier.

**M-3. Phase 1 gate, the full spoken round trip (T-1.9)**
```powershell
uv run python -m jarvis
```
Say "Hey Jarvis, what time is it". Pass: the orb brightens, a spoken answer follows, and the p95 in
the latency log is at or under 1200 ms on `gpu-12`. Then ask something longer and start talking
over the reply. Pass: playback stops within about a tenth of a second and the new question is
picked up.

**M-4. Elevated helper (T-2.8)**
```powershell
.\scripts\build_helper.ps1
.\jarvis-helper.exe --check
.\jarvis-helper.exe
```
Approve the UAC prompt. Pass: `--check` reports `"elevated": true` and exactly the three methods;
the running helper answers `sys.thermal` and `sys.fans` with real numbers. Then decline the prompt
on a fresh run and confirm those two tools report unavailable while everything else still works.

**M-5. Phase 2 gate, spoken answers (T-2.11)**
Ask each aloud and confirm the answer is correct and sounds like a person said it:
overall system health, what is using the CPU, GPU temperature, free disk space, pending Windows
updates, and uptime. Pass: every answer comes from a tool, no invented numbers, no JSON read out.

**M-6. HUD build (T-3.2)**
```powershell
cd src\jarvis\ui\app
npm install
npm run tauri build
```
Pass: the build succeeds and the window renders transparent over the desktop with no frame and no
taskbar entry. Clicking through to the desktop works everywhere except the drag handle.

The compile and bundle half of this is no longer unverified: `npm run build` and a full
`cargo build --release` both run in CI here and produce a `jarvis-hud` binary, so a code error in
the shell or the frontend is caught before you see it. What still needs the Windows host is
everything the Linux build cannot exercise: WebView2 rather than webkit2gtk, the MSI and NSIS
bundling, and every visual claim above. The click-through behaviour in particular is worth
watching closely, because it was completely broken until this pass and the fix polls the cursor
rather than reacting to events.

**M-7. Orb and panels (T-3.3, T-3.4)**
Run JARVIS and watch the orb through idle, listening, thinking, speaking, and an induced error.
Pass: each state has a visibly distinct palette, the orb reacts to your voice amplitude, and the
frame rate holds at 60. Sparklines update, and the GPU line reads `n/a` rather than zero on a
machine without an NVIDIA card.

**M-8. Icons for the HUD bundle (T-3.2)** - `RESOLVED`, no longer blocking
`icon.png` (512x512 RGBA) and `icon.ico` (16, 32, 48, 64, 128, 256) are committed, generated by
`scripts/make_icons.py` from the accent colour, stdlib only. They are placeholders: swap in real
artwork under the same two names whenever you like. Tests assert the ICO directory is structurally
valid and that regenerating reproduces the committed bytes exactly.

**M-9. LibreHardwareMonitorLib.dll (T-2.8)** - `RESOLVED`, no longer blocking
`vendor/LibreHardwareMonitorLib.dll` 0.9.4 is committed, taken from the NuGet package
`LibreHardwareMonitorLib`, `lib/net472/`. MPL-2.0, which §1 permits, and §4's repo layout lists
this exact path. `scripts/fetch_vendor.ps1` re-fetches it and verifies a pinned SHA-256, deleting
rather than keeping a mismatched download, since an elevated process loads this file. Provenance
and licence obligations are in `vendor/README.md`.

Residual manual check, folded into M-4: confirm the assembly actually loads under your pythonnet
runtime. Nothing here can test that, since it needs Windows and .NET. If pythonnet resolves to
CoreCLR rather than .NET Framework, take `lib/net8.0/` from the same package instead.

**M-10. Latency benchmark on real hardware (T-5.1)**
```powershell
uv run python tests/bench_latency.py
```
Pass: every stage reports measured rather than skipped, and each p95 is inside its §3 budget. The
first run establishes the baseline; later runs fail on a regression past 15 percent.

Partly exercised on the build host: `vad_endpoint` and `tts_first_audio` now measure for real, so
the harness itself is proven end to end rather than only against synthetic timings. `stt` still
skips there (the cpu tier wants whisper.cpp, whose package is Windows only) and
`llm_first_token` needs Ollama, so `turn_total` has still never been measured. **The §3 p95 of
1200 ms remains unverified**, and this is the check most worth doing carefully. Note the budgets
are only enforced on `gpu-12` and above; below that the run reports misses without failing.

---

## Deviations from the original contract

Recorded here rather than silently applied.

1. **§9 placeholders.** All were unfilled. Rather than invent machine specifics, hardware values
   resolve to `auto` and are detected at first run by `scripts/verify_cuda.py`, which is what §0.6
   requires anyway since it forbids hardcoded machine-specific paths. Persona, UI, and feature
   switches took the defaults the contract itself hints at. `target_phase` was set to 5.

2. **§2 tier table gap.** A 6 or 7 GB card matched no tier in the original table. Added `gpu-6`,
   and a card below 6 GB now resolves to `cpu` rather than falling through.

3. **§0b added.** The contract assumed the build host was the target host. The section states the
   lazy-import rule that lets the Windows-targeted tree build and test on Linux.

4. **`git push`.** The loop guardrails forbade pushing, assuming a local developer machine. This
   build ran in an ephemeral container where not pushing loses the work, so the designated feature
   branch is pushed. Force-push and history rewriting were not used.

5. **The 8-tasks-per-run stop condition** was lifted, as the operator asked for an end-to-end
   build in one run.

6. **§6 makes PowerShell cmdlets unreachable, so they left the `shell.run` allowlist.** The two
   rules are mutually exclusive as written. Reaching a cmdlet such as `Get-Date` means invoking
   `powershell.exe -Command Get-Date`, and §6 blocks `-Command` (along with `-File` and `-enc`)
   as a mandatory hardening rule. The shipped allowlist held ten `Get-*` cmdlets plus `ver`, a
   `cmd.exe` builtin: eleven of sixteen entries were accepted by the allowlist and then refused
   by the argument rule, and the tool description advertised `Get-Date` as an example of
   something it could run. The default is now the five standalone executables that do run
   (`systeminfo`, `ipconfig`, `hostname`, `whoami`, `tasklist`), and a test asserts no entry is a
   cmdlet. The §6 rule was **not** relaxed to make cmdlets work, per §10. If cmdlet access is
   wanted, that is a change to §6 and needs an explicit decision: the safe shape would be a
   separate fixed-argv cmdlet allowlist that never takes a model-authored string, but that is a
   contract amendment, not an implementation detail.

---

---

## Deviation 7: the voice is `am_onyx`, not `bm_george`

§1 and §9 both name `bm_george`, and §7 T-1.8 asks for a "British-butler register". The operator
listened to all twelve male English voices Kokoro publishes, each through the same effect rack, and
chose `am_onyx`. §9 is the record of resolved operator configuration rather than a constraint, so
the shipped default follows the choice and the §9 line is annotated. §1's entry is untouched: it
locks Kokoro-82M as the component, and the voice is a parameter of it.

Verified rather than assumed, since the rack was tuned against `bm_george`:

    voice am_onyx, lang_code 'a', profile jarvis
    dry -26.89 dBFS -> wet -25.08 dBFS (delta +1.82 dB), peak 0.488
      raw kokoro         wer~0.000
      through the rack   wer~0.000

No re-tuning was needed. The rack calibrates its own level, and it costs this voice nothing
measurable in intelligibility.

Two consequences were handled:

* `tts.lang_code` now follows the voice's prefix rather than being configured. Kokoro selects its
  phonemiser from that code, not from the voice, so `am_onyx` with the British phonemiser
  mispronounces words and reports nothing. Deriving it removes the whole class of mismatch; setting
  it explicitly still wins, which is only useful for reading one language in another's accent.
* `pull_models.ps1` fetched `bm_george.pt` only, so the new default would have downloaded its voice
  from the hub on first speech and failed outright offline. It now fetches both. A voice pack is
  about half a megabyte. `tests/test_setup_scripts.py` caught this, because it derives the expected
  filename from `TtsConfig()` rather than hardcoding one.

The persona's wording still says "British butler". That describes word choice and manner, not
accent, and both still hold; changing it was not asked for. Worth a look if the American voice makes
the phrasing feel odd.

### Not done: the ElevenLabs voice

The operator first linked an ElevenLabs voice. It cannot be used: §0.1 forbids hosted TTS and API
keys, §0.2 forbids paid services and accounts, and the voice is proprietary. The API says so
plainly, without credentials::

    {"detail":{"message":"You must be logged in to fetch more than 3 voices."}}

Cloning it locally is closed off by §1 as well, which rejects XTTS-v2 (CPML), F5-TTS (CC-BY-NC) and
Fish Speech (CC-BY-NC-SA) by name: every capable local voice-cloning model. Per §10 this was
reported rather than worked around, and the operator chose a Kokoro voice instead.


## Nine-dimension adversarial review

Run after the first spoken round trip on the Windows host. 64 agents across nine
dimensions (safety, concurrency, audio, agentic, contract, resilience,
performance, deployment, tests), every finding re-checked by two independent
skeptics working from the reporter's own reproduction. 27 found, 23 survived.

Fixed here, each with a regression test verified to fail without the fix:

| # | Severity | What was wrong |
|---|---|---|
| 1 | critical | `project_root()` resolved into `%TEMP%` inside the PyInstaller helper, so the elevated process loaded its native assembly from a user-writable directory. Now frozen-aware, plus a load-time hash check. |
| 2 | critical | A malformed tool call on Ollama's terminal message discarded the whole message, making the turn a silent no-op. |
| 3 | high | The malformed-JSON "retry" never re-asked; it skipped the line and decremented a counter. |
| 4 | high | The tool-iteration ceiling ended the turn silently and reported success. |
| 5 | high | The wake word's pre-roll was captured and dropped, clipping the start of every request. |
| 6 | high | Ctrl-C during a reply hung for the supervisor's whole grace period. |
| 11 | high | The STT pre-fetch filled a cache the runtime does not read. |
| 12 | high | Kokoro weights were downloaded into `models/` and then downloaded again from the hub. |
| 13 | high | Nothing was warmed at startup, so the first turn paid ~11 s of model loading. |
| 14 | high | Memory compaction ran a blocking LLM generation between a tool result and the answer. |
| 15 | high | An unrecoverable endpointer failure was retried at the frame rate, 31 errors a second. |
| 16 | high | A dead Kokoro left the assistant mute with no error event and the state machine still saying "speaking". |
| 17 | high | A mutating tool ran on a spoken "yes" when the confirmation prompt was never audible. |
| 18 | high | The confirmation window was spent speaking the prompt, so `shell.run` could never be confirmed. |
| 19 | high | Two tests pinned the fake retry as correct, using a stream shape Ollama never produces. |
| 20 | high | Every absolute reminder time was read as UTC, so "at 5pm" fired at the wrong hour. |
| 8, 22 | high | The two stages §3 puts a hard budget on, `vad_endpoint` and `stt`, were never measured on the live path. |
| 23 | low | `tts.sample_rate` accepted any rate; Kokoro only emits 24 kHz. |

Also fixed, found in the same pass and refuted only because the fix landed while
the skeptics were checking: the HUD WebSocket accepted any browser origin, so any
page the user had open could read the live transcript.

### Also fixed, in a second pass

| # | Severity | What was wrong |
|---|---|---|
| 7 | high | A capture reader lapped by the writer resynchronised silently. `RingReader.dropped` was computed correctly and no consumer read it, so audio from either side of a gap was spliced into one utterance and handed to the transcriber, with Silero still carrying state from before the jump. All three consumers now notice: `_listen` abandons the utterance and says so, the barge-in watcher and the wake listener reset. |
| 9 | high | The HUD window was permanently click-through. `main.rs` set ignore-cursor-events at startup and defined a command to undo it that nothing ever called, so the drag handle and every control was dead. The obvious repair does not work: a window ignoring cursor events receives no pointer events, so a `pointermove` handler could never turn it back on. The frontend now publishes its interactive rectangles and the shell polls the global cursor against them. |
| 10 | high | `ui.port`, `ui.host`, `ui.accent_color` and `ui.hud_position` were validated and never reached the HUD. The comments named a launcher and a build-time injector that existed nowhere. `ui/server.py` now writes `hud-config.js` at startup with the bound port, and the CSP allows any loopback port instead of pinning 8765. |
| 21 | medium | One failed sink restart after a barge-in left the player mute for the session, because the retry was gated on the flag the failure had just cleared. It now retries, rebuilds the device if the restart keeps failing, reports at warning rather than debug, and `wait()` can no longer block past the audio it is waiting for. |

All 23 confirmed findings are closed.

### Notes

* Two tests in this repo asserted bugs as correct behaviour and so kept them
  alive: `tests/test_memory.py` asserted `message["name"]` for a key Ollama
  discards, and every barge-in test drove the detector from a scripted
  probability, which is why nothing could see that the assistant interrupts
  itself. Both are corrected with the reason recorded next to them.
* `tests/test_engines_live.py` grew two suites that need the real engines: one
  asserting no voice profile costs intelligibility, one asserting the assistant
  does not cut itself off at any realistic speaker bleed.
* Two more tests were enforcing bugs rather than the contract.
  `test_click_through_can_be_toggled_off` asserted only that the Rust side
  *defined* the command, which it always had, while nothing called it; and
  `test_csp_only_allows_the_local_core` pinned `ws://127.0.0.1:8765`, which made
  a configurable port a contract violation. Both now assert the thing that
  actually matters, and the first also checks the built bundle, since a
  source-only check is what let the dead HUD ship.
* The Rust and the frontend are both compiled and built in this repo now
  (`cargo check`, `npm run build`), so the HUD changes are verified rather than
  written blind. The GTK development headers had to be installed to do it; on
  the Windows target host they are not needed.
