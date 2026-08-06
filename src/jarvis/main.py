"""Entrypoint.

T-1.9 and T-3.5. Builds the whole assistant, wires the threads, and runs until
interrupted. ``--headless`` skips the HUD so the core runs without the UI.

Thread layout, all fed from one continuous capture (§5):

* PortAudio's callback thread writes microphone audio into the ring buffer.
* The wake listener reads its own cursor and fires on the wake phrase.
* The turn thread endpoints, transcribes, runs the turn, and plays speech.
* A barge-in reader watches for the user talking over playback.
* The UI server broadcasts state on its own asyncio loop.

Nothing downstream can stall capture, because every consumer has an independent
reader and the bus drops rather than blocks.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from typing import Any

from jarvis.config import JarvisConfig, load_config, set_config
from jarvis.state import AssistantState, EventBus, EventType
from jarvis.util.errors import JarvisError
from jarvis.util.logging import setup_logging
from jarvis.util.resilience import ShutdownCoordinator, Supervisor

__all__ = ["Assistant", "main"]

_log = logging.getLogger(__name__)

#: How often the metrics sampler pushes a HUD update.
_METRICS_INTERVAL_S = 1.0


def _register_tools(config: JarvisConfig) -> int:
    """Import every tool module so its decorators register.

    Import order does not matter, but import failure does: a tool whose optional
    dependency is missing should not take the assistant down with it.

    Returns:
        How many tools ended up registered.
    """
    import importlib

    from jarvis.tools.registry import registry

    modules = [
        "jarvis.tools.sys_cpu",
        "jarvis.tools.sys_memory",
        "jarvis.tools.sys_disk",
        "jarvis.tools.sys_gpu",
        "jarvis.tools.sys_network",
        "jarvis.tools.sys_process",
        "jarvis.tools.sys_thermal",
        "jarvis.tools.sys_updates",
        "jarvis.tools.sys_events",
        "jarvis.tools.files",
        "jarvis.tools.reminders",
        "jarvis.tools.apps",
        "jarvis.tools.media",
        "jarvis.tools.vision",
        "jarvis.tools.websearch",
        "jarvis.tools.shell",
    ]
    for name in modules:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001 - one bad tool must not stop the rest
            _log.exception("could not load a tool module", extra={"context": {"module": name}})

    offered = registry.select(config=config)
    _log.info(
        "tools registered",
        extra={"context": {"total": len(registry), "offered": len(offered)}},
    )
    return len(offered)


class Assistant:
    """Owns every subsystem and the turn thread."""

    def __init__(self, config: JarvisConfig, *, headless: bool = False) -> None:
        self.config = config
        self.headless = headless
        self.bus = EventBus()
        self.shutdown = ShutdownCoordinator()
        self.supervisor = Supervisor(self.bus)

        self._capture: Any = None
        self._wake: Any = None
        self._player: Any = None
        self._synth: Any = None
        self._transcriber: Any = None
        self._endpointer: Any = None
        self._barge: Any = None
        self._ui: Any = None
        self._orchestrator: Any = None

        self._wake_event = threading.Event()
        self._wake_detection: Any = None
        self._stop = threading.Event()
        # One barge-in watcher per reply, not one per spoken chunk.
        self._barge_thread: threading.Thread | None = None
        self._barge_lock = threading.Lock()

    # -- construction ------------------------------------------------------

    def build(self) -> None:
        """Construct every subsystem. Nothing starts until :meth:`start`."""
        from jarvis.audio.ring import AudioCapture
        from jarvis.audio.stt import build_transcriber
        from jarvis.audio.tts import build_synthesizer
        from jarvis.audio.vad import BargeInDetector, Endpointer
        from jarvis.audio.wake import WakeListener
        from jarvis.brain.orchestrator import Orchestrator

        _register_tools(self.config)

        self._capture = AudioCapture(self.config, self.bus)
        self.shutdown.register("capture", self._capture.stop)

        self._endpointer = Endpointer(self.config)
        self._barge = BargeInDetector(self.config)
        self._transcriber = build_transcriber(self.config)
        self._synth = build_synthesizer(self.config)

        from jarvis.audio.player import StreamingPlayer

        self._player = StreamingPlayer(self.config)
        self.shutdown.register("player", self._player.close)

        self._orchestrator = Orchestrator(self.config, self.bus)
        self._orchestrator.set_listener(self._listen_for_confirmation)
        self.shutdown.register("orchestrator", self._orchestrator.close)

        if self.config.wake.enabled and not self.config.orchestrator.always_listening:
            self._wake = WakeListener(
                self.config,
                self._capture.reader(),
                self.bus,
                on_detected=self._on_wake,
            )
            self.shutdown.register("wake", self._wake.stop)

        if self.config.ui.enabled and not self.headless:
            from jarvis.ui.server import UiServer

            self._ui = UiServer(self.config, self.bus)
            self.shutdown.register("ui", self._ui.stop)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Start every subsystem."""
        self.config.ensure_directories()

        if self._ui is not None:
            try:
                self._ui.start()
            except Exception:  # noqa: BLE001 - the HUD is optional, the voice is not
                _log.exception("the HUD failed to start, continuing headless")
                self._ui = None

        self._capture.start()
        self._player.start()
        if self._wake is not None:
            self._wake.start()

        self.supervisor.add("turn-loop", self._turn_loop, critical=True)
        self.supervisor.add("metrics", self._metrics_loop)
        self.supervisor.start()
        self.shutdown.register("supervisor", self.supervisor.stop)
        # Registered last so it runs first: shutdown steps run in reverse.
        # The loops must be told to stop before anything they use is closed.
        self.shutdown.register("stop-flag", self._request_stop)

        greeting = self.config.orchestrator.greeting.strip()
        if greeting:
            self._speak(greeting)

        _log.info(
            "jarvis is running",
            extra={
                "context": {
                    "tier": str(self.config.effective_tier()),
                    "model": self.config.llm_model(),
                    "headless": self.headless or self._ui is None,
                    "wake_word": self.config.persona.wake_word,
                }
            },
        )

    def _request_stop(self) -> None:
        """Ask every loop to exit. Safe to call more than once."""
        self._stop.set()
        self._wake_event.set()
        if self._orchestrator is not None:
            self._orchestrator.interrupt()

    def stop(self) -> None:
        """Stop everything, in reverse order of construction."""
        self._request_stop()
        self.bus.emit(EventType.SHUTDOWN)
        self.shutdown.shutdown()
        self.bus.close()

    # -- the loops ---------------------------------------------------------

    def _on_wake(self, detection: Any) -> None:
        """Wake listener callback. Hands the turn thread its pre-roll."""
        self._wake_detection = detection
        self._wake_event.set()

    def _turn_loop(self) -> None:
        """Wait for a wake word, then run turns until the user goes quiet."""
        always = self.config.orchestrator.always_listening
        while not self.supervisor.should_stop("turn-loop") and not self._stop.is_set():
            if not always:
                self._orchestrator.state.set(AssistantState.IDLE)
                if not self._wake_event.wait(0.25):
                    continue
                self._wake_event.clear()

            deadline = time.monotonic() + self.config.orchestrator.idle_timeout_s
            while time.monotonic() < deadline:
                if self._stop.is_set() or self.supervisor.should_stop("turn-loop"):
                    return
                text = self._listen(self.config.vad.max_utterance_s)
                if not text:
                    if always:
                        break
                    continue

                self._orchestrator.run_turn(text, speak=self._speak)
                self._player.wait(timeout=self.config.orchestrator.max_turn_seconds)
                # Each answered question extends the window, so a follow-up
                # does not need the wake word again.
                deadline = time.monotonic() + self.config.orchestrator.idle_timeout_s

    def _listen(self, timeout_s: float) -> str:
        """Endpoint one utterance and transcribe it. Empty string on silence."""
        reader = self._capture.reader()
        self._endpointer.reset()
        frame_samples = self.config.vad.frame_samples
        deadline = time.monotonic() + timeout_s

        self._orchestrator.state.set(AssistantState.LISTENING)
        while time.monotonic() < deadline:
            if self._stop.is_set() or self.supervisor.should_stop("turn-loop"):
                return ""
            frame = reader.read(frame_samples)
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                endpoint = self._endpointer.process(frame)
            except Exception:  # noqa: BLE001 - a bad frame must not end listening
                _log.exception("the endpointer failed on a frame")
                continue
            if endpoint is None:
                continue

            try:
                transcript = self._transcriber.transcribe(
                    endpoint.audio, self.config.audio.sample_rate
                )
            except JarvisError as exc:
                _log.warning("transcription failed: %s", exc.message)
                self.bus.emit(EventType.ERROR, message=exc.message, speakable=exc.speakable)
                return ""
            return str(transcript.text).strip()
        return ""

    def _listen_for_confirmation(self, timeout_s: float) -> str:
        """Collect a spoken yes or no. Silence returns empty, which is a no.

        Waits for the prompt to finish playing first. Opening the microphone
        while the assistant is still speaking risks transcribing its own prompt
        and treating that as the answer.
        """
        deadline = time.monotonic() + max(2.0, self._player.queued_seconds + 1.0)
        while self._player.is_playing and time.monotonic() < deadline:
            if not self._player.wait(timeout=0.25):
                continue
        return self._listen(timeout_s)

    def _speak(self, text: str) -> None:
        """Synthesise and queue one chunk, watching for barge-in.

        The generation is captured before synthesis and handed back to the
        player, so a chunk that was still being synthesised when the user
        interrupted is discarded rather than starting to play a moment after
        the barge-in stopped everything else.
        """
        if not text.strip() or not self.config.tts.enabled:
            return
        if self._stop.is_set() or self._orchestrator.interrupted:
            return

        generation = self._player.generation
        try:
            audio = self._synth.synthesize(text)
        except JarvisError as exc:
            _log.warning("synthesis failed: %s", exc.message)
            return
        if audio.size == 0 or self._orchestrator.interrupted:
            return

        self._orchestrator.state.set(AssistantState.SPEAKING)
        self._player.play(audio, generation=generation)
        if self.config.orchestrator.barge_in:
            self._watch_for_barge_in()

    def _watch_for_barge_in(self) -> None:
        """Start a watcher that cuts playback when the user speaks over it.

        At most one watcher runs at a time. A reply is spoken as several chunks,
        and starting a thread per chunk would put five readers on the ring
        buffer all racing to call interrupt() for the same utterance.
        """
        if not self.config.orchestrator.barge_in:
            return

        with self._barge_lock:
            if self._barge_thread is not None and self._barge_thread.is_alive():
                return

        def watch() -> None:
            reader = self._capture.reader()
            self._barge.reset()
            frame_samples = self.config.vad.frame_samples
            while self._player.is_playing and not self._stop.is_set():
                frame = reader.read(frame_samples)
                if frame is None:
                    time.sleep(0.005)
                    continue
                try:
                    if self._barge.process(frame):
                        self._player.stop()
                        self._orchestrator.interrupt()
                        return
                except Exception:  # noqa: BLE001 - never kill this thread
                    _log.exception("the barge-in detector failed")
                    return

        with self._barge_lock:
            self._barge_thread = threading.Thread(
                target=watch, name="jarvis-bargein", daemon=True
            )
            self._barge_thread.start()

    def _metrics_loop(self) -> None:
        """Sample system metrics for the HUD."""
        from jarvis.tools import sys_cpu, sys_disk, sys_gpu, sys_memory, sys_network

        previous: dict[str, Any] = {}
        while not self.supervisor.should_stop("metrics") and not self._stop.is_set():
            try:
                sample: dict[str, Any] = {}
                sample.update(sys_cpu.snapshot())
                sample.update(sys_memory.snapshot())
                sample.update(sys_disk.snapshot())
                sample.update(sys_gpu.snapshot())
                previous = sys_network.snapshot(previous)
                sample.update(
                    {k: v for k, v in previous.items() if not k.startswith("_")}
                )
                self.bus.emit(EventType.METRICS, **sample)
            except Exception:  # noqa: BLE001 - metrics are cosmetic
                _log.debug("a metrics sample failed", exc_info=True)
            time.sleep(_METRICS_INTERVAL_S)


def build_parser() -> argparse.ArgumentParser:
    """Command line interface."""
    parser = argparse.ArgumentParser(
        prog="jarvis",
        description="A fully local, voice-first assistant that monitors this machine.",
    )
    parser.add_argument(
        "--headless", action="store_true", help="Run without the HUD (T-3.5)."
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml.")
    parser.add_argument("--log-level", default=None, help="Override the configured log level.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what is installed and reachable, then exit without listening.",
    )
    parser.add_argument(
        "--say",
        default=None,
        help="Run a single turn from this text and print the reply. No microphone needed.",
    )
    return parser


def run_check(config: JarvisConfig) -> int:
    """Report readiness without starting the assistant.

    Only the things the voice loop cannot run without count towards readiness.
    A capability the resolved tier deliberately turns off is reported, because
    it is worth knowing, but it is not a failure and no amount of re-running
    the setup scripts would change it.
    """
    from jarvis.brain.llm import OllamaClient
    from jarvis.util.platform import has_cuda, has_module, is_windows

    tools = _register_tools(config)
    ollama_up = False
    try:
        with OllamaClient(config) as client:
            ollama_up = client.is_available()
    except Exception:  # noqa: BLE001
        ollama_up = False

    tier = str(config.effective_tier())
    faster_whisper = has_module("faster_whisper")
    whispercpp = has_module("pywhispercpp")

    if faster_whisper or whispercpp:
        engine = "faster-whisper" if faster_whisper else "whisper.cpp"
        stt = f"yes, {engine}"
    else:
        stt = "no"

    if config.vision_available():
        vision = "yes"
    elif not config.tier_profile().supports_vision:
        # §2: the VLM wants about 6 GB of VRAM. Below that the tool turns
        # itself off by design, so say that rather than implying a fault.
        vision = f"off, the {tier} tier has too little VRAM"
    else:
        vision = "off in the configuration"

    #: (label, value, counts towards readiness)
    rows: list[tuple[str, str, bool]] = [
        ("platform", "Windows" if is_windows() else sys.platform, False),
        ("hardware tier", tier, False),
        ("llm model", config.llm_model(), False),
        ("ollama reachable", "yes" if ollama_up else "no", True),
        ("cuda", "yes" if has_cuda() else "no", False),
        ("sounddevice", "yes" if has_module("sounddevice") else "no", True),
        ("openwakeword", "yes" if has_module("openwakeword") else "no", True),
        ("speech to text", stt, True),
        ("kokoro", "yes" if has_module("kokoro") else "no", True),
        ("tools offered", str(tools), False),
        ("vision", vision, False),
        ("shell tool", "enabled" if config.shell.enabled else "disabled", False),
    ]
    width = max(len(name) for name, _, _ in rows)
    for name, value, _required in rows:
        print(f"{name:<{width}}  {value}")  # noqa: T201

    missing = [name for name, value, required in rows if required and value == "no"]
    if missing:
        print(f"\nNot ready: {', '.join(missing)}")  # noqa: T201
        print("Run scripts\\setup_env.ps1 and scripts\\pull_models.ps1.")  # noqa: T201
        return 1
    print("\nReady.")  # noqa: T201
    return 0


def run_once(config: JarvisConfig, text: str) -> int:
    """Run a single text turn and print the reply.

    The whole conversational core without the microphone, which makes it the
    fastest way to check that tools, memory, and the model are working.
    """
    from jarvis.brain.orchestrator import Orchestrator

    _register_tools(config)
    with Orchestrator(config) as orchestrator:
        def say(chunk: str) -> None:
            print(chunk)  # noqa: T201

        result = orchestrator.run_turn(text, speak=say)
        if result.tool_calls:
            print(f"\n[tools: {', '.join(result.tool_calls)}]")  # noqa: T201
        if result.error:
            print(f"[error: {result.error}]")  # noqa: T201
            return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    """Start the assistant."""
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except JarvisError as exc:
        print(f"jarvis: {exc.message}", file=sys.stderr)  # noqa: T201
        return 2
    set_config(config)

    setup_logging(
        level=args.log_level or config.logging.level,
        log_dir=config.log_dir,
        json_file=config.logging.json_file,
        console=config.logging.console,
        console_level=config.logging.console_level,
        max_bytes=config.logging.max_bytes,
        backup_count=config.logging.backup_count,
        force=True,
    )

    if args.check:
        return run_check(config)
    if args.say:
        return run_once(config, args.say)

    assistant = Assistant(config, headless=args.headless)
    try:
        assistant.build()
        assistant.start()
    except JarvisError as exc:
        _log.error("could not start: %s", exc.message)
        print(f"jarvis: {exc.message}", file=sys.stderr)  # noqa: T201
        assistant.stop()
        return 1

    assistant.shutdown.install_signal_handlers()
    try:
        # Block until a signal arrives. The work happens on the supervised
        # threads, so the main thread only needs to stay alive and responsive
        # to SIGINT.
        while not assistant.shutdown.wait(0.5):
            if not assistant.supervisor.health().healthy:
                _log.error("a critical subsystem failed, shutting down")
                break
    except KeyboardInterrupt:
        pass
    finally:
        assistant.stop()

    _log.info("jarvis stopped")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
