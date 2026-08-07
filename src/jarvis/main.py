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

import numpy as np

from jarvis.config import JarvisConfig, load_config, set_config
from jarvis.state import AssistantState, EventBus, EventType
from jarvis.util.errors import AudioError, DependencyMissingError, JarvisError
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
        "jarvis.tools.sys_time",
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
        self._endpointer_failed: JarvisError | None = None
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

        self._warm_engines()

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

    def _warm_engines(self) -> None:
        """Load every model now rather than during the user's first sentence.

        All three engines load lazily, and measured on a machine with the
        weights already on disk the first call to each costs:

            Silero VAD        86 ms   (steady 0.2 ms, §3 budget 250 ms)
            faster-whisper  3,722 ms  (steady 306 ms, §3 budget 200 ms)
            Kokoro          7,706 ms  (steady 568 ms, §3 budget 300 ms)

        So the first question of every session took about eleven seconds longer
        than every question after it, which reads as the assistant being broken
        rather than cold. ``Transcriber.warmup`` was written for exactly this and
        was never called from anywhere but its own test.

        Runs on a daemon thread so startup is not blocked: the wake word still
        has to fire before any of it is needed, and a user who speaks
        immediately simply waits where they would have waited anyway.
        """

        def warm() -> None:
            started = time.perf_counter()
            try:
                frame = np.zeros(self.config.vad.frame_samples, dtype=np.float32)
                for _ in range(3):
                    self._endpointer.process(frame)
                self._endpointer.reset()
            except Exception:  # noqa: BLE001 - warmup must never take the loop down
                _log.debug("could not warm the endpointer", exc_info=True)

            try:
                self._transcriber.warmup()
            except Exception:  # noqa: BLE001
                _log.debug("could not warm the transcriber", exc_info=True)

            try:
                # Real text, because Kokoro returns immediately for whitespace
                # and would never load. The audio is discarded, and the rack is
                # reset afterwards so this leaves no reverb tail in the greeting.
                self._synth.synthesize("Ready.")
                self._synth.begin_utterance()
            except Exception:  # noqa: BLE001
                _log.debug("could not warm the synthesiser", exc_info=True)

            _log.info(
                "engines warm",
                extra={"context": {"seconds": round(time.perf_counter() - started, 1)}},
            )

        threading.Thread(target=warm, name="jarvis-warmup", daemon=True).start()

    def _request_stop(self) -> None:
        """Ask every loop to exit. Safe to call more than once.

        The player has to be told too. During a reply the turn loop is parked in
        ``player.wait(max_turn_seconds)``, on an event only the audio callback or
        ``player.stop()`` ever sets, so a Ctrl-C mid-sentence left it there:
        shutdown blocked for its full five second grace period and then reported
        the turn loop as stalled. Barge-in already treats "cut playback" and
        "abandon the turn" as one operation. Shutdown is the same operation.
        """
        self._stop.set()
        self._wake_event.set()
        if self._orchestrator is not None:
            self._orchestrator.interrupt()
        if self._player is not None:
            self._player.stop()

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
            if self._endpointer_failed is not None:
                # Nothing here can recover it, and calling back in would only
                # restart the flood. WakeListener handles a fatal load the same
                # way: report it once and stop, rather than fail at frame rate.
                _log.error(
                    "the turn loop is stopping: %s", self._endpointer_failed.message
                )
                return
            if not always:
                self._orchestrator.state.set(AssistantState.IDLE)
                if not self._wake_event.wait(0.25):
                    continue
                self._wake_event.clear()

            deadline = time.monotonic() + self.config.orchestrator.idle_timeout_s
            while time.monotonic() < deadline:
                if self._stop.is_set() or self.supervisor.should_stop("turn-loop"):
                    return
                preroll, self._wake_detection = self._wake_detection, None
                text = self._listen(
                    self.config.vad.max_utterance_s,
                    getattr(preroll, "preroll", None),
                )
                if not text:
                    if always:
                        break
                    continue

                # One reply is one utterance for the voice rack: its delay lines
                # carry across the sentences within a reply, but the previous
                # reply's tail must not lead into this one.
                self._synth.begin_utterance()
                self._orchestrator.run_turn(text, speak=self._speak)
                self._player.wait(timeout=self.config.orchestrator.max_turn_seconds)
                # Each answered question extends the window, so a follow-up
                # does not need the wake word again.
                deadline = time.monotonic() + self.config.orchestrator.idle_timeout_s

    def _listen(self, timeout_s: float, preroll: Any = None) -> str:
        """Endpoint one utterance and transcribe it. Empty string on silence.

        Args:
            timeout_s: How long to wait for the utterance to close.
            preroll: Audio captured before the wake word fired, from
                :class:`~jarvis.audio.wake.WakeDetection`. Fed through the
                endpointer before the live stream so the utterance starts
                unclipped.

        The pre-roll matters more than it looks. The ring buffer keeps a second
        of it precisely so the wake word's trailing audio is not lost, and
        WakeDetection's own docstring says to prepend it, but nothing ever did:
        the turn loop attached a fresh cursor at the write head, so the
        endpointer only saw audio captured after the trigger. Anyone who says
        "hey jarvis what time is it" as one phrase lost the front of the
        question, and the transcriber was handed a sentence starting mid-word.
        """
        reader = self._capture.reader()
        self._endpointer.reset()
        frame_samples = self.config.vad.frame_samples
        deadline = time.monotonic() + timeout_s
        errors = 0

        self._orchestrator.state.set(AssistantState.LISTENING)

        endpoint = self._feed_preroll(preroll, frame_samples)
        if endpoint is not None:
            return self._transcribe(endpoint)
        while time.monotonic() < deadline:
            if self._stop.is_set() or self.supervisor.should_stop("turn-loop"):
                return ""
            frame = reader.read(frame_samples)
            if frame is None:
                time.sleep(0.005)
                continue
            try:
                endpoint = self._endpointer.process(frame)
            except (DependencyMissingError, AudioError) as exc:
                # Neither recovers between frames: the package will not install
                # itself and the checkpoint will not appear. Logging once per
                # frame buried the one line worth reading, and returning "" only
                # moved the spin one frame up: the turn loop reads that as
                # ordinary silence and calls straight back in, so the rate
                # stayed at the frame rate, 31 errors and 11 KiB of log a
                # second. Latching it is what actually stops it.
                self._endpointer_failed = exc
                _log.error(
                    "cannot endpoint speech, listening is disabled: %s",
                    exc.message,
                    extra={"context": {"error": type(exc).__name__, **exc.context}},
                )
                self.bus.emit(
                    EventType.ERROR, message=exc.message, speakable=exc.speakable
                )
                return ""
            except Exception:  # noqa: BLE001 - a bad frame must not end listening
                errors += 1
                if errors <= _MAX_FRAME_ERRORS:
                    _log.exception("the endpointer failed on a frame")
                elif errors == _MAX_FRAME_ERRORS + 1:
                    _log.error("further endpointer errors this turn will be logged at debug")
                else:
                    _log.debug("the endpointer failed on a frame (%d this turn)", errors)
                continue
            if endpoint is None:
                continue

            return self._transcribe(endpoint)

        # Timed out. Silence here is the worst outcome to debug, because the
        # user spoke, nothing happened, and nothing said why. Report what the
        # endpointer actually saw so the next step is obvious: no speech at all
        # points at the microphone or the VAD threshold, speech that never
        # ended points at trailing_silence_ms.
        _log.info(
            "heard no complete utterance before the listen window closed",
            extra={
                "context": {
                    "timeout_s": timeout_s,
                    "vad_threshold": self.config.vad.threshold,
                    **self._endpointer_state(),
                }
            },
        )
        return ""

    def _endpointer_state(self) -> dict[str, Any]:
        """What the endpointer saw, for a log line. Never raises.

        Read defensively on purpose. This exists only to explain a turn, and a
        diagnostic that can throw would lose the very utterance it is there to
        describe. Anything the endpointer does not expose is simply omitted.
        """
        fields: dict[str, Any] = {}
        for name in ("state", "speech_ms", "silence_ms", "frames_processed"):
            try:
                value = getattr(self._endpointer, name, None)
            except Exception:  # noqa: BLE001, S112 - see the docstring
                # Deliberately not logged. This runs while assembling a log
                # line, so reporting the failure here risks recursing into the
                # same problem for a field that is only ever diagnostic.
                continue
            if value is None:
                continue
            fields[name] = round(value) if isinstance(value, float) else str(value)
        return fields

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

    def _feed_preroll(self, preroll: Any, frame_samples: int) -> Any:
        """Run the wake word's pre-roll through the endpointer before live audio.

        Returns:
            An endpoint if the utterance somehow completed inside the pre-roll,
            which only happens when the user said nothing after the wake word,
            otherwise None.

        Never raises: a pre-roll that cannot be scored is not worth failing a
        turn over, and the live stream immediately after it will be scored by
        the same endpointer anyway.
        """
        if preroll is None:
            return None
        try:
            samples = np.asarray(preroll, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            _log.debug("the wake detection carried a pre-roll that was not audio")
            return None
        if samples.size < frame_samples:
            return None

        whole = (samples.size // frame_samples) * frame_samples
        for start in range(0, whole, frame_samples):
            try:
                endpoint = self._endpointer.process(samples[start : start + frame_samples])
            except JarvisError as exc:
                _log.debug("could not score the pre-roll: %s", exc.message)
                return None
            except Exception:  # noqa: BLE001 - the live stream is what matters
                _log.debug("the endpointer failed on a pre-roll frame", exc_info=True)
                return None
            if endpoint is not None:
                return endpoint
        _log.debug(
            "seeded the endpointer with the wake word's pre-roll",
            extra={"context": {"samples": int(whole)}},
        )
        return None

    def _transcribe(self, endpoint: Any) -> str:
        """Turn a closed utterance into text. Empty string on any failure."""
        # Only the length. The endpointer zeroes its counters as it emits, so
        # speech_ms and silence_ms read 0 here and would be worse than saying
        # nothing. audio.vad logs the reason and the decision time.
        _log.info(
            "utterance endpointed",
            extra={
                "context": {
                    "seconds": round(endpoint.audio.size / self.config.audio.sample_rate, 2),
                }
            },
        )
        try:
            transcript = self._transcriber.transcribe(
                endpoint.audio, self.config.audio.sample_rate
            )
        except JarvisError as exc:
            _log.warning("transcription failed: %s", exc.message)
            self.bus.emit(EventType.ERROR, message=exc.message, speakable=exc.speakable)
            return ""
        text = str(transcript.text).strip()
        if not text:
            # Endpointed but nothing came back. Worth a line: from outside it is
            # indistinguishable from never having heard anything.
            _log.info("the transcriber returned nothing for that utterance")
        else:
            _log.info("heard", extra={"context": {"text": text}})
        return text

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
                    # The reference is what makes this the *user* talking over
                    # the assistant rather than the assistant talking over
                    # itself. Without it the microphone hears the speakers, the
                    # model says "that is speech", and the reply is cut off half
                    # a second in, every turn.
                    if self._barge.process(frame, self._player.recent_output_level()):
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
    parser.add_argument(
        "--mic-test",
        nargs="?",
        type=float,
        const=6.0,
        default=None,
        metavar="SECONDS",
        help=(
            "Record from the configured microphone, report the level, and score the wake "
            "word against it. Use this when the assistant hears nothing."
        ),
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


#: Frame errors logged in full per turn before dropping to debug. A wedged
#: endpointer produces one per frame, about thirty a second.
_MAX_FRAME_ERRORS = 5

#: Peak below this over a whole recording means nothing reached the process.
_SILENCE_PEAK = 0.002

#: Peak below this is signal, but too quiet for the wake model to work with.
_QUIET_PEAK = 0.02

#: A wake score above this means the model recognised something, even if it did
#: not commit. The noise floor sits near zero, so anything here is a near miss
#: worth lowering the threshold for, not a failure to hear.
_MODEL_RESPONDED = 0.08


def _level_bar(peak: float, width: int = 30) -> str:
    """A crude meter, so a glance says more than a float does."""
    filled = min(width, round(peak * width * 2))
    return "#" * filled + "." * (width - filled)


def run_mic_test(config: JarvisConfig, seconds: float) -> int:
    """Record from the configured microphone and report what actually arrived.

    The wake word failing is ambiguous from the outside: a muted microphone, a
    device that captures silence, and speech the model simply scores low all
    look identical in the log as ``detections=0 errors=0``. This separates
    them, by measuring the signal and then scoring the same audio the listener
    would have seen.
    """
    import numpy as np

    from jarvis.audio.devices import describe_devices
    from jarvis.audio.ring import AudioCapture
    from jarvis.audio.vad import SileroVad
    from jarvis.audio.wake import WakeWordDetector

    print(describe_devices())  # noqa: T201
    print()  # noqa: T201

    capture = AudioCapture(config)
    frame_samples = max(1, int(config.wake.frame_samples))
    rate = config.audio.sample_rate
    detector = WakeWordDetector(config)
    # The endpointer is the stage after the wake word, and it fails the same
    # silent way: it either never confirms speech or never sees it end.
    vad = SileroVad(config)
    vad_frame = max(1, int(config.vad.frame_samples))

    collected: list[Any] = []
    best_score = 0.0
    best_speech = 0.0
    peak = 0.0

    try:
        capture.start()
    except JarvisError as exc:
        print(f"could not open the microphone: {exc.message}", file=sys.stderr)  # noqa: T201
        return 1

    try:
        reader = capture.reader()
        print(f"Speak now. Say your wake word a few times. Listening for {seconds:.0f}s.\n")  # noqa: T201
        deadline = time.monotonic() + seconds
        next_report = time.monotonic() + 0.5
        window_peak = 0.0

        while time.monotonic() < deadline:
            frame = reader.read(frame_samples)
            if frame is None:
                time.sleep(0.005)
                continue
            collected.append(frame)
            window_peak = max(window_peak, float(np.abs(frame).max()))
            peak = max(peak, window_peak)
            try:
                detector.process(frame)
                best_score = max(best_score, max(detector.scores.values(), default=0.0))
            except JarvisError as exc:
                print(f"\nwake model unavailable: {exc.message}")  # noqa: T201
                break

            now = time.monotonic()
            if now >= next_report:
                print(f"  {_level_bar(window_peak)}  peak {window_peak:.3f}")  # noqa: T201
                window_peak = 0.0
                next_report = now + 0.5
    finally:
        capture.stop()

    if not collected:
        print("\nNo audio blocks arrived at all. The stream opened but delivered nothing.")  # noqa: T201
        return 1

    audio = np.concatenate(collected)
    rms = float(np.sqrt(np.mean(np.square(audio))))

    # Score the same audio through the endpointer's own model, in its frame
    # size rather than the wake word's.
    for start in range(0, audio.size - vad_frame + 1, vad_frame):
        try:
            best_speech = max(best_speech, float(vad.probability(audio[start : start + vad_frame])))
        except JarvisError:
            best_speech = -1.0
            break

    print(f"\ncaptured   {audio.size / rate:.1f}s at {rate} Hz")  # noqa: T201
    print(f"peak       {peak:.4f}")  # noqa: T201
    print(f"rms        {rms:.4f}")  # noqa: T201
    print(f"wake score {best_score:.3f}  (threshold {config.wake.threshold:.2f})")  # noqa: T201
    if best_speech >= 0.0:
        print(  # noqa: T201
            f"speech     {best_speech:.3f}  (vad threshold {config.vad.threshold:.2f})"
        )
    else:
        print("speech     unavailable, the VAD model would not load")  # noqa: T201
    print()  # noqa: T201

    if 0.0 <= best_speech < config.vad.threshold:
        # The wake word can fire and the turn still go nowhere, silently,
        # because the endpointer never confirms speech.
        print("Note: the wake word may trigger, but the endpointer will not.")  # noqa: T201
        print(f"  Speech peaked at {best_speech:.2f}, under the {config.vad.threshold:.2f}")  # noqa: T201
        print("  threshold, so no utterance is ever completed and the turn ends")  # noqa: T201
        print("  in silence with nothing to transcribe.")  # noqa: T201
        print(f"  Lower vad.threshold toward {max(0.2, best_speech * 0.8):.2f}.")  # noqa: T201
        print()  # noqa: T201

    if peak < _SILENCE_PEAK:
        print("Verdict: silence. Nothing is reaching the process.")  # noqa: T201
        print("  Windows privacy: Settings, Privacy and security, Microphone,")  # noqa: T201
        print("    turn on 'Let desktop apps access your microphone'.")  # noqa: T201
        print("  Check the microphone is not muted and is the Windows default input.")  # noqa: T201
        print("  Or name a specific device in config.yaml under audio.input_device,")  # noqa: T201
        print("    using a name or index from the table above.")  # noqa: T201
        return 1

    if peak < _QUIET_PEAK:
        print("Verdict: signal present but very quiet.")  # noqa: T201
        print("  Raise the input level in Windows sound settings, or move closer.")  # noqa: T201
        return 1

    if best_score >= config.wake.threshold:
        print("Verdict: the wake word was detected. Audio and model are both fine.")  # noqa: T201
        return 0

    if best_score >= _MODEL_RESPONDED:
        # Judged against the noise floor, not against the threshold. A score
        # well clear of zero means the model heard the phrase and was merely
        # unsure, which is a tuning problem rather than a broken microphone.
        suggested = max(0.2, round(best_score * 0.8, 2))
        print("Verdict: audio is good and the wake word nearly triggered.")  # noqa: T201
        print(f"  The model scored {best_score:.2f} against a floor near zero,")  # noqa: T201
        print("  so it recognised the phrase but was not confident enough.")  # noqa: T201
        print(f"  Set wake.threshold to about {suggested:.2f} in config.yaml,")  # noqa: T201
        print("  then re-run this test. Raise it again if you get false triggers.")  # noqa: T201
        print("  A different capture path can also help: the table above lists")  # noqa: T201
        print("  the same microphone under several host APIs, and WASAPI")  # noqa: T201
        print("  resamples better than MME. Set audio.input_device to its index.")  # noqa: T201
        return 1

    print("Verdict: audio is good but the wake word did not register at all.")  # noqa: T201
    print("  Say 'hey jarvis' as one phrase, at a normal speaking pace.")  # noqa: T201
    print("  If it still will not score, the microphone may be picking up mostly noise.")  # noqa: T201
    return 1


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
    if args.mic_test is not None:
        return run_mic_test(config, max(1.0, float(args.mic_test)))
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
