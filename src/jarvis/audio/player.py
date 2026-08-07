"""Streaming playback with an immediate stop.

T-1.5 and T-1.9. ``stop()`` is the barge-in primitive: when the user starts
speaking over the assistant, playback has to die within
``tts.stop_latency_ms``, default 100 ms. That rules out the obvious
implementation of "set a flag and let the current buffer finish", because a
single Kokoro chunk can be several seconds long.

The design that meets it: audio is held as a queue of chunks, and the callback
reads from the current chunk a frame at a time. ``stop()`` drops the queue and
the chunk under an atomic swap, so the very next callback invocation, at most
one buffer period later, is already writing silence.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from jarvis.config import JarvisConfig
from jarvis.util.errors import AudioError
from jarvis.util.platform import require_module

__all__ = ["NullSink", "PlaybackSink", "StreamingPlayer"]

_log = logging.getLogger(__name__)

Samples = npt.NDArray[np.float32]

#: How many output blocks of level history to keep. At the default 1024 sample
#: block and 24 kHz that is roughly two seconds, comfortably more than the echo
#: window below ever asks for.
_LEVEL_HISTORY_BLOCKS = 64

#: Default lookback for :meth:`StreamingPlayer.recent_output_level`. Wide enough
#: to cover the output device's buffering plus the trip through the room.
_ECHO_WINDOW_S = 0.4

#: Slack added to the queued duration when wait() is given no timeout. Covers
#: scheduling and the device's own buffering without letting a stalled device
#: park the caller indefinitely.
_DRAIN_GRACE_S = 2.0


class PlaybackSink(Protocol):
    """Where audio frames go. sounddevice in production, a fake in tests."""

    def start(self) -> None:
        """Open the device and begin pulling frames."""
        ...

    def stop(self) -> None:
        """Stop pulling and close the device."""
        ...

    def abort(self) -> None:
        """Stop immediately, discarding anything already handed to the device."""
        ...


class NullSink:
    """Test double that records what was written and honours abort semantics."""

    def __init__(self) -> None:
        self.written: list[Samples] = []
        self.started = False
        self.aborted = False
        self.stopped = False

    def start(self) -> None:
        """Mark the sink as running."""
        self.started = True

    def stop(self) -> None:
        """Mark the sink as stopped."""
        self.stopped = True
        self.started = False

    def abort(self) -> None:
        """Mark the sink as aborted."""
        self.aborted = True
        self.started = False

    def write(self, frames: Samples) -> None:
        """Record frames the player pushed."""
        self.written.append(np.asarray(frames, dtype=np.float32).copy())

    @property
    def total_samples(self) -> int:
        """How many samples were written in total."""
        return sum(block.size for block in self.written)


class StreamingPlayer:
    """Plays queued audio chunks, interruptibly.

    Args:
        config: Supplies the output device, sample rate, and stop deadline.
        sink: Playback sink. Defaults to sounddevice. Tests inject a NullSink.
        blocksize: Frames per callback. Smaller means a tighter stop deadline
            and more callback overhead.
    """

    def __init__(
        self,
        config: JarvisConfig,
        sink: Any | None = None,
        *,
        blocksize: int = 1024,
    ) -> None:
        self._config = config
        self._sample_rate = config.tts.sample_rate
        self._blocksize = blocksize
        self._sink = sink
        self._owns_sink = sink is None

        self._lock = threading.Lock()
        # Chunks carry the generation they were queued in. stop() bumps it, so
        # audio synthesised for a turn the user cut off cannot start playing a
        # moment later, which is what made barge-in feel like it had not worked.
        self._queue: deque[tuple[int, Samples]] = deque()
        self._current: Samples | None = None
        self._offset = 0
        self._generation = 0

        self._drained = threading.Event()
        self._drained.set()
        self._started = False

        # What the speaker is actually producing, as (monotonic time, rms) per
        # block. This is the reference signal the barge-in detector needs: the
        # microphone hears whatever comes out of here, and without knowing how
        # loud that was there is no way to tell the assistant's own voice from
        # the user's. One float per block, so the audio thread pays almost
        # nothing for it.
        self._levels: deque[tuple[float, float]] = deque(maxlen=_LEVEL_HISTORY_BLOCKS)

    # -- lifecycle ---------------------------------------------------------

    @property
    def sample_rate(self) -> int:
        """Playback sample rate in hertz."""
        return self._sample_rate

    @property
    def is_playing(self) -> bool:
        """True while there is audio queued or mid-chunk."""
        with self._lock:
            return self._current is not None or bool(self._queue)

    @property
    def queued_seconds(self) -> float:
        """How much audio is waiting, in seconds."""
        with self._lock:
            pending = sum(chunk.size for _generation, chunk in self._queue)
            if self._current is not None:
                pending += self._current.size - self._offset
        return pending / float(self._sample_rate)

    def start(self) -> None:
        """Open the output device."""
        if self._started:
            return
        if self._sink is None:
            self._sink = self._build_sink()
        self._sink.start()
        self._started = True

    def close(self) -> None:
        """Stop playback and close the device."""
        self.stop()
        if self._sink is not None and self._started:
            try:
                self._sink.stop()
            except Exception:  # noqa: BLE001 - closing a dead device is routine
                _log.debug("closing the playback sink failed", exc_info=True)
        self._started = False

    def __enter__(self) -> StreamingPlayer:
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _build_sink(self) -> Any:
        """Build a sounddevice output stream driven by our callback."""
        sd = require_module("sounddevice", feature="audio playback")

        from jarvis.audio.devices import find_output_device

        device = find_output_device(self._config.audio.output_device)
        index = device.index if device is not None else None

        def callback(outdata: Any, frames: int, _time: Any, status: Any) -> None:
            if status:
                _log.debug("playback status: %s", status)
            try:
                block = self._next_block(frames)
                outdata[:, 0] = block
            except Exception:  # noqa: BLE001 - never raise into PortAudio
                _log.exception("playback callback failed")
                outdata.fill(0)

        try:
            return sd.OutputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
                blocksize=self._blocksize,
                device=index,
                callback=callback,
            )
        except Exception as exc:
            raise AudioError(
                f"could not open the output device: {exc}",
                speakable="I could not open the speakers.",
            ) from exc

    # -- queueing ----------------------------------------------------------

    @property
    def generation(self) -> int:
        """Current playback generation. Bumped by every :meth:`stop`."""
        with self._lock:
            return self._generation

    def play(self, chunk: Samples, *, generation: int | None = None) -> None:
        """Queue a chunk. Never blocks.

        Args:
            chunk: float32 mono audio.
            generation: The generation this audio belongs to, from
                :attr:`generation` before synthesis started. A chunk from a
                superseded generation is discarded, so audio that was still
                being synthesised when the user interrupted never reaches the
                speaker.
        """
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        with self._lock:
            if generation is not None and generation != self._generation:
                _log.debug("dropped audio from a superseded generation")
                return
            self._queue.append((self._generation, audio))
            self._drained.clear()

    def _next_block(self, frames: int) -> Samples:
        """Produce the next ``frames`` samples. Called from the audio thread.

        Holds the lock only long enough to slice, and never allocates more than
        one output block, so it cannot stall PortAudio.
        """
        out = np.zeros(frames, dtype=np.float32)
        filled = 0

        with self._lock:
            while filled < frames:
                if self._current is None or self._offset >= self._current.size:
                    if not self._queue:
                        self._current = None
                        self._offset = 0
                        if filled == 0:
                            self._drained.set()
                        break
                    queued_generation, chunk = self._queue.popleft()
                    if queued_generation != self._generation:
                        continue
                    self._current = chunk
                    self._offset = 0

                available = self._current.size - self._offset
                take = min(available, frames - filled)
                out[filled : filled + take] = self._current[self._offset : self._offset + take]
                self._offset += take
                filled += take

            # Release a chunk the moment it is exhausted rather than on the
            # next callback. Otherwise is_playing reports True, and wait()
            # blocks, when there is nothing left to play.
            if self._current is not None and self._offset >= self._current.size:
                self._current = None
                self._offset = 0

            if self._current is None and not self._queue:
                self._drained.set()

            self._levels.append((time.monotonic(), float(np.sqrt(np.mean(out**2)))))

        return out

    def recent_output_level(self, window_s: float = _ECHO_WINDOW_S) -> float:
        """Loudest thing the speaker has produced in the last ``window_s``.

        This is the reference the barge-in detector measures the microphone
        against. It is a peak over a window rather than a level at an instant
        because the echo arrives late and smeared: the output device buffers
        tens of milliseconds, the room adds a few more, and a reflection adds
        more again. A window wide enough to cover all of that errs towards
        calling a loud microphone frame an echo, which is the safe direction.
        Being slightly deaf during the assistant's loudest syllable costs one
        missed interruption; being wrong the other way cuts the assistant off
        mid-sentence on every single turn.

        Returns:
            RMS in 0..1, or 0.0 when nothing has been played recently.
        """
        cutoff = time.monotonic() - max(window_s, 0.0)
        with self._lock:
            recent = [level for stamp, level in self._levels if stamp >= cutoff]
        return max(recent) if recent else 0.0

    # -- interruption ------------------------------------------------------

    def stop(self) -> None:
        """Cut playback immediately.

        Safe to call when idle and safe to call twice. The queue and the current
        chunk are dropped under the lock, so the next callback, at most one
        block period away, already produces silence.
        """
        with self._lock:
            self._queue.clear()
            self._current = None
            self._offset = 0
            self._generation += 1
            self._drained.set()

        # Ask the device to discard anything it has already buffered. Without
        # this the hardware buffer alone can hold tens of milliseconds of the
        # assistant's voice after the queue is empty.
        #
        # Deliberately not gated on self._started. A restart that failed leaves
        # it False, and gating on it meant the next barge-in did not even try:
        # one failed restart, from an abort that left the stream active or a
        # Bluetooth headset walking out of range, made the player mute for the
        # rest of the session. play() kept queueing, nothing consumed it,
        # is_playing stayed True and wait() never returned, all reported at
        # debug level.
        sink = self._sink
        if sink is None:
            return
        abort = getattr(sink, "abort", None)
        if not callable(abort):
            return
        try:
            abort()
        except Exception:  # noqa: BLE001 - the restart below is what matters
            _log.debug("aborting the playback sink failed", exc_info=True)
        self._restart_sink()

    def _restart_sink(self) -> None:
        """Bring the device back after an abort, rebuilding it if it will not.

        sounddevice needs an explicit restart after ``abort()``. When that
        fails the sink itself is suspect, so the second attempt builds a new
        one: an aborted stream that refuses to restart is exactly the shape a
        removed device leaves behind, and reusing it can only fail again.
        """
        try:
            if self._sink is not None:
                self._sink.start()
                self._started = True
                return
        except Exception:  # noqa: BLE001 - rebuilding is the next thing to try
            _log.warning("restarting the playback sink failed", exc_info=True)
        self._started = False

        if not self._owns_sink:
            # An injected sink is the caller's to manage, and replacing it would
            # throw away whatever the test or the embedder is watching.
            return
        try:
            self._sink = self._build_sink()
            self._sink.start()
            self._started = True
            _log.info("rebuilt the playback device after a failed restart")
        except Exception:  # noqa: BLE001 - §5, a dead speaker must not end the turn
            _log.error(
                "the playback device could not be reopened, jarvis is mute until it recovers"
            )
            self._started = False

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queue drains.

        Args:
            timeout: Seconds to wait, or None to wait as long as the audio
                could possibly take.

        Returns:
            True when playback finished, False on timeout.

        A dead device drains nothing, so an unbounded wait here parks the turn
        loop until something else interrupts it. The queue knows how long it
        would take to play; anything much past that means the audio is not
        moving, and reporting a timeout is both true and recoverable.
        """
        if timeout is None:
            timeout = self.queued_seconds + _DRAIN_GRACE_S
        drained = self._drained.wait(timeout)
        if not drained and self.queued_seconds > 0:
            _log.warning(
                "playback did not drain in time, the device may have stopped",
                extra={
                    "context": {
                        "queued_seconds": round(self.queued_seconds, 2),
                        "waited_s": round(timeout, 2),
                        "started": self._started,
                    }
                },
            )
        return drained

    def measure_stop_latency(self) -> float:
        """Time a stop() call in milliseconds. Used by the tests and the bench."""
        started = time.perf_counter()
        self.stop()
        return (time.perf_counter() - started) * 1000.0
