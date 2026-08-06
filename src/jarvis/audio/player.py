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
        self._queue: deque[Samples] = deque()
        self._current: Samples | None = None
        self._offset = 0
        self._generation = 0

        self._drained = threading.Event()
        self._drained.set()
        self._started = False

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
            pending = sum(chunk.size for chunk in self._queue)
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

    def play(self, chunk: Samples) -> None:
        """Queue a chunk. Never blocks."""
        audio = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if audio.size == 0:
            return
        with self._lock:
            self._queue.append(audio)
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
                    self._current = self._queue.popleft()
                    self._offset = 0

                available = self._current.size - self._offset
                take = min(available, frames - filled)
                out[filled : filled + take] = self._current[self._offset : self._offset + take]
                self._offset += take
                filled += take

            if self._current is None and not self._queue:
                self._drained.set()

        return out

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
        sink = self._sink
        if sink is not None and self._started:
            abort = getattr(sink, "abort", None)
            if callable(abort):
                try:
                    abort()
                except Exception:  # noqa: BLE001
                    _log.debug("aborting the playback sink failed", exc_info=True)
                # sounddevice needs an explicit restart after abort().
                try:
                    sink.start()
                except Exception:  # noqa: BLE001
                    _log.debug("restarting the playback sink failed", exc_info=True)
                    self._started = False

    def wait(self, timeout: float | None = None) -> bool:
        """Block until the queue drains.

        Returns:
            True when playback finished, False on timeout.
        """
        return self._drained.wait(timeout)

    def measure_stop_latency(self) -> float:
        """Time a stop() call in milliseconds. Used by the tests and the bench."""
        started = time.perf_counter()
        self.stop()
        return (time.perf_counter() - started) * 1000.0
