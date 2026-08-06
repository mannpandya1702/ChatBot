"""Continuous circular capture buffer and the sounddevice capture thread.

The wake word fires a little after the word was actually spoken, so the audio
that matters is already in the past by the time anything asks for it. The fix is
the pattern used by GLaDOS: capture never stops, everything lands in a circular
buffer holding a couple of seconds of history (``audio.ring_seconds``), and each
consumer walks that history at its own pace through an independent cursor.

Three properties are load bearing:

* The PortAudio callback never blocks, never allocates without bound, and never
  raises back into the driver. A crash there kills the stream for the whole
  process (§5).
* A reader that falls behind the writer must notice, count what it lost, and
  resynchronise to the oldest sample still held. Returning half-old, half-new
  interleaved audio would corrupt an utterance silently, which is far worse than
  a logged gap.
* Reads copy under the same lock the writer takes, so a block is never observed
  half overwritten.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt

from jarvis.audio import devices as _devices
from jarvis.state import EventType
from jarvis.util.errors import AudioError, JarvisError
from jarvis.util.platform import require_module

if TYPE_CHECKING:  # pragma: no cover - typing only
    from jarvis.config import JarvisConfig
    from jarvis.state import EventBus

__all__ = [
    "AudioCapture",
    "AudioStream",
    "RingBuffer",
    "RingReader",
    "StreamSpec",
    "rms_level",
]

_log = logging.getLogger(__name__)

#: Full tracebacks from inside the audio callback stop after this many, so a
#: persistently broken stream cannot flood the log at block rate.
_MAX_LOGGED_CALLBACK_ERRORS = 5

#: Scale that maps a sample magnitude onto 0 to 1 for the level meter.
_INT16_FULL_SCALE = 32768.0

Samples = npt.NDArray[Any]


def rms_level(block: Samples, *, scale: float = 1.0) -> float:
    """Root mean square of ``block``, clamped to 0.0 to 1.0.

    Args:
        block: Audio samples, any shape. Multi channel input is flattened.
        scale: Multiplier applied after the square root. Use ``1.0`` for float
            samples already in -1 to 1, and ``1 / 32768`` for int16.

    Returns:
        The level. Empty input, all zeroes, and non-finite input all give 0.0,
        because the HUD meter must never show a NaN.
    """
    if block.size == 0:
        return 0.0
    data = np.asarray(block, dtype=np.float64).reshape(-1)
    mean_square = float(np.mean(np.square(data)))
    if not math.isfinite(mean_square) or mean_square <= 0.0:
        return 0.0
    return min(1.0, math.sqrt(mean_square) * scale)


class RingBuffer:
    """Fixed size circular buffer of mono samples, safe across threads.

    Writing past the end overwrites the oldest data. Positions are tracked as
    absolute sample counts since construction, so a reader's cursor stays
    meaningful across any number of wraps and lapping is a simple comparison.
    """

    def __init__(self, capacity_samples: int, dtype: npt.DTypeLike = np.float32) -> None:
        """Allocate the buffer.

        Args:
            capacity_samples: How many samples the buffer holds. At 16 kHz,
                ``2.0 * 16000`` gives the two seconds of pre-roll the wake word
                needs.
            dtype: Sample type. ``float32`` for the normal path, ``int16`` when
                the capture device is configured for it.

        Raises:
            ValueError: ``capacity_samples`` is not positive.
        """
        if capacity_samples <= 0:
            raise ValueError(f"capacity_samples must be positive, got {capacity_samples}")
        self._dtype: np.dtype[Any] = np.dtype(dtype)
        self._capacity = int(capacity_samples)
        self._buf: Samples = np.zeros(self._capacity, dtype=self._dtype)
        self._lock = threading.RLock()
        self._total_written = 0
        #: Absolute index below which data is deliberately discarded (clear()).
        self._base = 0
        #: Bumped by clear() so readers can tell a flush from being lapped.
        self._generation = 0

    # -- introspection -----------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum samples held at once."""
        return self._capacity

    @property
    def dtype(self) -> np.dtype[Any]:
        """Sample type of the stored audio."""
        return self._dtype

    @property
    def filled(self) -> int:
        """Samples currently held, at most :attr:`capacity`."""
        with self._lock:
            return self._total_written - self._oldest_locked()

    @property
    def total_written(self) -> int:
        """Samples ever written, including those already overwritten."""
        with self._lock:
            return self._total_written

    def __len__(self) -> int:
        """Samples currently held. Same as :attr:`filled`."""
        return self.filled

    # -- internals, caller holds the lock ----------------------------------

    def _oldest_locked(self) -> int:
        """Absolute index of the oldest sample still readable."""
        return max(self._base, self._total_written - self._capacity)

    def _read_locked(self, start_abs: int, count: int) -> Samples:
        """Copy ``count`` samples from absolute index ``start_abs``.

        The caller must have validated the range against the live window and
        must hold the lock, so the copy cannot straddle a concurrent write.
        """
        if count <= 0:
            return np.zeros(0, dtype=self._dtype)
        pos = start_abs % self._capacity
        end = pos + count
        if end <= self._capacity:
            return self._buf[pos:end].copy()
        head = self._capacity - pos
        out: Samples = np.empty(count, dtype=self._dtype)
        out[:head] = self._buf[pos:]
        out[head:] = self._buf[: count - head]
        return out

    def _as_mono(self, samples: Samples) -> Samples:
        """Flatten and cast an incoming block to one channel of :attr:`dtype`.

        Raises:
            AudioError: The array has more than two dimensions, which means the
                caller handed us something that is not an audio block.
        """
        block = np.asarray(samples)
        if block.ndim == 2:
            if block.shape[1] == 1:
                block = block[:, 0]
            elif block.shape[1] == 0:
                block = block.reshape(-1)
            else:
                # Downmix. Capture is configured mono, but a device that only
                # offers stereo must not corrupt the timeline by interleaving.
                block = block.mean(axis=1)
        elif block.ndim != 1:
            raise AudioError(
                f"audio block must be 1-D or 2-D, got shape {block.shape}",
                context={"shape": str(block.shape)},
            )
        if block.dtype != self._dtype:
            block = block.astype(self._dtype, copy=False)
        return block

    # -- writing -----------------------------------------------------------

    def write(self, samples: Samples) -> None:
        """Append ``samples``, overwriting the oldest data when full.

        A block longer than the whole buffer keeps only its tail, but still
        advances the write position by the full length so readers account for
        every sample that went past them.

        Args:
            samples: 1-D samples, or 2-D ``(frames, channels)`` as sounddevice
                delivers. Multi channel input is downmixed to mono.

        Raises:
            AudioError: The array shape is not audio shaped.
        """
        block = self._as_mono(samples)
        count = int(block.size)
        if count == 0:
            return
        with self._lock:
            skipped = 0
            if count > self._capacity:
                skipped = count - self._capacity
                block = block[skipped:]
            start_abs = self._total_written + skipped
            pos = start_abs % self._capacity
            end = pos + block.size
            if end <= self._capacity:
                self._buf[pos:end] = block
            else:
                head = self._capacity - pos
                self._buf[pos:] = block[:head]
                self._buf[: block.size - head] = block[head:]
            self._total_written += count

    def clear(self) -> None:
        """Drop everything held. Readers resync without counting a loss."""
        with self._lock:
            self._buf.fill(0)
            self._base = self._total_written
            self._generation += 1

    # -- reading -----------------------------------------------------------

    def latest(self, n: int) -> Samples:
        """The most recent ``n`` samples, zero padded at the front if short.

        Padding goes at the front so the returned array is always ``n`` samples
        of correctly ordered timeline: this is what the wake word hands to STT
        as pre-roll, and a short buffer must read as leading silence, not as
        audio shifted in time.

        Args:
            n: How many samples to return. Zero or negative gives an empty array.

        Returns:
            Exactly ``n`` samples, oldest first.
        """
        if n <= 0:
            return np.zeros(0, dtype=self._dtype)
        with self._lock:
            available = self._total_written - self._oldest_locked()
            take = min(n, available)
            out: Samples = np.zeros(n, dtype=self._dtype)
            if take > 0:
                out[n - take :] = self._read_locked(self._total_written - take, take)
            return out

    def snapshot(self) -> Samples:
        """Everything currently held, oldest first."""
        with self._lock:
            oldest = self._oldest_locked()
            return self._read_locked(oldest, self._total_written - oldest)

    def reader(self, *, at_oldest: bool = False) -> RingReader:
        """Create an independent cursor into this buffer.

        Args:
            at_oldest: Start at the oldest sample still held rather than at the
                write head. The default, starting at the head, means a consumer
                attached mid-run does not first receive a burst of stale audio.

        Returns:
            A new reader. Each reader belongs to one consumer thread.
        """
        with self._lock:
            position = self._oldest_locked() if at_oldest else self._total_written
            return RingReader(self, position, self._generation)


class RingReader:
    """An independent cursor over a :class:`RingBuffer`.

    Readers do not consume: several may sit at different positions in the same
    buffer, which is how the wake word, the VAD, and a debug recorder all read
    one capture stream.

    If the writer laps a reader, the samples in between are gone. The reader
    detects that on its next operation, adds the gap to :attr:`dropped`, and
    jumps to the oldest sample still held. It never returns a block spanning the
    gap. Every method here syncs first, including the properties, so
    :attr:`dropped` is accurate whenever it is asked for.

    One reader is meant for one thread. The buffer is thread safe; a reader's
    cursor is not.
    """

    def __init__(self, buffer: RingBuffer, position: int, generation: int) -> None:
        """Bind a cursor to ``buffer``. Use :meth:`RingBuffer.reader` instead."""
        self._buffer = buffer
        self._cursor = position
        self._generation = generation
        self._dropped = 0

    def _sync_locked(self) -> None:
        """Resynchronise after a flush or after being lapped. Lock held."""
        buffer = self._buffer
        oldest = buffer._oldest_locked()
        if self._generation != buffer._generation:
            # clear() is a deliberate flush by the consumer side, not data loss,
            # so absorb the flush without counting it. Absorbing only up to the
            # flush point rather than returning here means samples lost to
            # lapping after the flush are still counted below, keeping the
            # invariant read + dropped == produced true across a clear().
            self._generation = buffer._generation
            self._cursor = max(self._cursor, buffer._base)
        if self._cursor < oldest:
            self._dropped += oldest - self._cursor
            self._cursor = oldest

    @property
    def available(self) -> int:
        """Samples readable right now."""
        with self._buffer._lock:
            self._sync_locked()
            return self._buffer._total_written - self._cursor

    @property
    def dropped(self) -> int:
        """Samples lost because the writer lapped this reader."""
        with self._buffer._lock:
            self._sync_locked()
            return self._dropped

    @property
    def position(self) -> int:
        """Absolute index of the next sample this reader will return."""
        with self._buffer._lock:
            self._sync_locked()
            return self._cursor

    def read(self, n: int) -> Samples | None:
        """Read exactly ``n`` samples.

        Args:
            n: Samples wanted. Zero or negative returns an empty array.

        Returns:
            ``n`` samples, or None when fewer than ``n`` are available. The
            cursor only advances when a full block is returned, so a caller that
            needs fixed size frames (openWakeWord wants 1280, Silero wants 512)
            can simply retry.

        Raises:
            ValueError: ``n`` exceeds the buffer capacity. Such a read can never
                be satisfied, since the writer laps the reader before that many
                samples accumulate, and a retry loop would spin forever.
        """
        if n <= 0:
            return np.zeros(0, dtype=self._buffer.dtype)
        if n > self._buffer.capacity:
            raise ValueError(
                f"cannot read {n} samples from a buffer holding {self._buffer.capacity}"
            )
        with self._buffer._lock:
            self._sync_locked()
            if self._buffer._total_written - self._cursor < n:
                return None
            out = self._buffer._read_locked(self._cursor, n)
            self._cursor += n
            return out

    def read_available(self) -> Samples:
        """Read everything available, possibly nothing.

        Returns:
            The samples between the cursor and the write head, oldest first.
            Empty when the reader is caught up.
        """
        with self._buffer._lock:
            self._sync_locked()
            count = self._buffer._total_written - self._cursor
            out = self._buffer._read_locked(self._cursor, count)
            self._cursor += count
            return out

    def skip_to_latest(self) -> None:
        """Jump to the write head, discarding anything unread.

        Used when a stage resumes after a long pause, for example returning from
        speaking to listening, where stale audio would be replayed as if it were
        new. Deliberate discards are not counted in :attr:`dropped`.
        """
        with self._buffer._lock:
            self._sync_locked()
            self._cursor = self._buffer._total_written


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------

#: PortAudio calls this as ``(indata, frames, time_info, status)``. The types are
#: driver structures, so they stay ``Any`` at this boundary.
CaptureCallback = Callable[[Any, int, Any, Any], None]


class AudioStream(Protocol):
    """The part of ``sounddevice.InputStream`` that :class:`AudioCapture` uses.

    Declaring it as a protocol is what lets the tests inject a fake stream and
    drive the callback by hand, so the suite needs neither PortAudio nor a
    microphone (§0b).
    """

    def start(self) -> None:
        """Begin delivering audio to the callback."""

    def stop(self) -> None:
        """Stop delivering audio."""

    def close(self) -> None:
        """Release the device."""


@dataclass(frozen=True, slots=True)
class StreamSpec:
    """Everything needed to open a capture stream.

    Attributes:
        sample_rate: Capture rate in hertz.
        block_samples: Frames per callback.
        channels: Capture channels, always 1 for JARVIS.
        dtype: Sample format name, ``float32`` or ``int16``.
        device: PortAudio device index, or None for the system default.
        callback: The function PortAudio invokes per block.
    """

    sample_rate: int
    block_samples: int
    channels: int
    dtype: str
    device: int | None
    callback: CaptureCallback


def _sounddevice_stream(spec: StreamSpec) -> AudioStream:
    """Open a real ``sounddevice.InputStream``. Imported lazily per §0b."""
    sd = require_module("sounddevice", feature="microphone capture")
    stream = sd.InputStream(
        samplerate=spec.sample_rate,
        blocksize=spec.block_samples,
        device=spec.device,
        channels=spec.channels,
        dtype=spec.dtype,
        callback=spec.callback,
        latency="low",
    )
    return cast("AudioStream", stream)


StreamFactory = Callable[[StreamSpec], AudioStream]


class AudioCapture:
    """Continuous microphone capture into a :class:`RingBuffer`.

    Capture runs on PortAudio's own thread and outlives any single turn: the
    wake word, the VAD, and the transcriber all attach readers rather than
    taking turns owning the device. A slow LLM therefore cannot stall the
    microphone (§5).
    """

    def __init__(
        self,
        config: JarvisConfig,
        bus: EventBus | None = None,
        *,
        stream_factory: StreamFactory | None = None,
        level_interval_s: float = 0.05,
    ) -> None:
        """Prepare capture. No device is opened until :meth:`start`.

        Args:
            config: Source of sample rate, block size, ring depth, and the
                input device spec.
            bus: Event bus for :attr:`~jarvis.state.EventType.AUDIO_LEVEL`.
                None disables level events entirely.
            stream_factory: Builds the stream from a :class:`StreamSpec`.
                Defaults to sounddevice. Tests inject a fake.
            level_interval_s: Minimum gap between level events. Blocks arrive
                every 32 ms at the default settings, which is far more often
                than any consumer needs.
        """
        audio = config.audio
        self._config = config
        self._bus = bus
        self._factory = stream_factory or _sounddevice_stream
        self._sample_rate = int(audio.sample_rate)
        self._block_samples = int(audio.block_samples)
        self._channels = int(audio.channels)
        self._dtype_name = str(audio.dtype)
        self._scale = 1.0 if self._dtype_name == "float32" else 1.0 / _INT16_FULL_SCALE
        self._level_interval_s = max(0.0, level_interval_s)

        capacity = max(
            self._block_samples,
            round(audio.ring_seconds * self._sample_rate),
        )
        self.ring = RingBuffer(capacity, dtype=np.dtype(self._dtype_name))

        self._lock = threading.RLock()
        self._stream: AudioStream | None = None
        self._device: _devices.DeviceInfo | None = None
        self._level = 0.0
        self._last_level_emit = 0.0
        self._blocks = 0
        self._overflows = 0
        self._callback_errors = 0

    # -- introspection -----------------------------------------------------

    @property
    def is_running(self) -> bool:
        """True between a successful :meth:`start` and :meth:`stop`."""
        with self._lock:
            return self._stream is not None

    @property
    def level(self) -> float:
        """RMS of the most recent block, 0.0 to 1.0. Zero while stopped."""
        return self._level

    @property
    def sample_rate(self) -> int:
        """Capture rate in hertz."""
        return self._sample_rate

    @property
    def block_samples(self) -> int:
        """Frames per callback."""
        return self._block_samples

    @property
    def device(self) -> _devices.DeviceInfo | None:
        """Device chosen at :meth:`start`, or None for the system default."""
        return self._device

    @property
    def blocks_captured(self) -> int:
        """Callbacks handled since construction."""
        return self._blocks

    @property
    def overflow_count(self) -> int:
        """Callbacks PortAudio flagged, usually input overflow."""
        return self._overflows

    @property
    def callback_error_count(self) -> int:
        """Exceptions swallowed inside the callback. Non-zero means a bug."""
        return self._callback_errors

    def reader(self, *, at_oldest: bool = False) -> RingReader:
        """Attach a new cursor to the capture buffer."""
        return self.ring.reader(at_oldest=at_oldest)

    # -- lifecycle ---------------------------------------------------------

    def _resolve_device(self) -> int | None:
        """Config spec to PortAudio index. None means the system default.

        Raises:
            AudioError: The configured device name or index does not exist.
        """
        spec = self._config.audio.input_device
        # The default path must not need PortAudio enumeration at all, so an
        # auto spec short circuits before any device query.
        found = None if _devices.is_auto_spec(spec) else _devices.find_input_device(spec)
        self._device = found
        return found.index if found is not None else None

    def start(self) -> None:
        """Open the device and begin filling the ring. Idempotent.

        Raises:
            AudioError: The device could not be resolved or the stream refused
                to open.
            DependencyMissingError: sounddevice is not installed and no stream
                factory was injected.
        """
        with self._lock:
            if self._stream is not None:
                return
            index = self._resolve_device()
            spec = StreamSpec(
                sample_rate=self._sample_rate,
                block_samples=self._block_samples,
                channels=self._channels,
                dtype=self._dtype_name,
                device=index,
                callback=self._callback,
            )
            stream = self._open(spec)
            self._stream = stream
        _log.info(
            "audio capture started",
            extra={
                "context": {
                    "device": self._device.label() if self._device else "system default",
                    "sample_rate": self._sample_rate,
                    "block_samples": self._block_samples,
                    "ring_seconds": round(self.ring.capacity / self._sample_rate, 2),
                }
            },
        )

    def _open(self, spec: StreamSpec) -> AudioStream:
        """Build and start a stream, translating any failure into AudioError."""
        try:
            stream = self._factory(spec)
        except JarvisError:
            raise
        except Exception as exc:
            raise AudioError(
                f"could not open the audio input device: {exc}",
                speakable="I could not open the microphone.",
                context={"device": spec.device, "error": str(exc)},
            ) from exc
        try:
            stream.start()
        except Exception as exc:
            try:
                stream.close()
            except Exception:  # noqa: BLE001 - already failing, keep the first cause
                _log.debug("closing the failed stream also failed", exc_info=True)
            raise AudioError(
                f"could not start the audio input stream: {exc}",
                speakable="I could not start listening on the microphone.",
                context={"device": spec.device, "error": str(exc)},
            ) from exc
        return stream

    def stop(self) -> None:
        """Stop capture and release the device. Idempotent and never raises."""
        with self._lock:
            stream = self._stream
            self._stream = None
        if stream is None:
            return
        for step, action in (("stop", stream.stop), ("close", stream.close)):
            try:
                action()
            except Exception:  # noqa: BLE001 - shutdown must always complete
                _log.warning("audio stream %s failed", step, exc_info=True)
        self._level = 0.0
        _log.info(
            "audio capture stopped",
            extra={
                "context": {
                    "blocks": self._blocks,
                    "overflows": self._overflows,
                    "callback_errors": self._callback_errors,
                }
            },
        )

    def __enter__(self) -> AudioCapture:
        """Start capture and return self."""
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Stop capture."""
        self.stop()

    # -- the audio thread --------------------------------------------------

    def _callback(self, indata: Any, frames: int, time_info: Any, status: Any) -> None:
        """PortAudio callback. Must not block, allocate wildly, or raise.

        Everything is caught here. An exception escaping into PortAudio kills
        the stream for the rest of the process, which would take the assistant
        deaf until restart, so a bad block is logged and dropped instead.
        """
        try:
            if status:
                self._overflows += 1
                if self._overflows <= _MAX_LOGGED_CALLBACK_ERRORS:
                    _log.warning("audio input status %s", status)
            block = np.asarray(indata)
            self.ring.write(block)
            self._blocks += 1
            level = rms_level(block, scale=self._scale)
            self._level = level
            self._emit_level(level)
        except Exception:  # noqa: BLE001 - nothing may propagate into PortAudio
            self._callback_errors += 1
            if self._callback_errors <= _MAX_LOGGED_CALLBACK_ERRORS:
                _log.exception(
                    "audio callback failed, block dropped",
                    extra={"context": {"frames": frames}},
                )

    def _emit_level(self, level: float) -> None:
        """Publish AUDIO_LEVEL, rate limited to :attr:`_level_interval_s`."""
        bus = self._bus
        if bus is None:
            return
        now = time.monotonic()
        if now - self._last_level_emit < self._level_interval_s:
            return
        self._last_level_emit = now
        bus.emit(EventType.AUDIO_LEVEL, level=round(level, 4), source="capture")
