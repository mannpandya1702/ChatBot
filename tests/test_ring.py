"""T-1.1: circular capture buffer, reader cursors, device selection, capture thread.

The property that matters most here is that a reader which falls behind the
writer can never return interleaved audio. A silently corrupted utterance would
surface much later as a nonsense transcription, so the lapping path is tested
deterministically, under concurrency, and for exact drop accounting.

Nothing in this file needs PortAudio, a microphone, or Windows. The capture
stream is injected, and the two tests that genuinely need hardware are marked
``manual`` and written so they pass on the Windows host.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
import pytest

from jarvis.audio import devices as devices_module
from jarvis.audio.devices import (
    DeviceInfo,
    describe_devices,
    find_input_device,
    find_output_device,
    is_auto_spec,
    list_devices,
)
from jarvis.audio.ring import (
    AudioCapture,
    RingBuffer,
    RingReader,
    StreamSpec,
    rms_level,
)
from jarvis.config import JarvisConfig
from jarvis.state import EventType
from jarvis.util.errors import AudioError, DependencyMissingError
from jarvis.util.platform import has_module

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def ramp(start: int, count: int) -> np.ndarray[Any, Any]:
    """Samples whose value equals their absolute index in the stream.

    Every integrity assertion in this file relies on that identity: any block a
    reader hands back must equal ``arange(first_value, first_value + n)``, which
    is only true if the block is a contiguous, correctly ordered slice of what
    was written. float32 represents integers exactly up to 2**24, far beyond the
    sample counts used here.
    """
    return np.arange(start, start + count, dtype=np.float32)


class FakeStream:
    """Stands in for ``sounddevice.InputStream``.

    Holds the :class:`StreamSpec` it was built from so tests can assert on the
    stream parameters, and exposes :meth:`feed` to drive the capture callback by
    hand on the calling thread.
    """

    def __init__(self, spec: StreamSpec, *, fail_on_start: bool = False) -> None:
        self.spec = spec
        self.fail_on_start = fail_on_start
        self.starts = 0
        self.stops = 0
        self.closes = 0

    def start(self) -> None:
        if self.fail_on_start:
            raise RuntimeError("PortAudio error -9996")
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def close(self) -> None:
        self.closes += 1

    def feed(self, block: np.ndarray[Any, Any], status: Any = None) -> None:
        """Invoke the capture callback exactly as PortAudio would."""
        frames = int(np.asarray(block).shape[0]) if np.asarray(block).ndim else 0
        self.spec.callback(block, frames, None, status)


class RecordingFactory:
    """Stream factory that records what it built, for dependency injection."""

    def __init__(
        self, *, build_error: Exception | None = None, fail_on_start: bool = False
    ) -> None:
        self.build_error = build_error
        self.fail_on_start = fail_on_start
        self.created: list[FakeStream] = []

    def __call__(self, spec: StreamSpec) -> FakeStream:
        if self.build_error is not None:
            raise self.build_error
        stream = FakeStream(spec, fail_on_start=self.fail_on_start)
        self.created.append(stream)
        return stream

    @property
    def stream(self) -> FakeStream:
        """The most recently built stream."""
        assert self.created, "no stream was built"
        return self.created[-1]


#: A plausible Windows device list: the same microphone appears under more than
#: one host API, and one entry differs from another only by case.
FAKE_DEVICES = [
    DeviceInfo(0, "Microsoft Sound Mapper - Input", 2, 0, 44100.0, 0),
    DeviceInfo(1, "Microphone (Realtek(R) Audio)", 2, 0, 44100.0, 0, is_default_input=True),
    DeviceInfo(2, "Speakers (Realtek(R) Audio)", 0, 2, 48000.0, 0, is_default_output=True),
    DeviceInfo(3, "Microphone (Blue Yeti)", 2, 0, 48000.0, 1),
    DeviceInfo(4, "microphone (blue yeti)", 1, 0, 44100.0, 2),
    DeviceInfo(5, "Headset Earphone (Bluetooth)", 0, 1, 16000.0, 1),
]


@pytest.fixture
def capture_config(cfg: JarvisConfig) -> JarvisConfig:
    """Config with a small, fast ring so capture tests stay cheap."""
    cfg.audio.sample_rate = 16_000
    cfg.audio.block_samples = 512
    cfg.audio.ring_seconds = 1.0
    return cfg


# ---------------------------------------------------------------------------
# RingBuffer: capacity, wraparound, oversized writes
# ---------------------------------------------------------------------------


def test_new_buffer_is_empty_with_the_requested_capacity() -> None:
    ring = RingBuffer(1024)
    assert ring.capacity == 1024
    assert ring.filled == 0
    assert ring.total_written == 0
    assert len(ring) == 0
    assert ring.dtype == np.dtype(np.float32)
    assert ring.snapshot().size == 0


@pytest.mark.parametrize("capacity", [0, -1])
def test_capacity_must_be_positive(capacity: int) -> None:
    with pytest.raises(ValueError, match="capacity_samples must be positive"):
        RingBuffer(capacity)


def test_write_then_snapshot_preserves_order() -> None:
    ring = RingBuffer(10)
    ring.write(ramp(0, 4))
    ring.write(ramp(4, 3))
    assert np.array_equal(ring.snapshot(), ramp(0, 7))
    assert ring.filled == 7
    assert ring.total_written == 7


def test_wraparound_keeps_only_the_newest_capacity_samples() -> None:
    ring = RingBuffer(5)
    ring.write(ramp(0, 3))
    ring.write(ramp(3, 4))
    # Seven samples went in, five fit, the oldest two are gone.
    assert np.array_equal(ring.snapshot(), ramp(2, 5))
    assert ring.filled == 5
    assert ring.total_written == 7


def test_repeated_wraparound_stays_aligned() -> None:
    """Many small writes across many wraps must not drift the write position."""
    ring = RingBuffer(7)
    written = 0
    for _ in range(50):
        ring.write(ramp(written, 3))
        written += 3
    assert ring.total_written == written
    assert np.array_equal(ring.snapshot(), ramp(written - 7, 7))


def test_write_larger_than_capacity_keeps_the_tail() -> None:
    ring = RingBuffer(4)
    ring.write(ramp(0, 10))
    assert np.array_equal(ring.snapshot(), ramp(6, 4))
    assert ring.filled == 4
    # The whole block counts as written, so readers can account for the loss.
    assert ring.total_written == 10


def test_oversized_write_leaves_the_buffer_aligned_for_later_writes() -> None:
    """An oversized write must not shift the modulo mapping of later writes."""
    ring = RingBuffer(4)
    ring.write(ramp(0, 3))
    ring.write(ramp(3, 9))  # nine samples into a four sample buffer
    ring.write(ramp(12, 2))
    assert ring.total_written == 14
    assert np.array_equal(ring.snapshot(), ramp(10, 4))


def test_write_exactly_capacity() -> None:
    ring = RingBuffer(6)
    ring.write(ramp(0, 6))
    assert np.array_equal(ring.snapshot(), ramp(0, 6))
    assert ring.filled == 6


def test_empty_write_is_a_no_op() -> None:
    ring = RingBuffer(4)
    ring.write(np.zeros(0, dtype=np.float32))
    assert ring.total_written == 0
    assert ring.filled == 0


def test_snapshot_returns_a_copy() -> None:
    ring = RingBuffer(4)
    ring.write(ramp(0, 4))
    taken = ring.snapshot()
    taken[0] = 999.0
    assert ring.snapshot()[0] == 0.0


# ---------------------------------------------------------------------------
# RingBuffer: latest() padding and shape handling
# ---------------------------------------------------------------------------


def test_latest_pads_at_the_front_when_short() -> None:
    """Pre-roll must read as leading silence, never as audio shifted in time."""
    ring = RingBuffer(10)
    ring.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert np.array_equal(ring.latest(5), np.array([0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))


def test_latest_returns_the_newest_samples_when_full() -> None:
    ring = RingBuffer(5)
    ring.write(ramp(0, 7))
    assert np.array_equal(ring.latest(3), ramp(4, 3))
    assert np.array_equal(ring.latest(5), ramp(2, 5))


def test_latest_caps_at_capacity_and_pads_the_rest() -> None:
    ring = RingBuffer(4)
    ring.write(ramp(0, 4))
    out = ring.latest(6)
    assert out.size == 6
    assert np.array_equal(out, np.array([0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32))


def test_latest_of_an_empty_buffer_is_silence() -> None:
    ring = RingBuffer(8)
    out = ring.latest(4)
    assert out.size == 4
    assert not out.any()


@pytest.mark.parametrize("n", [0, -5])
def test_latest_of_zero_or_negative_is_empty(n: int) -> None:
    ring = RingBuffer(8)
    ring.write(ramp(0, 8))
    assert ring.latest(n).size == 0


def test_two_dimensional_mono_block_is_flattened() -> None:
    """sounddevice delivers (frames, channels), which is (n, 1) for mono."""
    ring = RingBuffer(8)
    ring.write(ramp(0, 4).reshape(4, 1))
    assert np.array_equal(ring.snapshot(), ramp(0, 4))


def test_stereo_block_is_downmixed_not_interleaved() -> None:
    ring = RingBuffer(8)
    ring.write(np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32))
    assert np.array_equal(ring.snapshot(), np.array([2.0, 3.0], dtype=np.float32))


def test_three_dimensional_block_is_rejected() -> None:
    ring = RingBuffer(8)
    with pytest.raises(AudioError, match="1-D or 2-D"):
        ring.write(np.zeros((2, 2, 2), dtype=np.float32))


def test_int16_buffer_casts_incoming_samples() -> None:
    ring = RingBuffer(8, dtype=np.int16)
    ring.write(np.array([1.9, -2.9], dtype=np.float32))
    out = ring.snapshot()
    assert out.dtype == np.dtype(np.int16)
    assert np.array_equal(out, np.array([1, -2], dtype=np.int16))


# ---------------------------------------------------------------------------
# RingReader: cursors
# ---------------------------------------------------------------------------


def test_reader_starts_at_the_write_head() -> None:
    """A consumer attached mid-run must not be handed a burst of stale audio."""
    ring = RingBuffer(16)
    ring.write(ramp(0, 8))
    reader = ring.reader()
    assert reader.available == 0
    assert reader.read(1) is None
    ring.write(ramp(8, 4))
    assert np.array_equal(reader.read_available(), ramp(8, 4))


def test_reader_at_oldest_sees_the_existing_history() -> None:
    ring = RingBuffer(16)
    ring.write(ramp(0, 8))
    reader = ring.reader(at_oldest=True)
    assert reader.available == 8
    assert np.array_equal(reader.read(8), ramp(0, 8))


def test_read_returns_none_and_does_not_advance_when_short() -> None:
    """Fixed frame consumers (openWakeWord wants 1280) simply retry."""
    ring = RingBuffer(16)
    reader = ring.reader()
    ring.write(ramp(0, 3))
    assert reader.read(4) is None
    assert reader.available == 3
    ring.write(ramp(3, 1))
    assert np.array_equal(reader.read(4), ramp(0, 4))
    assert reader.available == 0


def test_read_larger_than_the_buffer_is_a_loud_error_not_a_spin() -> None:
    """It could never be satisfied, so a retry loop would hang forever."""
    ring = RingBuffer(16)
    reader = ring.reader()
    with pytest.raises(ValueError, match="cannot read 17 samples"):
        reader.read(17)


def test_read_of_zero_is_an_empty_array_not_none() -> None:
    ring = RingBuffer(8)
    reader = ring.reader()
    out = reader.read(0)
    assert out is not None
    assert out.size == 0


def test_read_available_drains_to_the_head() -> None:
    ring = RingBuffer(16)
    reader = ring.reader()
    assert reader.read_available().size == 0
    ring.write(ramp(0, 5))
    assert np.array_equal(reader.read_available(), ramp(0, 5))
    assert reader.read_available().size == 0


def test_reader_cursor_is_independent_of_the_buffer() -> None:
    """Reading does not consume: the buffer still holds the history."""
    ring = RingBuffer(16)
    reader = ring.reader()
    ring.write(ramp(0, 8))
    assert np.array_equal(reader.read(8), ramp(0, 8))
    assert ring.filled == 8
    assert np.array_equal(ring.snapshot(), ramp(0, 8))


def test_two_readers_hold_different_positions() -> None:
    ring = RingBuffer(32)
    fast = ring.reader()
    slow = ring.reader()
    ring.write(ramp(0, 10))

    assert np.array_equal(fast.read(10), ramp(0, 10))
    assert np.array_equal(slow.read(4), ramp(0, 4))
    assert fast.position == 10
    assert slow.position == 4
    assert fast.available == 0
    assert slow.available == 6

    ring.write(ramp(10, 5))
    assert np.array_equal(fast.read_available(), ramp(10, 5))
    assert np.array_equal(slow.read_available(), ramp(4, 11))
    assert fast.position == slow.position == 15


def test_a_third_reader_starting_at_oldest_coexists() -> None:
    ring = RingBuffer(32)
    first = ring.reader()
    ring.write(ramp(0, 12))
    second = ring.reader()
    third = ring.reader(at_oldest=True)

    assert first.available == 12
    assert second.available == 0
    assert third.available == 12
    assert np.array_equal(third.read(12), ramp(0, 12))
    assert first.available == 12  # untouched by the other readers


def test_skip_to_latest_discards_without_counting_a_drop() -> None:
    ring = RingBuffer(32)
    reader = ring.reader()
    ring.write(ramp(0, 20))
    reader.skip_to_latest()
    assert reader.available == 0
    assert reader.dropped == 0
    ring.write(ramp(20, 3))
    assert np.array_equal(reader.read_available(), ramp(20, 3))


# ---------------------------------------------------------------------------
# RingReader: lapping, the single most important property in this module
# ---------------------------------------------------------------------------


def test_lapped_reader_counts_the_loss_and_resynchronises() -> None:
    ring = RingBuffer(8)
    reader = ring.reader()

    ring.write(ramp(0, 4))
    assert np.array_equal(reader.read(4), ramp(0, 4))
    assert reader.dropped == 0

    # Twenty more samples into an eight sample buffer: the reader sat at
    # absolute index 4, the oldest surviving sample is now 16, so 12 are gone.
    ring.write(ramp(4, 20))
    assert reader.dropped == 12
    assert reader.position == 16
    assert reader.available == 8

    recovered = reader.read_available()
    assert np.array_equal(recovered, ramp(16, 8))
    assert reader.dropped == 12


def test_lapped_reader_never_returns_a_block_spanning_the_gap() -> None:
    """The corruption this whole design exists to prevent.

    Old and new samples share physical slots. A naive implementation would hand
    back a block whose first half is from the previous lap, producing audio that
    is silently wrong rather than loudly missing.
    """
    ring = RingBuffer(64)
    reader = ring.reader()
    written = 0
    consumed = 0
    read_total = 0

    for _ in range(40):
        ring.write(ramp(written, 100))  # laps the buffer on every iteration
        written += 100
        chunk = reader.read(32)
        assert chunk is not None
        start = int(chunk[0])
        assert np.array_equal(chunk, ramp(start, 32)), "block spans a discontinuity"
        assert start >= consumed, "reader went backwards"
        consumed = start + 32
        read_total += 32

    # Every sample the cursor travelled past was either handed over or counted
    # as dropped. Nothing was invented and nothing was double counted.
    assert read_total + reader.dropped == reader.position
    assert reader.position <= written


def test_drop_accounting_is_exact() -> None:
    ring = RingBuffer(16)
    reader = ring.reader()
    ring.write(ramp(0, 100))
    assert reader.dropped == 84  # 100 written, 16 survive, reader was at zero
    assert np.array_equal(reader.read_available(), ramp(84, 16))
    assert reader.position == 100
    assert reader.dropped + 16 == 100


def test_dropped_is_reported_without_needing_a_read() -> None:
    ring = RingBuffer(8)
    reader = ring.reader()
    ring.write(ramp(0, 30))
    assert reader.dropped == 22
    assert reader.available == 8


def test_a_reader_that_keeps_up_never_drops() -> None:
    ring = RingBuffer(64)
    reader = ring.reader()
    for block in range(50):
        ring.write(ramp(block * 16, 16))
        assert np.array_equal(reader.read(16), ramp(block * 16, 16))
    assert reader.dropped == 0
    assert reader.position == 800


def test_clear_flushes_without_charging_readers_a_drop() -> None:
    """A deliberate flush is not data loss, so it must not inflate ``dropped``."""
    ring = RingBuffer(16)
    reader = ring.reader()
    ring.write(ramp(0, 10))
    ring.clear()

    assert ring.filled == 0
    assert not ring.snapshot().size
    assert reader.available == 0
    assert reader.dropped == 0

    ring.write(ramp(100, 4))
    assert np.array_equal(reader.read_available(), ramp(100, 4))
    assert reader.dropped == 0


def test_clear_zeroes_the_stored_audio() -> None:
    ring = RingBuffer(8)
    ring.write(np.full(8, 0.5, dtype=np.float32))
    ring.clear()
    assert not ring.latest(8).any()


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def _drain(reader: RingReader, chunk_size: int, writer_done: threading.Event) -> int:
    """Consume ``reader`` until the writer finishes, checking every block.

    Returns:
        How many samples were read.

    Raises:
        AssertionError: A block was torn, out of order, or went backwards.
    """
    read_total = 0
    last_end = 0
    while True:
        chunk = reader.read(chunk_size)
        if chunk is None:
            if writer_done.is_set() and reader.available < chunk_size:
                break
            time.sleep(0.0002)
            continue
        start = int(chunk[0])
        if not np.array_equal(chunk, ramp(start, chunk_size)):
            raise AssertionError(f"torn block at absolute index {start}")
        if start < last_end:
            raise AssertionError(f"reader went backwards: {start} < {last_end}")
        last_end = start + chunk_size
        read_total += chunk_size

    tail = reader.read_available()
    if tail.size:
        start = int(tail[0])
        if not np.array_equal(tail, ramp(start, int(tail.size))):
            raise AssertionError(f"torn tail at absolute index {start}")
        if start < last_end:
            raise AssertionError("tail went backwards")
        read_total += int(tail.size)
    return read_total


@pytest.mark.slow
def test_concurrent_writer_and_readers_keep_the_stream_intact() -> None:
    """A live writer and two live readers, checked for tearing and for accounting.

    Whatever each reader returns must be a contiguous slice of what was written,
    and read plus dropped must equal exactly the distance its cursor travelled.
    """
    ring = RingBuffer(4_096)
    readers = [ring.reader(), ring.reader()]
    block = 128
    total = 128_000  # a whole number of blocks
    writer_done = threading.Event()
    results: dict[int, int] = {}
    failures: list[BaseException] = []

    def consume(slot: int) -> None:
        try:
            results[slot] = _drain(readers[slot], 256 if slot == 0 else 384, writer_done)
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            failures.append(exc)

    consumers = [threading.Thread(target=consume, args=(slot,)) for slot in range(len(readers))]
    for thread in consumers:
        thread.start()

    written = 0
    while written < total:
        ring.write(ramp(written, block))
        written += block
        if written % 8192 == 0:
            time.sleep(0.0005)  # let the consumers run, and let them fall behind
    writer_done.set()

    for thread in consumers:
        thread.join(timeout=30)
        assert not thread.is_alive(), "consumer thread did not finish"

    assert not failures, f"integrity failure: {failures[0]}"
    assert ring.total_written == total
    for slot, reader in enumerate(readers):
        assert reader.position == total, "reader did not finish at the write head"
        assert results[slot] + reader.dropped == total, "samples are unaccounted for"


@pytest.mark.slow
def test_concurrent_snapshot_never_observes_a_torn_block() -> None:
    """snapshot() copies under the writer's lock, so it is always self-consistent."""
    ring = RingBuffer(1_024)
    stop = threading.Event()
    failures: list[str] = []

    def write_forever() -> None:
        written = 0
        while not stop.is_set():
            ring.write(ramp(written, 64))
            written += 64

    writer = threading.Thread(target=write_forever)
    writer.start()
    try:
        for _ in range(400):
            taken = ring.snapshot()
            if taken.size and not np.array_equal(taken, ramp(int(taken[0]), int(taken.size))):
                failures.append(f"torn snapshot starting at {taken[0]}")
                break
    finally:
        stop.set()
        writer.join(timeout=10)
    assert not failures, failures[0]


# ---------------------------------------------------------------------------
# Level metering
# ---------------------------------------------------------------------------


def test_rms_of_full_scale_square_wave_is_one() -> None:
    assert rms_level(np.array([1.0, -1.0, 1.0, -1.0], dtype=np.float32)) == pytest.approx(1.0)


def test_rms_of_silence_is_zero(silence: Any) -> None:
    assert rms_level(silence(0.1)) == 0.0


def test_rms_of_a_sine_is_amplitude_over_root_two(sine_audio: Any) -> None:
    level = rms_level(sine_audio(0.5, amplitude=0.2))
    assert level == pytest.approx(0.2 / np.sqrt(2), abs=0.005)


def test_rms_of_an_empty_block_is_zero() -> None:
    assert rms_level(np.zeros(0, dtype=np.float32)) == 0.0


def test_rms_is_clamped_to_one() -> None:
    assert rms_level(np.full(8, 4.0, dtype=np.float32)) == 1.0


def test_rms_of_non_finite_input_is_zero() -> None:
    """The HUD meter must never be handed a NaN."""
    assert rms_level(np.array([np.nan, np.nan], dtype=np.float32)) == 0.0
    assert rms_level(np.array([np.inf, 0.0], dtype=np.float32)) == 0.0


def test_rms_scales_int16_to_the_same_range() -> None:
    block = np.full(16, 16_384, dtype=np.int16)
    assert rms_level(block, scale=1.0 / 32_768.0) == pytest.approx(0.5)


def test_rms_flattens_multichannel_input() -> None:
    stereo = np.array([[1.0, -1.0], [1.0, -1.0]], dtype=np.float32)
    assert rms_level(stereo) == pytest.approx(1.0)


def test_rms_rises_with_amplitude(sine_audio: Any) -> None:
    quiet = rms_level(sine_audio(0.1, amplitude=0.01))
    loud = rms_level(sine_audio(0.1, amplitude=0.5))
    assert 0.0 < quiet < loud < 1.0


# ---------------------------------------------------------------------------
# AudioCapture, with an injected stream
# ---------------------------------------------------------------------------


def test_capture_sizes_the_ring_from_the_configured_preroll(capture_config: JarvisConfig) -> None:
    capture_config.audio.ring_seconds = 2.0
    capture = AudioCapture(capture_config, stream_factory=RecordingFactory())
    assert capture.ring.capacity == 32_000
    assert capture.sample_rate == 16_000
    assert capture.block_samples == 512
    assert not capture.is_running
    assert capture.level == 0.0


def test_capture_ring_is_never_smaller_than_one_block(cfg: JarvisConfig) -> None:
    cfg.audio.sample_rate = 8_000
    cfg.audio.block_samples = 4_096
    cfg.audio.ring_seconds = 0.5  # 4000 samples, smaller than the block
    capture = AudioCapture(cfg, stream_factory=RecordingFactory())
    assert capture.ring.capacity == 4_096


def test_start_opens_a_stream_with_the_configured_parameters(
    capture_config: JarvisConfig,
) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        spec = factory.stream.spec
        assert spec.sample_rate == 16_000
        assert spec.block_samples == 512
        assert spec.channels == 1
        assert spec.dtype == "float32"
        assert spec.device is None  # the system default
        assert factory.stream.starts == 1
        assert capture.is_running
    finally:
        capture.stop()


def test_start_is_idempotent(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    capture.start()
    try:
        assert len(factory.created) == 1
    finally:
        capture.stop()


def test_stop_closes_the_stream_and_is_idempotent(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    capture.stop()
    capture.stop()
    assert factory.stream.stops == 1
    assert factory.stream.closes == 1
    assert not capture.is_running


def test_stop_survives_a_stream_that_raises_on_close(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    stream = factory.stream

    def boom() -> None:
        raise RuntimeError("device already gone")

    stream.close = boom  # type: ignore[method-assign]
    capture.stop()  # must not raise: shutdown always completes
    assert not capture.is_running


def test_context_manager_starts_and_stops(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    with AudioCapture(capture_config, stream_factory=factory) as capture:
        assert capture.is_running
    assert not capture.is_running
    assert factory.stream.stops == 1
    assert factory.stream.closes == 1


def test_capture_can_restart_after_stopping(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    capture.stop()
    capture.start()
    try:
        assert len(factory.created) == 2
        assert capture.is_running
    finally:
        capture.stop()


def test_callback_fills_the_ring_and_updates_the_level(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        reader = capture.reader()
        block = np.full((512, 1), 0.5, dtype=np.float32)
        factory.stream.feed(block)
        factory.stream.feed(block)

        assert capture.blocks_captured == 2
        assert capture.ring.filled == 1024
        assert capture.level == pytest.approx(0.5)
        got = reader.read(1024)
        assert got is not None
        assert np.allclose(got, 0.5)
    finally:
        capture.stop()


def test_capture_of_a_long_stream_keeps_only_the_preroll(capture_config: JarvisConfig) -> None:
    """Capacity is the pre-roll window, so old audio rolls off on its own."""
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        for index in range(40):  # 20480 samples into a 16000 sample ring
            factory.stream.feed(ramp(index * 512, 512))
        assert capture.ring.filled == 16_000
        assert np.array_equal(capture.ring.latest(4), ramp(40 * 512 - 4, 4))
    finally:
        capture.stop()


def test_callback_never_raises_into_portaudio(capture_config: JarvisConfig) -> None:
    """A bad block is logged and dropped: the stream must stay alive."""
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        factory.stream.feed(np.zeros((4, 2, 2), dtype=np.float32))  # not audio shaped
        assert capture.callback_error_count == 1
        assert capture.is_running

        factory.stream.feed(np.full(512, 0.25, dtype=np.float32))
        assert capture.callback_error_count == 1
        assert capture.blocks_captured == 1
        assert capture.level == pytest.approx(0.25)
    finally:
        capture.stop()


def test_callback_counts_portaudio_status_flags(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        factory.stream.feed(np.zeros(512, dtype=np.float32), status="input overflow")
        assert capture.overflow_count == 1
        assert capture.blocks_captured == 1  # the block is still kept
        assert capture.callback_error_count == 0
    finally:
        capture.stop()


def test_audio_level_events_are_rate_limited(capture_config: JarvisConfig, bus: Any) -> None:
    """One event per block at 32 ms would swamp the HUD and the audio thread."""
    seen: list[float] = []
    bus.subscribe(lambda event: seen.append(event.payload["level"]), [EventType.AUDIO_LEVEL])

    factory = RecordingFactory()
    capture = AudioCapture(capture_config, bus=bus, stream_factory=factory, level_interval_s=10.0)
    capture.start()
    try:
        for _ in range(20):
            factory.stream.feed(np.full(512, 0.5, dtype=np.float32))
        assert len(seen) == 1
        assert seen[0] == pytest.approx(0.5)
        assert capture.blocks_captured == 20
    finally:
        capture.stop()


def test_audio_level_events_are_emitted_when_the_interval_allows(
    capture_config: JarvisConfig, bus: Any
) -> None:
    seen: list[Any] = []
    bus.subscribe(seen.append, [EventType.AUDIO_LEVEL])

    factory = RecordingFactory()
    capture = AudioCapture(capture_config, bus=bus, stream_factory=factory, level_interval_s=0.0)
    capture.start()
    try:
        for _ in range(5):
            factory.stream.feed(np.full(512, 0.1, dtype=np.float32))
        assert len(seen) == 5
        assert seen[0].payload["source"] == "capture"
    finally:
        capture.stop()


def test_capture_without_a_bus_still_works(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        factory.stream.feed(np.full(512, 0.2, dtype=np.float32))
        assert capture.blocks_captured == 1
    finally:
        capture.stop()


def test_start_wraps_a_failing_factory_in_audio_error(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory(build_error=OSError("no such device"))
    capture = AudioCapture(capture_config, stream_factory=factory)
    with pytest.raises(AudioError, match="could not open the audio input device"):
        capture.start()
    assert not capture.is_running


def test_start_closes_the_stream_when_starting_it_fails(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory(fail_on_start=True)
    capture = AudioCapture(capture_config, stream_factory=factory)
    with pytest.raises(AudioError) as excinfo:
        capture.start()
    assert "could not start" in str(excinfo.value)
    assert excinfo.value.speakable
    assert factory.stream.closes == 1  # the half-open device was released
    assert not capture.is_running


def test_start_propagates_a_missing_dependency_unchanged(capture_config: JarvisConfig) -> None:
    """DependencyMissingError already carries an install hint, do not bury it."""
    factory = RecordingFactory(build_error=DependencyMissingError("sounddevice", extra="audio"))
    capture = AudioCapture(capture_config, stream_factory=factory)
    with pytest.raises(DependencyMissingError):
        capture.start()


def test_named_input_device_is_resolved_to_an_index(
    capture_config: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devices_module, "list_devices", lambda: list(FAKE_DEVICES))
    capture_config.audio.input_device = "Blue Yeti"
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        assert factory.stream.spec.device == 3
        assert capture.device is not None
        assert capture.device.name == "Microphone (Blue Yeti)"
    finally:
        capture.stop()


def test_unknown_input_device_fails_before_opening_a_stream(
    capture_config: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(devices_module, "list_devices", lambda: list(FAKE_DEVICES))
    capture_config.audio.input_device = "Nonexistent Array Mic"
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    with pytest.raises(AudioError, match="no input device matches"):
        capture.start()
    assert not factory.created
    assert not capture.is_running


def test_default_device_path_does_not_enumerate_devices(
    capture_config: JarvisConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Enumeration needs PortAudio; the default path must not require it."""

    def explode() -> list[DeviceInfo]:
        raise AssertionError("list_devices must not be called for an auto spec")

    monkeypatch.setattr(devices_module, "list_devices", explode)
    capture_config.audio.input_device = "auto"
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        assert factory.stream.spec.device is None
        assert capture.device is None
    finally:
        capture.stop()


def test_capture_readers_are_independent(capture_config: JarvisConfig) -> None:
    factory = RecordingFactory()
    capture = AudioCapture(capture_config, stream_factory=factory)
    capture.start()
    try:
        wake = capture.reader()
        vad = capture.reader()
        factory.stream.feed(ramp(0, 512))
        assert np.array_equal(wake.read(512), ramp(0, 512))
        assert vad.available == 512
        assert np.array_equal(vad.read(512), ramp(0, 512))
    finally:
        capture.stop()


def test_int16_capture_scales_the_level(cfg: JarvisConfig) -> None:
    cfg.audio.dtype = "int16"
    cfg.audio.block_samples = 512
    factory = RecordingFactory()
    capture = AudioCapture(cfg, stream_factory=factory)
    capture.start()
    try:
        assert factory.stream.spec.dtype == "int16"
        assert capture.ring.dtype == np.dtype(np.int16)
        factory.stream.feed(np.full(512, 16_384, dtype=np.int16))
        assert capture.level == pytest.approx(0.5)
    finally:
        capture.stop()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [None, "auto", "", "AUTO", " default "])
def test_auto_specs_are_recognised(spec: str | None) -> None:
    assert is_auto_spec(spec)


@pytest.mark.parametrize("spec", ["Blue Yeti", 3, 0])
def test_concrete_specs_are_not_auto(spec: str | int) -> None:
    assert not is_auto_spec(spec)


def test_auto_resolves_to_the_system_default() -> None:
    found = find_input_device(None, devices=FAKE_DEVICES)
    assert found is not None
    assert found.index == 1
    assert find_output_device("auto", devices=FAKE_DEVICES) == FAKE_DEVICES[2]


def test_auto_falls_back_to_the_first_usable_device_when_none_is_default() -> None:
    pool = [
        DeviceInfo(7, "Some Capture Card", 2, 0, 48000.0, 0),
        DeviceInfo(8, "Another Mic", 1, 0, 44100.0, 0),
    ]
    found = find_input_device(None, devices=pool)
    assert found is not None
    assert found.index == 7


def test_auto_returns_none_when_the_machine_has_no_input() -> None:
    outputs_only = [d for d in FAKE_DEVICES if not d.is_input]
    assert find_input_device(None, devices=outputs_only) is None


def test_exact_name_wins_over_case_insensitive_match() -> None:
    found = find_input_device("microphone (blue yeti)", devices=FAKE_DEVICES)
    assert found is not None
    assert found.index == 4


def test_case_insensitive_exact_match_is_next() -> None:
    found = find_input_device("MICROPHONE (BLUE YETI)", devices=FAKE_DEVICES)
    assert found is not None
    assert found.index == 3  # lowest index of the two case-insensitive matches


def test_substring_match_is_last() -> None:
    found = find_input_device("Realtek", devices=FAKE_DEVICES)
    assert found is not None
    assert found.index == 1


def test_substring_match_prefers_the_system_default() -> None:
    """Windows lists one microphone per host API; the default is the right one."""
    found = find_input_device("Microphone", devices=FAKE_DEVICES)
    assert found is not None
    assert found.index == 1


def test_name_match_is_scoped_to_the_direction() -> None:
    assert find_output_device("Realtek", devices=FAKE_DEVICES) == FAKE_DEVICES[2]
    assert find_output_device("Headset", devices=FAKE_DEVICES) == FAKE_DEVICES[5]


def test_leading_and_trailing_space_is_ignored() -> None:
    found = find_input_device("  Microphone (Blue Yeti)  ", devices=FAKE_DEVICES)
    assert found is not None
    assert found.index == 3


def test_unknown_name_raises_with_candidates() -> None:
    with pytest.raises(AudioError) as excinfo:
        find_input_device("Yamaha AG03", devices=FAKE_DEVICES)
    message = str(excinfo.value)
    assert "no input device matches" in message
    assert "Microphone" in message  # the message names real alternatives
    assert excinfo.value.speakable == "I could not find that microphone."
    assert excinfo.value.context["direction"] == "input"


def test_unknown_output_name_speaks_about_speakers() -> None:
    with pytest.raises(AudioError) as excinfo:
        find_output_device("Studio Monitors", devices=FAKE_DEVICES)
    assert excinfo.value.speakable == "I could not find that speaker."


def test_name_lookup_when_there_are_no_devices_at_all() -> None:
    with pytest.raises(AudioError, match="no input devices at all"):
        find_input_device("Anything", devices=[])


def test_index_spec_is_validated_against_the_device_list() -> None:
    found = find_input_device(3, devices=FAKE_DEVICES)
    assert found is not None
    assert found.name == "Microphone (Blue Yeti)"


def test_index_that_does_not_exist_is_refused() -> None:
    with pytest.raises(AudioError) as excinfo:
        find_input_device(99, devices=FAKE_DEVICES)
    assert "no audio device with index 99" in str(excinfo.value)
    assert excinfo.value.context["usable"] == [0, 1, 3, 4]


def test_index_pointing_at_the_wrong_direction_is_refused() -> None:
    with pytest.raises(AudioError, match="has no input channels"):
        find_input_device(2, devices=FAKE_DEVICES)
    with pytest.raises(AudioError, match="has no output channels"):
        find_output_device(1, devices=FAKE_DEVICES)


def test_boolean_spec_is_refused_rather_than_read_as_device_one() -> None:
    with pytest.raises(AudioError, match="must be a name, an index, or null"):
        find_input_device(True, devices=FAKE_DEVICES)


def test_device_info_direction_helpers() -> None:
    mic = FAKE_DEVICES[1]
    assert mic.is_input and not mic.is_output
    assert mic.supports("input") and not mic.supports("output")
    assert mic.is_default_for("input") and not mic.is_default_for("output")
    assert mic.label() == "1: Microphone (Realtek(R) Audio)"


# ---------------------------------------------------------------------------
# Device enumeration and the troubleshooting table
# ---------------------------------------------------------------------------


class FakeSounddevice:
    """The slice of the sounddevice module that ``list_devices`` touches."""

    def __init__(self, rows: list[dict[str, Any]], *, error: Exception | None = None) -> None:
        self._rows = rows
        self._error = error
        self.default = type("Default", (), {"device": [1, 2]})()

    def query_devices(self) -> list[dict[str, Any]]:
        if self._error is not None:
            raise self._error
        return self._rows

    def query_hostapis(self) -> list[dict[str, Any]]:
        return [{"name": "MME"}, {"name": "Windows WASAPI"}]


def test_list_devices_needs_sounddevice(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str, *, feature: str | None = None) -> Any:
        raise DependencyMissingError(name, extra="audio")

    monkeypatch.setattr(devices_module, "require_module", missing)
    with pytest.raises(DependencyMissingError) as excinfo:
        list_devices()
    assert "uv sync --extra audio" in str(excinfo.value)


def test_list_devices_parses_portaudio_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "name": "Sound Mapper",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
            "hostapi": 0,
        },
        {
            "name": "Microphone (Realtek)",
            "max_input_channels": 2,
            "max_output_channels": 0,
            "default_samplerate": 44100.0,
            "hostapi": 0,
        },
        {
            "name": "Speakers (Realtek)",
            "max_input_channels": 0,
            "max_output_channels": 2,
            "default_samplerate": 48000.0,
            "hostapi": 1,
        },
    ]
    fake = FakeSounddevice(rows)
    monkeypatch.setattr(devices_module, "require_module", lambda *a, **k: fake)

    found = list_devices()
    assert [d.index for d in found] == [0, 1, 2]
    assert found[1].is_default_input
    assert found[2].is_default_output
    assert not found[0].is_default_input
    assert found[2].max_output_channels == 2


def test_list_devices_tolerates_sparse_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSounddevice([{"name": "Mystery Device"}])
    monkeypatch.setattr(devices_module, "require_module", lambda *a, **k: fake)
    found = list_devices()
    assert found[0].name == "Mystery Device"
    assert found[0].max_input_channels == 0
    assert found[0].default_samplerate == 0.0


def test_list_devices_wraps_a_portaudio_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeSounddevice([], error=OSError("PortAudio host error"))
    monkeypatch.setattr(devices_module, "require_module", lambda *a, **k: fake)
    with pytest.raises(AudioError, match="could not enumerate audio devices"):
        list_devices()


def test_describe_devices_renders_a_table() -> None:
    table = describe_devices(FAKE_DEVICES, hostapi_names={0: "MME", 1: "WASAPI", 2: "DirectSound"})
    lines = table.splitlines()
    assert lines[0].split() == ["idx", "in", "out", "rate", "host", "api", "name"]
    assert len(lines) == len(FAKE_DEVICES) + 2  # header plus rule
    assert "[default in]" in table
    assert "[default out]" in table
    assert "WASAPI" in table
    assert "Microphone (Blue Yeti)" in table
    assert "44100" in table


def test_describe_devices_reports_an_empty_machine() -> None:
    assert describe_devices([]) == "no audio devices were found"


def test_describe_devices_never_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """This is what a user is told to run when audio is already broken."""

    def broken() -> list[DeviceInfo]:
        raise AudioError("PortAudio is not initialised")

    monkeypatch.setattr(devices_module, "list_devices", broken)
    text = describe_devices()
    assert "could not be listed" in text


# ---------------------------------------------------------------------------
# Manual, Windows host with a real microphone
# ---------------------------------------------------------------------------


@pytest.mark.manual
def test_manual_real_devices_include_a_microphone() -> None:
    """Windows host: enumerate real devices and confirm a capture device exists.

    Check: ``uv run pytest tests/test_ring.py -m manual`` on the Windows box with
    the audio extra installed. Skips where PortAudio is absent, since there is
    nothing to verify there.
    """
    if not has_module("sounddevice"):
        pytest.skip("sounddevice is not installed, run this on the Windows target")

    found = list_devices()
    assert found, "PortAudio reported no devices at all"
    assert any(device.is_input for device in found), describe_devices(found)
    assert any(device.is_output for device in found), describe_devices(found)


@pytest.mark.manual
def test_manual_real_microphone_capture(cfg: JarvisConfig) -> None:
    """Windows host: capture one second from the default microphone.

    Check: ``uv run pytest tests/test_ring.py -m manual`` on the Windows box with
    the audio extra installed, speaking while it runs. It asserts that blocks
    arrived, that the ring filled, and that the callback never errored, which is
    the whole T-1.1 contract on real hardware.
    """
    if not has_module("sounddevice"):
        pytest.skip("sounddevice is not installed, run this on the Windows target")

    capture = AudioCapture(cfg)
    with capture:
        reader = capture.reader()
        time.sleep(1.0)
        assert capture.is_running
        heard = reader.read_available()

    expected_blocks = cfg.audio.sample_rate // cfg.audio.block_samples // 2
    assert capture.blocks_captured >= expected_blocks
    assert capture.callback_error_count == 0
    assert heard.size >= cfg.audio.sample_rate // 2
    assert capture.ring.filled == capture.ring.capacity
    assert rms_level(heard) > 0.0, "the microphone delivered pure digital silence"
