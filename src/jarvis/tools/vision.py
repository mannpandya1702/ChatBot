"""Screen understanding via a local vision model.

T-4.5: screenshot with ``mss``, send to a local VLM through Ollama, answer
"what's on my screen". Read-only.

Two constraints shape this. First, §0.1: the image never leaves the machine, it
goes to Ollama on localhost. Second, §2: the VLM needs roughly 6 GB of free
VRAM, so on the cpu and gpu-6 tiers the tool reports that it cannot run rather
than thrashing or inventing an answer.
"""

from __future__ import annotations

import base64
import io
import logging

from pydantic import Field

from jarvis.config import get_config
from jarvis.tools.registry import ToolCategory, ToolInput, ToolOutput, tool
from jarvis.util.errors import LlmError, ToolExecutionError
from jarvis.util.platform import require_module

__all__ = ["VisionInput", "VisionOutput", "capture_screen", "describe_screen"]

_log = logging.getLogger(__name__)

#: Longest edge of the image sent to the model. Full 4K frames waste tokens and
#: add latency without improving the answer.
_MAX_EDGE_PX = 1280
_JPEG_QUALITY = 80


class VisionInput(ToolInput):
    """Arguments for the screen description tool."""

    question: str = Field(
        default="Describe what is on this screen in one or two sentences.",
        max_length=500,
        description=(
            "What to ask about the screen. Defaults to a general description. Use a specific "
            "question when the user asked one, for example what does this error say."
        ),
    )
    monitor: int = Field(
        default=0,
        ge=0,
        le=8,
        description="Which monitor to capture. Zero means all monitors combined.",
    )


class VisionOutput(ToolOutput):
    """What the vision model saw, or why it could not look."""

    available: bool = Field(
        description="False when the vision tool is disabled or the tier cannot host the model."
    )
    reason: str | None = Field(default=None, description="Why the tool did not run.")
    answer: str = Field(default="", description="The model's description of the screen.")
    model: str | None = Field(default=None, description="Which vision model answered.")
    width: int | None = Field(default=None, description="Captured image width in pixels.")
    height: int | None = Field(default=None, description="Captured image height in pixels.")


def capture_screen(monitor: int = 0) -> tuple[bytes, int, int]:
    """Grab the screen as JPEG bytes.

    Args:
        monitor: Monitor index. Zero captures every monitor as one image.

    Returns:
        ``(jpeg_bytes, width, height)``.

    Raises:
        ToolExecutionError: The screen could not be captured.
    """
    mss_module = require_module("mss", feature="screen capture")

    try:
        with mss_module.mss() as sct:
            monitors = sct.monitors
            index = monitor if 0 <= monitor < len(monitors) else 0
            shot = sct.grab(monitors[index])
            width, height = shot.width, shot.height
            raw = bytes(shot.rgb)
    except Exception as exc:
        _log.exception("screen capture failed")
        raise ToolExecutionError(
            "vision.screen",
            f"could not capture the screen: {exc}",
            speakable="I could not see the screen.",
        ) from exc

    return _encode_jpeg(raw, width, height), width, height


def _encode_jpeg(rgb: bytes, width: int, height: int) -> bytes:
    """Downscale and JPEG-encode a raw RGB frame.

    Pillow is not a declared dependency, so this degrades to a raw BMP when it
    is absent. Ollama accepts either, and the BMP path only costs bandwidth over
    a loopback socket.
    """
    try:
        from PIL import Image
    except ImportError:
        _log.debug("Pillow is not installed, sending an uncompressed capture")
        return _encode_bmp(rgb, width, height)

    image = Image.frombytes("RGB", (width, height), rgb)
    longest = max(image.width, image.height)
    if longest > _MAX_EDGE_PX:
        scale = _MAX_EDGE_PX / longest
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.LANCZOS,
        )
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return buffer.getvalue()


def _encode_bmp(rgb: bytes, width: int, height: int) -> bytes:
    """Wrap raw RGB in a minimal BMP container."""
    import struct

    row_padded = (width * 3 + 3) & ~3
    pixel_bytes = row_padded * height
    header = struct.pack("<2sIHHI", b"BM", 14 + 40 + pixel_bytes, 0, 0, 14 + 40)
    info = struct.pack("<IiiHHIIiiII", 40, width, -height, 1, 24, 0, pixel_bytes, 0, 0, 0, 0)

    rows: list[bytes] = []
    padding = b"\x00" * (row_padded - width * 3)
    for y in range(height):
        start = y * width * 3
        row = rgb[start : start + width * 3]
        # BMP stores BGR, the capture is RGB.
        rows.append(_swap_rgb(row) + padding)
    return header + info + b"".join(rows)


def _swap_rgb(row: bytes) -> bytes:
    """Convert an RGB row to BGR."""
    out = bytearray(row)
    out[0::3], out[2::3] = row[2::3], row[0::3]
    return bytes(out)


@tool(
    name="vision.screen",
    description=(
        "Look at what is currently on the screen and answer a question about it. Use this "
        "for requests like what is on my screen, what does this error say, or read this for "
        "me. The screenshot is analysed by a model running on this machine and never leaves "
        "it. Needs about 6 gigabytes of free video memory, so it reports that it is "
        "unavailable on smaller graphics cards. Read-only."
    ),
    category=ToolCategory.VISION,
    read_only=True,
    is_enabled=lambda config: config.vision_available(),
)
def describe_screen(params: VisionInput) -> VisionOutput:
    """Describe the screen using the local vision model.

    Args:
        params: The question and which monitor to capture.

    Returns:
        The model's answer, or ``available=False`` with a reason.

    Raises:
        ToolExecutionError: The capture failed.
    """
    config = get_config()

    if not config.tools.enable_vision:
        return VisionOutput(
            available=False, reason="the vision tool is turned off in the configuration"
        )
    if not config.vision_available():
        return VisionOutput(
            available=False,
            reason=(
                "this graphics card does not have enough spare video memory for the vision "
                "model, which needs about 6 gigabytes"
            ),
        )

    image, width, height = capture_screen(params.monitor)
    encoded = base64.b64encode(image).decode("ascii")

    from jarvis.brain.llm import OllamaClient

    model = config.tools.vision_model
    try:
        with OllamaClient(config) as client:
            try:
                answer = client.vision(params.question, encoded, model=model)
            except LlmError as exc:
                # The primary VLM may not be pulled. Try the smaller fallback
                # before giving up, since it is a fraction of the size.
                fallback = config.tools.vision_fallback_model
                if fallback and fallback != model:
                    _log.info(
                        "vision model unavailable, trying the fallback",
                        extra={
                            "context": {
                                "model": model,
                                "fallback": fallback,
                                "error": str(exc),
                            }
                        },
                    )
                    answer = client.vision(params.question, encoded, model=fallback)
                    model = fallback
                else:
                    raise
    except LlmError as exc:
        return VisionOutput(
            available=False,
            reason=exc.speakable,
            width=width,
            height=height,
        )

    return VisionOutput(
        available=True,
        answer=answer.strip(),
        model=model,
        width=width,
        height=height,
    )
