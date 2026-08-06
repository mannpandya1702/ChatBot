"""Generate the HUD application icons.

``npm run tauri build`` fails outright without ``icon.png`` and ``icon.ico``, so
these have to exist before the bundle can be produced at all. Rather than commit
opaque binaries with no way to regenerate them, this script draws them from the
accent colour in ``config.example.yaml``: a soft arc-reactor ring on a
transparent field, matching the orb the HUD renders.

Stdlib only, deliberately. Pillow would be a build dependency carried forever
for two files that change approximately never.

Run with::

    uv run python scripts/make_icons.py
"""

from __future__ import annotations

import argparse
import math
import struct
import sys
import zlib
from pathlib import Path

#: Matches ``ui.accent_color`` in config.example.yaml.
ACCENT = (0x22, 0xD3, 0xEE)

#: Sizes Windows expects inside a .ico. 256 is what File Explorer shows at large
#: icon size; 16 is the title bar.
ICO_SIZES = (16, 32, 48, 64, 128, 256)

Pixel = tuple[int, int, int, int]


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """Hermite interpolation, used to keep every edge anti-aliased."""
    if edge0 == edge1:
        return 0.0 if x < edge0 else 1.0
    t = min(1.0, max(0.0, (x - edge0) / (edge1 - edge0)))
    return t * t * (3.0 - 2.0 * t)


def _pixel(x: int, y: int, size: int) -> Pixel:
    """Colour one pixel of the icon.

    The design is three concentric elements so it stays legible at 16 pixels:
    a filled core, a gap, and a bright ring. Below about 32 pixels the ring and
    core merge visually into a single dot, which is the intended fallback.
    """
    half = size / 2.0
    # Sample at pixel centres so the figure is symmetric.
    dx = (x + 0.5 - half) / half
    dy = (y + 0.5 - half) / half
    radius = math.hypot(dx, dy)

    core = 1.0 - _smoothstep(0.14, 0.26, radius)
    ring = _smoothstep(0.52, 0.60, radius) - _smoothstep(0.74, 0.84, radius)
    # A faint halo so the icon does not look cut out against dark themes. Kept
    # low: any stronger and it fills the gap between core and ring, and the
    # whole thing reads as one flat disc.
    halo = (1.0 - _smoothstep(0.10, 0.85, radius)) * 0.06

    intensity = min(1.0, core + ring + halo)
    if intensity <= 0.0:
        return (0, 0, 0, 0)

    # The core is lifted toward white but stays clearly cyan, so it reads as
    # a distinct element against the halo rather than dissolving into it.
    whiteness = core * 0.30
    red = round(ACCENT[0] + (255 - ACCENT[0]) * whiteness)
    green = round(ACCENT[1] + (255 - ACCENT[1]) * whiteness)
    blue = round(ACCENT[2] + (255 - ACCENT[2]) * whiteness)
    return (red, green, blue, round(255 * intensity))


def render(size: int) -> list[list[Pixel]]:
    """Render the icon at one size as rows of RGBA pixels."""
    return [[_pixel(x, y, size) for x in range(size)] for y in range(size)]


def _png_chunk(tag: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + tag
        + payload
        + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
    )


def to_png(rows: list[list[Pixel]]) -> bytes:
    """Encode RGBA rows as a PNG.

    Every scanline uses filter type 0. The image is small and mostly flat, so
    the filtering that a real encoder would choose buys very little here.
    """
    size = len(rows)
    raw = bytearray()
    for row in rows:
        raw.append(0)
        for red, green, blue, alpha in row:
            raw += bytes((red, green, blue, alpha))

    header = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + _png_chunk(b"IEND", b"")
    )


def _to_dib(rows: list[list[Pixel]]) -> bytes:
    """Encode RGBA rows as a 32-bit BMP/DIB for embedding in a .ico.

    ICO stores the DIB bottom-up, with the height field doubled to account for
    the AND mask. The mask is still required even for 32-bit entries: some
    Windows shell paths read it, and a missing one renders as a black box.
    """
    size = len(rows)
    pixels = bytearray()
    for row in reversed(rows):
        for red, green, blue, alpha in row:
            pixels += bytes((blue, green, red, alpha))

    # 1 bit per pixel, each row padded to a 4-byte boundary. All zero means
    # "opaque everywhere" and lets the alpha channel decide.
    mask_row_bytes = ((size + 31) // 32) * 4
    mask = bytes(mask_row_bytes * size)

    header = struct.pack(
        "<IiiHHIIiiII",
        40,          # header size
        size,        # width
        size * 2,    # height, doubled for the mask
        1,           # planes
        32,          # bits per pixel
        0,           # BI_RGB, uncompressed
        len(pixels) + len(mask),
        0, 0, 0, 0,  # resolution and palette fields, unused
    )
    return header + bytes(pixels) + mask


def to_ico(sizes: tuple[int, ...]) -> bytes:
    """Build a multi-resolution .ico.

    Entries up to 128 pixels are stored as DIBs, which every Windows version
    reads. The 256 pixel entry is stored as PNG, which is how it has been done
    since Vista and keeps the file from tripling in size.
    """
    images: list[bytes] = []
    for size in sizes:
        rows = render(size)
        images.append(to_png(rows) if size >= 256 else _to_dib(rows))

    offset = 6 + 16 * len(sizes)
    directory = struct.pack("<HHH", 0, 1, len(sizes))
    for size, blob in zip(sizes, images, strict=True):
        directory += struct.pack(
            "<BBBBHHII",
            0 if size >= 256 else size,  # 0 means 256
            0 if size >= 256 else size,
            0,   # no colour palette
            0,   # reserved
            1,   # planes
            32,  # bits per pixel
            len(blob),
            offset,
        )
        offset += len(blob)
    return directory + b"".join(images)


def main(argv: list[str] | None = None) -> int:
    """Write icon.png and icon.ico into the Tauri icons directory."""
    default = Path(__file__).resolve().parents[1] / "src/jarvis/ui/app/src-tauri/icons"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=default, help="Directory to write the icons into."
    )
    parser.add_argument(
        "--size", type=int, default=512, help="Edge length of icon.png in pixels."
    )
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    png_path = args.out / "icon.png"
    ico_path = args.out / "icon.ico"

    png_path.write_bytes(to_png(render(args.size)))
    ico_path.write_bytes(to_ico(ICO_SIZES))

    print(f"wrote {png_path} ({args.size}x{args.size}, {png_path.stat().st_size} bytes)")
    print(
        f"wrote {ico_path} ({', '.join(str(s) for s in ICO_SIZES)}, "
        f"{ico_path.stat().st_size} bytes)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
