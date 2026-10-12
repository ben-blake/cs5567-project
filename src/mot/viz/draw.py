"""Drawing tracked boxes and writing result videos."""

from __future__ import annotations

import colorsys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """A stable, well-separated colour per track id.

    The golden-ratio hue step keeps consecutive ids visually distinct instead
    of shading into one another.
    """
    hue = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def _font(size: int = 16) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_tracks(
    image: Image.Image,
    boxes: np.ndarray,
    track_ids: list[int],
    scores: list[float] | None = None,
    highlight_id: int | None = None,
    width: int = 3,
) -> Image.Image:
    """Overlay boxes and id labels on a copy of ``image``.

    ``highlight_id`` dims every other track, which is how we show the
    "single target pedestrian" the assignment asks for.
    """
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    font = _font()

    for index, (box, track_id) in enumerate(zip(boxes, track_ids, strict=True)):
        x1, y1, x2, y2 = (float(v) for v in box)
        is_target = highlight_id is None or track_id == highlight_id

        color = color_for_id(track_id) if is_target else (110, 110, 110)
        thickness = width + 2 if (highlight_id is not None and is_target) else width
        draw.rectangle([x1, y1, x2, y2], outline=color, width=thickness)

        label = f"ID {track_id}"
        if scores is not None and index < len(scores):
            label += f" {scores[index]:.2f}"

        text_box = draw.textbbox((0, 0), label, font=font)
        tw, th = text_box[2] - text_box[0], text_box[3] - text_box[1]
        # Keep the label on-screen when the box touches the top edge.
        ty = max(0.0, y1 - th - 6)
        draw.rectangle([x1, ty, x1 + tw + 8, ty + th + 6], fill=color)
        draw.text((x1 + 4, ty + 3), label, fill=(255, 255, 255), font=font)

    return canvas


def draw_frame_label(image: Image.Image, text: str) -> Image.Image:
    """Stamp a caption in the top-right (frame number, track count)."""
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)
    font = _font(18)
    box = draw.textbbox((0, 0), text, font=font)
    w, h = box[2] - box[0], box[3] - box[1]
    x = canvas.width - w - 16
    draw.rectangle([x - 8, 8, x + w + 8, h + 20], fill=(0, 0, 0))
    draw.text((x, 12), text, fill=(255, 255, 255), font=font)
    return canvas


#: Tried in order; which ones exist depends on how OpenCV was built, so we
#: probe rather than assume. mp4v is usually present on Colab, avc1 on macOS.
_MP4_CODECS = ("mp4v", "avc1", "H264")
_AVI_CODECS = ("MJPG", "XVID")


def _fourcc(codec: str) -> int:
    """Codec code, across OpenCV versions.

    OpenCV 4 exposes ``cv2.VideoWriter_fourcc``; OpenCV 5 moved it to
    ``cv2.VideoWriter.fourcc``. Colab currently ships 4.x and recent desktop
    wheels ship 5.x, so we support both.
    """
    import cv2

    factory = getattr(cv2, "VideoWriter_fourcc", None)
    if factory is None:
        factory = cv2.VideoWriter.fourcc
    return int(factory(*codec))


def write_video(
    frames: list[Image.Image] | list[Path],
    output_path: str | Path,
    fps: int = 30,
) -> Path:
    """Encode frames to a video file.

    Frames may be PIL images or paths to images on disk. OpenCV reports a
    missing codec by returning a writer that is simply not open and then
    silently discarding every frame, so we check ``isOpened()`` and move on to
    the next candidate instead of producing a zero-byte file.
    """
    import cv2

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not frames:
        raise ValueError("no frames to write")

    def as_bgr(frame):
        image = Image.open(frame) if isinstance(frame, (str, Path)) else frame
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)

    first = as_bgr(frames[0])
    height, width = first.shape[:2]

    candidates = _MP4_CODECS if output_path.suffix.lower() == ".mp4" else _AVI_CODECS
    attempted = []

    for codec in candidates:
        writer = cv2.VideoWriter(str(output_path), _fourcc(codec), fps, (width, height))
        if not writer.isOpened():
            writer.release()
            attempted.append(codec)
            continue

        try:
            writer.write(first)
            for frame in frames[1:]:
                writer.write(as_bgr(frame))
        finally:
            writer.release()

        if output_path.is_file() and output_path.stat().st_size > 0:
            return output_path
        attempted.append(codec)

    raise RuntimeError(
        f"could not encode {output_path.name}: no working codec among {attempted}. "
        "Install a fuller OpenCV build, or write .avi instead of .mp4."
    )


def contact_sheet(
    images: list[Image.Image], columns: int = 4, thumb_width: int = 320
) -> Image.Image:
    """Tile frames into one image -- useful as a static slide figure."""
    if not images:
        raise ValueError("no images")

    scale = thumb_width / images[0].width
    thumb_height = int(images[0].height * scale)
    rows = (len(images) + columns - 1) // columns

    sheet = Image.new("RGB", (columns * thumb_width, rows * thumb_height), (20, 20, 20))
    for index, image in enumerate(images):
        thumb = image.convert("RGB").resize((thumb_width, thumb_height))
        sheet.paste(thumb, ((index % columns) * thumb_width, (index // columns) * thumb_height))
    return sheet
