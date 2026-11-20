"""Synthetic camera frames for the simulated robots, encoded once as JPEG.

Builds the fixed frame set that every robot cycles through, and summarizes the
sizes of its frames so that a run can record what it sent.
"""

from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

DEFAULT_FRAME_COUNT = 32
DEFAULT_JPEG_QUALITY = 80
DEFAULT_SEED = 0

# ASSUMPTION: A23. Frames are synthetic and the set is fixed. Their content sets
# the JPEG size, and the JPEG size sets the transport and decode cost of every
# frame. The scene constants below are therefore a design choice: they aim at a
# mean between 25 KB and 60 KB at quality 80, a realistic size for a 640x480
# camera frame. FrameSet.stats() reports the sizes actually built, and every run
# records them.
_HORIZON_ROW_PX = 250
_WALL_COLOUR_BGR = (176, 184, 190)
_FLOOR_COLOUR_BGR = (96, 108, 120)
_TEXTURE_SPECTRUM_EXPONENT = 1.5
_TEXTURE_MAX_WAVELENGTH_PX = 160.0
# Levels are steps of 8-bit intensity, 0 to 255.
_TEXTURE_SIGMA_LEVELS = 16.0
_SENSOR_NOISE_SIGMA_LEVELS = 3.0
_OBJECT_COUNT = 6
_OBJECT_SIZE_MIN_PX = 48.0
_OBJECT_SIZE_MAX_PX = 144.0
_ORBIT_RADIUS_MIN_PX = 40.0


class FrameEncodeError(Exception):
    """The JPEG encoder failed on a rendered frame."""


@dataclass(frozen=True)
class FrameSetStats:
    """Sizes of the encoded frames of one frame set."""

    count: int
    jpeg_quality: int
    min_bytes: int
    mean_bytes: float
    max_bytes: int


@dataclass(frozen=True)
class FrameSet:
    """The pre-encoded frames that every robot cycles through."""

    jpegs: tuple[bytes, ...]
    jpeg_quality: int

    def stats(self) -> FrameSetStats:
        sizes = [len(jpeg) for jpeg in self.jpegs]
        return FrameSetStats(
            count=len(sizes),
            jpeg_quality=self.jpeg_quality,
            min_bytes=min(sizes),
            mean_bytes=sum(sizes) / len(sizes),
            max_bytes=max(sizes),
        )


def build_frame_set(count: int, seed: int, jpeg_quality: int) -> FrameSet:
    """Render and encode the frame set; equal arguments give equal bytes within one process."""
    if count < 1:
        raise ValueError(f"count must be at least 1, got {count}")
    if seed < 0:
        raise ValueError(f"seed must not be negative, got {seed}")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError(f"jpeg_quality must be between 1 and 100, got {jpeg_quality}")
    jpegs: list[bytes] = []
    for index, frame in enumerate(_render_frames(count, np.random.default_rng(seed))):
        # ASSUMPTION: A5. Each frame is encoded once, here at startup, and robots
        # cycle through the result. A real robot encodes on its own hardware, and
        # encoding at 30 fps per robot inside the container's 2 CPUs would measure
        # the simulator. Reuse is sound because the frame number and capture time
        # travel in the message header, not inside the JPEG (A4).
        encoded_ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not encoded_ok:
            raise FrameEncodeError(f"JPEG encoding failed for frame {index}")
        jpegs.append(encoded.tobytes())
    return FrameSet(tuple(jpegs), jpeg_quality)


def _render_frames(count: int, rng: np.random.Generator) -> Iterator[npt.NDArray[np.uint8]]:
    """Yield the BGR frames of one scene: a static background, moving objects, sensor noise."""
    background = _render_background(rng)
    frame_size_px = np.array((FRAME_WIDTH, FRAME_HEIGHT))
    # Orbit centers in the middle half of the frame, with orbit radii of at most a
    # quarter of it, keep the center of every object in view in every frame.
    orbit_centers_px = rng.uniform(
        0.25 * frame_size_px, 0.75 * frame_size_px, size=(_OBJECT_COUNT, 2)
    )
    orbit_radii_px = rng.uniform(
        _ORBIT_RADIUS_MIN_PX, 0.25 * frame_size_px, size=(_OBJECT_COUNT, 2)
    )
    orbit_phases_rad = rng.uniform(0.0, 2.0 * np.pi, size=_OBJECT_COUNT)
    object_sizes_px = rng.uniform(
        _OBJECT_SIZE_MIN_PX, _OBJECT_SIZE_MAX_PX, size=(_OBJECT_COUNT, 2)
    ).tolist()
    object_colours_bgr = rng.integers(0, 256, size=(_OBJECT_COUNT, 3)).tolist()
    # Each object makes one revolution over the whole set, so when a robot cycles,
    # the step from the last frame back to the first is a step like any other.
    angles_rad = 2.0 * np.pi * np.arange(count)[:, np.newaxis] / count + orbit_phases_rad
    directions = np.stack((np.cos(angles_rad), np.sin(angles_rad)), axis=-1)
    object_centers_px = (orbit_centers_px + orbit_radii_px * directions).tolist()
    for frame_centers_px in object_centers_px:
        frame = background.copy()
        for center_px, size_px, colour_bgr in zip(
            frame_centers_px, object_sizes_px, object_colours_bgr, strict=True
        ):
            # The box form of ellipse() takes sub-pixel coordinates, so objects also
            # move between neighbouring frames of a large set, where steps are small.
            cv2.ellipse(frame, (center_px, size_px, 0.0), colour_bgr, cv2.FILLED, cv2.LINE_AA)
        noise = rng.standard_normal(frame.shape, dtype=np.float32)
        yield np.clip(np.rint(frame + _SENSOR_NOISE_SIGMA_LEVELS * noise), 0, 255).astype(np.uint8)


def _render_background(rng: np.random.Generator) -> npt.NDArray[np.uint8]:
    """Render the static part of the scene: a wall above a floor, both with surface texture."""
    cycles_per_px = np.hypot(
        np.fft.rfftfreq(FRAME_WIDTH)[np.newaxis, :], np.fft.fftfreq(FRAME_HEIGHT)[:, np.newaxis]
    )
    # A power-law amplitude spectrum puts detail at every scale, like the surfaces
    # a camera sees. A steeper exponent moves detail to coarser scales and shrinks
    # the JPEG. The longest wavelengths, and the mean with them, get no amplitude:
    # they are few terms and would carry most of the contrast, so their random
    # strength would decide how much is left for fine detail, and the JPEG size
    # would swing with the seed.
    kept = cycles_per_px >= 1.0 / _TEXTURE_MAX_WAVELENGTH_PX
    amplitude = np.zeros_like(cycles_per_px)
    amplitude[kept] = cycles_per_px[kept] ** -_TEXTURE_SPECTRUM_EXPONENT
    white_spectrum = rng.standard_normal(kept.shape) + 1j * rng.standard_normal(kept.shape)
    texture = np.fft.irfft2(amplitude * white_spectrum, s=(FRAME_HEIGHT, FRAME_WIDTH))
    texture *= _TEXTURE_SIGMA_LEVELS / texture.std()
    rows = np.arange(FRAME_HEIGHT)[:, np.newaxis, np.newaxis]
    base = np.where(rows < _HORIZON_ROW_PX, _WALL_COLOUR_BGR, _FLOOR_COLOUR_BGR)
    return np.clip(np.rint(base + texture[:, :, np.newaxis]), 0, 255).astype(np.uint8)
