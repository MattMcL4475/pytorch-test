#!/usr/bin/env python3
"""Generate a deep Mandelbrot zoom as HD video (and optionally GIF).

Features:
    * PyTorch accelerated (GPU if available; falls back to CPU).
    * 1080p (1920x1080) default aspect-correct render (no stretching).
    * Adaptive iteration budget that increases with depth.
    * Specify either a per-frame zoom factor OR target final scale.
    * MP4 (H.264) output via imageio-ffmpeg; optional GIF for quick preview.

Examples:
    # Produce a 15 second 60 FPS deep zoom MP4 (~900 frames) ending at final scale 1e-9
    python test.py --frames 900 --fps 60 --final-scale 1e-9 --video-out zoom.mp4

    # Shallow quick test (first frame only)
    python test.py --dry-run

Default runtime profile (tuned for ~2 minutes on a high-end NVIDIA GPU):
    * 3600 frames @ 30 FPS => 120 seconds finished video.
    * final_scale=1e-12 yields very deep magnification (extreme detail).
    * Base max_iter kept modest; adaptive growth increases iterations for deep frames.

To shorten runtime:
    - Lower --frames (video length = frames / fps).
    - Use a larger --final-scale (e.g., 1e-9) for a shallower zoom.
    - Reduce resolution (e.g., 1280x720) or maximize GPU usage with --device cuda.

To lengthen / go deeper:
    - Increase --frames OR decrease --final-scale.
    - Raise --max-it cautiously (iteration cost grows with depth multiplier).

Potential future option (not yet implemented): --target-seconds to auto-derive frame count.

Key parameters:
    --frames        Number of frames.
    --width/--height  Output resolution (default 1920x1080).
    --max-it        Baseline max iterations (scaled automatically with depth).
    --zoom          Per-frame zoom factor (<1 zooms in) (mutually exclusive with --final-scale).
    --final-scale   Target final half-width of view; derive zoom automatically.
    --center        Complex plane center (real,imag).
    --colormap      Coloring mode: log|linear.
    --fps           Output video frames per second.
    --video-out     MP4 filename (omit to disable video export).
    --gif-out       GIF filename (omit to skip GIF).
    --device        cpu|cuda|auto.
    --dry-run       Generate a single frame and exit.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass
from typing import Any  # (Optional retained types removed in favor of PEP 604 unions)

import imageio.v2 as imageio  # type: ignore
import torch


@dataclass
class MandelbrotConfig:
    # Defaults tuned for an approximate 2-minute render on a high-end NVIDIA GPU (e.g., B200):
    #  - 3600 frames at 30 FPS => 120 seconds of video
    #  - final_scale drives zoom depth; 1e-12 provides a very deep dive
    #  - iteration count scales adaptively; base max_iter kept moderate for early frames
    frames: int = 3600
    width: int = 1920
    height: int = 1080
    max_iter: int = 400  # Lower base; adaptive growth will raise this substantially in deeper frames
    zoom_factor: float | None = None  # If None and final_scale provided, compute dynamically
    final_scale: float | None = 1e-12  # Deep zoom target half-width
    center: complex = complex(-0.743643887037151, 0.13182590420533)
    colormap: str = "log"
    device: str = "auto"
    output_dir: str = "/data/mattmcl"
    video_out: str | None = "mandelbrot_zoom.mp4"  # relative to output_dir if not absolute
    gif_out: str | None = None  # relative to output_dir if not absolute
    fps: int = 30
    dry_run: bool = False


def select_device(preference: str) -> torch.device:
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_grid(width: int, height: int, center: complex, scale: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    # Maintain aspect ratio: scale refers to half-width; derive half-height
    aspect = height / width
    half_w = scale
    half_h = scale * aspect
    xs = torch.linspace(-half_w, half_w, width, device=device)
    ys = torch.linspace(-half_h, half_h, height, device=device)
    x, y = torch.meshgrid(xs, ys, indexing="xy")
    real = x + center.real
    imag = y + center.imag
    return real, imag


def mandelbrot(real: torch.Tensor, imag: torch.Tensor, max_iter: int) -> torch.Tensor:
    # Initialize z = 0
    zr = torch.zeros_like(real)
    zi = torch.zeros_like(imag)
    # Iteration counts
    iters = torch.zeros(real.shape, dtype=torch.int32, device=real.device)
    mask = torch.ones_like(real, dtype=torch.bool)

    for i in range(max_iter):
        # z = z^2 + c (where c = real + i*imag)
        # (zr + i*zi)^2 = (zr^2 - zi^2) + i*(2*zr*zi)
        zr2 = zr * zr
        zi2 = zi * zi
        two_zr_zi = 2.0 * zr * zi
        zr_new = zr2 - zi2 + real
        zi_new = two_zr_zi + imag
        zr, zi = zr_new, zi_new

        # magnitude squared
        mag2 = zr2 + zi2
        escaped = mag2 > 4.0
        newly_escaped = escaped & mask
        iters[newly_escaped] = i
        mask = mask & (~escaped)
        if not mask.any():  # all points escaped
            break

    # Points that never escaped get max_iter
    iters[mask] = max_iter
    return iters


def normalize_iterations(iters: torch.Tensor, max_iter: int, mode: str) -> torch.Tensor:
    iters_f = iters.to(torch.float32)
    if mode == "log":
        # Smooth coloring: log scale
        return torch.log1p(iters_f) / math.log1p(max_iter)
    return iters_f / max_iter


def palette_map(norm: torch.Tensor) -> torch.Tensor:
    # Simple gradient: convert scalar field to RGB
    # Try a blue -> cyan -> yellow -> white mapping
    r = torch.clamp(3.0 * norm - 1.5, 0.0, 1.0)
    g = torch.clamp(3.0 * norm - 0.5, 0.0, 1.0)
    b = torch.clamp(3.0 * norm + 0.5, 0.0, 1.0)
    rgb = torch.stack([r, g, b], dim=-1)
    return (rgb * 255).to(torch.uint8)


def compute_zoom_factor(frames: int, final_scale: float, base_scale: float) -> float:
    # Solve: base_scale * zf**(frames-1) = final_scale -> zf = (final_scale/base_scale)**(1/(frames-1))
    # Guard against pathological input
    if frames < 2:
        return 1.0
    return (final_scale / base_scale) ** (1.0 / (frames - 1))


def generate_frame(cfg: MandelbrotConfig, frame_index: int, device: torch.device, base_scale: float, zoom_factor: float) -> torch.Tensor:
    scale = base_scale * (zoom_factor ** frame_index)
    # Iteration scaling: deeper => more iterations; power schedule
    depth_multiplier = (base_scale / scale)
    # Adjust growth exponent slightly downward for very large frame counts to stabilize runtime
    iter_growth = min(25.0, depth_multiplier ** 0.25)
    max_iter = int(cfg.max_iter * iter_growth)
    real, imag = generate_grid(cfg.width, cfg.height, cfg.center, scale, device)
    iters = mandelbrot(real, imag, max_iter=max_iter)
    norm = normalize_iterations(iters, max_iter, cfg.colormap)
    rgb = palette_map(norm)
    interior_mask = iters == max_iter
    if interior_mask.any():
        rgb[interior_mask] = 0
    return rgb.cpu()


def write_gif(frames, outfile: str, fps: int = 15):
    imageio.mimsave(outfile, frames, format="GIF", fps=fps)


def resolved_path(base_dir: str, path: str | None) -> str | None:
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.join(base_dir, path)


def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def write_video_stream(cfg: MandelbrotConfig, device: torch.device, base_scale: float, zoom_factor: float):
    assert cfg.video_out is not None
    import imageio
    video_path = resolved_path(cfg.output_dir, cfg.video_out)
    writer = imageio.get_writer(video_path, fps=cfg.fps, codec="libx264", quality=8, pixelformat="yuv420p")
    try:
        for i in range(cfg.frames):
            frame = generate_frame(cfg, i, device, base_scale, zoom_factor)
            writer.append_data(frame.numpy())
            if (i + 1) % max(1, cfg.fps) == 0 or i == 0:
                print(f"Frame {i+1}/{cfg.frames} (zoom scale={(base_scale * (zoom_factor ** i)):.3e})")
    finally:
        writer.close()
    print(f"Saved video to {video_path}")


def parse_args() -> MandelbrotConfig:
    parser = argparse.ArgumentParser(description="Generate a deep Mandelbrot zoom video using PyTorch.")
    parser.add_argument("--frames", type=int, default=3600)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--max-it", type=int, default=400)
    parser.add_argument("--zoom", type=float, default=None, help="Per-frame zoom factor (<1 zooms in). Mutually exclusive with --final-scale.")
    parser.add_argument("--final-scale", type=float, default=1e-12, help="Target final half-width of view; overrides --zoom if provided.")
    parser.add_argument("--center", type=str, default="-0.743643887037151,0.13182590420533")
    parser.add_argument("--colormap", type=str, default="log", choices=["log", "linear"])
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--output-dir", type=str, default="/data/mattmcl", help="Base directory for outputs (will be created if missing).")
    parser.add_argument("--video-out", type=str, default="mandelbrot_zoom.mp4", help="MP4 output filename (relative to output-dir unless absolute). Use empty to disable.")
    parser.add_argument("--gif-out", type=str, default=None, help="Optional GIF output filename (relative to output-dir unless absolute).")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for video (default 30 => frames/30 ~= seconds).")
    parser.add_argument("--dry-run", action="store_true", help="Generate only the first frame and exit.")
    args = parser.parse_args()

    try:
        real_str, imag_str = args.center.split(",")
        center = complex(float(real_str), float(imag_str))
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"Invalid --center format: {args.center} ({exc})")

    zoom = args.zoom
    if zoom is not None and not (0 < zoom < 1):
        raise SystemExit("--zoom must be between 0 and 1 (exclusive) if provided.")

    video_out = args.video_out if args.video_out else None
    gif_out = args.gif_out if args.gif_out else None

    return MandelbrotConfig(
        frames=args.frames,
        width=args.width,
        height=args.height,
        max_iter=args.max_it,
        zoom_factor=zoom,
        final_scale=args.final_scale,
        center=center,
        colormap=args.colormap,
        device=args.device,
        output_dir=args.output_dir,
        video_out=video_out,
        gif_out=gif_out,
        fps=args.fps,
        dry_run=args.dry_run,
    )


def main():
    cfg = parse_args()
    device = select_device(cfg.device)
    print(f"Device: {device}")
    base_scale = 1.8  # starting half-width

    # Determine zoom factor
    if cfg.final_scale and cfg.zoom_factor is None:
        zoom_factor = compute_zoom_factor(cfg.frames, cfg.final_scale, base_scale)
        print(f"Computed zoom_factor={zoom_factor:.6f} from final_scale={cfg.final_scale}")
    elif cfg.zoom_factor is not None:
        zoom_factor = cfg.zoom_factor
        print(f"Using provided zoom_factor={zoom_factor}")
    else:
        zoom_factor = 0.95
        print(f"Defaulting zoom_factor={zoom_factor}")

    ensure_output_dir(cfg.output_dir)

    if cfg.dry_run:
        print("Dry run: generating a single frame...")
        frame = generate_frame(cfg, 0, device, base_scale, zoom_factor)
        out_path = resolved_path(cfg.output_dir, "mandelbrot_frame0.png")
        imageio.imwrite(out_path, frame.numpy())
        print(f"Wrote {out_path}")
        return

    print(
        f"Rendering {cfg.frames} frames at {cfg.width}x{cfg.height}, fps={cfg.fps}, base max_iter={cfg.max_iter}, final scale ~{base_scale * (zoom_factor ** (cfg.frames-1)):.3e}"
    )

    # Optional: GIF collection (only if requested)
    gif_frames = [] if cfg.gif_out else None

    if cfg.video_out:
        write_video_stream(cfg, device, base_scale, zoom_factor)
    else:
        print("Video output disabled (no --video-out provided).")

    if gif_frames is not None:
        print("Generating frames for GIF (this may be memory intensive)...")
        for i in range(cfg.frames):
            frame = generate_frame(cfg, i, device, base_scale, zoom_factor)
            gif_frames.append(frame.numpy())  # type: ignore[arg-type]
        gif_path = resolved_path(cfg.output_dir, cfg.gif_out)
        write_gif(gif_frames, gif_path, fps=min(cfg.fps, 30))  # limit GIF fps for size
        print(f"Saved GIF to {gif_path}")


if __name__ == "__main__":  # pragma: no cover
    main()