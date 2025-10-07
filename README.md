# Mandelbrot Deep Zoom (PyTorch GPU)

This project generates a deep Mandelbrot set zoom as an HD video (optionally GIF) using PyTorch as a high‑throughput array math engine. It automatically uses a CUDA GPU if available, falling back to CPU with the same code path.

## Key Technical Ideas

### 1. Device Selection
`select_device()` picks `cuda` if `torch.cuda.is_available()` else `cpu`. All tensors are allocated directly on that device to avoid per-op transfers.

### 2. Pixel Grid as Tensors
For each frame we build two 2‑D tensors (`real`, `imag`) via `torch.linspace` + `torch.meshgrid`. Each element represents a complex constant \(c = x + i y\) corresponding to one pixel. Aspect ratio is preserved (1920×1080 by default) by scaling the vertical extent.

### 3. Fully Vectorized Iteration
Instead of looping per pixel in Python, we iterate over the entire grid simultaneously. We keep two tensors: `zr` and `zi` for the real and imaginary components of \(z\). One Mandelbrot iteration:
```
(zr + i*zi)^2 + (real + i*imag)
=> zr_new = zr^2 - zi^2 + real
   zi_new = 2*zr*zi + imag
```
All of these are tensor operations dispatched to the GPU (or optimized CPU kernels).

### 4. Escape Tracking Without Per‑Pixel Python Logic
A boolean `mask` marks pixels still “alive” (not yet escaped). Each loop:
1. Compute magnitude squared `mag2 = zr2 + zi2`.
2. Mark `escaped = mag2 > 4`.
3. For newly escaped pixels, record the current iteration in `iters` (first escape time).
4. Clear those from `mask`.
5. Break early if all pixels have escaped.

Interior (non-escaping) points are assigned the final iteration count and later colored black.

### 5. Adaptive Iteration Budget
Deeper zoom frames need more iterations to resolve fine structure. We scale `max_iter` by a power of the current zoom depth (capped) so:
- Early frames compute fast.
- Later frames gain detail without exploding runtime.

### 6. Normalized Coloring
We normalize iteration counts either linearly or with logarithmic smoothing:
```
log_norm = log1p(iters) / log1p(max_iter)
```
A small custom palette converts the normalized scalar to RGB (blue→cyan→yellow→white). Interior points (`iters == max_iter`) are forced to black. All color math stays on the device until the final `.cpu()` call per frame.

### 7. Minimal Host Transfers
Only the finished RGB frame is moved back to CPU memory (`rgb.cpu()`) for encoding—one transfer per frame—keeping PCIe overhead low.

### 8. Zoom Control
You can specify either:
- `--final-scale`: desired final half‑width of the viewing window. The code computes a geometric per-frame zoom factor.
- `--zoom`: explicit per-frame zoom multiplier (<1 zooms in).

### 9. Video & GIF Output
Frames are encoded via `imageio` / `imageio-ffmpeg`:
- MP4 (H.264) for efficient long sequences.
- Optional GIF (slower, large, for quick sharing).

### 10. Output Directory Management
All artifacts (video, GIF, dry-run frame) default to `/data/mattmcl` unless overridden with `--output-dir`. Relative filenames are resolved under that directory. The folder is created automatically.

## Default Deep Zoom Profile
Defaults target roughly a 2‑minute video on a high-end GPU (e.g., NVIDIA B200):
- Frames: 3600
- FPS: 30 (=> 120 seconds)
- Final scale: 1e-12 (very deep)
- Base max_iter: 400 (grows adaptively)
- Resolution: 1920×1080

## Usage
Basic (defaults produce MP4 at /data/mattmcl/mandelbrot_zoom.mp4):
```bash
python test.py
```

Shorter preview (30 seconds, shallower):
```bash
python test.py --frames 900 --final-scale 1e-9 --max-it 300 --width 1280 --height 720
```

Custom explicit zoom factor (no final-scale auto compute):
```bash
python test.py --frames 800 --zoom 0.985
```

Generate GIF as well:
```bash
python test.py --gif-out zoom.gif --frames 600 --final-scale 1e-10
```

Dry-run first frame only (writes PNG):
```bash
python test.py --dry-run
```

Force GPU (otherwise auto):
```bash
python test.py --device cuda
```

Change output directory:
```bash
python test.py --output-dir /data/run2 --video-out deep.mp4
```

## Performance Tuning
| Lever | Effect | Notes |
|-------|--------|-------|
| `--frames` | Video length & total work | Linear scaling of runtime. |
| `--final-scale` | Depth of zoom | Smaller => deeper => more adaptive iterations. |
| `--max-it` | Baseline iteration budget | Too high wastes early-frame time. |
| Resolution | Pixel count | Runtime roughly proportional to width * height. |
| GPU vs CPU | Throughput | GPU strongly recommended for deep zoom. |
| Log coloring | Smoother gradients | Slight extra math; negligible cost. |

### PyTorch Threading (CPU fallback)
If CPU-bound, you may tune:
```python
import torch
torch.set_num_threads(8)
```
Or environment variables: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`.

## Implementation Outline (Pseudo Flow)
```
parse_args()
select_device()
compute zoom_factor (from final_scale if needed)
for frame_index in 0..frames-1:
    scale = base_scale * zoom_factor**frame_index
    real, imag = generate_grid(...)
    iters = mandelbrot(real, imag, dynamic_max_iter)
    norm = normalize_iterations(iters)
    rgb  = palette_map(norm)
    rgb[iters==max_iter] = 0  # interior black
    frame_cpu = rgb.cpu().numpy()
    encode frame
```

## Color Mapping Notes
Current simple palette can be swapped for perceptual maps (e.g., matplotlib colormaps) by converting normalized values through a lookup table (LUT) tensor on the device for negligible overhead.

## Extending the Project
Potential enhancements:
- Continuous (smooth) coloring via fractional escape time.
- Alternate palette sets / external palette file.
- Dynamic camera path (panning + zoom). 
- Multi-precision arithmetic (e.g., mpmath) for ultra-deep (>1e-15) zooms.
- Progress bar & ETA (tqdm) or frame time logging.
- Batch frame generation in overlapping CUDA streams.

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| All black interior too large | Final scale too small with low iterations | Increase `--max-it` or let adaptive grow more (adjust exponent). |
| Banding / color steps | Using linear palette | Switch to `--colormap log` (default) or improve palette. |
| Very slow start frames | Base `--max-it` too high | Lower base or frames count. |
| CUDA OOM | Resolution * iterations too large | Reduce resolution or frames, increase final_scale. |

## License
MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments
Inspired by classic Mandelbrot zoom renderers; implemented with concise tensor math using PyTorch.

---
Feel free to open issues or request enhancements (palettes, smoothing, multi-precision, streaming stats).
