# PeakMuncher

A standalone offline waveshaping clipper for mastering and sound design, built in Rust with [iced](https://iced.rs/).

PeakMuncher loads an audio file, lets you shape and tame its peaks with a choice of clipping curves, and exports the result. Unlike a real-time plugin, it works on the whole file at once — so you can see the entire waveform, park a probe point anywhere to analyze it, and split the timeline into independent **zones**, each with its own clipping settings.

---

## Features

### Zone-based processing
Split the timeline at any point and give each **zone** its own independent settings — clipper type, ceiling, gain, fades, and DC correction. A quiet intro and a loud drop can be clipped completely differently in the same pass.

### Nine clipping curves
Each zone can use any of nine waveshaping curves, from transparent to aggressive:

| Curve | Character |
|-------|-----------|
| Hard | Brick-wall, hardest edge |
| Quintic | Smooth polynomial knee |
| Cubic | Gentle polynomial knee |
| Tangent | Rounded saturation |
| Algebraic | Soft algebraic curve |
| Arctangent | Smooth analog-style |
| Sine Fold | Wavefolder — bright, aggressive, harmonically rich |
| Tanh Drive | Warm, tape-like |
| Sigmoid | Clean, polished logistic curve |

All curves are bounded and pass cleanly through zero.

### Analysis and metering
- **Frozen probe-point spectrum** — park the playhead anywhere and see a high-resolution (16384-point, Blackman-Harris, frame-averaged) spectrum of that moment.
- **Input vs. output overlay** — the original and processed spectra are drawn together, with a difference fill (amber = harmonics the clipper *added*, blue = energy it *removed*) so you can see exactly what each mode does to the sound.
- **Spectrogram view** — a whole-file frequency heatmap.
- **Per-zone input peak marker** on the ceiling slider, showing exactly where clipping begins.
- **Live input/output level meters.**

### Level and cleanup tools
- **Input gain** to drive the clipper.
- **Ceiling** control (the clipping threshold).
- **Oversampling** (applied on export) to reduce aliasing.
- **Normalize** — peak or LUFS, measured over the trim region.
- **Fade in / fade out** per zone (anchored to the trim window).
- **DC offset** correction and a **DC blocker** (one-pole high-pass).

### Editing
- **Trim** handles to set the exported region; fades and normalization respect the trim boundaries.
- **A/B** toggle to compare original against processed.
- **Apply** to bake changes into the working buffer, with a multi-step undo/redo history.

### Interface
- Four-tab control panel: **Clipper**, **Levels**, **Fix**, **Output**.
- Waveform canvas with input/output overlay, clipping visualization, zoom, and sample-level detail.
- Dark theme with KDE-style accent coloring.

---

## Supported formats

**Input:** WAV, FLAC, MP3, AIFF
**Output:** WAV, FLAC, MP3

---

## Building

PeakMuncher is a standard Cargo project.

```bash
cargo build --release
```

The binary is written to `target/release/`.

### Requirements
- Rust (recent stable toolchain)
- A Linux desktop environment (developed and tested on CachyOS / KDE Plasma). The GUI uses [iced](https://iced.rs/) with a wgpu backend.

---

## Usage

1. **Open** an audio file.
2. **Split** the timeline into zones at the cursor if you want different settings for different sections (optional — one zone covers the whole file by default).
3. Pick a **clipper** curve and set the **ceiling** for the selected zone.
4. Use the **FFT/spectrum** view to see how the curve reshapes the harmonics — park the playhead on a telling moment (a kick, an exposed bass note) and compare input vs. output.
5. Set **fades**, **normalize**, and any **DC** cleanup as needed.
6. **Trim** to the region you want to export.
7. **Save** the result.

---

## Project layout

| File | Responsibility |
|------|----------------|
| `main.rs` | App state, UI, message handling, history, export |
| `waveform.rs` | Waveform, spectrum, and spectrogram rendering; canvas interaction |
| `zones.rs` | Per-zone DSP render (clipping, gain, fades, DC) |
| `dsp.rs` | The clipping curve functions |
| `fft.rs` | FFT spectrum and spectrogram analysis |
| `project.rs` | Project save/load |

---

## Status

Actively developed. Working build. See `PEAKMUNCHER_HANDOFF.md` for detailed development notes and the roadmap.

---

## License

See the [LICENSE](LICENSE) file in this repository.
