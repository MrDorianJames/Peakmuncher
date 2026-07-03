# PeakMuncher

A standalone, file-based waveshaping clipper inspired by [PeakEater](https://github.com/vvvar/PeakEater) — but built as an offline audio editor instead of a real-time plugin. Written in Rust with the [iced](https://github.com/iced-rs/iced) toolkit.

## What it does

1. Open a `.wav`, `.flac`, `.mp3`, or `.aiff` file.
2. See the waveform — input envelope plus a live processed envelope overlaid on top.
3. Drop **splits** at any point to divide the track into zones, each with independent clipper parameters. Splits snap to nearest zero-crossing by default.
4. Adjust **ceiling**, **input gain**, **output gain**, **clipper type**, and **oversampling** per zone.
5. Audition with **Play / Pause** and flip between original and processed mid-playback (**A/B**) to hear the difference instantly.
6. **Export** to WAV, FLAC, MP3, or AIFF when you're done. Bit depth (16/24-bit) is preserved from the source.

## Features

- **6 clipper curves** — Hard, Quintic, Cubic, Tangent, Algebraic, Arctangent.
- **Zone-based step automation** — drop splits, give each region its own clipper config.
- **Auto-detect zones** — analyzes dynamics + harmonic structure to suggest splits at musical boundaries.
- **Oversampling** — per-zone 2×/4×/8× polyphase FIR (Kaiser-windowed sinc), applied at export for cleaner saturation.
- **Two normalization modes** — Peak (dBFS) for headroom, LUFS (BS.1770/EBU R128) for loudness matching to streaming targets.
- **Real-time spectrum + spectrogram views** — toggle with `F`.
- **Reduction overlay** — visualize what the clipper is actually shaving off.
- **Customizable theme** — Dark / Light, accent colors (or follow KDE system accent), waveform color schemes.
- **Customizable keybindings** — edit `~/.config/peakmuncher/settings.json`.
- **Project files** — save zone setup + view state to `.pmproj`, reload later.
- **Zone presets** — save just the zone config to `.pmpreset`, apply to any compatible track.
- **Undo/redo**, **recent files**, **cmd-line file argument** for `xdg-open` integration.

## Keyboard shortcuts

All shortcuts are configurable in `~/.config/peakmuncher/settings.json`. Defaults:

| Key | Action |
|---|---|
| `Space` | Play / pause |
| `Shift+Space` | Rewind to start |
| `S` | Add split at playhead |
| `Z` | Toggle zero-crossing snap |
| `R` | Toggle reduction overlay |
| `F` | Cycle FFT view (off / spectrum / spectrogram) |
| `=` / `-` | Zoom in / out (around playhead) |
| `0` | Reset zoom (fit whole file) |
| `←` / `→` | Previous / next zone |
| `Del` | Remove selected split |
| `Ctrl+Z` / `Ctrl+Shift+Z` | Undo / redo |

Mouse: middle-click drag to pan, scroll wheel to zoom, two-finger horizontal scroll to pan, right-click a zone for copy/paste/delete-split.

## Audio formats

| Format | Read | Write |
|---|---|---|
| WAV | 8/16/24-bit int, 32-bit float | 16/24-bit int (matches source), 32-bit float fallback |
| FLAC | 16/24-bit | 16/24-bit (matches source) |
| MP3 | up to 320kbps | 320kbps CBR |
| AIFF / AIFC | 16/24/32-bit BE int, 32/64-bit BE float, sowt LE int | 16/24-bit BE int (matches source) |

Bit depth is preserved across formats — open a 24-bit AIFF, export to WAV, you'll get 24-bit WAV.

## Clipper types

All six are reimplemented from first principles (no PeakEater code copied):

| Type | Curve |
|---|---|
| Hard | `clamp(x, ±C)` — brick wall |
| Quintic | 5th-order polynomial, smooth saturation |
| Cubic | `1.5x − 0.5x³`, smooth saturation |
| Tangent | `tanh(x)`, classic warm soft-clip |
| Algebraic | `x / √(1 + x²)`, asymptotic |
| Arctangent | `(2/π)·atan(x)`, asymptotic |

## Build

You'll need a recent stable Rust toolchain (1.78+). On Arch:

```bash
sudo pacman -S --needed base-devel pkgconf flac lame libxkbcommon wayland alsa-lib

git clone <wherever you put this>
cd peakmuncher
cargo build --release
./target/release/peakmuncher
```

On Debian/Ubuntu:
```bash
sudo apt install build-essential pkg-config libflac-dev libmp3lame-dev \
                 libxkbcommon-dev libwayland-dev libasound2-dev
```

The binary is fully standalone after build — no DAW required.

`alsa-lib` is needed by cpal for audio output. FLAC encode/decode are pure-Rust (no libFLAC linking required at runtime, but pkg-config picks up headers at build time).

## Configuration

PeakMuncher writes config files to `~/.config/peakmuncher/`:

- `settings.json` — theme, accent, waveform scheme, default folders, keybindings.
- `recent.json` — recent file list.

Delete either to reset to defaults. Both are auto-created on first run.

## Project files

Save a `.pmproj` to capture: the audio file path (absolute + relative — whichever resolves), all splits and zone parameters, and view state (zoom, scroll, normalization mode and target). Open from File menu or by passing on the command line.

`.pmpreset` files store just the zone configuration, applicable to any track of compatible duration.

## KDE Plasma notes

`rfd` (the file dialog) auto-detects the desktop environment and uses xdg-portal on KDE — you'll get the native Plasma file picker, not a fallback GTK one. If for some reason you want to force it, set `XDG_CURRENT_DESKTOP=KDE`.

The "System" accent color option reads `~/.config/kdeglobals` for KDE's chosen accent. Other DEs fall back to the default Blue.

If you want a `.desktop` launcher entry, drop this in `~/.local/share/applications/peakmuncher.desktop`:

```ini
[Desktop Entry]
Type=Application
Name=PeakMuncher
Exec=/path/to/peakmuncher %f
MimeType=audio/wav;audio/flac;audio/mpeg;audio/aiff;
Icon=audio-x-generic
Categories=AudioVideo;Audio;
```

## Architecture

```
src/
├── main.rs        — iced app, state, controls panel, playback wiring
├── waveform.rs    — canvas widget: envelope, zones, splits, playhead, rulers, meters, FFT
├── zones.rs       — split-point model, per-zone params, render() function
├── dsp.rs         — the six clipper curves
├── audio_io.rs    — WAV (hound), FLAC (claxon + flacenc), MP3 (minimp3 + mp3lame), AIFF (custom)
├── playback.rs    — cpal output stream, playhead state, A/B source switching
├── fft.rs         — real-FFT spectrum + spectrogram via rustfft
├── oversample.rs  — polyphase FIR upsampler/downsampler
├── detect.rs      — auto-zone-detection: dynamics + chroma novelty
├── structure.rs   — chroma vectors, self-similarity matrix, Foote checkerboard novelty
├── recent.rs      — recent-files persistence
├── project.rs     — .pmproj + .pmpreset serialization
├── settings.rs    — theme/accent/scheme/folder persistence + KDE accent reader
└── keybindings.rs — JSON-customizable shortcut parser
```

The DSP runs on a dedicated worker thread fed by an mpsc channel; if multiple parameter changes pile up, only the newest job runs (older ones are dropped). Per-zone clipping is parallelized across zones with `rayon`.

LUFS measurement uses the [`ebur128`](https://github.com/sdroege/ebur128) crate — a pure-Rust port of the reference C library that passes the EBU R128 test set. Verified against ffmpeg's `loudnorm`: 0.03 LU error on a real master.

## License

GPL-3.0, matching PeakEater's license — though no PeakEater source is included or derived.
