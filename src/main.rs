//! PeakMuncher — standalone offline waveshaping clipper with zone automation.
//!
//! Layout:
//!   ┌────────────────────────────────────────────────────────────────┐
//!   │  [Open] [Save As…] [Add split @ cursor] [Render]   00:00.000   │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │                                                                │
//!   │                   WAVEFORM CANVAS                              │
//!   │   • gray = input, orange = processed                           │
//!   │   • red horizontal lines = zone ceilings                       │
//!   │   • blue vertical lines = split points (drag to move)          │
//!   │                                                                │
//!   ├────────────────────────────────────────────────────────────────┤
//!   │  Selected zone N:  Clipper [Tangent ▾]                         │
//!   │  Input gain  [────●──── ] +0.0 dB                              │
//!   │  Ceiling     [───●───── ] -1.0 dB                              │
//!   │  Output gain [────●──── ] +0.0 dB                              │
//!   │  [Delete selected split]                                       │
//!   └────────────────────────────────────────────────────────────────┘

mod audio_io;
mod detect;
mod dsp;
mod fft;
mod keybindings;
mod oversample;
mod playback;
mod project;
mod recent;
mod settings;
mod structure;
mod waveform;
mod zones;

use audio_io::AudioBuffer;
use dsp::ClipperType;
use iced::time;
use iced::widget::{
    button, canvas, column, container, horizontal_rule, mouse_area, pick_list, row, slider, stack,
    svg, text, Space,
};
use iced::{Color, Element, Length, Point, Rectangle, Renderer, Size, Subscription, Task, Theme};
use playback::{Player, Source};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use waveform::{build_envelope, CanvasEvent, Waveform};
use zones::{ZoneMap, ZoneParams};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SpectrumMode {
    Off,
    Spectrum,
    Spectrogram,
}

impl SpectrumMode {
    fn next(self) -> Self {
        match self {
            SpectrumMode::Off => SpectrumMode::Spectrum,
            SpectrumMode::Spectrum => SpectrumMode::Spectrogram,
            SpectrumMode::Spectrogram => SpectrumMode::Off,
        }
    }
    fn label(self) -> &'static str {
        match self {
            SpectrumMode::Off => "FFT: Off",
            SpectrumMode::Spectrum => "FFT: Spectrum",
            SpectrumMode::Spectrogram => "FFT: Spectrogram",
        }
    }
}

const ENVELOPE_WIDTH: usize = 2000; // resolution for the rendered envelope arrays

/// Embedded SVG icon data — baked into the binary so the app stays a single
/// portable executable.
const ICON_ZOOM_IN: &[u8] = include_bytes!("../assets/icons/zoom-in.svg");
const ICON_ZOOM_OUT: &[u8] = include_bytes!("../assets/icons/zoom-out.svg");
const ICON_ZOOM_FIT: &[u8] = include_bytes!("../assets/icons/zoom-fit-best.svg");
// "Stop" in this app's UX semantics actually means "rewind to start", which
// the skip-backward icon expresses better than a stop square.
const ICON_STOP: &[u8] = include_bytes!("../assets/icons/media-skip-backward.svg");
const ICON_PLAY: &[u8] = include_bytes!("../assets/icons/media-playback-start.svg");
const ICON_PAUSE: &[u8] = include_bytes!("../assets/icons/media-playback-pause.svg");
// File menu / app menu burger icon.
const ICON_MENU: &[u8] = include_bytes!("../assets/icons/menu.svg");
// Pulled-out menu items, now living in the top bar as icons.
const ICON_ZERO_CROSSING: &[u8] = include_bytes!("../assets/icons/zero-crossing.svg");
const ICON_REDUCTION: &[u8] = include_bytes!("../assets/icons/reduction.svg");
const ICON_GUIDES: &[u8] = include_bytes!("../assets/icons/guides.svg");
const ICON_SETTINGS: &[u8] = include_bytes!("../assets/icons/settings.svg");
// FFT / spectrogram view cycle (Off → Spectrum → Spectrogram).
const ICON_FFT_VIEW: &[u8] = include_bytes!("../assets/icons/view.svg");
// Small scissors glyph shown in front of the trim readout (not a button).
const ICON_SCISSORS: &[u8] = include_bytes!("../assets/icons/scissors.svg");
// Small arrow shown between the trim start and end times.
const ICON_ARROW_RIGHT: &[u8] = include_bytes!("../assets/icons/arrow-right.svg");
// A/B compare toggle (git-compare) and Apply (arrows-up-from-line).
const ICON_COMPARE: &[u8] = include_bytes!("../assets/icons/git-compare.svg");
const ICON_APPLY: &[u8] = include_bytes!("../assets/icons/arrows-up-from-line.svg");
// Zone navigation chevrons (prev/next zone).
const ICON_CHEVRON_LEFT: &[u8] = include_bytes!("../assets/icons/chevron-left.svg");
const ICON_CHEVRON_RIGHT: &[u8] = include_bytes!("../assets/icons/chevron-right.svg");
// Split-at-cursor and delete-selected-split, grouped left of the zone chevrons.
const ICON_SPLIT: &[u8] = include_bytes!("../assets/icons/split.svg");
const ICON_TRASH: &[u8] = include_bytes!("../assets/icons/trash-2.svg");
// Apply tracker dots: filled = active Apply at-or-before history_pos,
// empty = unused slot. Tabler icons, currentColor — pick up surrounding text color.
const ICON_CIRCLE_FILLED: &[u8] = include_bytes!("../assets/icons/circle_filled.svg");
const ICON_CIRCLE_EMPTY: &[u8] = include_bytes!("../assets/icons/circle_empty.svg");

/// Which control-panel tab is active. The tabs sit horizontally next to the
/// zone label; only the active tab's controls render below.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlTab {
    Clipper,
    Levels,
    Fix,
    Output,
}

impl ControlTab {
    const ALL: [ControlTab; 4] = [
        ControlTab::Clipper,
        ControlTab::Levels,
        ControlTab::Fix,
        ControlTab::Output,
    ];
    fn label(self) -> &'static str {
        match self {
            ControlTab::Clipper => "CLIPPER",
            ControlTab::Levels => "LEVELS",
            ControlTab::Fix => "FIX",
            ControlTab::Output => "OUTPUT",
        }
    }
}

/// Normalization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizeMode {
    /// Peak normalization: scale so the loudest |sample| hits the target dBFS.
    /// Cheap, works on any length, but two files with the same peak can sound
    /// very different in loudness.
    Peak,
    /// LUFS / EBU R128 / ITU-R BS.1770 integrated loudness. Scales so the
    /// perceived loudness hits the target LUFS. Better for matching tracks
    /// to streaming targets (-14 LUFS) or playlists. Requires ~3+ seconds
    /// of audio for a stable measurement.
    Lufs,
}

/// 3-way normalization state shown in the UI: Off / Peak / LUFS. The
/// existing `NormalizeMode` enum + `normalize_enabled` bool model the
/// same thing internally; this is just the user-facing picker value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NormalizeState {
    Off,
    Peak,
    Lufs,
}

impl NormalizeState {
    const ALL: [NormalizeState; 3] = [
        NormalizeState::Off,
        NormalizeState::Peak,
        NormalizeState::Lufs,
    ];
}

impl std::fmt::Display for NormalizeState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            NormalizeState::Off => "Off",
            NormalizeState::Peak => "Peak",
            NormalizeState::Lufs => "LUFS",
        })
    }
}

impl Default for NormalizeMode {
    fn default() -> Self {
        Self::Peak
    }
}

impl NormalizeMode {
    fn label(self) -> &'static str {
        match self {
            Self::Peak => "Peak",
            Self::Lufs => "LUFS",
        }
    }
    /// Slider range for this mode. Returns (min, max, default).
    fn range(self) -> (f32, f32, f32) {
        match self {
            Self::Peak => (-12.0, 0.0, -0.3),
            Self::Lufs => (-23.0, -9.0, -14.0),
        }
    }
    fn unit(self) -> &'static str {
        match self {
            Self::Peak => "dBFS",
            Self::Lufs => "LUFS",
        }
    }
}

/// Apply normalization in place. Returns the gain (in dB) that was applied,
/// or 0.0 if disabled / silent / measurement failed.
///
/// For LUFS, additionally clamps the gain so the resulting peak doesn't
/// exceed -1 dBFS (a true-peak limiter would be more correct, but for an
/// offline tool, sample-peak is close enough).
/// Apply normalization in-place and return the LINEAR gain that was
/// applied (1.0 = no change). Used both to scale the audio buffer and
/// to scale the input reference shown in the waveform (so red and blue
/// layers grow together when normalize amplifies).
fn apply_normalization(
    samples: &mut [f32],
    channels: u16,
    sample_rate: u32,
    enabled: bool,
    mode: NormalizeMode,
    target_db: f32,
    // Optional measurement window in FRAME indices [start_frame, end_frame).
    // When Some, peak/LUFS are measured only over this region (used to
    // exclude trimmed audio from the normalize calculation), but the
    // resulting gain is still applied to the WHOLE buffer. When None, the
    // whole buffer is measured (original behavior).
    measure_window: Option<(usize, usize)>,
) -> f32 {
    if !enabled || samples.is_empty() {
        return 1.0;
    }
    let ch = channels.max(1) as usize;
    let total_frames = samples.len() / ch;
    // Resolve the measurement slice (interleaved sample indices). Clamp the
    // window to valid bounds; fall back to the whole buffer if degenerate.
    let (m_lo, m_hi) = match measure_window {
        Some((f0, f1)) => {
            let f0 = f0.min(total_frames);
            let f1 = f1.min(total_frames).max(f0);
            if f1 > f0 {
                (f0 * ch, f1 * ch)
            } else {
                (0, samples.len())
            }
        }
        None => (0, samples.len()),
    };
    let measure_slice = &samples[m_lo..m_hi];
    let gain = match mode {
        NormalizeMode::Peak => {
            let mut peak = 0.0f32;
            for &s in measure_slice.iter() {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }
            if peak < 1e-6 {
                return 1.0;
            }
            let target_amp = 10f32.powf(target_db / 20.0);
            target_amp / peak
        }
        NormalizeMode::Lufs => {
            // ebur128 wants per-channel planar samples, but our buffer is
            // interleaved. Either de-interleave (allocation) or feed
            // interleaved using `add_frames_*` which the crate supports.
            let mut analyzer = match ebur128::EbuR128::new(
                ch as u32,
                sample_rate,
                ebur128::Mode::I,
            ) {
                Ok(a) => a,
                Err(_) => return 1.0,
            };
            // add_frames_f32 takes interleaved samples — exactly our layout.
            // Feed only the measurement window.
            if analyzer.add_frames_f32(measure_slice).is_err() {
                return 1.0;
            }
            let measured = match analyzer.loudness_global() {
                Ok(v) if v.is_finite() => v as f32,
                _ => return 1.0, // silent / too short for valid measurement
            };
            let mut gain_db = target_db - measured;
            // Clamp so resulting peak ≤ -1 dBFS. Measure peak over the same
            // window so the clamp matches what we normalized to.
            let mut peak = 0.0f32;
            for &s in measure_slice.iter() {
                let a = s.abs();
                if a > peak {
                    peak = a;
                }
            }
            if peak > 1e-6 {
                let post_peak_db = 20.0 * peak.log10() + gain_db;
                if post_peak_db > -1.0 {
                    gain_db -= post_peak_db - (-1.0);
                }
            }
            10f32.powf(gain_db / 20.0)
        }
    };
    if (gain - 1.0).abs() < 1e-6 {
        return 1.0;
    }
    for s in samples.iter_mut() {
        *s *= gain;
    }
    gain
}

/// Like `icon_button` but accepts an optional on_press. When None, the
/// button is rendered "disabled" — same transparent ghost styling but
/// dimmed and not clickable. Useful for nav arrows at list boundaries.
fn icon_button_opt<'a>(
    icon_bytes: &'static [u8],
    on_press: Option<Msg>,
    accent_hover: Color,
) -> button::Button<'a, Msg> {
    let handle = svg::Handle::from_memory(icon_bytes);
    let icon = svg(handle).width(16).height(16);
    let mut b = button(icon).padding([4, 8]).style(
        move |_theme: &Theme, status: button::Status| {
            let bg = match status {
                button::Status::Hovered | button::Status::Pressed => {
                    iced::Background::Color(accent_hover)
                }
                _ => iced::Background::Color(Color::TRANSPARENT),
            };
            // Disabled state: render with a dimmed icon by passing a
            // mid-gray text color so currentColor falls back to that.
            let text_color = match status {
                button::Status::Disabled => Color::from_rgba(0.6, 0.6, 0.6, 0.5),
                _ => Color::WHITE,
            };
            button::Style {
                background: Some(bg),
                text_color,
                border: iced::Border {
                    radius: 3.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        },
    );
    if let Some(msg) = on_press {
        b = b.on_press(msg);
    }
    b
}

/// Build an SVG icon button styled as a "ghost" button — no background
/// in the default state, transparent border, and a faint accent-tinted
/// background only on hover/press. The icon SVG should use
/// `stroke="currentColor"` (or `fill="currentColor"`) so it picks up
/// the theme color. The button passes `accent_hover` from settings as
/// the hover-tint color.
fn icon_button<'a>(
    icon_bytes: &'static [u8],
    on_press: Msg,
    accent_hover: Color,
) -> button::Button<'a, Msg> {
    let handle = svg::Handle::from_memory(icon_bytes);
    let icon = svg(handle).width(16).height(16);
    button(icon)
        .on_press(on_press)
        .padding([4, 8])
        .style(move |_theme: &Theme, status: button::Status| {
            let bg = match status {
                button::Status::Hovered | button::Status::Pressed => {
                    // Faint accent tint on hover, slightly more opaque
                    // when pressed. accent_hover already has alpha
                    // applied by the settings module.
                    iced::Background::Color(accent_hover)
                }
                _ => iced::Background::Color(Color::TRANSPARENT),
            };
            button::Style {
                background: Some(bg),
                text_color: Color::WHITE,
                border: iced::Border {
                    radius: 3.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
}

/// Variant of `icon_button` for toggle settings — when `enabled` is
/// true, the button shows a solid accent background so the user can see
/// the toggle is on at a glance. When disabled, it behaves like a
/// regular ghost icon_button (transparent except on hover).
fn toggle_icon_button<'a>(
    icon_bytes: &'static [u8],
    on_press: Msg,
    enabled: bool,
    accent: Color,
    accent_hover: Color,
) -> button::Button<'a, Msg> {
    let handle = svg::Handle::from_memory(icon_bytes);
    let icon = svg(handle).width(16).height(16);
    button(icon)
        .on_press(on_press)
        .padding([4, 8])
        .style(move |_theme: &Theme, status: button::Status| {
            // Base color depends on enabled state. Hover/press blends
            // toward accent_hover regardless of enabled state.
            let bg = match (enabled, status) {
                (_, button::Status::Hovered) | (_, button::Status::Pressed) => accent_hover,
                (true, _) => accent,
                (false, _) => Color::TRANSPARENT,
            };
            button::Style {
                background: Some(iced::Background::Color(bg)),
                text_color: Color::WHITE,
                border: iced::Border {
                    radius: 3.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        })
}

pub fn main() -> iced::Result {
    // Quick CLI handling — anything more elaborate goes through App::new.
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "PeakMuncher — offline waveshaping clipper\n\n\
             USAGE:\n    peakmuncher [PATH]\n\n\
             ARGS:\n    PATH    Audio file to open at startup (.wav, .flac, .mp3)\n\n\
             OPTIONS:\n    -h, --help    Show this help and exit"
        );
        return Ok(());
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("peakmuncher {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }

    // Print panic location/message instead of just "Aborted".
    std::panic::set_hook(Box::new(|info| {
        eprintln!("\n=== PeakMuncher panic ===");
        eprintln!("{info}");
        if let Ok(bt) = std::env::var("RUST_BACKTRACE") {
            if bt == "1" || bt == "full" {
                eprintln!("{}", std::backtrace::Backtrace::force_capture());
            }
        }
        eprintln!("Re-run with RUST_BACKTRACE=1 for a backtrace.");
    }));

    iced::application("PeakMuncher", App::update, App::view)
        .subscription(App::subscription)
        .theme(|app: &App| app.settings.build_theme())
        .window_size((1100.0, 650.0))
        .run_with(App::new)
}

/// State for the "Auto-detect zones" modal.
#[derive(Debug, Clone)]
struct AutoDetectState {
    /// Sensitivity slider, 0.0 (few splits) to 1.0 (many splits).
    sensitivity: f32,
    /// Cached preview of detected splits at the current sensitivity.
    preview_splits: Vec<f32>,
}

impl Default for AutoDetectState {
    fn default() -> Self {
        Self {
            sensitivity: 0.55,
            preview_splits: Vec::new(),
        }
    }
}

struct App {
    input: Option<AudioBuffer>,
    /// Where the current input file came from on disk. None until a file is
    /// loaded. Saved into project files so projects remember which audio
    /// they refer to.
    input_path: Option<PathBuf>,
    processed: Option<AudioBuffer>,
    envelope_in: Vec<(f32, f32)>,
    envelope_out: Vec<(f32, f32)>,
    /// Cached mono mixdown of the input — used by zero-crossing snap and the
    /// per-pixel envelope when zoomed in.
    mono_input: Vec<f32>,
    /// Raw input mono, cached once on load and shared with the DSP worker
    /// by Arc so it isn't rebuilt from interleaved on every render.
    raw_input_mono: Arc<Vec<f32>>,
    /// Cached mono mixdown of the processed output — used by per-pixel envelope.
    mono_output: Vec<f32>,
    zones: ZoneMap,
    selected_split: Option<usize>,
    /// Export trim boundaries, in seconds. Non-destructive: audio outside
    /// `[trim_start, trim_end]` is excluded only on export, and excluded
    /// from the normalize MEASUREMENT, but never removed from the working
    /// buffer (drag the handles anytime, before or after other ops).
    /// Defaulted to (0.0, duration) on load so the handles park at the
    /// edges and trim nothing until dragged inward. `None` only before a
    /// file is loaded.
    trim_start: Option<f32>,
    trim_end: Option<f32>,
    selected_zone: Option<usize>,
    /// Active control-panel tab (Clipper/Levels/Fix/Output).
    active_tab: ControlTab,
    /// Cached input peak (dBFS) of the selected zone, for the Ceiling
    /// slider's peak marker. Recomputed on zone switch / file load /
    /// input-gain change (input gain shifts the peak the clipper sees).
    /// None when no file or silent zone.
    zone_input_peak_db: Option<f32>,
    cursor_time: Option<f32>, // hover position from canvas
    status: String,
    canvas_cache: canvas::Cache,
    overlay_cache: canvas::Cache,
    /// Separate cache for the bottom strip (peak meters + FFT spectrum line).
    /// These update at ~30 Hz during playback; keeping them in their own
    /// cache means the expensive waveform layer doesn't have to redraw
    /// every tick.
    meter_cache: canvas::Cache,
    player: Option<Player>,
    playhead_secs: f32,
    /// Waveform view: 1.0 = whole file fits, 10.0 = 10x zoom, etc.
    zoom: f32,
    /// Seconds at the left edge of the visible window.
    scroll: f32,
    /// If true, snap split adds/moves to the nearest zero crossing.
    snap_enabled: bool,
    show_reduction: bool,
    /// Spectrum view mode (None = off, Spectrum = real-time line, Spectrogram = heatmap)
    spectrum_mode: SpectrumMode,
    /// Lazily-computed spectrogram for the input audio.
    spectrogram_in: Option<Arc<fft::Spectrogram>>,
    /// Spectrogram for the processed audio (rebuilt on DspDone).
    spectrogram_out: Option<Arc<fft::Spectrogram>>,
    /// Cached single-frame spectrum LINE at the current probe point, for the
    /// OUTPUT audio. Computed by `refresh_spectrum_line()` only when the probe
    /// (playhead) or audio changes — NOT on every 30 Hz meter repaint, since
    /// it's now a large averaged FFT (see fft::spectrum_at). This is the first
    /// step toward the frozen probe-point comparison model.
    spectrum_line_out: Option<Arc<Vec<f32>>>,
    /// Same, for the INPUT audio — enables the input-vs-output overlay.
    spectrum_line_in: Option<Arc<Vec<f32>>>,
    /// The probe frame the cached spectrum lines were computed at, so we can
    /// detect when the probe moved enough to warrant a recompute.
    spectrum_line_frame: Option<usize>,
    /// Smoothed peak meter values, in linear amplitude [0..1+]. Decayed on
    /// each Tick toward the latest sample.
    meter_in: f32,
    meter_out: f32,
    /// Unified undo/redo history. Each entry is either a parameter change
    /// (tiny ZoneMap snapshot) or an Apply operation (carries the
    /// pre-Apply audio buffer so the bake can be reversed). `history_pos`
    /// indexes the current state — undo decrements, redo increments.
    /// Apply entries are capped at 5 to bound memory: beyond that, the
    /// oldest Apply (and everything before it) gets pruned.
    history: Vec<HistoryEntry>,
    history_pos: usize,
    /// Whether the File menu is currently open (drop-down visible).
    file_menu_open: bool,
    menu_close_abort: Option<iced::task::Handle>,
    recent_files: recent::RecentFiles,
    /// Optional draggable horizontal guide line, in dB (negative). None = hidden.
    guide_db: Option<f32>,
    /// Right-click context menu on a zone: `(zone_index, split_index_if_on_split, canvas_x, canvas_y)`.
    /// `split_index_if_on_split` is Some(i) when the right-click landed on
    /// or very near split i, enabling the "Delete split" item.
    zone_ctx_menu: Option<(usize, Option<usize>, f32, f32)>,
    /// Clipboard slot for copy/paste of zone parameters.
    zone_clipboard: Option<ZoneParams>,
    /// Master normalization toggle. Mode determines whether we measure peak
    /// (dBFS) or integrated loudness (LUFS) before scaling.
    normalize_enabled: bool,
    /// What kind of normalization to apply.
    normalize_mode: NormalizeMode,
    /// Target value. Interpretation depends on `normalize_mode`:
    ///   Peak: dBFS, range -12..0 (e.g. -0.3)
    ///   Lufs: LUFS, range -23..-9 (e.g. -14 for streaming)
    normalize_target_db: f32,
    /// When `Some`, the auto-detect zones modal is open.
    auto_detect: Option<AutoDetectState>,
    /// Persisted user settings (theme, accent color, default paths).
    settings: settings::Settings,
    /// When `Some`, the Settings modal is open. Holds an in-flight draft
    /// so the user can Cancel without saving partial changes.
    settings_draft: Option<settings::Settings>,
    dsp_gen: u64,
    dsp_tx: Sender<DspJob>,
    dsp_rx: Arc<Mutex<Receiver<DspResult>>>,
    dsp_pending: bool,
}

/// Snapshot of "everything an Apply rewinds and re-applies" — the full
/// DSP-relevant state on both sides of an Apply operation. Stored on
/// the history stack so we can navigate to either side cheaply (no
/// reprocessing needed).
///
/// Memory: each ApplyFrame holds two full audio buffers (pre and post).
/// Capped at 5 Apply entries in history to bound total cost.
#[derive(Clone)]
struct ApplyFrame {
    // ---- Pre-Apply state (restored on undo across this Apply) ----
    prev_input: Arc<Vec<f32>>,
    prev_mono_input: Vec<f32>,
    prev_envelope_in: Vec<(f32, f32)>,
    prev_zones: ZoneMap,
    prev_normalize_enabled: bool,
    prev_normalize_mode: NormalizeMode,
    prev_normalize_target_db: f32,

    // ---- Post-Apply state (restored on redo across this Apply) ----
    /// The baked audio that became the new input. This is the result of
    /// running the prev_zones / prev_normalize processing on prev_input.
    post_input: Arc<Vec<f32>>,
    post_mono_input: Vec<f32>,
    post_envelope_in: Vec<(f32, f32)>,
}

/// One entry in the unified undo/redo history. `ParamChange` is the
/// existing lightweight snapshot used for every slider commit, etc.
/// `Apply` is heavier — it carries the audio buffer that was the input
/// before the Apply ran, so the Apply can be reversed.
#[derive(Clone)]
enum HistoryEntry {
    ParamChange(ZoneMap),
    Apply(ApplyFrame),
}

/// Job sent to the DSP worker.
struct DspJob {
    gen: u64,
    samples: Arc<Vec<f32>>,
    channels: u16,
    sample_rate: u32,
    zones: ZoneMap,
    /// If true, scale the post-clip output to hit normalize_target_db.
    normalize: bool,
    normalize_mode: NormalizeMode,
    normalize_target_db: f32,
    /// Normalize measurement window in FRAME indices [start, end), derived
    /// from the trim boundaries. None = measure whole buffer.
    trim_window: Option<(usize, usize)>,
    /// Raw input mixed to mono, computed ONCE on file load and shared by
    /// Arc. The raw input never changes during parameter edits (only the
    /// zones/normalize do), so rebuilding it per render was a wasted full
    /// mono pass every ceiling-slider tick. The worker only applies the
    /// cheap normalize-gain scaling to this.
    raw_input_mono: Arc<Vec<f32>>,
    /// When true, render at reduced resolution (every Nth frame) for a fast
    /// preview during slider drags. The result is display-only; a full-res
    /// render follows on slider release (Msg::Commit). Playback/Apply/export
    /// never use preview output.
    preview: bool,
}

/// Result from the DSP worker.
struct DspResult {
    gen: u64,
    processed: Arc<Vec<f32>>,
    mono: Vec<f32>,
    envelope: Vec<(f32, f32)>,
    /// Mono representation of the RAW INPUT scaled by the same normalize
    /// gain applied to `processed`. Used as the red reference layer in
    /// the waveform so the input/clipping visualization grows together
    /// with the output when normalize amplifies.
    mono_in_scaled: Vec<f32>,
    /// Input envelope, prebuilt in the worker (was previously rebuilt on
    /// the main thread every poll — moving it here keeps the UI thread
    /// from stuttering during slider drags on long files).
    envelope_in: Vec<(f32, f32)>,
    /// True if this result came from a decimated preview render. The result
    /// handler then updates ONLY the display envelopes and leaves the
    /// full-resolution processed/mono buffers untouched (they stay valid
    /// from the last full render and refresh on slider release).
    preview: bool,
}

/// Spawn the dedicated DSP worker. It owns ONE OS thread for the lifetime
/// of the app. New jobs sent down `tx` while one is running queue up; we
/// drain the queue and only run the most recent job (older gens are
/// implicitly stale and will be dropped by the App side).
fn spawn_dsp_worker() -> (Sender<DspJob>, Receiver<DspResult>) {
    let (job_tx, job_rx) = mpsc::channel::<DspJob>();
    let (res_tx, res_rx) = mpsc::channel::<DspResult>();
    std::thread::Builder::new()
        .name("peakmuncher-dsp".into())
        .spawn(move || loop {
            // Block waiting for a job.
            let mut job = match job_rx.recv() {
                Ok(j) => j,
                Err(_) => return, // App dropped the sender → quit
            };
            // Drain any queued jobs and keep only the newest — no point
            // running a stale one.
            while let Ok(newer) = job_rx.try_recv() {
                job = newer;
            }
            if job.preview {
                // ---- Decimated preview path (slider drag) ----
                // Render only every Nth frame and build the display
                // envelopes from that. Skips the full mono/processed
                // construction entirely — those stay valid from the last
                // full render and refresh on release. The envelope is what
                // the user sees at full-file zoom during a drag.
                const DECIM: usize = 4;
                let env_out =
                    zones::render_envelope_decimated(
                        &job.samples,
                        job.channels,
                        job.sample_rate,
                        &job.zones,
                        job.normalize,
                        job.normalize_mode,
                        job.normalize_target_db,
                        job.trim_window,
                        ENVELOPE_WIDTH,
                        DECIM,
                    );
                // Input envelope: decimate the cached raw mono directly.
                let norm_gain_prev = env_out.1;
                let env_in = envelope_decimated_mono(
                    &job.raw_input_mono,
                    norm_gain_prev,
                    ENVELOPE_WIDTH,
                    DECIM,
                );
                if res_tx
                    .send(DspResult {
                        gen: job.gen,
                        processed: Arc::new(Vec::new()),
                        mono: Vec::new(),
                        envelope: env_out.0,
                        mono_in_scaled: Vec::new(),
                        envelope_in: env_in,
                        preview: true,
                    })
                    .is_err()
                {
                    return;
                }
                continue;
            }

            let mut processed_vec = zones::render(
                &job.samples,
                job.channels,
                job.sample_rate,
                &job.zones,
                job.trim_window,
            );
            let norm_gain = apply_normalization(
                &mut processed_vec,
                job.channels,
                job.sample_rate,
                job.normalize,
                job.normalize_mode,
                job.normalize_target_db,
                job.trim_window,
            );
            let processed = Arc::new(processed_vec);
            let mono = mono_from_interleaved(&processed, job.channels);
            let envelope = build_envelope(&mono, ENVELOPE_WIDTH);
            // Scale the cached raw input mono by the normalize gain. The
            // raw mono itself was built once on load (see raw_input_mono),
            // so we skip a full mono mixdown pass over the interleaved
            // input on every render — only the cheap scaling remains.
            let mono_in_scaled: Vec<f32> = if (norm_gain - 1.0).abs() < 1e-6 {
                job.raw_input_mono.as_ref().clone()
            } else {
                job.raw_input_mono.iter().map(|&v| v * norm_gain).collect()
            };
            // Prebuild the input envelope here (off the main thread).
            let envelope_in = build_envelope(&mono_in_scaled, ENVELOPE_WIDTH);
            if res_tx
                .send(DspResult {
                    gen: job.gen,
                    processed,
                    mono,
                    envelope,
                    mono_in_scaled,
                    envelope_in,
                    preview: false,
                })
                .is_err()
            {
                return;
            }
        })
        .expect("spawn DSP thread");
    (job_tx, res_rx)
}

#[derive(Debug, Clone)]
enum Msg {
    OpenFile,
    OpenPath(PathBuf),
    Loaded(Option<(PathBuf, Result<AudioBuffer, String>)>),
    SaveFile,
    Saved(Result<PathBuf, String>),
    AddSplitAtCursor,
    Canvas(CanvasEvent),
    DeleteSelectedSplit,
    SetCeiling(f32),
    SetInputGain(f32),
    SetOutputGain(f32),
    SetClipper(ClipperType),
    SetOversampling(u8),
    SetDcOffset(f32),
    /// Measure the DC offset (mean value) of the selected zone's samples
    /// and set the zone's dc_offset to negate it.
    AutoDetectDc,
    /// Per-zone fade-in length in seconds (0 = off).
    SetFadeIn(f32),
    /// Per-zone fade-out length in seconds (0 = off).
    SetFadeOut(f32),
    /// Toggle the per-zone DC blocker (one-pole high-pass).
    ToggleDcBlocker,
    /// Set DC blocker cutoff frequency in Hz.
    SetDcBlockerHz(f32),
    /// Bake the current processed output into the input — "Apply".
    /// The current output becomes the new source of truth; all
    /// processing params reset to defaults. Reversible via undo.
    ApplyProcessing,
    // Playback
    Play,
    Pause,
    TogglePlay,      // spacebar
    Stop,
    /// Raw keyboard event — bindings lookup happens in update().
    KeyPressed(iced::keyboard::Key, iced::keyboard::Modifiers),
    ToggleSource,    // A/B
    Tick,            // ~30Hz, advances playhead display
    // Zone nav
    PrevZone,
    NextZone,
    SelectZoneIdx(usize),
    SelectTab(ControlTab),
    // DSP background result
    DspPoll,
    // View / view controls
    ZoomIn,
    ZoomOut,
    ZoomReset,
    ToggleSnap,
    ToggleReduction,
    // Undo / redo
    Undo,
    Redo,
    /// User finished editing a parameter (e.g. slider release). Snapshots
    /// the current ZoneMap into history.
    Commit,
    // FFT
    CycleSpectrumMode,
    SpectrogramInDone(Arc<fft::Spectrogram>),
    SpectrogramOutDone(Arc<fft::Spectrogram>),
    // Menu
    ToggleFileMenu,
    CloseFileMenu,
    /// Mouse left the menu — schedule a delayed close.
    MenuMouseExit,
    /// Mouse re-entered the menu — cancel the pending close.
    MenuMouseEnter,
    // Guide line
    ToggleGuide,
    SetGuideDb(f32),
    // Zone context menu (right-click)
    ZoneContextMenu(usize, Option<usize>, f32, f32),
    CloseZoneContextMenu,
    /// Delete the given split (from the right-click context menu).
    DeleteSplitAt(usize),
    CopyZoneParams,
    PasteZoneParams,
    // Normalization
    /// Set normalization to a specific state (Off, Peak, or LUFS) — used by
    /// the 3-way picker in the controls panel. Snaps target to the mode's
    /// default when switching modes so units stay sensible.
    SetNormalizeState(NormalizeState),
    SetNormalizeTarget(f32),
    /// Scroll the waveform to a specific time (used by the bottom scrollbar).
    SetScroll(f32),
    // Presets
    SavePreset,
    LoadPreset,
    PresetSaved(Result<PathBuf, String>),
    PresetLoaded(Result<(PathBuf, ZoneMap), String>),
    // Project files
    SaveProject,
    OpenProject,
    ProjectSaved(Result<PathBuf, String>),
    /// Project loaded: project path, parsed project, audio buffer.
    ProjectLoaded(Result<(PathBuf, project::Project, AudioBuffer), String>),
    // Auto-detect zones
    /// Open the auto-detect modal.
    OpenAutoDetect,
    /// Close it without applying.
    CloseAutoDetect,
    /// User adjusted the sensitivity slider — recompute preview.
    SetDetectSensitivity(f32),
    /// Apply the previewed splits, replacing existing splits.
    ApplyAutoDetectReplace,
    /// Apply the previewed splits, merged with existing.
    ApplyAutoDetectAdd,
    // Settings modal
    OpenSettings,
    CloseSettings,
    DraftSetTheme(settings::ThemeChoice),
    DraftSetAccent(settings::AccentColor),
    DraftSetWaveformScheme(settings::WaveformScheme),
    PickDefaultOpenDir,
    PickDefaultExportDir,
    DraftSetOpenDir(Option<PathBuf>),
    DraftSetExportDir(Option<PathBuf>),
    SaveSettings,
}

impl App {
    fn new() -> (Self, Task<Msg>) {
        // Optional command-line argument: a path to open at startup.
        // Usage: `peakmuncher [PATH]`. Bail if the file doesn't exist instead
        // of starting empty + showing an error in the status bar — that just
        // looks like a typo silently being eaten.
        let cli_path: Option<PathBuf> = std::env::args()
            .skip(1)
            .find(|a| !a.starts_with('-')) // ignore -h / --help / future flags
            .map(PathBuf::from);
        let initial_task = match cli_path {
            Some(p) if p.exists() => Task::done(Msg::OpenPath(p)),
            Some(p) => {
                eprintln!("PeakMuncher: file not found: {}", p.display());
                Task::none()
            }
            None => Task::none(),
        };

        let (player, status) = match Player::new() {
            Ok(p) => (Some(p), "Open an audio file to begin.".into()),
            Err(e) => (
                None,
                format!("Audio output unavailable ({e}). Open a file — playback disabled."),
            ),
        };
        let (dsp_tx, dsp_rx) = spawn_dsp_worker();
        (
            Self {
                input: None,
                input_path: None,
                processed: None,
                envelope_in: Vec::new(),
                envelope_out: Vec::new(),
                mono_input: Vec::new(),
                raw_input_mono: Arc::new(Vec::new()),
                mono_output: Vec::new(),
                zones: ZoneMap::new(),
                selected_split: None,
                trim_start: None,
                trim_end: None,
                selected_zone: Some(0),
                active_tab: ControlTab::Clipper,
                zone_input_peak_db: None,
                cursor_time: None,
                status,
                canvas_cache: canvas::Cache::new(),
                overlay_cache: canvas::Cache::new(),
                meter_cache: canvas::Cache::new(),
                player,
                playhead_secs: 0.0,
                zoom: 1.0,
                scroll: 0.0,
                snap_enabled: true,
                show_reduction: false,
                spectrum_mode: SpectrumMode::Off,
                spectrogram_in: None,
                spectrogram_out: None,
                spectrum_line_out: None,
                spectrum_line_in: None,
                spectrum_line_frame: None,
                meter_in: 0.0,
                meter_out: 0.0,
                history: vec![HistoryEntry::ParamChange(ZoneMap::new())],
                history_pos: 0,
                file_menu_open: false,
                menu_close_abort: None,
                recent_files: recent::RecentFiles::load(),
                guide_db: None,
                zone_ctx_menu: None,
                zone_clipboard: None,
                normalize_enabled: false,
                normalize_mode: NormalizeMode::Peak,
                normalize_target_db: -0.3,
                auto_detect: None,
                settings: settings::Settings::load(),
                settings_draft: None,
                dsp_gen: 0,
                dsp_tx,
                dsp_rx: Arc::new(Mutex::new(dsp_rx)),
                dsp_pending: false,
            },
            initial_task,
        )
    }

    fn subscription(&self) -> Subscription<Msg> {
        // Bindings lookup happens in update() since on_key_press requires a
        // non-capturing fn pointer; we just forward the raw event.
        let keys = iced::keyboard::on_key_press(|key, modifiers| {
            Some(Msg::KeyPressed(key, modifiers))
        });

        let is_playing = self
            .player
            .as_ref()
            .map(|p| p.is_playing())
            .unwrap_or(false);

        let mut subs: Vec<Subscription<Msg>> = vec![keys];
        // Tick while playing OR while peak meters still need to decay.
        let needs_tick = is_playing || self.meter_in > 0.0 || self.meter_out > 0.0;
        if needs_tick {
            subs.push(time::every(Duration::from_millis(33)).map(|_| Msg::Tick));
        }
        if self.dsp_pending {
            subs.push(time::every(Duration::from_millis(16)).map(|_| Msg::DspPoll));
        }
        Subscription::batch(subs)
    }

    /// Schedule a background DSP rebuild. Returns a Task that spawns the
    /// computation on a thread; when it completes, `Msg::DspDone(gen, samples,
    /// envelope)` arrives. Stale results (gen != current) are dropped.
    fn rebuild_processed(&mut self) -> Task<Msg> {
        self.rebuild_processed_ex(false)
    }

    /// Like rebuild_processed but renders a fast decimated preview (used
    /// during slider drags). Falls back to a full render when normalize is
    /// enabled, because normalize depends on a whole-file peak/LUFS
    /// measurement that the decimated path can't reproduce faithfully —
    /// using the preview there makes the waveform jump on release.
    fn rebuild_processed_preview(&mut self) -> Task<Msg> {
        if self.normalize_enabled {
            self.rebuild_processed_ex(false)
        } else {
            self.rebuild_processed_ex(true)
        }
    }

    fn rebuild_processed_ex(&mut self, preview: bool) -> Task<Msg> {
        let Some(input) = &self.input else { return Task::none() };
        self.dsp_gen = self.dsp_gen.wrapping_add(1);
        let trim_window = self.trim_window_frames();
        // If a job is already pending/running, the worker will drain queued
        // jobs and run only the newest. Just enqueue the new params.
        let job = DspJob {
            gen: self.dsp_gen,
            samples: input.samples.clone(),
            channels: input.channels,
            sample_rate: input.sample_rate,
            zones: self.zones.clone(),
            normalize: self.normalize_enabled,
            normalize_mode: self.normalize_mode,
            normalize_target_db: self.normalize_target_db,
            trim_window,
            raw_input_mono: self.raw_input_mono.clone(),
            preview,
        };
        // Send is fire-and-forget; if the worker died, ignore.
        let _ = self.dsp_tx.send(job);
        self.dsp_pending = true;
        Task::none()
    }

    /// Recompute the cached input peak (dBFS) for the selected zone, as
    /// seen by the clipper — i.e. input × the zone's input gain. Used by the
    /// Ceiling slider's peak marker (drag the ceiling to this dB and you're
    /// just touching the loudest peak; below it you start clipping).
    /// True if any zone has a nonzero fade-in or fade-out. Used to decide
    /// whether moving a trim handle needs a processed-buffer rebuild (fades
    /// anchor to the trim edges, so trim changes the output when fades exist).
    fn any_fade_active(&self) -> bool {
        self.zones
            .zones
            .iter()
            .any(|z| z.fade_in_secs > 0.0 || z.fade_out_secs > 0.0)
    }

    fn recompute_zone_input_peak(&mut self) {
        self.zone_input_peak_db = (|| {
            let input = self.input.as_ref()?;
            let sel = self.selected_zone.unwrap_or(0);
            let dur = input.duration_secs();
            let sr = input.sample_rate as f32;
            // Resolve the selected zone's frame range from splits.
            let (s_secs, e_secs, zparams) = self
                .zones
                .iter_zones(dur)
                .nth(sel)
                .map(|(s, e, z)| (s, e, *z))?;
            if self.raw_input_mono.is_empty() {
                return None;
            }
            let f0 = ((s_secs * sr) as usize).min(self.raw_input_mono.len());
            let f1 = ((e_secs * sr) as usize).min(self.raw_input_mono.len()).max(f0);
            if f1 <= f0 {
                return None;
            }
            let in_gain = crate::dsp::db_to_amp(zparams.input_gain_db);
            let mut peak = 0.0f32;
            for &s in &self.raw_input_mono[f0..f1] {
                let a = (s * in_gain).abs();
                if a > peak {
                    peak = a;
                }
            }
            if peak < 1e-6 {
                None
            } else {
                Some(crate::dsp::amp_to_db(peak))
            }
        })();
    }

    /// Compute the normalize/export measurement window in FRAME indices
    /// from the current trim boundaries. Returns None when trim covers the
    /// whole file (the common case). Thin wrapper over `compute_trim_window`.
    fn trim_window_frames(&self) -> Option<(usize, usize)> {
        let input = self.input.as_ref()?;
        let total_frames = if input.channels > 0 {
            input.samples.len() / input.channels.max(1) as usize
        } else {
            0
        };
        compute_trim_window(
            input.sample_rate,
            total_frames,
            input.duration_secs(),
            self.trim_start,
            self.trim_end,
        )
    }

    fn update(&mut self, msg: Msg) -> Task<Msg> {
        // Auto-close the File menu whenever any of its items are clicked.
        // (Toggles like Snap/Reduction/FFT and actions like Open/Save/Undo/Redo
        // all dismiss the dropdown the way a native menu would.)
        match &msg {
            Msg::OpenFile
            | Msg::OpenPath(_)
            | Msg::SaveFile
            | Msg::Undo
            | Msg::Redo
            | Msg::AddSplitAtCursor
            | Msg::ToggleSnap
            | Msg::ToggleReduction
            | Msg::ToggleGuide
            | Msg::SavePreset
            | Msg::LoadPreset
            | Msg::SaveProject
            | Msg::OpenProject
            | Msg::OpenAutoDetect
            | Msg::OpenSettings
            | Msg::CycleSpectrumMode => {
                self.file_menu_open = false;
            }
            _ => {}
        }
        match msg {
            Msg::OpenFile => {
                let default_dir = self.settings.default_open_dir.clone();
                Task::perform(
                    async move {
                        let mut dialog = rfd::AsyncFileDialog::new()
                            .add_filter("Audio", &["wav", "flac", "mp3", "aiff", "aif", "aifc"]);
                        if let Some(d) = default_dir {
                            dialog = dialog.set_directory(d);
                        }
                        let file = dialog.pick_file().await?;
                        let path = file.path().to_path_buf();
                        match audio_io::load(&path) {
                            Ok(buf) => Some((path, Ok(buf))),
                            Err(e) => Some((path, Err(e.to_string()))),
                        }
                    },
                    Msg::Loaded,
                )
            }
            Msg::OpenPath(path) => Task::perform(
                async move {
                    match audio_io::load(&path) {
                        Ok(buf) => Some((path, Ok(buf))),
                        Err(e) => Some((path, Err(e.to_string()))),
                    }
                },
                Msg::Loaded,
            ),
            Msg::Loaded(Some((path, Err(e)))) => {
                self.status = format!(
                    "Failed to load {}: {e}",
                    path.file_name().unwrap_or_default().to_string_lossy()
                );
                Task::none()
            }
            Msg::Loaded(Some((path, Ok(buf)))) => {
                self.status = format!(
                    "Loaded {} — {:.2}s, {} ch, {} Hz",
                    path.file_name().unwrap_or_default().to_string_lossy(),
                    buf.duration_secs(),
                    buf.channels,
                    buf.sample_rate
                );
                self.recent_files.push(&path);
                self.recent_files.save();
                self.input_path = Some(path.clone());
                let loaded_mono = Arc::new(buf.to_mono());
                self.envelope_in = build_envelope(&loaded_mono, ENVELOPE_WIDTH);
                self.mono_input = loaded_mono.as_ref().clone();
                self.raw_input_mono = loaded_mono;
                self.zones = ZoneMap::new();
                self.selected_split = None;
                // Park trim handles at the file edges (trim nothing yet).
                self.trim_start = Some(0.0);
                self.trim_end = Some(buf.duration_secs());
                self.selected_zone = Some(0);
                self.playhead_secs = 0.0;
                self.zoom = 1.0;
                self.scroll = 0.0;
                // Reset normalize state on file load so settings from a
                // previous file don't silently affect the new one. The
                // project-file load path (further down) intentionally
                // does NOT reset, since that path's whole point is to
                // restore saved state.
                self.normalize_enabled = false;
                self.normalize_mode = NormalizeMode::Peak;
                self.normalize_target_db = -0.3;

                // Initial DSP synchronously so the player has something to
                // play right away. With one zone and no clipping happening,
                // this is essentially a copy and is fast.
                let mut initial = zones::render(
                    &buf.samples,
                    buf.channels,
                    buf.sample_rate,
                    &self.zones,
                    None,
                );
                if self.normalize_enabled {
                    let mut peak = 0.0f32;
                    for &s in &initial {
                        let a = s.abs();
                        if a > peak {
                            peak = a;
                        }
                    }
                    if peak > 1e-6 {
                        let target_amp = 10f32.powf(self.normalize_target_db / 20.0);
                        let gain = target_amp / peak;
                        for s in initial.iter_mut() {
                            *s *= gain;
                        }
                    }
                }
                let processed_samples = Arc::new(initial);
                let mono_out = mono_from_interleaved(&processed_samples, buf.channels);
                self.envelope_out = build_envelope(&mono_out, ENVELOPE_WIDTH);
                self.mono_output = mono_out;
                if let Some(p) = &self.player {
                    p.load(
                        buf.samples.clone(),
                        processed_samples.clone(),
                        buf.channels,
                        buf.sample_rate,
                    );
                }
                self.processed = Some(AudioBuffer {
                    samples: processed_samples,
                    sample_rate: buf.sample_rate,
                    channels: buf.channels,
                    source_bit_depth: buf.source_bit_depth,
                });
                self.input = Some(buf);
                self.recompute_zone_input_peak();
                self.canvas_cache.clear();
                self.overlay_cache.clear();
                // Reset undo history with the empty zone state.
                self.history = vec![HistoryEntry::ParamChange(self.zones.clone())];
                self.history_pos = 0;
                self.spectrogram_in = None;
                self.spectrogram_out = None;
                self.invalidate_spectrum_line();
                // Spawn spectrograms only if the user is currently looking
                // at the spectrogram view; otherwise wait until they ask.
                if self.spectrum_mode == SpectrumMode::Spectrogram {
                    return Task::batch([
                        self.spawn_spectrogram_in(),
                        self.spawn_spectrogram_out(),
                    ]);
                }
                Task::none()
            }
            Msg::Loaded(None) => Task::none(),
            Msg::SaveFile => {
                let Some(input) = self.input.clone() else {
                    self.status = "Nothing to save yet — load a file first.".into();
                    return Task::none();
                };
                // If any zone uses oversampling, re-render the file at high
                // quality before saving. Otherwise reuse the live preview.
                let needs_os = self.zones.zones.iter().any(|z| z.oversampling > 1);
                let zones = self.zones.clone();
                let normalize = self.normalize_enabled;
                let normalize_mode = self.normalize_mode;
                let normalize_target_db = self.normalize_target_db;
                let live = self.processed.clone();
                let default_dir = self.settings.default_export_dir.clone();
                // Trim window in frames (None = export whole file). Same
                // window used for the normalize measurement, so what you
                // hear in the trimmed preview is what gets written.
                let trim_window = self.trim_window_frames();
                Task::perform(
                    async move {
                        let mut dialog = rfd::AsyncFileDialog::new()
                            .add_filter("WAV", &["wav"])
                            .add_filter("FLAC", &["flac"])
                            .add_filter("MP3", &["mp3"])
                            .add_filter("AIFF", &["aiff", "aif"]);
                        if let Some(d) = default_dir {
                            dialog = dialog.set_directory(d);
                        }
                        let file = dialog
                            .save_file()
                            .await
                            .ok_or_else(|| "cancelled".to_string())?;
                        let path = file.path().to_path_buf();
                        // Build the buffer to save.
                        let to_save: AudioBuffer = if needs_os {
                            // Render with oversampling on a worker thread.
                            let result = tokio::task::spawn_blocking(move || {
                                let mut samples = zones::render_with_oversampling(
                                    &input.samples,
                                    input.channels,
                                    input.sample_rate,
                                    &zones,
                                    trim_window,
                                );
                                apply_normalization(
                                    &mut samples,
                                    input.channels,
                                    input.sample_rate,
                                    normalize,
                                    normalize_mode,
                                    normalize_target_db,
                                    trim_window,
                                );
                                AudioBuffer {
                                    samples: Arc::new(samples),
                                    sample_rate: input.sample_rate,
                                    channels: input.channels,
                                    source_bit_depth: input.source_bit_depth,
                                }
                            })
                            .await
                            .map_err(|e| format!("Render failed: {e}"))?;
                            result
                        } else {
                            live.ok_or_else(|| "Nothing to save".to_string())?
                        };
                        // Apply trim: slice the interleaved buffer to the
                        // [start, end) frame window. This is the only
                        // destructive step, and only affects the exported
                        // file — the editor buffer is untouched.
                        let to_save = if let Some((f0, f1)) = trim_window {
                            let ch = to_save.channels.max(1) as usize;
                            let total_frames = to_save.samples.len() / ch;
                            let f0 = f0.min(total_frames);
                            let f1 = f1.min(total_frames).max(f0);
                            if f1 > f0 && (f0 > 0 || f1 < total_frames) {
                                let sliced: Vec<f32> =
                                    to_save.samples[f0 * ch..f1 * ch].to_vec();
                                AudioBuffer {
                                    samples: Arc::new(sliced),
                                    sample_rate: to_save.sample_rate,
                                    channels: to_save.channels,
                                    source_bit_depth: to_save.source_bit_depth,
                                }
                            } else {
                                to_save
                            }
                        } else {
                            to_save
                        };
                        audio_io::save(&path, &to_save).map_err(|e| e.to_string())?;
                        Ok(path)
                    },
                    Msg::Saved,
                )
            }
            Msg::Saved(Ok(p)) => {
                self.status = format!("Exported {}", p.display());
                self.recent_files.push(&p);
                self.recent_files.save();
                Task::none()
            }
            Msg::Saved(Err(e)) => {
                self.status = format!("Save failed: {e}");
                Task::none()
            }
            Msg::AddSplitAtCursor => {
                if let Some(buf) = &self.input {
                    let t = self.maybe_snap(self.playhead_secs);
                    if t > 0.0 && t < buf.duration_secs() {
                        self.zones.add_split(t);
                        self.snapshot();
                        return self.rebuild_processed();
                    }
                }
                Task::none()
            }
            Msg::Canvas(ev) => {
                let mut needs_rebuild = false;
                match ev {
                    CanvasEvent::AddSplit(t) => {
                        let t = self.maybe_snap(t);
                        self.zones.add_split(t);
                        needs_rebuild = true;
                    }
                    CanvasEvent::SelectSplit(i) => {
                        // Snapshot the pre-drag state so undo restores the
                        // split's position before the user grabbed it.
                        self.snapshot();
                        self.selected_split = Some(i);
                        self.canvas_cache.clear();
                    }
                    CanvasEvent::SelectZone(i) => {
                        self.selected_zone = Some(i);
                        self.selected_split = None;
                        self.canvas_cache.clear();
                    }
                    CanvasEvent::Seek(t) => {
                        let zi = self.zones.zone_at(t);
                        self.selected_zone = Some(zi);
                        self.selected_split = None;
                        self.canvas_cache.clear();
                        self.overlay_cache.clear();
                        self.playhead_secs = t.max(0.0);
                        if let (Some(p), Some(buf)) = (&self.player, &self.input) {
                            let frame = (t.max(0.0) * buf.sample_rate as f32) as u64;
                            p.seek(frame);
                        }
                        // Parking the playhead = moving the probe point.
                        self.refresh_spectrum_line();
                    }
                    CanvasEvent::MoveSplit(i, t) => {
                        let lo = if i == 0 { 0.0 } else { self.zones.splits[i - 1] + 1e-3 };
                        let hi = self
                            .zones
                            .splits
                            .get(i + 1)
                            .copied()
                            .unwrap_or_else(|| {
                                self.input
                                    .as_ref()
                                    .map(|b| b.duration_secs())
                                    .unwrap_or(0.0)
                            })
                            - 1e-3;
                        let t = self.maybe_snap(t).clamp(lo, hi);
                        self.zones.splits[i] = t;
                        self.canvas_cache.clear();
                        needs_rebuild = true;
                    }
                    CanvasEvent::Zoom(factor, anchor_secs) => {
                        let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(1.0);
                        let new_zoom = (self.zoom * factor).clamp(1.0, 1_000_000.0);
                        if (new_zoom - self.zoom).abs() > 1e-3 {
                            // Keep `anchor_secs` at the same fractional x of the view.
                            let old_vis = dur / self.zoom;
                            let new_vis = dur / new_zoom;
                            let anchor_frac = ((anchor_secs - self.scroll) / old_vis)
                                .clamp(0.0, 1.0);
                            self.scroll = (anchor_secs - anchor_frac * new_vis)
                                .clamp(0.0, (dur - new_vis).max(0.0));
                            self.zoom = new_zoom;
                            self.canvas_cache.clear();
                            self.overlay_cache.clear();
                        }
                    }
                    CanvasEvent::Pan(delta_secs) => {
                        let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(1.0);
                        let vis = dur / self.zoom;
                        let new_scroll = (self.scroll + delta_secs)
                            .clamp(0.0, (dur - vis).max(0.0));
                        if (new_scroll - self.scroll).abs() > 1e-6 {
                            self.scroll = new_scroll;
                            self.canvas_cache.clear();
                            self.overlay_cache.clear();
                        }
                    }
                    CanvasEvent::Hover(t) => {
                        let threshold = self
                            .input
                            .as_ref()
                            .map(|b| b.duration_secs() / 1500.0)
                            .unwrap_or(0.01);
                        let changed = match (self.cursor_time, t) {
                            (Some(a), Some(b)) => (a - b).abs() > threshold,
                            (None, Some(_)) | (Some(_), None) => true,
                            (None, None) => false,
                        };
                        if changed {
                            self.cursor_time = t;
                            self.overlay_cache.clear();
                        }
                    }
                    CanvasEvent::DragGuide(db) => {
                        let clamped = db.clamp(-60.0, 0.0);
                        if self.guide_db.map(|g| (g - clamped).abs() > 1e-3).unwrap_or(true) {
                            self.guide_db = Some(clamped);
                            self.canvas_cache.clear();
                        }
                    }
                    CanvasEvent::MoveTrimStart(t) => {
                        // Clamp to [0, trim_end] so start can't cross end.
                        let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(t);
                        let end = self.trim_end.unwrap_or(dur);
                        let clamped = t.clamp(0.0, end);
                        if self.trim_start.map(|s| (s - clamped).abs() > 1e-4).unwrap_or(true) {
                            self.trim_start = Some(clamped);
                            self.canvas_cache.clear();
                            // A rebuild is needed when trim affects the live
                            // processed output: that's true for normalize
                            // (whole-file measurement) AND for fades, which
                            // now anchor to the trim edges — moving the trim
                            // moves the fade, so the processed buffer changes.
                            if self.normalize_enabled || self.any_fade_active() {
                                needs_rebuild = true;
                            }
                        }
                    }
                    CanvasEvent::MoveTrimEnd(t) => {
                        // Clamp to [trim_start, duration] so end can't cross start.
                        let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(t);
                        let start = self.trim_start.unwrap_or(0.0);
                        let clamped = t.clamp(start, dur);
                        if self.trim_end.map(|e| (e - clamped).abs() > 1e-4).unwrap_or(true) {
                            self.trim_end = Some(clamped);
                            self.canvas_cache.clear();
                            if self.normalize_enabled || self.any_fade_active() {
                                needs_rebuild = true;
                            }
                        }
                    }
                    CanvasEvent::RightClickZone(zi, split_idx, x, y) => {
                        self.selected_zone = Some(zi);
                        self.zone_ctx_menu = Some((zi, split_idx, x, y));
                        self.canvas_cache.clear();
                    }
                }
                if needs_rebuild {
                    self.rebuild_processed()
                } else {
                    Task::none()
                }
            }
            Msg::DeleteSelectedSplit => {
                if let Some(i) = self.selected_split.take() {
                    self.zones.remove_split(i);
                    self.snapshot();
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::SetCeiling(db) => {
                if let Some(z) = self.current_zone_mut() {
                    z.ceiling_db = db;
                    return self.rebuild_processed_preview();
                }
                Task::none()
            }
            Msg::SetInputGain(db) => {
                if let Some(z) = self.current_zone_mut() {
                    z.input_gain_db = db;
                    self.recompute_zone_input_peak();
                    return self.rebuild_processed_preview();
                }
                Task::none()
            }
            Msg::SetOutputGain(db) => {
                if let Some(z) = self.current_zone_mut() {
                    z.output_gain_db = db;
                    return self.rebuild_processed_preview();
                }
                Task::none()
            }
            Msg::SetDcOffset(v) => {
                if let Some(z) = self.current_zone_mut() {
                    z.dc_offset = v.clamp(-0.5, 0.5);
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::AutoDetectDc => {
                // Measure the mean sample value across the zone, then
                // set dc_offset to its negation. Walks the raw input
                // samples (interleaved across channels) within the
                // currently-selected zone's bounds.
                let Some(sel) = self.selected_zone else {
                    return Task::none();
                };
                let Some(input) = &self.input else {
                    return Task::none();
                };
                let dur = input.frames() as f32 / input.sample_rate.max(1) as f32;
                let zones = &self.zones;
                let (start_s, end_s) =
                    match zones.iter_zones(dur).nth(sel) {
                        Some((s, e, _)) => (s, e),
                        None => return Task::none(),
                    };
                let ch = input.channels.max(1) as usize;
                let frames = input.frames();
                let s0 = ((start_s * input.sample_rate as f32) as usize).min(frames);
                let s1 = ((end_s * input.sample_rate as f32) as usize).min(frames);
                if s1 <= s0 {
                    return Task::none();
                }
                let slice = &input.samples[s0 * ch..s1 * ch];
                if slice.is_empty() {
                    return Task::none();
                }
                let mean: f32 = slice.iter().sum::<f32>() / slice.len() as f32;
                if let Some(z) = self.zones.zones.get_mut(sel) {
                    z.dc_offset = (-mean).clamp(-0.5, 0.5);
                    self.snapshot();
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::SetClipper(c) => {
                if let Some(z) = self.current_zone_mut() {
                    z.clipper = c;
                    self.snapshot();
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::SetFadeIn(secs) => {
                if let Some(z) = self.current_zone_mut() {
                    z.fade_in_secs = secs.max(0.0);
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::SetFadeOut(secs) => {
                if let Some(z) = self.current_zone_mut() {
                    z.fade_out_secs = secs.max(0.0);
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::ToggleDcBlocker => {
                if let Some(z) = self.current_zone_mut() {
                    z.dc_blocker_enabled = !z.dc_blocker_enabled;
                    self.snapshot();
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::SetDcBlockerHz(hz) => {
                if let Some(z) = self.current_zone_mut() {
                    z.dc_blocker_hz = hz.clamp(5.0, 60.0);
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::ApplyProcessing => self.apply_processing(),
            Msg::SetOversampling(n) => {
                if let Some(z) = self.current_zone_mut() {
                    z.oversampling = n.clamp(1, 8);
                    self.snapshot();
                    // Note: live preview never oversamples (too slow on
                    // slider drags). The setting is applied at Save time.
                    // No rebuild needed.
                }
                Task::none()
            }
            Msg::Play => {
                if let Some(p) = &self.player {
                    p.play();
                }
                Task::none()
            }
            Msg::Pause => {
                if let Some(p) = &self.player {
                    p.pause();
                }
                Task::none()
            }
            Msg::Stop => {
                if let Some(p) = &self.player {
                    p.stop();
                }
                self.playhead_secs = 0.0;
                self.overlay_cache.clear();
                Task::none()
            }
            Msg::ToggleSource => {
                if let Some(p) = &self.player {
                    let new = match p.current_source() {
                        Source::Input => Source::Processed,
                        Source::Processed => Source::Input,
                    };
                    p.set_source(new);
                }
                Task::none()
            }
            Msg::KeyPressed(key, modifiers) => {
                use crate::keybindings::KeyAction;
                let Some(action) = self.settings.keybindings.lookup(modifiers, &key)
                else {
                    return Task::none();
                };
                let mapped = match action {
                    KeyAction::TogglePlay => Msg::TogglePlay,
                    KeyAction::Stop => Msg::Stop,
                    KeyAction::AddSplit => Msg::AddSplitAtCursor,
                    KeyAction::ToggleSnap => Msg::ToggleSnap,
                    KeyAction::ToggleReduction => Msg::ToggleReduction,
                    KeyAction::CycleFft => Msg::CycleSpectrumMode,
                    KeyAction::ZoomIn => Msg::ZoomIn,
                    KeyAction::ZoomOut => Msg::ZoomOut,
                    KeyAction::ZoomReset => Msg::ZoomReset,
                    KeyAction::PrevZone => Msg::PrevZone,
                    KeyAction::NextZone => Msg::NextZone,
                    KeyAction::DeleteSplit => Msg::DeleteSelectedSplit,
                    KeyAction::Undo => Msg::Undo,
                    KeyAction::Redo => Msg::Redo,
                };
                Task::done(mapped)
            }
            Msg::Tick => {
                let mut probe_moved = false;
                if let Some(p) = &self.player {
                    let new_pos = p.position_secs();
                    if (new_pos - self.playhead_secs).abs() > 0.01 {
                        self.playhead_secs = new_pos;
                        self.overlay_cache.clear();
                        probe_moved = true;
                    }
                    let (pi, po) = p.take_peaks();
                    let prev_in = self.meter_in;
                    let prev_out = self.meter_out;
                    self.meter_in = self.meter_in.max(pi) * 0.85 + pi * 0.15;
                    self.meter_out = self.meter_out.max(po) * 0.85 + po * 0.15;
                    if self.meter_in < 1e-4 {
                        self.meter_in = 0.0;
                    }
                    if self.meter_out < 1e-4 {
                        self.meter_out = 0.0;
                    }
                    // Repaint the static layer for meter bars and (if active)
                    // the real-time spectrum line. They live inside that
                    // cached layer so without invalidation they wouldn't move.
                    if (self.meter_in - prev_in).abs() > 1e-4
                        || (self.meter_out - prev_out).abs() > 1e-4
                        || self.spectrum_mode == SpectrumMode::Spectrum
                    {
                        self.meter_cache.clear();
                    }
                }
                // Done outside the `&self.player` borrow: probe follows the
                // playhead, recompute the cached spectrum line (no-ops unless
                // the probe moved enough).
                if probe_moved {
                    self.refresh_spectrum_line();
                }
                Task::none()
            }
            Msg::TogglePlay => {
                if let Some(p) = &self.player {
                    if p.is_playing() {
                        p.pause();
                    } else {
                        p.play();
                    }
                }
                Task::none()
            }
            Msg::PrevZone => {
                let n = self.zones.zones.len();
                if n > 0 {
                    let cur = self.selected_zone.unwrap_or(0);
                    self.selected_zone = Some(cur.saturating_sub(1));
                    self.selected_split = None;
                    self.recompute_zone_input_peak();
                    self.canvas_cache.clear();
                }
                Task::none()
            }
            Msg::NextZone => {
                let n = self.zones.zones.len();
                if n > 0 {
                    let cur = self.selected_zone.unwrap_or(0);
                    self.selected_zone = Some((cur + 1).min(n - 1));
                    self.selected_split = None;
                    self.recompute_zone_input_peak();
                    self.canvas_cache.clear();
                }
                Task::none()
            }
            Msg::SelectTab(tab) => {
                self.active_tab = tab;
                Task::none()
            }
            Msg::SelectZoneIdx(i) => {
                if i < self.zones.zones.len() {
                    self.selected_zone = Some(i);
                    self.selected_split = None;
                    self.recompute_zone_input_peak();
                    self.canvas_cache.clear();
                }
                Task::none()
            }
            Msg::DspPoll => {
                let rx = self.dsp_rx.clone();
                let mut latest: Option<DspResult> = None;
                {
                    let rx = rx.lock().unwrap();
                    while let Ok(r) = rx.try_recv() {
                        latest = Some(r);
                    }
                }
                if let Some(res) = latest {
                    // Honor the generation guard (see `rebuild_processed`):
                    // a result from a superseded job must NOT touch the
                    // display buffers. Writing a stale `mono_output` pairs
                    // it with the current-generation `mono_input`, so the
                    // yellow clipping-shadow caps (in_max - out_max) flip
                    // between absent (stale output ≈ input) and full-size
                    // (fresh output) as the ceiling moves — the "no yellow
                    // at -5.2 dB, explosion at -5.3 dB" artifact. Drop the
                    // stale result; the matching-gen result arrives on a
                    // later poll, because the 50 ms DspPoll subscription
                    // stays active while `dsp_pending` is true.
                    if res.gen != self.dsp_gen {
                        return Task::none();
                    }
                    self.dsp_pending = false;
                    if self.input.is_some() {
                        if res.preview {
                            // Decimated preview: update ONLY the display
                            // envelopes. Leave processed/mono buffers intact
                            // (they stay valid from the last full render and
                            // refresh on slider release via Msg::Commit).
                            self.envelope_out = res.envelope;
                            self.envelope_in = res.envelope_in;
                            self.canvas_cache.clear();
                            return Task::none();
                        }
                    }
                    if let Some(input) = &self.input {
                        if let Some(p) = &self.player {
                            p.update_processed(res.processed.clone());
                        }
                        self.envelope_out = res.envelope;
                        self.mono_output = res.mono;
                        // Use the worker-computed scaled input as the red
                        // reference layer (so it tracks normalize gain).
                        self.mono_input = res.mono_in_scaled;
                        // Input envelope is now prebuilt in the worker
                        // (was build_envelope here, which stuttered the UI
                        // thread on long files during slider drags).
                        self.envelope_in = res.envelope_in;
                        self.processed = Some(AudioBuffer {
                            samples: res.processed,
                            sample_rate: input.sample_rate,
                            channels: input.channels,
                            source_bit_depth: input.source_bit_depth,
                        });
                        self.canvas_cache.clear();
                        // Output changed → invalidate output spectrogram.
                        self.spectrogram_out = None;
                        // Output audio changed → recompute the probe-point
                        // spectrum line so the overlay reflects the new
                        // processing (this is the "recompute on processing
                        // change" half of the freeze model).
                        self.invalidate_spectrum_line();
                        self.refresh_spectrum_line();
                        if self.spectrum_mode == SpectrumMode::Spectrogram {
                            return self.spawn_spectrogram_out();
                        }
                    }
                }
                Task::none()
            }
            Msg::ZoomIn => {
                self.zoom_around_playhead(2.0);
                Task::none()
            }
            Msg::ZoomOut => {
                self.zoom_around_playhead(0.5);
                Task::none()
            }
            Msg::ZoomReset => {
                self.zoom = 1.0;
                self.scroll = 0.0;
                self.canvas_cache.clear();
                self.overlay_cache.clear();
                Task::none()
            }
            Msg::ToggleSnap => {
                self.snap_enabled = !self.snap_enabled;
                Task::none()
            }
            Msg::ToggleReduction => {
                self.show_reduction = !self.show_reduction;
                self.canvas_cache.clear();
                Task::none()
            }
            Msg::Commit => {
                self.snapshot();
                // Slider released — replace the decimated preview with a
                // full-resolution render.
                self.rebuild_processed()
            }
            Msg::Undo => {
                if self.history_pos > 0 {
                    self.history_pos -= 1;
                    let task = self.restore_history_entry();
                    self.recompute_zone_input_peak();
                    return task;
                }
                Task::none()
            }
            Msg::Redo => {
                if self.history_pos + 1 < self.history.len() {
                    self.history_pos += 1;
                    let task = self.restore_history_entry();
                    self.recompute_zone_input_peak();
                    return task;
                }
                Task::none()
            }
            Msg::CycleSpectrumMode => {
                self.spectrum_mode = self.spectrum_mode.next();
                self.canvas_cache.clear();
                self.meter_cache.clear();
                if self.spectrum_mode == SpectrumMode::Spectrum {
                    // Compute the probe-point lines now so the view isn't blank.
                    self.refresh_spectrum_line();
                }
                if self.spectrum_mode == SpectrumMode::Spectrogram {
                    let mut tasks = Vec::new();
                    if self.spectrogram_in.is_none() {
                        tasks.push(self.spawn_spectrogram_in());
                    }
                    if self.spectrogram_out.is_none() {
                        tasks.push(self.spawn_spectrogram_out());
                    }
                    if !tasks.is_empty() {
                        return Task::batch(tasks);
                    }
                }
                Task::none()
            }
            Msg::SpectrogramInDone(sg) => {
                self.spectrogram_in = Some(sg);
                self.canvas_cache.clear();
                Task::none()
            }
            Msg::SpectrogramOutDone(sg) => {
                self.spectrogram_out = Some(sg);
                self.canvas_cache.clear();
                Task::none()
            }
            Msg::ToggleFileMenu => {
                self.file_menu_open = !self.file_menu_open;
                // Cancel any pending close from a previous open.
                if let Some(h) = self.menu_close_abort.take() {
                    h.abort();
                }
                Task::none()
            }
            Msg::CloseFileMenu => {
                self.file_menu_open = false;
                self.menu_close_abort = None;
                Task::none()
            }
            Msg::MenuMouseExit => {
                // Schedule a delayed close after 300 ms unless the mouse
                // returns to the menu in the meantime.
                if let Some(h) = self.menu_close_abort.take() {
                    h.abort();
                }
                let close_task = Task::perform(
                    async {
                        tokio::time::sleep(Duration::from_millis(300)).await;
                    },
                    |_| Msg::CloseFileMenu,
                );
                let (task, handle) = Task::abortable(close_task);
                self.menu_close_abort = Some(handle);
                task
            }
            Msg::MenuMouseEnter => {
                if let Some(h) = self.menu_close_abort.take() {
                    h.abort();
                }
                Task::none()
            }
            Msg::ToggleGuide => {
                self.guide_db = match self.guide_db {
                    Some(_) => None,
                    None => Some(-3.0),
                };
                self.canvas_cache.clear();
                Task::none()
            }
            Msg::SetGuideDb(db) => {
                let clamped = db.clamp(-60.0, 0.0);
                if self.guide_db.map(|g| (g - clamped).abs() > 1e-3).unwrap_or(true) {
                    self.guide_db = Some(clamped);
                    self.canvas_cache.clear();
                }
                Task::none()
            }
            Msg::ZoneContextMenu(zi, split_idx, x, y) => {
                self.selected_zone = Some(zi);
                self.zone_ctx_menu = Some((zi, split_idx, x, y));
                self.canvas_cache.clear();
                Task::none()
            }
            Msg::CloseZoneContextMenu => {
                self.zone_ctx_menu = None;
                Task::none()
            }
            Msg::DeleteSplitAt(idx) => {
                if idx < self.zones.splits.len() {
                    self.zones.remove_split(idx);
                    self.snapshot();
                    self.zone_ctx_menu = None;
                    self.selected_split = None;
                    return self.rebuild_processed();
                }
                self.zone_ctx_menu = None;
                Task::none()
            }
            Msg::CopyZoneParams => {
                if let Some(z) = self.selected_zone.and_then(|i| self.zones.zones.get(i)) {
                    self.zone_clipboard = Some(z.clone());
                    self.status = "Copied zone parameters".into();
                }
                self.zone_ctx_menu = None;
                Task::none()
            }
            Msg::PasteZoneParams => {
                if let (Some(src), Some(i)) =
                    (self.zone_clipboard.clone(), self.selected_zone)
                {
                    if let Some(dst) = self.zones.zones.get_mut(i) {
                        *dst = src;
                        self.snapshot();
                        self.zone_ctx_menu = None;
                        return self.rebuild_processed();
                    }
                }
                self.zone_ctx_menu = None;
                Task::none()
            }
            Msg::SetNormalizeState(state) => {
                let new_mode = match state {
                    NormalizeState::Off => {
                        self.normalize_enabled = false;
                        return self.rebuild_processed();
                    }
                    NormalizeState::Peak => NormalizeMode::Peak,
                    NormalizeState::Lufs => NormalizeMode::Lufs,
                };
                let mode_changed = self.normalize_mode != new_mode;
                self.normalize_enabled = true;
                self.normalize_mode = new_mode;
                // Snap target to the new mode's default when switching
                // modes — units differ wildly (-0.3 dBFS vs -0.3 LUFS),
                // carrying the old value over is misleading.
                if mode_changed {
                    let (_, _, default) = self.normalize_mode.range();
                    self.normalize_target_db = default;
                }
                self.rebuild_processed()
            }
            Msg::SetNormalizeTarget(db) => {
                let (lo, hi, _) = self.normalize_mode.range();
                self.normalize_target_db = db.clamp(lo, hi);
                if self.normalize_enabled {
                    // Full render (not preview): normalize depends on a
                    // whole-file peak/LUFS measurement, which the decimated
                    // preview can't approximate faithfully, so dragging the
                    // target slider must use the real render.
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::SetScroll(t) => {
                let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(1.0);
                let vis = dur / self.zoom;
                let max_scroll = (dur - vis).max(0.0);
                let new_scroll = t.clamp(0.0, max_scroll);
                if (new_scroll - self.scroll).abs() > 1e-6 {
                    self.scroll = new_scroll;
                    self.canvas_cache.clear();
                    self.overlay_cache.clear();
                }
                Task::none()
            }
            Msg::SavePreset => {
                let zones = self.zones.clone();
                Task::perform(
                    async move {
                        let file = rfd::AsyncFileDialog::new()
                            .add_filter("PeakMuncher preset", &["pmpreset", "json"])
                            .set_file_name("preset.pmpreset")
                            .save_file()
                            .await
                            .ok_or_else(|| "cancelled".to_string())?;
                        let path = file.path().to_path_buf();
                        let json = serde_json::to_vec_pretty(&zones)
                            .map_err(|e| format!("Serialize failed: {e}"))?;
                        std::fs::write(&path, json)
                            .map_err(|e| format!("Write failed: {e}"))?;
                        Ok(path)
                    },
                    Msg::PresetSaved,
                )
            }
            Msg::LoadPreset => Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("PeakMuncher preset", &["pmpreset", "json"])
                        .pick_file()
                        .await
                        .ok_or_else(|| "cancelled".to_string())?;
                    let path = file.path().to_path_buf();
                    let bytes = std::fs::read(&path)
                        .map_err(|e| format!("Read failed: {e}"))?;
                    let zones: ZoneMap = serde_json::from_slice(&bytes)
                        .map_err(|e| format!("Parse failed: {e}"))?;
                    Ok((path, zones))
                },
                Msg::PresetLoaded,
            ),
            Msg::PresetSaved(Ok(p)) => {
                self.status = format!("Saved preset {}", p.display());
                Task::none()
            }
            Msg::PresetSaved(Err(e)) => {
                if e != "cancelled" {
                    self.status = format!("Preset save failed: {e}");
                }
                Task::none()
            }
            Msg::PresetLoaded(Ok((p, zones))) => {
                // Validate split times against the current file's duration.
                let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(0.0);
                if dur > 0.0 {
                    let mut valid = zones;
                    valid.splits.retain(|s| *s > 0.0 && *s < dur);
                    // Ensure zones.len() == splits.len() + 1.
                    let expected = valid.splits.len() + 1;
                    if valid.zones.len() < expected {
                        valid.zones.resize(expected, ZoneParams::default());
                    } else if valid.zones.len() > expected {
                        valid.zones.truncate(expected);
                    }
                    self.zones = valid;
                } else {
                    self.zones = zones;
                }
                self.selected_zone = Some(0);
                self.selected_split = None;
                self.recompute_zone_input_peak();
                self.snapshot();
                self.status = format!("Loaded preset {}", p.display());
                self.canvas_cache.clear();
                return self.rebuild_processed();
            }
            Msg::PresetLoaded(Err(e)) => {
                if e != "cancelled" {
                    self.status = format!("Preset load failed: {e}");
                }
                Task::none()
            }
            Msg::SaveProject => {
                let Some(audio_path) = self.input_path.clone() else {
                    self.status = "Load a file first.".into();
                    return Task::none();
                };
                let zones = self.zones.clone();
                let view = project::ViewState {
                    zoom: self.zoom,
                    scroll: self.scroll,
                    selected_zone: self.selected_zone.unwrap_or(0),
                    snap_enabled: self.snap_enabled,
                    normalize_enabled: self.normalize_enabled,
                    normalize_target_db: self.normalize_target_db,
                    trim_start: self.trim_start,
                    trim_end: self.trim_end,
                };
                Task::perform(
                    async move {
                        let suggested_name = audio_path
                            .file_stem()
                            .map(|s| format!("{}.pmproj", s.to_string_lossy()))
                            .unwrap_or_else(|| "project.pmproj".to_string());
                        let file = rfd::AsyncFileDialog::new()
                            .add_filter("PeakMuncher project", &["pmproj"])
                            .set_file_name(&suggested_name)
                            .save_file()
                            .await
                            .ok_or_else(|| "cancelled".to_string())?;
                        let project_path = file.path().to_path_buf();
                        let audio_path_relative = project_path
                            .parent()
                            .map(|d| project::make_relative(d, &audio_path));
                        let project = project::Project {
                            version: project::PROJECT_FORMAT_VERSION,
                            audio_path: audio_path.clone(),
                            audio_path_relative,
                            zones,
                            view,
                        };
                        let json = serde_json::to_vec_pretty(&project)
                            .map_err(|e| format!("Serialize failed: {e}"))?;
                        std::fs::write(&project_path, json)
                            .map_err(|e| format!("Write failed: {e}"))?;
                        Ok(project_path)
                    },
                    Msg::ProjectSaved,
                )
            }
            Msg::OpenProject => Task::perform(
                async {
                    let file = rfd::AsyncFileDialog::new()
                        .add_filter("PeakMuncher project", &["pmproj"])
                        .pick_file()
                        .await
                        .ok_or_else(|| "cancelled".to_string())?;
                    let project_path = file.path().to_path_buf();
                    let bytes = std::fs::read(&project_path)
                        .map_err(|e| format!("Read failed: {e}"))?;
                    let project: project::Project = serde_json::from_slice(&bytes)
                        .map_err(|e| format!("Parse failed: {e}"))?;
                    let audio_path = project::resolve_audio_path(&project, &project_path)
                        .ok_or_else(|| {
                            format!("Audio file not found: {}", project.audio_path.display())
                        })?;
                    let buf = audio_io::load(&audio_path).map_err(|e| e.to_string())?;
                    Ok((project_path, project, buf))
                },
                Msg::ProjectLoaded,
            ),
            Msg::ProjectSaved(Ok(p)) => {
                self.status = format!("Saved project {}", p.display());
                Task::none()
            }
            Msg::ProjectSaved(Err(e)) => {
                if e != "cancelled" {
                    self.status = format!("Project save failed: {e}");
                }
                Task::none()
            }
            Msg::ProjectLoaded(Ok((project_path, project, buf))) => {
                // Hand the audio off through the same code path as a regular
                // file load (so envelopes / DSP / player all initialize), then
                // overlay the project's zones and view state.
                let audio_path = project::resolve_audio_path(&project, &project_path)
                    .unwrap_or_else(|| project.audio_path.clone());
                self.recent_files.push(&audio_path);
                self.recent_files.save();
                self.input_path = Some(audio_path.clone());
                self.status = format!(
                    "Loaded project {} — {:.2}s, {} ch, {} Hz",
                    project_path.file_name().unwrap_or_default().to_string_lossy(),
                    buf.duration_secs(),
                    buf.channels,
                    buf.sample_rate
                );
                let loaded_mono = Arc::new(buf.to_mono());
                self.envelope_in = build_envelope(&loaded_mono, ENVELOPE_WIDTH);
                self.mono_input = loaded_mono.as_ref().clone();
                self.raw_input_mono = loaded_mono;
                // Apply zones from the project, validating against the file.
                let dur = buf.duration_secs();
                let mut zones = project.zones;
                zones.splits.retain(|s| *s > 0.0 && *s < dur);
                let expected = zones.splits.len() + 1;
                if zones.zones.len() < expected {
                    zones.zones.resize(expected, ZoneParams::default());
                } else if zones.zones.len() > expected {
                    zones.zones.truncate(expected);
                }
                self.zones = zones;
                self.selected_zone = Some(project.view.selected_zone.min(self.zones.zones.len().saturating_sub(1)));
                self.selected_split = None;
                self.zoom = project.view.zoom.clamp(1.0, 1_000_000.0);
                self.scroll = project.view.scroll.clamp(0.0, dur);
                self.snap_enabled = project.view.snap_enabled;
                self.normalize_enabled = project.view.normalize_enabled;
                self.normalize_target_db = project.view.normalize_target_db;
                // Restore trim, clamped to the actual file duration. Older
                // projects (None) re-park at the edges. Guard ordering so a
                // corrupt pair can't invert.
                let ts = project.view.trim_start.unwrap_or(0.0).clamp(0.0, dur);
                let te = project.view.trim_end.unwrap_or(dur).clamp(0.0, dur);
                self.trim_start = Some(ts.min(te));
                self.trim_end = Some(ts.max(te).max(ts));
                self.playhead_secs = 0.0;
                // Reset undo history with the loaded zone state.
                self.history = vec![HistoryEntry::ParamChange(self.zones.clone())];
                self.history_pos = 0;
                self.spectrogram_in = None;
                self.spectrogram_out = None;
                self.invalidate_spectrum_line();

                // Run the initial DSP synchronously.
                // Window from the just-restored trim, computed against buf
                // (self.input isn't assigned until below). Uses the shared
                // helper so the clamp logic matches trim_window_frames().
                let init_trim_window = compute_trim_window(
                    buf.sample_rate,
                    buf.samples.len() / buf.channels.max(1) as usize,
                    buf.duration_secs(),
                    self.trim_start,
                    self.trim_end,
                );
                let mut initial = zones::render(
                    &buf.samples,
                    buf.channels,
                    buf.sample_rate,
                    &self.zones,
                    init_trim_window,
                );
                apply_normalization(
                    &mut initial,
                    buf.channels,
                    buf.sample_rate,
                    self.normalize_enabled,
                    self.normalize_mode,
                    self.normalize_target_db,
                    init_trim_window,
                );
                let processed_samples = Arc::new(initial);
                let mono_out = mono_from_interleaved(&processed_samples, buf.channels);
                self.envelope_out = build_envelope(&mono_out, ENVELOPE_WIDTH);
                self.mono_output = mono_out;
                if let Some(p) = &self.player {
                    p.load(
                        buf.samples.clone(),
                        processed_samples.clone(),
                        buf.channels,
                        buf.sample_rate,
                    );
                }
                self.processed = Some(AudioBuffer {
                    samples: processed_samples,
                    sample_rate: buf.sample_rate,
                    channels: buf.channels,
                    source_bit_depth: buf.source_bit_depth,
                });
                self.input = Some(buf);
                self.recompute_zone_input_peak();
                self.canvas_cache.clear();
                self.overlay_cache.clear();
                Task::none()
            }
            Msg::ProjectLoaded(Err(e)) => {
                if e != "cancelled" {
                    self.status = format!("Project open failed: {e}");
                }
                Task::none()
            }
            Msg::OpenAutoDetect => {
                if self.input.is_none() {
                    self.status = "Load a file first.".into();
                    return Task::none();
                }
                let mut state = AutoDetectState::default();
                // Compute initial preview at default sensitivity.
                if let Some(input) = &self.input {
                    state.preview_splits =
                        detect::detect_splits(&self.mono_input, input.sample_rate, state.sensitivity);
                }
                self.auto_detect = Some(state);
                Task::none()
            }
            Msg::CloseAutoDetect => {
                self.auto_detect = None;
                Task::none()
            }
            Msg::SetDetectSensitivity(s) => {
                if let Some(state) = &mut self.auto_detect {
                    state.sensitivity = s.clamp(0.0, 1.0);
                    if let Some(input) = &self.input {
                        state.preview_splits = detect::detect_splits(
                            &self.mono_input,
                            input.sample_rate,
                            state.sensitivity,
                        );
                    }
                }
                Task::none()
            }
            Msg::ApplyAutoDetectReplace => {
                if let Some(state) = self.auto_detect.take() {
                    let n = state.preview_splits.len();
                    self.zones.splits = state.preview_splits;
                    // Reset zones to one default per region.
                    self.zones.zones = vec![ZoneParams::default(); n + 1];
                    self.selected_zone = Some(0);
                    self.selected_split = None;
                    self.snapshot();
                    self.status = format!("Auto-detected {n} split(s)");
                    self.canvas_cache.clear();
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::ApplyAutoDetectAdd => {
                if let Some(state) = self.auto_detect.take() {
                    let mut added = 0;
                    for t in state.preview_splits {
                        // Use the same min-spacing rule as add_split, which
                        // refuses near-duplicates already.
                        let before = self.zones.splits.len();
                        self.zones.add_split(t);
                        if self.zones.splits.len() > before {
                            added += 1;
                        }
                    }
                    self.snapshot();
                    self.status = format!("Added {added} auto-detected split(s)");
                    self.canvas_cache.clear();
                    return self.rebuild_processed();
                }
                Task::none()
            }
            Msg::OpenSettings => {
                self.settings_draft = Some(self.settings.clone());
                Task::none()
            }
            Msg::CloseSettings => {
                self.settings_draft = None;
                Task::none()
            }
            Msg::DraftSetTheme(t) => {
                if let Some(d) = &mut self.settings_draft {
                    d.theme = t;
                }
                Task::none()
            }
            Msg::DraftSetAccent(a) => {
                if let Some(d) = &mut self.settings_draft {
                    d.accent = a;
                }
                Task::none()
            }
            Msg::DraftSetWaveformScheme(w) => {
                if let Some(d) = &mut self.settings_draft {
                    d.waveform_scheme = w;
                }
                Task::none()
            }
            Msg::PickDefaultOpenDir => Task::perform(
                async {
                    rfd::AsyncFileDialog::new()
                        .pick_folder()
                        .await
                        .map(|f| f.path().to_path_buf())
                },
                Msg::DraftSetOpenDir,
            ),
            Msg::PickDefaultExportDir => Task::perform(
                async {
                    rfd::AsyncFileDialog::new()
                        .pick_folder()
                        .await
                        .map(|f| f.path().to_path_buf())
                },
                Msg::DraftSetExportDir,
            ),
            Msg::DraftSetOpenDir(p) => {
                if let (Some(d), Some(p)) = (&mut self.settings_draft, p) {
                    d.default_open_dir = Some(p);
                }
                Task::none()
            }
            Msg::DraftSetExportDir(p) => {
                if let (Some(d), Some(p)) = (&mut self.settings_draft, p) {
                    d.default_export_dir = Some(p);
                }
                Task::none()
            }
            Msg::SaveSettings => {
                if let Some(draft) = self.settings_draft.take() {
                    self.settings = draft;
                    self.settings.save();
                    self.status = "Settings saved.".into();
                }
                Task::none()
            }
        }
    }

    /// Zoom by `factor` keeping the playhead position fixed in the view
    /// (or centered if no file/playhead).
    fn zoom_around_playhead(&mut self, factor: f32) {
        let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(1.0);
        let new_zoom = (self.zoom * factor).clamp(1.0, 1_000_000.0);
        if (new_zoom - self.zoom).abs() < 1e-3 {
            return;
        }
        let anchor = self.playhead_secs.clamp(0.0, dur);
        let old_vis = dur / self.zoom;
        let new_vis = dur / new_zoom;
        let anchor_frac = ((anchor - self.scroll) / old_vis).clamp(0.0, 1.0);
        self.scroll = (anchor - anchor_frac * new_vis)
            .clamp(0.0, (dur - new_vis).max(0.0));
        self.zoom = new_zoom;
        self.canvas_cache.clear();
        self.overlay_cache.clear();
    }

    fn current_zone_mut(&mut self) -> Option<&mut ZoneParams> {
        let i = self.selected_zone?;
        self.zones.zones.get_mut(i)
    }

    /// Push the current zone state onto the undo stack as a ParamChange.
    /// Drops any redo history past the current position.
    fn snapshot(&mut self) {
        // No-op if nothing changed since the current snapshot. Only
        // compare against ParamChange entries — an Apply entry stores a
        // pre-Apply state that's different by design.
        if let Some(HistoryEntry::ParamChange(cur)) = self.history.get(self.history_pos) {
            if zones_eq(cur, &self.zones) {
                return;
            }
        }
        self.history.truncate(self.history_pos + 1);
        self.history.push(HistoryEntry::ParamChange(self.zones.clone()));
        self.history_pos = self.history.len() - 1;
        // Cap total entries at 100 to bound memory. Apply entries take
        // far more memory than ParamChange entries, so we also enforce a
        // separate 5-Apply cap (handled in apply_processing()).
        if self.history.len() > 100 {
            let drop = self.history.len() - 100;
            self.history.drain(0..drop);
            self.history_pos -= drop;
        }
    }

    /// "Apply" — bake the current processed output into the input, then
    /// reset processing params to defaults. The current visualization
    /// state (yellow reference, blue output) becomes uniform — both
    /// trace the same audio, since no processing is applied yet on top
    /// of the new input.
    ///
    /// Reversible: a history entry is pushed that captures both the
    /// pre-bake and post-bake state, so undo/redo navigate cleanly
    /// across the Apply.
    /// Build the "Apply tracker" widget: a row of 5 small SVG dots.
    /// Filled dots represent Applies currently baked into the audio
    /// (entries at or before history_pos); empty dots represent unused
    /// capacity. Applies that exist in history but are "ahead" of
    /// history_pos (recoverable via redo) are drawn as empty — they
    /// aren't currently affecting the audio.
    fn apply_indicator_row(&self) -> Element<'_, Msg> {
        let mut active: usize = 0;
        for (i, entry) in self.history.iter().enumerate() {
            if i > self.history_pos {
                break;
            }
            if matches!(entry, HistoryEntry::Apply(_)) {
                active += 1;
            }
        }
        let cap: usize = 5;
        let active = active.min(cap);
        let mut r = row![].spacing(2).align_y(iced::Alignment::Center);
        for i in 0..cap {
            let bytes = if i < active {
                ICON_CIRCLE_FILLED
            } else {
                ICON_CIRCLE_EMPTY
            };
            let handle = svg::Handle::from_memory(bytes);
            r = r.push(svg(handle).width(12).height(12));
        }
        r.into()
    }

    fn apply_processing(&mut self) -> Task<Msg> {
        let Some(input) = self.input.clone() else {
            return Task::none();
        };
        let trim_window = self.trim_window_frames();
        // Compute the processed output now (synchronous, since we need
        // it to bake — the worker pipeline is async and would race).
        let mut processed_vec = zones::render(
            &input.samples,
            input.channels,
            input.sample_rate,
            &self.zones,
            trim_window,
        );
        let _ = apply_normalization(
            &mut processed_vec,
            input.channels,
            input.sample_rate,
            self.normalize_enabled,
            self.normalize_mode,
            self.normalize_target_db,
            trim_window,
        );
        let post_input = Arc::new(processed_vec);
        let post_mono_input = mono_from_interleaved(&post_input, input.channels);
        let post_envelope_in = build_envelope(&post_mono_input, ENVELOPE_WIDTH);

        // Build the history frame.
        let frame = ApplyFrame {
            prev_input: input.samples.clone(),
            prev_mono_input: self.mono_input.clone(),
            prev_envelope_in: self.envelope_in.clone(),
            prev_zones: self.zones.clone(),
            prev_normalize_enabled: self.normalize_enabled,
            prev_normalize_mode: self.normalize_mode,
            prev_normalize_target_db: self.normalize_target_db,
            post_input: post_input.clone(),
            post_mono_input: post_mono_input.clone(),
            post_envelope_in: post_envelope_in.clone(),
        };

        // Push to history (truncate redo first).
        self.history.truncate(self.history_pos + 1);
        self.history.push(HistoryEntry::Apply(frame));
        self.history_pos = self.history.len() - 1;

        // Enforce the 5-Apply cap. If we now have more than 5 Applies,
        // drop everything up to and including the oldest Apply entry.
        let apply_count = self
            .history
            .iter()
            .filter(|e| matches!(e, HistoryEntry::Apply(_)))
            .count();
        if apply_count > 5 {
            if let Some(idx) = self
                .history
                .iter()
                .position(|e| matches!(e, HistoryEntry::Apply(_)))
            {
                self.history.drain(0..=idx);
                self.history_pos = self.history_pos.saturating_sub(idx + 1);
            }
        }

        // Swap in the post-bake state.
        if let Some(inp) = self.input.as_mut() {
            inp.samples = post_input;
        }
        self.mono_input = post_mono_input.clone();
        self.raw_input_mono = Arc::new(post_mono_input);
        self.envelope_in = post_envelope_in;
        // Reset processing params. ZoneParams::default() is now true
        // identity (Hard clipper @ 0 dB) — running it on the baked
        // input produces exactly the baked input, so rebuild_processed()
        // below won't shave any signal off.
        self.zones = ZoneMap::new();
        self.normalize_enabled = false;
        self.selected_zone = if self.zones.zones.is_empty() { None } else { Some(0) };
        self.selected_split = None;
        // The baked buffer is the new input; recompute the peak marker to
        // reflect the applied state (zones are now default/identity).
        self.recompute_zone_input_peak();
        self.canvas_cache.clear();

        // Trigger a rebuild — with zones=default + normalize=off, the
        // worker will produce processed == input. This synchronizes
        // mono_output and envelope_out to match the new state.
        self.rebuild_processed()
    }

    /// Restore app state to whatever history[history_pos] describes.
    /// Called by Undo and Redo after they move history_pos.
    fn restore_history_entry(&mut self) -> Task<Msg> {
        let pos = self.history_pos;
        let entry = self.history.get(pos).cloned();
        match entry {
            Some(HistoryEntry::ParamChange(z)) => {
                self.zones = z;
                // Audio input: walk back to find most recent Apply <= pos.
                // If found, its post_input is the current input. If not,
                // the original loaded file is the input.
                self.restore_input_from_history(pos);
                // Normalize state: walk back from pos+1 looking for the
                // next Apply. If found, restore from its prev_* fields
                // (those capture the pre-Apply normalize state, which is
                // what we want at this position since we just undid that
                // Apply or are at a ParamChange before it). If no Apply
                // ahead, we're in the "before any Apply" region; restore
                // from the oldest Apply's prev_*, or leave alone if no
                // Apply exists in history at all.
                self.restore_normalize_state_for_pos(pos);
            }
            Some(HistoryEntry::Apply(frame)) => {
                // Position i is the post-bake state.
                if let Some(input) = self.input.as_mut() {
                    input.samples = frame.post_input.clone();
                }
                self.mono_input = frame.post_mono_input.clone();
                self.raw_input_mono = Arc::new(frame.post_mono_input.clone());
                self.envelope_in = frame.post_envelope_in.clone();
                // After Apply, zones reset to defaults — a single zone
                // covering the whole file with default params. Normalize
                // also resets (it was baked into the audio).
                self.zones = ZoneMap::new();
                self.normalize_enabled = false;
            }
            None => {}
        }
        // Selection sanity
        let n = self.zones.zones.len();
        if let Some(i) = self.selected_zone {
            if i >= n {
                self.selected_zone = if n > 0 { Some(n - 1) } else { None };
            }
        }
        self.selected_split = None;
        self.canvas_cache.clear();
        self.rebuild_processed()
    }

    /// Restore normalize_enabled / mode / target to the state that was
    /// active at position `pos`. The strategy:
    ///   - Look at history[pos+1..]: if the next Apply ahead of us has
    ///     prev_* fields, those describe the normalize state that was
    ///     active immediately before that Apply — which is also the
    ///     state at pos (since param changes don't touch normalize and
    ///     normalize toggles aren't snapshotted).
    ///   - If no Apply exists ahead, look for any Apply (it must be at
    ///     or before pos, meaning we already moved past it — but that
    ///     case shouldn't reach here because the Apply branch above
    ///     would have handled it. Defensive fallback: leave state alone.)
    ///   - If no Apply anywhere, leave alone (initial state was already
    ///     correct at load and never changed history-wise).
    fn restore_normalize_state_for_pos(&mut self, pos: usize) {
        for i in (pos + 1)..self.history.len() {
            if let HistoryEntry::Apply(frame) = &self.history[i] {
                self.normalize_enabled = frame.prev_normalize_enabled;
                self.normalize_mode = frame.prev_normalize_mode;
                self.normalize_target_db = frame.prev_normalize_target_db;
                return;
            }
        }
        // No Apply ahead — meaning all of history's Apply entries (if
        // any) are at-or-before pos. In that case the "current" normalize
        // state should be what it was after the most recent Apply — i.e.
        // false (defaults), since Apply resets normalize. Leave alone
        // if there are no Applies in history at all.
        for i in (0..=pos).rev() {
            if matches!(self.history.get(i), Some(HistoryEntry::Apply(_))) {
                self.normalize_enabled = false;
                return;
            }
        }
    }

    /// Walk back from `pos` (inclusive) to find the most recent Apply
    /// entry, and set the input audio accordingly:
    ///   - If found: use that Apply's post_input as the current input.
    ///   - If none: restore from the oldest Apply's prev_input (which
    ///     captures the original file). If there's no Apply anywhere,
    ///     self.input is already the original — nothing to do.
    fn restore_input_from_history(&mut self, pos: usize) {
        // Most-recent-Apply-at-or-before-pos path.
        for i in (0..=pos).rev() {
            if let Some(HistoryEntry::Apply(frame)) = self.history.get(i) {
                let frame = frame.clone();
                if let Some(input) = self.input.as_mut() {
                    input.samples = frame.post_input;
                }
                self.mono_input = frame.post_mono_input.clone();
                self.raw_input_mono = Arc::new(frame.post_mono_input);
                self.envelope_in = frame.post_envelope_in;
                return;
            }
        }
        // No Apply at-or-before pos. Look for one *anywhere* in history
        // — its prev_input is the original file. (User undid back past
        // their first Apply.)
        for entry in self.history.clone().into_iter() {
            if let HistoryEntry::Apply(frame) = entry {
                if let Some(input) = self.input.as_mut() {
                    input.samples = frame.prev_input;
                }
                self.mono_input = frame.prev_mono_input.clone();
                self.raw_input_mono = Arc::new(frame.prev_mono_input);
                self.envelope_in = frame.prev_envelope_in;
                return;
            }
        }
        // No Apply anywhere — input is still the original file.
    }

    /// Recompute the cached probe-point spectrum lines (in + out) IF the probe
    /// moved or audio changed. Cheap to call often — it no-ops when the frame
    /// hasn't changed. The actual FFT (large, averaged) runs only on change,
    /// NOT on the 30 Hz meter repaint. Probe = playhead if present, else the
    /// left edge of the current view (scroll).
    fn refresh_spectrum_line(&mut self) {
        if self.spectrum_mode != SpectrumMode::Spectrum || self.mono_output.is_empty() {
            return;
        }
        let sr = self.input.as_ref().map(|b| b.sample_rate).unwrap_or(44100);
        let probe_secs = self.playhead_secs;
        let frame = (probe_secs * sr as f32) as usize;
        // Only recompute if the probe moved by a meaningful amount (≥ a small
        // fraction of the FFT window) or nothing is cached yet.
        let moved = match self.spectrum_line_frame {
            Some(prev) => frame.abs_diff(prev) >= fft::SPECTRUM_FFT_SIZE / 16,
            None => true,
        };
        if !moved && self.spectrum_line_out.is_some() {
            return;
        }
        self.spectrum_line_out = fft::spectrum_at(&self.mono_output, frame).map(Arc::new);
        self.spectrum_line_in = if self.mono_input.is_empty() {
            None
        } else {
            fft::spectrum_at(&self.mono_input, frame).map(Arc::new)
        };
        self.spectrum_line_frame = Some(frame);
        // The spectrum line is drawn into meter_cache (the fast bottom strip).
        // Updating the data alone isn't enough — without clearing the cache the
        // old image is redrawn. Clearing here is what makes parking the
        // playhead (not just playback) update the spectrum.
        self.meter_cache.clear();
    }

    /// Invalidate the cached spectrum lines (call when audio changes — load,
    /// DSP done, Apply, undo/redo — so the next refresh recomputes).
    fn invalidate_spectrum_line(&mut self) {
        self.spectrum_line_out = None;
        self.spectrum_line_in = None;
        self.spectrum_line_frame = None;
    }

    /// Apply spectrogram computation in the background after a file load
    /// or DSP completion.
    fn spawn_spectrogram_in(&self) -> Task<Msg> {
        if self.mono_input.is_empty() {
            return Task::none();
        }
        let mono = self.mono_input.clone();
        let sr = self.input.as_ref().map(|b| b.sample_rate).unwrap_or(44100);
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || Arc::new(fft::compute_spectrogram(&mono, sr)))
                    .await
                    .ok()
            },
            |maybe_sg| {
                Msg::SpectrogramInDone(maybe_sg.unwrap_or_else(|| {
                    Arc::new(fft::Spectrogram {
                        slices: Vec::new(),
                        hop: 1024,
                        sample_rate: 44100,
                    })
                }))
            },
        )
    }

    fn spawn_spectrogram_out(&self) -> Task<Msg> {
        if self.mono_output.is_empty() {
            return Task::none();
        }
        let mono = self.mono_output.clone();
        let sr = self.input.as_ref().map(|b| b.sample_rate).unwrap_or(44100);
        Task::perform(
            async move {
                tokio::task::spawn_blocking(move || Arc::new(fft::compute_spectrogram(&mono, sr)))
                    .await
                    .ok()
            },
            |maybe_sg| {
                Msg::SpectrogramOutDone(maybe_sg.unwrap_or_else(|| {
                    Arc::new(fft::Spectrogram {
                        slices: Vec::new(),
                        hop: 1024,
                        sample_rate: 44100,
                    })
                }))
            },
        )
    }
    /// 15 ms is roughly one period of 67 Hz, so we always find a crossing
    /// even on bass-heavy material; small enough that the snap stays close
    /// to where the user intended.
    fn maybe_snap(&self, t: f32) -> f32 {
        if !self.snap_enabled {
            return t;
        }
        let Some(input) = &self.input else { return t };
        if self.mono_input.is_empty() {
            return t;
        }
        waveform::snap_to_zero_crossing(&self.mono_input, input.sample_rate, t, 15.0)
    }

    fn view(&self) -> Element<'_, Msg> {
        let is_playing = self
            .player
            .as_ref()
            .map(|p| p.is_playing())
            .unwrap_or(false);

        let play_pause = if is_playing {
            icon_button(ICON_PAUSE, Msg::Pause, self.settings.accent.color_hover())
        } else {
            icon_button(ICON_PLAY, Msg::Play, self.settings.accent.color_hover())
        };

        let time_label = fmt_time(self.playhead_secs);

        // Trim readout pieces, shown only when trim is active (handles
        // moved off the file edges). Rendered as: [scissors] start [arrow]
        // end   out <duration>.
        let trim_parts: Option<(String, String, String)> = self.input.as_ref().and_then(|buf| {
            let dur = buf.duration_secs();
            let ts = self.trim_start.unwrap_or(0.0);
            let te = self.trim_end.unwrap_or(dur);
            if ts <= 1e-4 && te >= dur - 1e-4 {
                None
            } else {
                let out = (te - ts).max(0.0);
                Some((fmt_time(ts), fmt_time(te), fmt_time(out)))
            }
        });

        // ---- Top bar: File menu + file info | playhead | zoom | transport | A/B label ----
        let accent = self.settings.accent.color();
        let accent_hover = self.settings.accent.color_hover();
        let file_menu_btn = icon_button(ICON_MENU, Msg::ToggleFileMenu, accent_hover);

        let top_bar = row![
            file_menu_btn,
            // Pulled-out menu items. Toggle icons light up when their
            // setting is enabled (snap, reduction overlay, guide line).
            // Settings is a non-toggle — just opens the modal.
            toggle_icon_button(
                ICON_ZERO_CROSSING,
                Msg::ToggleSnap,
                self.snap_enabled,
                accent,
                accent_hover,
            ),
            toggle_icon_button(
                ICON_REDUCTION,
                Msg::ToggleReduction,
                self.show_reduction,
                accent,
                accent_hover,
            ),
            toggle_icon_button(
                ICON_GUIDES,
                Msg::ToggleGuide,
                self.guide_db.is_some(),
                accent,
                accent_hover,
            ),
            // FFT/spectrogram view cycle. Lit whenever a spectrum view is
            // active (Spectrum or Spectrogram); click cycles through the
            // three modes (Off → Spectrum → Spectrogram → Off).
            toggle_icon_button(
                ICON_FFT_VIEW,
                Msg::CycleSpectrumMode,
                self.spectrum_mode != SpectrumMode::Off,
                accent,
                accent_hover,
            ),
            icon_button(ICON_SETTINGS, Msg::OpenSettings, accent_hover),
            Space::with_width(12),
            text(&self.status).size(13),
            Space::with_width(Length::Fill),
            // Trim readout (only present when trimming): scissors icon,
            // start time, arrow, end time, then the output duration. All
            // accent-tinted so the cluster stands apart from the playhead
            // time next to it.
            {
                let el: Element<'_, Msg> = match &trim_parts {
                    Some((start, end, out)) => {
                        let scissors = svg(svg::Handle::from_memory(ICON_SCISSORS))
                            .width(14)
                            .height(14)
                            .style(move |_theme: &Theme, _status| {
                                iced::widget::svg::Style { color: Some(accent) }
                            });
                        let arrow = svg(svg::Handle::from_memory(ICON_ARROW_RIGHT))
                            .width(14)
                            .height(14)
                            .style(move |_theme: &Theme, _status| {
                                iced::widget::svg::Style { color: Some(accent) }
                            });
                        row![
                            scissors,
                            Space::with_width(5),
                            text(start.clone()).size(13).color(accent),
                            Space::with_width(5),
                            arrow,
                            Space::with_width(5),
                            text(end.clone()).size(13).color(accent),
                            Space::with_width(12),
                            text(format!("out {out}")).size(13).color(accent),
                            Space::with_width(16),
                        ]
                        .align_y(iced::Alignment::Center)
                        .into()
                    }
                    None => Space::with_width(0).into(),
                };
                el
            },
            text(time_label).size(13),
            Space::with_width(16),
            text(format!("{:.1}x", self.zoom)).size(12).width(40),
            icon_button(ICON_ZOOM_OUT, Msg::ZoomOut, accent_hover),
            icon_button(ICON_ZOOM_IN, Msg::ZoomIn, accent_hover),
            icon_button(ICON_ZOOM_FIT, Msg::ZoomReset, accent_hover),
            Space::with_width(12),
            play_pause,
            icon_button(ICON_STOP, Msg::Stop, accent_hover),
            // A/B compare: icon toggle. Lit when listening to the PROCESSED
            // (B) output, unlit on the ORIGINAL (A) — the icon itself shows
            // which side is active, so no text label is needed.
            toggle_icon_button(
                ICON_COMPARE,
                Msg::ToggleSource,
                self.player
                    .as_ref()
                    .map(|p| matches!(p.current_source(), Source::Processed))
                    .unwrap_or(false),
                accent,
                accent_hover,
            ),
            Space::with_width(12),
            // Apply: bake current processed output into a new "input",
            // resetting parameters. Like a flatten/merge-down — lets the
            // user do another round of editing as if starting fresh,
            // while keeping the original recoverable via undo.
            icon_button(ICON_APPLY, Msg::ApplyProcessing, accent_hover),
            // Apply tracker: 5 small SVG dots showing how many Apply
            // operations are baked into the current audio state.
            // Filled = active (at or before history_pos), empty = unused
            // slot. Applies that are "ahead" of history_pos (recoverable
            // via redo) are shown as empty here since they're not
            // currently affecting the audio.
            self.apply_indicator_row(),
        ]
        .spacing(8)
        .padding(8)
        .align_y(iced::Alignment::Center);

        // ---- File dropdown menu (conditionally inserted) ----
        let file_menu: Option<Element<'_, Msg>> = if self.file_menu_open {
            // Each menu item closes the menu via a sequence: send the action
            // then immediately CloseFileMenu. iced 0.13 doesn't have a clean
            // "do two things" message API, so we encode the close inline by
            // having items send a wrapper message — but to keep things simple,
            // each menu button just sends its primary action; the menu is
            // hidden by user clicking File again or by the click landing on
            // the menu item triggering a state change. For UX, we explicitly
            // listen for clicks outside via a transparent overlay button row
            // below the menu.
            // Menu-item style: matches the dropdown menus — dark background
            // (from the container), accent text, accent-tinted hover.
            let accent = self.settings.accent.color();
            let menu_item_style = move |_theme: &Theme, status: button::Status| {
                let bg = match status {
                    button::Status::Hovered | button::Status::Pressed => {
                        Some(iced::Background::Color(Color { a: 0.22, ..accent }))
                    }
                    _ => None, // transparent — the container's dark shows through
                };
                button::Style {
                    background: bg,
                    text_color: accent,
                    border: iced::Border {
                        radius: 2.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }
            };
            // Disabled-look style for unavailable items (dim text, no hover).
            let menu_item_disabled_style = move |_theme: &Theme, _status: button::Status| {
                button::Style {
                    background: None,
                    text_color: Color { a: 0.35, ..accent },
                    border: iced::Border {
                        radius: 2.0.into(),
                        ..Default::default()
                    },
                    ..Default::default()
                }
            };
            let mi = |label: String, msg: Msg| -> Element<'_, Msg> {
                button(text(label).size(13))
                    .on_press(msg)
                    .padding([4, 14])
                    .width(Length::Fill)
                    .style(menu_item_style)
                    .into()
            };
            let mi_disabled = |label: String| -> Element<'_, Msg> {
                button(text(label).size(13))
                    .padding([4, 14])
                    .width(Length::Fill)
                    .style(menu_item_disabled_style)
                    .into()
            };
            let undo_item: Element<'_, Msg> = if self.history_pos > 0 {
                mi("Undo            Ctrl+Z".to_string(), Msg::Undo)
            } else {
                mi_disabled("Undo            Ctrl+Z".to_string())
            };
            let redo_item: Element<'_, Msg> = if self.history_pos + 1 < self.history.len() {
                mi("Redo      Ctrl+Shift+Z".to_string(), Msg::Redo)
            } else {
                mi_disabled("Redo      Ctrl+Shift+Z".to_string())
            };
            // Recent files entries (rendered as a sub-section between Open and Save).
            let recent_items: Vec<Element<'_, Msg>> = self
                .recent_files
                .paths
                .iter()
                .take(8)
                .map(|p| {
                    let display = p
                        .file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_else(|| p.display().to_string());
                    let label = format!("  {}", display);
                    mi(label, Msg::OpenPath(p.clone()))
                })
                .collect();

            let mut menu_items: Vec<Element<'_, Msg>> = Vec::new();
            menu_items.push(mi("Open…                Ctrl+O".to_string(), Msg::OpenFile));
            if !recent_items.is_empty() {
                menu_items.push(
                    text("  Recent")
                        .size(11)
                        .color(Color::from_rgba(1.0, 1.0, 1.0, 0.45))
                        .into(),
                );
                menu_items.extend(recent_items);
            }
            menu_items.push(mi(
                "Export…                       ".to_string(),
                Msg::SaveFile,
            ));
            menu_items.push(horizontal_rule(1).into());
            menu_items.push(mi(
                "Open project…                ".to_string(),
                Msg::OpenProject,
            ));
            menu_items.push(mi(
                "Save project…                ".to_string(),
                Msg::SaveProject,
            ));
            menu_items.push(horizontal_rule(1).into());
            menu_items.push(mi(
                "Save zone preset…            ".to_string(),
                Msg::SavePreset,
            ));
            menu_items.push(mi(
                "Load zone preset…            ".to_string(),
                Msg::LoadPreset,
            ));
            menu_items.push(horizontal_rule(1).into());
            menu_items.push(undo_item);
            menu_items.push(redo_item);
            menu_items.push(horizontal_rule(1).into());
            menu_items.push(mi(
                "Add split @ playhead       S".to_string(),
                Msg::AddSplitAtCursor,
            ));
            menu_items.push(mi(
                "Auto-detect zones…           ".to_string(),
                Msg::OpenAutoDetect,
            ));
            let menu = column(menu_items).spacing(0).width(280);
            Some(
                mouse_area(
                    container(menu)
                        .padding(2)
                        .style(|_theme: &Theme| container::Style {
                            background: Some(iced::Background::Color(Color::from_rgb(
                                0.1255, 0.1333, 0.1451,
                            ))),
                            text_color: Some(Color::WHITE),
                            border: iced::Border {
                                color: Color::from_rgba(1.0, 1.0, 1.0, 0.25),
                                width: 1.0,
                                radius: 4.0.into(),
                            },
                            ..Default::default()
                        }),
                )
                .on_enter(Msg::MenuMouseEnter)
                .on_exit(Msg::MenuMouseExit)
                .into(),
            )
        } else {
            None
        };

        let canvas_widget: Element<'_, Msg> = if let Some(buf) = &self.input {
            let fft_mode = match self.spectrum_mode {
                SpectrumMode::Off => 0,
                SpectrumMode::Spectrum => 1,
                SpectrumMode::Spectrogram => 2,
            };
            let (env_in_color, env_out_color) = self.settings.waveform_scheme.colors(
                self.settings.accent.color(),
                self.settings.custom_input_color.into(),
                self.settings.custom_output_color.into(),
            );
            let c = canvas(Waveform {
                envelope_in: &self.envelope_in,
                envelope_out: &self.envelope_out,
                mono_in: &self.mono_input,
                mono_out: &self.mono_output,
                sample_rate: buf.sample_rate,
                duration: buf.duration_secs(),
                zoom: self.zoom,
                scroll: self.scroll,
                zones: &self.zones,
                selected_split: self.selected_split,
                selected_zone: self.selected_zone,
                playhead_secs: Some(self.playhead_secs),
                hover_secs: self.cursor_time,
                show_reduction: self.show_reduction,
                show_time_ruler: true,
                show_db_ruler: true,
                meter_in: self.meter_in,
                meter_out: self.meter_out,
                fft_mode,
                spectrogram_in: self.spectrogram_in.as_deref(),
                spectrogram_out: self.spectrogram_out.as_deref(),
                spectrum_line_out: self.spectrum_line_out.as_deref().map(|v| v.as_slice()),
                spectrum_line_in: self.spectrum_line_in.as_deref().map(|v| v.as_slice()),
                guide_db: self.guide_db,
                trim_start: self.trim_start,
                trim_end: self.trim_end,
                envelope_in_color: env_in_color,
                envelope_out_color: env_out_color,
                cache: &self.canvas_cache,
                overlay_cache: &self.overlay_cache,
                meter_cache: &self.meter_cache,
            })
            .width(Length::Fill)
            .height(Length::Fill);
            let elem: Element<'_, CanvasEvent> = c.into();
            elem.map(Msg::Canvas)
        } else {
            container(text("No file loaded").size(20))
                .center_x(Length::Fill)
                .center_y(Length::Fill)
                .into()
        };

        let controls = self.controls_panel();

        // The base column: top bar, canvas area, controls. The File menu (when
        // open) is layered as a floating overlay on top of the canvas so it
        // doesn't push other widgets down.
        // Horizontal scrollbar — shown only when zoomed in enough that there's
        // somewhere to scroll. Repurposed slider: tracks scroll position in
        // seconds, range 0..=max_scroll.
        let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(0.0);
        let vis = if self.zoom > 0.0 { dur / self.zoom } else { dur };
        let max_scroll = (dur - vis).max(0.0);
        let scrollbar: Element<'_, Msg> = if max_scroll > 1e-3 {
            container(
                slider(0.0..=max_scroll, self.scroll, Msg::SetScroll)
                    .step(vis / 200.0),
            )
            .padding(iced::padding::left(60).right(20).top(2).bottom(2))
            .into()
        } else {
            // No scroll possible — render a same-height spacer so the layout
            // doesn't jump when zoom changes.
            Space::with_height(0).into()
        };

        let base = column![
            top_bar,
            container(canvas_widget)
                .width(Length::Fill)
                .height(Length::Fill),
            scrollbar,
            container(controls)
                .padding(10)
                .width(Length::Fill)
                .height(Length::Shrink),
        ];

        if let Some(menu) = file_menu {
            // Invisible full-window mouse-area underneath the menu — any click
            // that doesn't land on the menu itself dismisses the dropdown.
            let dismiss_layer = mouse_area(
                container(Space::with_width(Length::Fill))
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .on_press(Msg::CloseFileMenu);

            // The menu, anchored to the top-left below the File button.
            let menu_overlay = container(
                row![menu, Space::with_width(Length::Fill)]
                    .padding(iced::padding::top(44).left(8).right(8)),
            )
            .width(Length::Fill)
            .height(Length::Fill);

            stack![base, dismiss_layer, menu_overlay].into()
        } else if let Some((zi, split_idx, x, y)) = self.zone_ctx_menu {
            // Floating context menu for a right-clicked zone. Layered:
            //   base UI → invisible dismiss layer → the menu itself.
            let dismiss_layer = mouse_area(
                container(Space::with_width(Length::Fill))
                    .width(Length::Fill)
                    .height(Length::Fill),
            )
            .on_press(Msg::CloseZoneContextMenu);

            let ctx_menu = self.build_zone_context_menu(zi, split_idx);

            // Position via padding from the top-left so the menu opens at the
            // click coordinates. Cap so the menu doesn't go off the right edge.
            let pad_left = x.max(0.0) as u16;
            let pad_top = y.max(0.0) as u16;
            let positioned = container(
                row![ctx_menu, Space::with_width(Length::Fill)]
                    .padding(iced::padding::top(pad_top).left(pad_left)),
            )
            .width(Length::Fill)
            .height(Length::Fill);

            stack![base, dismiss_layer, positioned].into()
        } else if let Some(ad) = &self.auto_detect {
            // Modal: dim backdrop with click-to-close + centered panel.
            let dismiss_layer = mouse_area(
                container(Space::with_width(Length::Fill))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .style(|_t: &Theme| container::Style {
                        background: Some(iced::Background::Color(Color::from_rgba(
                            0.0, 0.0, 0.0, 0.55,
                        ))),
                        ..Default::default()
                    }),
            )
            .on_press(Msg::CloseAutoDetect);

            let modal = self.build_auto_detect_modal(ad);
            let centered = container(modal).center_x(Length::Fill).center_y(Length::Fill);
            stack![base, dismiss_layer, centered].into()
        } else if let Some(draft) = &self.settings_draft {
            // Settings modal — same dim-backdrop pattern.
            let dismiss_layer = mouse_area(
                container(Space::with_width(Length::Fill))
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .style(|_t: &Theme| container::Style {
                        background: Some(iced::Background::Color(Color::from_rgba(
                            0.0, 0.0, 0.0, 0.55,
                        ))),
                        ..Default::default()
                    }),
            )
            .on_press(Msg::CloseSettings);

            let modal = self.build_settings_modal(draft);
            let centered = container(modal).center_x(Length::Fill).center_y(Length::Fill);
            stack![base, dismiss_layer, centered].into()
        } else {
            base.into()
        }
    }

    /// Build the auto-detect-zones modal panel.
    fn build_auto_detect_modal(&self, ad: &AutoDetectState) -> Element<'_, Msg> {
        let dur = self.input.as_ref().map(|b| b.duration_secs()).unwrap_or(0.0);
        let n = ad.preview_splits.len();
        let zones_after = n + 1;
        let preview_text = if n == 0 {
            "No transitions detected at this sensitivity. Try increasing it.".to_string()
        } else if n <= 8 {
            // Show the actual times.
            let times: Vec<String> = ad
                .preview_splits
                .iter()
                .map(|t| fmt_time(*t))
                .collect();
            format!(
                "Found {n} split(s) → {zones_after} zone(s):  {}",
                times.join(", ")
            )
        } else {
            format!("Found {n} split(s) → {zones_after} zone(s).")
        };

        let title: Element<'_, Msg> = text("Auto-detect zones").size(18).into();
        let blurb: Element<'_, Msg> = text(format!(
            "Analyzes the song's loudness and dynamic density to find natural section boundaries. Audio: {:.1}s.",
            dur
        ))
        .size(12)
        .color(Color::from_rgba(1.0, 1.0, 1.0, 0.7))
        .into();

        let sens_row: Element<'_, Msg> = row![
            text("Sensitivity").width(110),
            slider(0.0..=1.0, ad.sensitivity, Msg::SetDetectSensitivity).step(0.05),
            text(format!("{:.0}%", ad.sensitivity * 100.0)).width(50),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center)
        .into();

        let preview: Element<'_, Msg> = text(preview_text)
            .size(13)
            .color(Color::from_rgba(1.0, 1.0, 1.0, 0.85))
            .into();

        let buttons: Element<'_, Msg> = {
            let cancel = button(text("Cancel").size(13))
                .on_press(Msg::CloseAutoDetect)
                .padding([6, 14]);
            let add = if n > 0 {
                button(text("Add to existing").size(13))
                    .on_press(Msg::ApplyAutoDetectAdd)
                    .padding([6, 14])
            } else {
                button(text("Add to existing").size(13)).padding([6, 14])
            };
            let replace = if n > 0 {
                button(text("Replace existing").size(13))
                    .on_press(Msg::ApplyAutoDetectReplace)
                    .padding([6, 14])
            } else {
                button(text("Replace existing").size(13)).padding([6, 14])
            };
            row![Space::with_width(Length::Fill), cancel, add, replace]
                .spacing(8)
                .align_y(iced::Alignment::Center)
                .into()
        };

        let inner = column![
            title,
            blurb,
            Space::with_height(8),
            sens_row,
            Space::with_height(4),
            preview,
            Space::with_height(12),
            buttons,
        ]
        .spacing(6)
        .width(540);

        container(inner)
            .padding(20)
            .style(|_t: &Theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.10, 0.11, 0.13))),
                text_color: Some(Color::WHITE),
                border: iced::Border {
                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.25),
                    width: 1.0,
                    radius: 6.0.into(),
                },
                ..Default::default()
            })
            .into()
    }

    /// Build the Settings modal. Reads from `draft` so the user can
    /// flip controls without committing until they hit Save.
    fn build_settings_modal(&self, draft: &settings::Settings) -> Element<'_, Msg> {
        // Row of theme buttons; current selection is highlighted with the
        // accent color.
        let accent = draft.accent.color();
        let accent_h = draft.accent.color_hover();
        let theme_btn = |t: settings::ThemeChoice, current: settings::ThemeChoice| -> Element<'_, Msg> {
            let selected = t == current;
            let acc = accent;
            let acc_h = accent_h;
            button(text(t.label()).size(13))
                .on_press(Msg::DraftSetTheme(t))
                .padding([6, 14])
                .style(move |_th: &Theme, status: button::Status| {
                    let bg = if selected {
                        match status {
                            button::Status::Hovered | button::Status::Pressed => acc_h,
                            _ => acc,
                        }
                    } else {
                        match status {
                            button::Status::Hovered | button::Status::Pressed => {
                                Color::from_rgb(0.30, 0.30, 0.32)
                            }
                            _ => Color::from_rgb(0.20, 0.20, 0.22),
                        }
                    };
                    button::Style {
                        background: Some(iced::Background::Color(bg)),
                        text_color: Color::WHITE,
                        border: iced::Border {
                            radius: 3.0.into(),
                            ..Default::default()
                        },
                        ..Default::default()
                    }
                })
                .into()
        };
        let theme_row = row![
            text("Theme").width(140),
            theme_btn(settings::ThemeChoice::Dark, draft.theme),
            theme_btn(settings::ThemeChoice::Light, draft.theme),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        // Row of accent color swatches; selected one has a brighter border.
        let accent_swatch = |a: settings::AccentColor, current: settings::AccentColor| -> Element<'_, Msg> {
            let selected = a == current;
            let c = a.color();
            let ch = a.color_hover();
            button(text(" ").size(12))
                .on_press(Msg::DraftSetAccent(a))
                .padding([10, 18])
                .style(move |_th: &Theme, status: button::Status| {
                    let bg = match status {
                        button::Status::Hovered | button::Status::Pressed => ch,
                        _ => c,
                    };
                    let border_color = if selected {
                        Color::WHITE
                    } else {
                        Color::from_rgba(1.0, 1.0, 1.0, 0.15)
                    };
                    button::Style {
                        background: Some(iced::Background::Color(bg)),
                        text_color: Color::WHITE,
                        border: iced::Border {
                            color: border_color,
                            width: if selected { 2.0 } else { 1.0 },
                            radius: 4.0.into(),
                        },
                        ..Default::default()
                    }
                })
                .into()
        };
        let mut accent_items: Vec<Element<'_, Msg>> = vec![text("Accent color").width(140).into()];
        for a in settings::AccentColor::ALL {
            accent_items.push(accent_swatch(a, draft.accent));
        }
        let accent_row = row(accent_items)
            .spacing(8)
            .align_y(iced::Alignment::Center);

        // Waveform color scheme row — pick_list to match the Clipper /
        // Normalize controls and avoid overflow when many options exist.
        let scheme_pick = pick_list(
            &settings::WaveformScheme::ALL[..],
            Some(draft.waveform_scheme),
            Msg::DraftSetWaveformScheme,
        )
        .style(dropdown_style(accent))
        .menu_style(dropdown_menu_style(accent));
        let scheme_row = row![
            text("Waveform scheme").width(140),
            scheme_pick,
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        // Default open dir row.
        let open_label = draft
            .default_open_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(system default)".to_string());
        let open_row = row![
            text("Default open folder").width(140),
            text(open_label).size(12).width(Length::Fill),
            button(text("Browse…").size(13))
                .on_press(Msg::PickDefaultOpenDir)
                .padding([4, 12]),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        let export_label = draft
            .default_export_dir
            .as_ref()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| "(system default)".to_string());
        let export_row = row![
            text("Default export folder").width(140),
            text(export_label).size(12).width(Length::Fill),
            button(text("Browse…").size(13))
                .on_press(Msg::PickDefaultExportDir)
                .padding([4, 12]),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        // Read-only display of current keybindings. Editable only by hand
        // in `~/.config/peakmuncher/settings.json` for now.
        let mut kb_rows: Vec<Element<'_, Msg>> = Vec::new();
        kb_rows.push(
            text("Keyboard shortcuts (edit settings.json to change)")
                .size(12)
                .color(Color::from_rgba(1.0, 1.0, 1.0, 0.6))
                .into(),
        );
        for action in crate::keybindings::KeyAction::ALL {
            let combo = draft.keybindings.display(action);
            let row_el: Element<'_, Msg> = row![
                text(action.label()).size(12).width(220),
                text(combo).size(12).color(Color::from_rgba(1.0, 1.0, 1.0, 0.85)),
            ]
            .spacing(8)
            .into();
            kb_rows.push(row_el);
        }
        let kb_block: Element<'_, Msg> = column(kb_rows).spacing(2).into();

        let buttons = row![
            Space::with_width(Length::Fill),
            button(text("Cancel").size(13))
                .on_press(Msg::CloseSettings)
                .padding([6, 14]),
            {
                let acc = accent;
                let acc_h = accent_h;
                button(text("Save").size(13))
                    .on_press(Msg::SaveSettings)
                    .padding([6, 14])
                    .style(move |_th: &Theme, status: button::Status| {
                        let bg = match status {
                            button::Status::Hovered | button::Status::Pressed => acc_h,
                            _ => acc,
                        };
                        button::Style {
                            background: Some(iced::Background::Color(bg)),
                            text_color: Color::WHITE,
                            border: iced::Border {
                                radius: 3.0.into(),
                                ..Default::default()
                            },
                            ..Default::default()
                        }
                    })
            },
        ]
        .spacing(8);

        let inner = column![
            text("Settings").size(18),
            text("Changes apply when you click Save.")
                .size(12)
                .color(Color::from_rgba(1.0, 1.0, 1.0, 0.7)),
            Space::with_height(8),
            theme_row,
            accent_row,
            scheme_row,
            open_row,
            export_row,
            Space::with_height(8),
            kb_block,
            Space::with_height(12),
            buttons,
        ]
        .spacing(10)
        .width(620);

        container(inner)
            .padding(20)
            .style(|_t: &Theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.10, 0.11, 0.13))),
                text_color: Some(Color::WHITE),
                border: iced::Border {
                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.25),
                    width: 1.0,
                    radius: 6.0.into(),
                },
                ..Default::default()
            })
            .into()
    }

    /// Build the small floating menu shown on right-click of a zone.
    fn build_zone_context_menu(
        &self,
        zi: usize,
        split_idx: Option<usize>,
    ) -> Element<'_, Msg> {
        // Same styles as the File menu items.
        let menu_item_style = |_theme: &Theme, status: button::Status| {
            let bg = match status {
                button::Status::Hovered | button::Status::Pressed => {
                    Some(iced::Background::Color(Color::from_rgb(0.30, 0.30, 0.32)))
                }
                _ => None,
            };
            button::Style {
                background: bg,
                text_color: Color::WHITE,
                border: iced::Border {
                    radius: 2.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        };
        let menu_item_disabled_style = |_theme: &Theme, _status: button::Status| {
            button::Style {
                background: None,
                text_color: Color::from_rgba(1.0, 1.0, 1.0, 0.35),
                border: iced::Border {
                    radius: 2.0.into(),
                    ..Default::default()
                },
                ..Default::default()
            }
        };

        let copy_btn: Element<'_, Msg> = button(text("Copy zone params").size(13))
            .on_press(Msg::CopyZoneParams)
            .padding([4, 14])
            .width(Length::Fill)
            .style(menu_item_style)
            .into();

        let paste_btn: Element<'_, Msg> = if self.zone_clipboard.is_some() {
            button(text("Paste zone params").size(13))
                .on_press(Msg::PasteZoneParams)
                .padding([4, 14])
                .width(Length::Fill)
                .style(menu_item_style)
                .into()
        } else {
            button(text("Paste zone params").size(13))
                .padding([4, 14])
                .width(Length::Fill)
                .style(menu_item_disabled_style)
                .into()
        };

        // "Delete split" — only shown when right-click landed on a split.
        let mut items: Vec<Element<'_, Msg>> = Vec::new();
        let header: Element<'_, Msg> = text(format!("  Zone {}", zi + 1))
            .size(11)
            .color(Color::from_rgba(1.0, 1.0, 1.0, 0.45))
            .into();
        items.push(header);
        items.push(copy_btn);
        items.push(paste_btn);
        if let Some(si) = split_idx {
            items.push(horizontal_rule(1).into());
            let del_btn: Element<'_, Msg> = button(
                text(format!("Delete split {}", si + 1)).size(13),
            )
            .on_press(Msg::DeleteSplitAt(si))
            .padding([4, 14])
            .width(Length::Fill)
            .style(menu_item_style)
            .into();
            items.push(del_btn);
        }
        let menu_col = column(items).spacing(0).width(190);

        container(menu_col)
            .padding(2)
            .style(|_theme: &Theme| container::Style {
                background: Some(iced::Background::Color(Color::from_rgb(0.1255, 0.1333, 0.1451))),
                text_color: Some(Color::WHITE),
                border: iced::Border {
                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.25),
                    width: 1.0,
                    radius: 4.0.into(),
                },
                ..Default::default()
            })
            .into()
    }

    fn controls_panel(&self) -> Element<'_, Msg> {
        let zone = self
            .selected_zone
            .and_then(|i| self.zones.zones.get(i))
            .copied()
            .unwrap_or_default();
        let total = self.zones.zones.len();
        let cur = self.selected_zone.unwrap_or(0);
        let zone_label = if total == 0 {
            "(no zone)".to_string()
        } else {
            format!("Zone {} of {}", cur + 1, total)
        };

        // Group prev/next chevrons together on the left, then the
        // "Zone N of M" text. Chevrons are auto-disabled when at the
        // start/end of the zone list.
        let accent_hover_for_nav = self.settings.accent.color_hover();
        let accent = self.settings.accent.color();

        // Split-at-cursor: enabled whenever a file is loaded (the handler
        // itself validates the cursor position). Placed leftmost.
        let split_btn = icon_button_opt(
            ICON_SPLIT,
            if self.input.is_some() {
                Some(Msg::AddSplitAtCursor)
            } else {
                None
            },
            accent_hover_for_nav,
        );
        // Delete-selected-split: enabled only when a split is selected.
        // Sits between the split button and the prev-zone chevron.
        let delete_btn = icon_button_opt(
            ICON_TRASH,
            if self.selected_split.is_some() {
                Some(Msg::DeleteSelectedSplit)
            } else {
                None
            },
            accent_hover_for_nav,
        );

        let prev_btn = icon_button_opt(
            ICON_CHEVRON_LEFT,
            if cur > 0 { Some(Msg::PrevZone) } else { None },
            accent_hover_for_nav,
        );
        let next_btn = icon_button_opt(
            ICON_CHEVRON_RIGHT,
            if cur + 1 < total { Some(Msg::NextZone) } else { None },
            accent_hover_for_nav,
        );

        // Horizontal tab bar (CLIPPER | LEVELS | FIX | OUTPUT). Each tab is
        // a label with a thin underline bar beneath it — accent when active,
        // faint grey when inactive — for a classic tabbed look. (iced's
        // Border is all-sides, so the underline is a stacked 2px bar rather
        // than a bottom border.)
        let active_tab = self.active_tab;
        let tab_button = |tab: ControlTab| -> Element<'_, Msg> {
            let is_active = tab == active_tab;
            let txt_color = if is_active {
                accent
            } else {
                Color::from_rgba(1.0, 1.0, 1.0, 0.45)
            };
            let underline_color = if is_active {
                accent
            } else {
                Color::from_rgba(1.0, 1.0, 1.0, 0.10)
            };
            let underline = container(Space::with_height(0))
                .width(Length::Fill)
                .height(Length::Fixed(2.0))
                .style(move |_theme: &Theme| container::Style {
                    background: Some(iced::Background::Color(underline_color)),
                    ..Default::default()
                });
            let label_btn = button(
                text(tab.label())
                    .size(12)
                    .color(txt_color)
                    .align_x(iced::alignment::Horizontal::Center)
                    .width(Length::Fill),
            )
            .on_press(Msg::SelectTab(tab))
            .padding([4, 0])
            .width(Length::Fill)
            .style(move |_theme: &Theme, status: button::Status| {
                let hovered = matches!(
                    status,
                    button::Status::Hovered | button::Status::Pressed
                );
                button::Style {
                    background: if hovered && !is_active {
                        Some(iced::Background::Color(Color { a: 0.08, ..accent }))
                    } else {
                        None
                    },
                    text_color: txt_color,
                    ..Default::default()
                }
            });
            column![label_btn, underline]
                .spacing(3)
                .width(Length::Fixed(78.0))
                .into()
        };
        let tab_bar = row(ControlTab::ALL.iter().map(|&t| tab_button(t)))
            .spacing(4)
            .align_y(iced::Alignment::Center);

        let zone_nav = row![
            split_btn,
            delete_btn,
            prev_btn,
            next_btn,
            Space::with_width(4),
            text(zone_label).size(15).width(140),
            Space::with_width(20),
            tab_bar,
            Space::with_width(Length::Fill),
        ]
        .spacing(2)
        .align_y(iced::Alignment::Center);

        let clipper_pick =
            pick_list(ClipperType::ALL, Some(zone.clipper), Msg::SetClipper)
                .style(dropdown_style(accent))
                .menu_style(dropdown_menu_style(accent));

        // Oversampling is a save-time only setting; clearly label it.
        // 'static slice so the pick_list's borrow lives long enough to be
        // returned as part of the Element.
        const OS_OPTIONS: &[u8] = &[1, 2, 4, 8];
        let os_pick = pick_list(
            OS_OPTIONS,
            Some(zone.oversampling),
            Msg::SetOversampling,
        )
        .style(dropdown_style(accent))
        .menu_style(dropdown_menu_style(accent));

        let ceiling_strip = canvas(CeilingStrip {
            ceiling_db: zone.ceiling_db,
            peak_db: self.zone_input_peak_db,
            range: (-30.0, 0.0),
        })
        .width(Length::Fill)
        .height(Length::Fixed(10.0));
        let ceiling = column![
            row![
                text("Ceiling").width(110),
                slider(-30.0..=0.0, zone.ceiling_db, Msg::SetCeiling)
                    .step(0.1)
                    .on_release(Msg::Commit),
                text(format!("{:+.1} dB", zone.ceiling_db)).width(80),
            ]
            .spacing(8),
            // Peak marker + clip-amount strip, indented to align under the
            // slider track (label 110 + spacing 8 on the left, readout 80 +
            // spacing 8 on the right).
            row![
                Space::with_width(118),
                ceiling_strip,
                Space::with_width(88),
            ],
        ]
        .spacing(2);

        let in_gain = row![
            text("Input gain").width(110),
            slider(-12.0..=24.0, zone.input_gain_db, Msg::SetInputGain)
                .step(0.1)
                .on_release(Msg::Commit),
            text(format!("{:+.1} dB", zone.input_gain_db)).width(80),
        ]
        .spacing(8);

        // ── OUTPUT GAIN — HIDDEN FROM UI (2026-06-15) ──────────────────
        // Output gain is a vestige of PeakMuncher's origins as a live-input
        // plugin, where the output level was abstract and a manual trim was
        // the only way to set final level. Now that we work on a complete
        // file with Normalize available, a post-clipper output trim is
        // redundant with Normalize (both just scale the final signal), and
        // because it sits AFTER the clipper it breaks the amber "clipped at
        // the ceiling" relationship (pushing output gain up moves the final
        // signal above the ceiling, making amber appear/disappear oddly).
        //
        // DECISION: hide the slider from the LEVELS tab for now, but KEEP the
        // `output_gain_db` field, its `Msg::SetOutputGain` handler, and the
        // DSP `* z.out_gain` stages intact. With the slider gone the field
        // stays at its default 0.0 → out_gain = 1.0 → a no-op multiply, so
        // nothing changes audibly and old projects/presets still load.
        //
        // FUTURE (see handoff "Output gain" note): either remove it fully
        // (DSP + ZoneParams + serialization), OR repurpose it as a PRE-clipper
        // "Drive"/"Makeup" control — which would make the ceiling a true
        // final wall, fix the amber, and give it a real job Normalize can't do.
        // The `out_gain` row builder is intentionally not constructed here.

        // DC offset row: a slider in the linear-amplitude domain plus an
        // "Auto-detect" button that measures the zone's sample mean and
        // sets the slider to negate it. DC offset is applied before input
        // gain in the DSP chain, so it cleans the signal up front.
        let dc_offset = row![
            text("DC offset").width(110),
            slider(-0.5_f32..=0.5_f32, zone.dc_offset, Msg::SetDcOffset)
                .step(0.0001_f32)
                .on_release(Msg::Commit),
            text(format!("{:+.4}", zone.dc_offset)).width(80),
            button(text("Auto").size(13))
                .on_press(Msg::AutoDetectDc)
                .padding([4, 10]),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        // DC blocker row: a one-pole high-pass filter that handles
        // wandering / drifting DC that a constant offset can't fix. The
        // toggle button reads "On" or "Off"; the slider is disabled when
        // the blocker is off (still visible so the user sees the cutoff
        // setting will be applied when they enable). Default 20 Hz —
        // below most musical bass content.
        let blocker_label = if zone.dc_blocker_enabled { "On" } else { "Off" };
        let dc_blocker = row![
            text("DC blocker").width(110),
            button(text(blocker_label).size(13))
                .on_press(Msg::ToggleDcBlocker)
                .padding([4, 10]),
            slider(5.0_f32..=60.0_f32, zone.dc_blocker_hz, Msg::SetDcBlockerHz)
                .step(1.0_f32)
                .on_release(Msg::Commit),
            text(format!("{:.0} Hz", zone.dc_blocker_hz)).width(70),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        // Normalize state picker — `pick_list` to match the Clipper /
        // Oversampling controls' visual language.
        let current_state = if !self.normalize_enabled {
            NormalizeState::Off
        } else {
            match self.normalize_mode {
                NormalizeMode::Peak => NormalizeState::Peak,
                NormalizeMode::Lufs => NormalizeState::Lufs,
            }
        };
        let norm_pick = pick_list(
            &NormalizeState::ALL[..],
            Some(current_state),
            Msg::SetNormalizeState,
        )
        .style(dropdown_style(accent))
        .menu_style(dropdown_menu_style(accent));
        let (norm_lo, norm_hi, _) = self.normalize_mode.range();
        let value_color = if self.normalize_enabled {
            Color::WHITE
        } else {
            Color::from_rgba(1.0, 1.0, 1.0, 0.4)
        };
        let normalize = row![
            text("Normalize").width(110),
            norm_pick,
            slider(
                norm_lo..=norm_hi,
                self.normalize_target_db,
                Msg::SetNormalizeTarget
            )
            .step(0.1),
            text(format!(
                "{:+.1} {}",
                self.normalize_target_db,
                self.normalize_mode.unit()
            ))
            .color(value_color)
            .width(110),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        // Per-zone fade-in and fade-out: linear ramps applied at the
        // zone's boundaries. 0 = no fade. Capped at 10s in the slider;
        // also auto-capped at half the zone's length in DSP so they
        // don't overlap inside short zones.
        let fade_in = row![
            text("Fade in").width(110),
            slider(0.0_f32..=10.0_f32, zone.fade_in_secs, Msg::SetFadeIn)
                .step(0.05_f32)
                .on_release(Msg::Commit),
            text(format!("{:.2} s", zone.fade_in_secs)).width(80),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);
        let fade_out = row![
            text("Fade out").width(110),
            slider(0.0_f32..=10.0_f32, zone.fade_out_secs, Msg::SetFadeOut)
                .step(0.05_f32)
                .on_release(Msg::Commit),
            text(format!("{:.2} s", zone.fade_out_secs)).width(80),
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);

        // Active-tab content. Only the selected tab's controls render.
        // (The tab bar in zone_nav is the section header now, so no inline
        // section labels here.)
        let os_row = row![
            text("Oversampling").width(110),
            os_pick,
            text("(applied on Save)").size(11)
        ]
        .spacing(8)
        .align_y(iced::Alignment::Center);
        let clipper_row = row![text("Clipper").width(110), clipper_pick].spacing(8);

        let tab_content: Element<'_, Msg> = match self.active_tab {
            ControlTab::Clipper => column![clipper_row, os_row, ceiling]
                .spacing(6)
                .into(),
            ControlTab::Levels => column![in_gain].spacing(6).into(),
            ControlTab::Fix => column![dc_offset, dc_blocker].spacing(6).into(),
            ControlTab::Output => column![normalize, fade_in, fade_out]
                .spacing(6)
                .into(),
        };
        // Pin the control area to a consistent height (sized to the tallest
        // tab, Clipper) so the panel doesn't resize as tabs switch. Content
        // aligns to the top; shorter tabs just leave space below.
        let tab_content = container(tab_content)
            .height(Length::Fixed(132.0))
            .align_y(iced::alignment::Vertical::Top);

        column![
            zone_nav,
            Space::with_height(8),
            tab_content,
        ]
        .spacing(6)
        .into()
    }
}

/// Format seconds as M:SS.mmm.
/// Compute a trim measurement/export window in FRAME indices from explicit
/// parameters. Returns None when the trim covers (effectively) the whole
/// file — the no-op fast path used everywhere trim is applied. Single source
/// of truth shared by `App::trim_window_frames` and the project-load path,
/// so the clamp/edge logic can't drift between call sites.
fn compute_trim_window(
    sample_rate: u32,
    total_frames: usize,
    duration_secs: f32,
    trim_start: Option<f32>,
    trim_end: Option<f32>,
) -> Option<(usize, usize)> {
    let sr = sample_rate as f32;
    let ts = trim_start.unwrap_or(0.0);
    let te = trim_end.unwrap_or(duration_secs);
    // No-op if the window is effectively the whole file.
    if ts <= 1e-4 && te >= duration_secs - 1e-4 {
        return None;
    }
    let f0 = ((ts * sr) as usize).min(total_frames);
    let f1 = ((te * sr).ceil() as usize).min(total_frames).max(f0);
    if f1 > f0 {
        Some((f0, f1))
    } else {
        None
    }
}

/// Thin strip drawn under the Ceiling slider showing the selected zone's
/// input peak as a fixed tick, plus a bar indicating how many dB are being
/// clipped (ceiling below the peak). Display-only; emits no messages.
struct CeilingStrip {
    ceiling_db: f32,
    peak_db: Option<f32>,
    range: (f32, f32),
}

impl<Message> canvas::Program<Message> for CeilingStrip {
    type State = ();

    fn draw(
        &self,
        _state: &(),
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: iced::mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = canvas::Frame::new(renderer, bounds.size());
        let w = bounds.width;
        let h = bounds.height;
        let (lo, hi) = self.range;
        let span = (hi - lo).max(1e-6);
        // iced's default slider handle is ~16px wide; its CENTER travels
        // from handle_w/2 to width-handle_w/2. Inset the strip's usable
        // span by that half-width so a given dB maps to the same x as the
        // slider handle's center.
        let margin = 8.0_f32;
        let usable = (w - margin * 2.0).max(1.0);
        let db_to_x = |db: f32| -> f32 {
            let t = ((db - lo) / span).clamp(0.0, 1.0);
            margin + t * usable
        };

        // Neutral grey track/background so the strip reads as a passive
        // METER, not a second slider (the accent-blue echoed the slider too
        // strongly). Only the peak tick and clip bar carry color.
        let track_bg = Color::from_rgba(1.0, 1.0, 1.0, 0.06);
        let track_line = Color::from_rgba(1.0, 1.0, 1.0, 0.20);
        let amber = Color::from_rgba(1.0, 0.65, 0.15, 1.0);

        frame.fill_rectangle(
            Point::new(margin, 1.0),
            Size::new(usable, h - 2.0),
            track_bg,
        );
        let track_y = h * 0.5;
        frame.fill_rectangle(
            Point::new(margin, track_y - 0.5),
            Size::new(usable, 1.0),
            track_line,
        );

        if let Some(peak) = self.peak_db {
            let peak_x = db_to_x(peak);
            let ceil_x = db_to_x(self.ceiling_db);

            // Clip-amount bar: peak tick → ceiling handle, only when the
            // ceiling is BELOW the peak (actually clipping). Length = dB clipped.
            if self.ceiling_db < peak - 1e-3 {
                let x0 = ceil_x.min(peak_x);
                let x1 = ceil_x.max(peak_x);
                frame.fill_rectangle(
                    Point::new(x0, 1.0),
                    Size::new((x1 - x0).max(1.0), h - 2.0),
                    Color { a: 0.55, ..amber },
                );
            }

            // Peak marker tick (full height, amber), on top. Clamp inward so
            // it isn't clipped at the extreme edges (e.g. a 0 dBFS peak).
            let tick_w = 3.0_f32;
            let tx = (peak_x - tick_w * 0.5)
                .clamp(margin, (margin + usable) - tick_w);
            frame.fill_rectangle(
                Point::new(tx, 0.0),
                Size::new(tick_w, h),
                amber,
            );
        }

        vec![frame.into_geometry()]
    }
}

/// Shared style for the controls-panel dropdowns (`pick_list`). Dark
/// background blending into the panel with a muted grey border; accent text
/// and dropdown triangle. Hover/open lifts the border toward the accent.
fn dropdown_style(
    accent: Color,
) -> impl Fn(&Theme, pick_list::Status) -> pick_list::Style {
    move |_theme, status| {
        let border_color = match status {
            pick_list::Status::Hovered | pick_list::Status::Opened => {
                Color { a: 0.55, ..accent }
            }
            _ => Color::from_rgba(1.0, 1.0, 1.0, 0.12),
        };
        pick_list::Style {
            // Dark, matching the window background #202225.
            background: iced::Background::Color(Color::from_rgb(0.1255, 0.1333, 0.1451)),
            border: iced::Border {
                color: border_color,
                width: 1.0,
                radius: 6.0.into(),
            },
            text_color: accent,
            placeholder_color: Color { a: 0.5, ..accent },
            handle_color: accent,
        }
    }
}

/// Matching style for the OPEN dropdown menu (the popup list). Same dark
/// background as the closed control and the file menu, accent text, with a
/// subtle accent-tinted highlight on the hovered/selected row.
fn dropdown_menu_style(accent: Color) -> impl Fn(&Theme) -> iced::widget::overlay::menu::Style {
    move |_theme| iced::widget::overlay::menu::Style {
        background: iced::Background::Color(Color::from_rgb(0.1255, 0.1333, 0.1451)),
        border: iced::Border {
            color: Color::from_rgba(1.0, 1.0, 1.0, 0.18),
            width: 1.0,
            radius: 4.0.into(),
        },
        text_color: accent,
        selected_text_color: accent,
        selected_background: iced::Background::Color(Color { a: 0.22, ..accent }),
    }
}

fn fmt_time(s: f32) -> String {
    let total_ms = (s * 1000.0).max(0.0) as u64;
    let m = total_ms / 60_000;
    let sec = (total_ms % 60_000) / 1000;
    let ms = total_ms % 1000;
    format!("{m}:{sec:02}.{ms:03}")
}

/// Mix-down interleaved samples to mono for envelope display.
/// Build a display envelope from a mono buffer, sampling every `decim`th
/// frame within each bucket. Used for the fast preview input envelope —
/// the cached raw mono scaled by `gain`. Visually near-identical to the
/// full envelope at full-file zoom while reading 1/decim of the data.
fn envelope_decimated_mono(
    mono: &[f32],
    gain: f32,
    width: usize,
    decim: usize,
) -> Vec<(f32, f32)> {
    use rayon::prelude::*;
    if mono.is_empty() || width == 0 {
        return Vec::new();
    }
    let decim = decim.max(1);
    let bucket = (mono.len() as f32 / width as f32).max(1.0);
    (0..width)
        .into_par_iter()
        .map(|i| {
            let start = (i as f32 * bucket) as usize;
            let end = (((i as f32 + 1.0) * bucket) as usize)
                .min(mono.len())
                .max(start + 1);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            let mut j = start;
            while j < end {
                let s = mono[j] * gain;
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
                j += decim;
            }
            // Guard: if decim stepped past the whole (tiny) bucket, sample
            // the first frame so we never emit an empty bucket.
            if !lo.is_finite() {
                let s = mono[start] * gain;
                lo = s;
                hi = s;
            }
            (lo.clamp(-1.0, 1.0), hi.clamp(-1.0, 1.0))
        })
        .collect()
}

fn mono_from_interleaved(samples: &[f32], channels: u16) -> Vec<f32> {
    use rayon::prelude::*;
    let ch = channels.max(1) as usize;
    if ch == 1 {
        return samples.to_vec();
    }
    let frames = samples.len() / ch;
    // Parallel mono mixdown. This was previously a single-threaded loop
    // over every frame — on a 6-minute file that's ~18M iterations and was
    // the dominant cost in the envelope-build step (bigger than the render
    // itself). Mapping in parallel over frame indices cuts it across cores.
    let inv_ch = 1.0 / ch as f32;
    (0..frames)
        .into_par_iter()
        .map(|f| {
            let base = f * ch;
            let mut acc = 0.0;
            for c in 0..ch {
                acc += samples[base + c];
            }
            acc * inv_ch
        })
        .collect()
}

/// Build the mono mixdown AND its display envelope in a SINGLE parallel
/// pass over the interleaved buffer. Previously these were two separate
/// full sweeps of ~18M samples (mono_from_interleaved, then build_envelope
/// re-reading the whole mono Vec). Fusing them halves the memory traffic
/// on the hot path that runs on every ceiling-slider tick. Returns
/// (mono, envelope) where envelope has `width` (lo, hi) buckets.
/// Structural equality for ZoneMap (used by undo to skip duplicate snapshots).
fn zones_eq(a: &ZoneMap, b: &ZoneMap) -> bool {
    if a.splits.len() != b.splits.len() || a.zones.len() != b.zones.len() {
        return false;
    }
    for (x, y) in a.splits.iter().zip(b.splits.iter()) {
        if (x - y).abs() > 1e-6 {
            return false;
        }
    }
    for (x, y) in a.zones.iter().zip(b.zones.iter()) {
        if x.clipper != y.clipper
            || x.oversampling != y.oversampling
            || (x.ceiling_db - y.ceiling_db).abs() > 1e-4
            || (x.input_gain_db - y.input_gain_db).abs() > 1e-4
            || (x.output_gain_db - y.output_gain_db).abs() > 1e-4
            || (x.dc_offset - y.dc_offset).abs() > 1e-6
            || (x.fade_in_secs - y.fade_in_secs).abs() > 1e-4
            || (x.fade_out_secs - y.fade_out_secs).abs() > 1e-4
            || x.dc_blocker_enabled != y.dc_blocker_enabled
            || (x.dc_blocker_hz - y.dc_blocker_hz).abs() > 1e-3
        {
            return false;
        }
    }
    true
}
