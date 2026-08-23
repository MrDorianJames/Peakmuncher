//! Waveform canvas widget.
//!
//! Draws:
//!   • The input audio waveform (gray min/max envelope).
//!   • The processed waveform overlay (colored, semi-transparent).
//!   • Per-zone ceiling lines (horizontal, drawn at the zone's ceiling dB).
//!   • Vertical split lines, draggable.
//!   • Mirrored ceiling line below the zero-axis (negative side).
//!
//! Click on empty waveform space → adds a split at that time.
//! Click on existing split line → selects it (parent app then shows controls).
//! Drag a split → moves it.

use crate::zones::ZoneMap;
use iced::mouse;
use iced::widget::canvas::{self, Cache, Frame, Geometry, Path, Stroke};
use iced::{Color, Point, Rectangle, Renderer, Size, Theme};

#[derive(Debug, Clone, Copy)]
pub enum CanvasEvent {
    AddSplit(f32),
    SelectSplit(usize),
    MoveSplit(usize, f32),
    SelectZone(usize),
    Seek(f32),
    Hover(Option<f32>),
    Zoom(f32, f32),
    Pan(f32),
    /// Right-click on a zone — `(zone_idx, split_idx_if_on_split, canvas_x, canvas_y)` in canvas-local coords.
    RightClickZone(usize, Option<usize>, f32, f32),
    /// User dragged the guide line to a new dB value.
    DragGuide(f32),
    /// User dragged a trim handle to a new time (seconds). Clamping and the
    /// start<end constraint are enforced in the app update handler.
    MoveTrimStart(f32),
    MoveTrimEnd(f32),
    /// Click-repair mode: user clicked to repair a click at this INPUT frame
    /// index. Fired instead of Seek when click_repair_mode is on and the
    /// view is zoomed to sample resolution.
    RepairClick(usize),
    /// Mono-fold badge in the goniometer's corner was clicked.
    ToggleMonoFold,
    /// Frequency band selected by dragging on the correlation curve, in Hz.
    /// `None` clears the selection (a click without a drag).
    SelectBand(Option<(f32, f32)>),
    /// Clicked or dragged on the correlation overview strip: seek here AND
    /// bring the zoomed view to this point. Distinct from `Seek` because the
    /// strip always shows the whole file, so its target is usually outside
    /// the current view — seeking without scrolling would move the playhead
    /// somewhere you can't see.
    NavigateTo(f32),
}

/// Size of the mono-fold badge drawn in the goniometer's top-right corner.
/// Padding inside the correlation-curve panel. Shared by the drawing and
/// the drag hit-test, so a click lands on the frequency it looks like it
/// lands on.
const CORR_PAD_L: f32 = 26.0;
const CORR_PAD_R: f32 = 8.0;
const CORR_PAD_T: f32 = 18.0;
const CORR_PAD_B: f32 = 14.0;

const MONO_BADGE_W: f32 = 30.0;
const MONO_BADGE_H: f32 = 18.0;

/// Height of the always-on overview strip that sits directly under the
/// waveform. Fixed rather than proportional: it's a navigation bar, and a
/// navigation bar that changes size as the window resizes is harder to
/// build muscle memory for.
pub const OVERVIEW_H: f32 = 22.0;

/// Correlation → colour. Shared by the frequency curve and the timeline
/// strip so the same value reads as the same colour in both, which is what
/// lets you glance between them.
///
/// Stops are muted rather than saturated: this fills areas that sit behind
/// brighter foreground elements, and shouldn't shout. Flat green through the
/// whole +0.5..+1.0 range — normal material shouldn't look like it's doing
/// something.
fn corr_color_ramp(c: f32) -> (f32, f32, f32) {
    const STOPS: [(f32, [f32; 3]); 5] = [
        (1.00, [0.30, 0.72, 0.45]),  // green
        (0.50, [0.34, 0.70, 0.42]),  // green (flat through healthy)
        (0.25, [0.78, 0.62, 0.26]),  // amber
        (0.00, [0.84, 0.44, 0.24]),  // orange
        (-1.00, [0.86, 0.26, 0.26]), // red
    ];
    let c = c.clamp(-1.0, 1.0);
    let mut a = STOPS[0];
    let mut b = STOPS[1];
    for i in 0..STOPS.len() - 1 {
        if c <= STOPS[i].0 && c >= STOPS[i + 1].0 {
            a = STOPS[i];
            b = STOPS[i + 1];
            break;
        }
    }
    let span = (a.0 - b.0).max(1e-6);
    let t = ((a.0 - c) / span).clamp(0.0, 1.0);
    (
        a.1[0] + (b.1[0] - a.1[0]) * t,
        a.1[1] + (b.1[1] - a.1[1]) * t,
        a.1[2] + (b.1[2] - a.1[2]) * t,
    )
}

/// Where the mono badge sits, given the goniometer column's origin and width.
///
/// Drawing and hit-testing BOTH go through this, so the clickable rect can't
/// drift away from the painted one — the classic way canvas-drawn controls
/// break.
fn mono_badge_rect(gx: f32, gy: f32, gw: f32) -> (f32, f32, f32, f32) {
    (
        gx + gw - MONO_BADGE_W - 6.0,
        gy + 3.0,
        MONO_BADGE_W,
        MONO_BADGE_H,
    )
}

pub struct Waveform<'a> {
    pub envelope_in: &'a [(f32, f32)],
    pub envelope_out: &'a [(f32, f32)],
    pub mono_in: &'a [f32],
    pub mono_out: &'a [f32],
    pub sample_rate: u32,
    pub duration: f32,
    pub zoom: f32,
    pub scroll: f32,
    pub zones: &'a ZoneMap,
    pub selected_split: Option<usize>,
    pub selected_zone: Option<usize>,
    pub playhead_secs: Option<f32>,
    pub hover_secs: Option<f32>,
    pub show_reduction: bool,
    /// Show top time ruler.
    pub show_time_ruler: bool,
    /// Show left dB ruler.
    pub show_db_ruler: bool,
    /// Real-time peak meter values [0..1+] for input and output.
    pub meter_in: f32,
    pub meter_out: f32,
    /// FFT view mode: 0 = off, 1 = spectrum line, 2 = spectrogram heatmap.
    pub fft_mode: u8,
    /// When true, a click in the wave area (at sample zoom) fires
    /// RepairClick(frame) instead of Seek — the click-repair tool is active.
    pub click_repair_mode: bool,
    /// Frame indices of clicks found by the scanner, drawn as vertical
    /// markers so the user can find them.
    pub detected_clicks: &'a [usize],
    /// Index of the currently-navigated click (highlighted brighter).
    pub current_click: Option<usize>,
    /// Stereo correlation (-1..1) at the probe point, for Correlation mode.
    pub probe_correlation: Option<f32>,
    /// Goniometer L/R points at the probe point, for Goniometer mode.
    pub probe_gonio: &'a [(f32, f32)],
    /// Whole-file correlation over time, for the strip at the top of the
    /// Phase panel. `None` until computed.
    pub corr_timeline: Option<&'a crate::fft::CorrTimeline>,
    /// Selected frequency band on the correlation curve, in Hz.
    pub band_sel: Option<(f32, f32)>,
    /// Correlation measured on the selected band only.
    pub band_sel_corr: Option<f32>,
    /// Whether mono-fold monitoring is engaged (lights the goniometer badge).
    pub mono_fold: bool,
    /// Per-frequency correlation at the probe: `(corr, weight)` per FFT bin,
    /// bin `i` at `i / len * nyquist` Hz — the same frequency mapping the
    /// spectrum line uses, so the two views line up.
    pub probe_band_corr: &'a [(f32, f32)],
    /// Spectrogram for input (used in mode 2). None if not yet computed.
    pub spectrogram_in: Option<&'a crate::fft::Spectrogram>,
    pub spectrogram_out: Option<&'a crate::fft::Spectrogram>,
    /// Cached probe-point spectrum lines (dB per bin), computed by the App
    /// only when the probe/audio changes — NOT here in the draw. `out` is the
    /// processed signal, `in` the original (for the overlay). None until
    /// computed.
    pub spectrum_line_out: Option<&'a [f32]>,
    pub spectrum_line_in: Option<&'a [f32]>,
    /// Optional draggable horizontal guide line, in dB (negative).
    pub guide_db: Option<f32>,
    /// Export trim boundaries in seconds. Regions outside `[trim_start,
    /// trim_end]` are dimmed with a dark overlay and bounded by colored
    /// handles (green at start, red at end). Non-destructive — purely a
    /// view of what export/normalize will use.
    pub trim_start: Option<f32>,
    pub trim_end: Option<f32>,
    /// Colors for the input and processed waveform envelopes.
    pub envelope_in_color: Color,
    pub envelope_out_color: Color,
    pub cache: &'a Cache,
    pub overlay_cache: &'a Cache,
    pub meter_cache: &'a Cache,
}

impl<'a> Waveform<'a> {
    /// The visible window's duration in seconds.
    fn visible_duration(&self) -> f32 {
        (self.duration / self.zoom.max(0.001)).max(0.001)
    }

    /// Reserved widths for the left dB ruler and right space (none).
    fn left_inset(&self) -> f32 {
        if self.show_db_ruler { 44.0 } else { 0.0 }
    }
    /// Top time-ruler height.
    fn top_inset(&self) -> f32 {
        if self.show_time_ruler { 22.0 } else { 0.0 }
    }
    /// Height of the FFT/spectrum panel at the bottom, as a fraction of the
    /// usable area (canvas minus top ruler and the always-on meter strip).
    /// 0.5 = an even split with the waveform. Scales with the window so the
    /// spectrum stays readable at any size. Only nonzero when a spectrum view
    /// is active.
    const FFT_PANEL_FRACTION: f32 = 0.5;

    fn fft_panel_height(&self, canvas_h: f32) -> f32 {
        if self.fft_mode == 0 {
            return 0.0;
        }
        let meter = 18.0;
        let usable = (canvas_h - self.top_inset() - meter).max(1.0);
        // Give the FFT the configured fraction, but keep sane min/max so a
        // tiny or huge window still leaves both areas usable. Guard against a
        // window so small the max would fall below the min.
        let target = usable * Self::FFT_PANEL_FRACTION;
        let max_h = (usable - 60.0).max(1.0);
        let min_h = 120.0_f32.min(max_h);
        target.clamp(min_h, max_h)
    }

    /// Bottom inset: peak-meter strip + optional FFT panel.
    fn bottom_inset_h(&self, canvas_h: f32) -> f32 {
        let meter = 18.0; // always shown
        OVERVIEW_H + meter + self.fft_panel_height(canvas_h)
    }

    /// Top edge of the overview strip (first thing below the waveform).
    fn overview_y(&self, canvas_h: f32) -> f32 {
        canvas_h - self.bottom_inset_h(canvas_h)
    }

    /// Top edge of the peak-meter strip, which sits under the overview.
    fn meter_y(&self, canvas_h: f32) -> f32 {
        self.overview_y(canvas_h) + OVERVIEW_H
    }

    /// Bounds of the actual waveform drawing area (inset from canvas).
    fn wave_area(&self, canvas_w: f32, canvas_h: f32) -> (f32, f32, f32, f32) {
        let x = self.left_inset();
        let y = self.top_inset();
        let w = (canvas_w - x).max(1.0);
        let h = (canvas_h - y - self.bottom_inset_h(canvas_h)).max(1.0);
        (x, y, w, h)
    }

    /// Screen rect of the overview strip. Drawing, the overlay, and
    /// hit-testing all route through this so they can't disagree.
    fn overview_rect(&self, canvas_w: f32, canvas_h: f32) -> Option<(f32, f32, f32, f32)> {
        if self.duration <= 0.0 {
            return None;
        }
        let (wx, _wy, ww, _wh) = self.wave_area(canvas_w, canvas_h);
        Some((wx, self.overview_y(canvas_h), ww, OVERVIEW_H))
    }

    /// Rect of the FFT panel, whatever mode it's in. `None` when hidden.
    fn fft_panel_rect(&self, canvas_w: f32, canvas_h: f32) -> Option<(f32, f32, f32, f32)> {
        if self.fft_mode == 0 {
            return None;
        }
        let (wx, _wy, ww, _wh) = self.wave_area(canvas_w, canvas_h);
        let fft_y = self.meter_y(canvas_h) + 18.0;
        let fft_h = canvas_h - fft_y;
        if fft_h <= 10.0 {
            return None;
        }
        Some((wx, fft_y, ww, fft_h))
    }

    /// Rect of the Phase panel (the whole FFT area when Phase is active).
    fn phase_panel_rect(&self, canvas_w: f32, canvas_h: f32) -> Option<(f32, f32, f32, f32)> {
        if self.fft_mode != 3 {
            return None;
        }
        self.fft_panel_rect(canvas_w, canvas_h)
    }

    /// Rect of the correlation-curve PLOT area (inside its padding), when
    /// Phase is the active panel.
    fn corr_plot_rect(&self, canvas_w: f32, canvas_h: f32) -> Option<(f32, f32, f32, f32)> {
        let panel = self.phase_panel_rect(canvas_w, canvas_h)?;
        let (gx, _gy, _gw, _gh) = Self::gonio_rect(panel);
        let (pxx, pyy, _pw, ph) = panel;
        let band_w = (gx - pxx).max(40.0);
        let x = pxx + CORR_PAD_L;
        let w = (band_w - CORR_PAD_L - CORR_PAD_R).max(1.0);
        let y = pyy + CORR_PAD_T;
        let h = (ph - CORR_PAD_T - CORR_PAD_B).max(1.0);
        Some((x, y, w, h))
    }

    /// Frequency range covered by the correlation curve's x axis.
    fn corr_freq_range(&self) -> Option<(f32, f32)> {
        if self.sample_rate == 0 || self.probe_band_corr.is_empty() {
            return None;
        }
        let nyq = self.sample_rate as f32 / 2.0;
        let lo = 20.0_f32.max(nyq / self.probe_band_corr.len() as f32);
        Some((lo, nyq))
    }

    /// Map an x position inside the curve plot to a frequency in Hz.
    fn corr_x_to_hz(&self, x_frac: f32) -> Option<f32> {
        let (lo, hi) = self.corr_freq_range()?;
        let log_lo = lo.log10();
        let span = (hi.log10() - log_lo).max(1e-6);
        Some(10f32.powf(log_lo + x_frac.clamp(0.0, 1.0) * span))
    }

    /// Rect of the spectrogram, or `None` when it isn't the active panel.
    ///
    /// The spectrogram is the ONLY analysis panel whose x axis is time — the
    /// spectrum line and the phase curve are both frequency-on-x. So it's
    /// the only one where waveform gestures (click to seek, drag to scrub,
    /// wheel to zoom) have a meaning, and the only one that gets a playhead.
    fn spectrogram_rect(&self, canvas_w: f32, canvas_h: f32) -> Option<(f32, f32, f32, f32)> {
        if self.fft_mode != 2 {
            return None;
        }
        self.fft_panel_rect(canvas_w, canvas_h)
    }

    /// Goniometer column within the Phase panel: `(x, y, w, h)`.
    ///
    /// The single source for this split. `draw_phase` and the badge
    /// hit-test both go through it, so a clickable control drawn in the
    /// column can't drift away from its own click target.
    fn gonio_rect(panel: (f32, f32, f32, f32)) -> (f32, f32, f32, f32) {
        let (px, py, pw, ph) = panel;
        let gw = ph.min(pw * 0.40).max(60.0);
        let band_w = (pw - gw).max(40.0);
        (px + band_w, py, gw, ph)
    }

    /// Hit-test the mono-fold badge.
    fn mono_badge_hit(&self, canvas_w: f32, canvas_h: f32, px: f32, py: f32) -> bool {
        let Some(panel) = self.phase_panel_rect(canvas_w, canvas_h) else {
            return false;
        };
        let (gx, gy, gw, _gh) = Self::gonio_rect(panel);
        let (bx, by, bw, bh) = mono_badge_rect(gx, gy, gw);
        px >= bx && px <= bx + bw && py >= by && py <= by + bh
    }

    /// Convert sample-time to canvas x within the wave area.
    fn t_to_x(&self, t: f32, wave_w: f32) -> f32 {
        let vis = self.visible_duration();
        ((t - self.scroll) / vis) * wave_w
    }
    /// Convert canvas-relative x (already offset by left_inset) to sample-time.
    fn x_to_t(&self, x_in_area: f32, wave_w: f32) -> f32 {
        let vis = self.visible_duration();
        (self.scroll + (x_in_area / wave_w) * vis).clamp(0.0, self.duration)
    }
    /// Convert wave-area-relative y to dB, where 0 dB = top edge, -inf = mid.
    /// Symmetric: top half maps positive dB-to-amplitude up from center; we
    /// only deal with the upper half (negative dB ceiling line, conventionally).
    fn y_to_db(&self, y_in_area: f32, wh: f32) -> f32 {
        let mid = wh * 0.5;
        // Distance above center, normalized.
        let dist = (mid - y_in_area).abs() / mid.max(1.0);
        let amp = dist.clamp(1e-6, 1.0);
        20.0 * amp.log10()
    }
    /// Convert dB (negative) to wave-area-relative y at the top half of the area.
    fn db_to_y_top(&self, db: f32, wy: f32, wh: f32) -> f32 {
        let mid = wy + wh * 0.5;
        let half_h = wh * 0.5;
        let amp = 10f32.powf(db.min(0.0) / 20.0);
        mid - amp * half_h
    }
    /// Slice the envelope for the visible window.
    fn visible_envelope_slice<'b>(&self, env: &'b [(f32, f32)]) -> &'b [(f32, f32)] {
        if env.is_empty() {
            return env;
        }
        let n = env.len();
        let start = ((self.scroll / self.duration) * n as f32) as usize;
        let end = (((self.scroll + self.visible_duration()) / self.duration) * n as f32)
            as usize;
        let start = start.min(n);
        let end = end.min(n).max(start);
        &env[start..end]
    }

    /// Pick the right rendering strategy for the current zoom level and
    /// stroke the envelope into `frame`.
    fn draw_adaptive_envelope(
        &self,
        frame: &mut Frame,
        env: &[(f32, f32)],
        mono: &[f32],
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
        color: Color,
    ) {
        if mono.is_empty() {
            draw_envelope(frame, self.visible_envelope_slice(env), ox, oy, w, h, color);
            return;
        }
        let pixels = w.max(1.0) as usize;
        let vis_secs = self.visible_duration();
        let vis_samples = (vis_secs * self.sample_rate as f32) as usize;
        let samples_per_pixel = vis_samples as f32 / pixels as f32;

        if samples_per_pixel < 3.0 {
            self.draw_waveform_line(frame, mono, ox, oy, w, h, color);
            return;
        }
        let precomputed_slice = self.visible_envelope_slice(env);
        let buckets_per_pixel = precomputed_slice.len() as f32 / pixels as f32;
        if buckets_per_pixel < 3.0 {
            self.draw_per_pixel_envelope(frame, mono, ox, oy, w, h, color);
        } else {
            draw_envelope(frame, precomputed_slice, ox, oy, w, h, color);
        }
    }

    /// Two-layer renderer matching the user's mockup model.
    ///
    /// Mental model:
    ///   - BLUE = OUTPUT body (post all processing including normalize).
    ///     Drawn from center outward. Grows when normalize amplifies,
    ///     shrinks when clipping pulls peaks down.
    ///   - RED = CLIPPING SHADOW: shown in the vertical space between
    ///     the blue's edge and the canvas edge (or the unclipped
    ///     reference, whichever is closer). Represents "what got eaten."
    ///   - The two layers COMPETE for vertical space. When blue fills
    ///     the canvas, red gets squeezed into a thin sliver at the very
    ///     top and bottom, ensuring the user can still see clipping is
    ///     occurring even at full normalization.
    ///   - When there's no clipping (output ≥ input in this column),
    ///     no red is drawn — pure blue body.
    ///
    /// `mono_in` here is the input scaled by the same normalize gain as
    /// the output (computed in the DSP worker), so the clipping shadow
    /// scales together with the output when normalize amplifies.
    fn draw_input_with_eaten(
        &self,
        frame: &mut Frame,
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
        out_color: Color,
        in_color: Color,
    ) {
        if self.mono_in.is_empty() || self.mono_in.len() != self.mono_out.len() {
            return;
        }
        let pixels = w.max(1.0) as usize;
        let mid = oy + h * 0.5;
        let half_h = h * 0.5;
        let min_indicator_px: f32 = 2.0;
        let start_frame = (self.scroll * self.sample_rate as f32) as usize;
        let end_frame =
            ((self.scroll + self.visible_duration()) * self.sample_rate as f32) as usize;
        let end_frame = end_frame.min(self.mono_in.len()).min(self.mono_out.len());
        if end_frame <= start_frame {
            return;
        }
        let span = (end_frame - start_frame) as f32;
        let threshold = 0.0002;

        // ---- Pass 1: bucket samples per pixel column into min/max arrays. ----
        let mut in_max_v: Vec<f32> = Vec::with_capacity(pixels);
        let mut in_min_v: Vec<f32> = Vec::with_capacity(pixels);
        let mut out_max_v: Vec<f32> = Vec::with_capacity(pixels);
        let mut out_min_v: Vec<f32> = Vec::with_capacity(pixels);
        for px in 0..pixels {
            let s0 = start_frame + ((px as f32 / pixels as f32) * span) as usize;
            let s1 = start_frame + (((px + 1) as f32 / pixels as f32) * span) as usize;
            let s1 = s1.min(end_frame).max(s0 + 1);
            let mut in_max = f32::NEG_INFINITY;
            let mut in_min = f32::INFINITY;
            let mut out_max = f32::NEG_INFINITY;
            let mut out_min = f32::INFINITY;
            for i in s0..s1 {
                let vi = self.mono_in[i];
                let vo = self.mono_out[i];
                if vi > in_max { in_max = vi; }
                if vi < in_min { in_min = vi; }
                if vo > out_max { out_max = vo; }
                if vo < out_min { out_min = vo; }
            }
            if !out_max.is_finite() {
                out_max = 0.0; out_min = 0.0;
            }
            if !in_max.is_finite() {
                in_max = 0.0; in_min = 0.0;
            }
            in_max_v.push(in_max);
            in_min_v.push(in_min);
            out_max_v.push(out_max);
            out_min_v.push(out_min);
        }

        // ---- Pass 2: adaptive smoothing across pixels. ----
        // At low samples_per_pixel (zoomed in) — no smoothing, want
        // sample-accurate edges. At high samples_per_pixel (zoomed out),
        // each pixel's bucket has lots of samples so neighboring buckets
        // jitter independently; smoothing fixes the "hair" noise that
        // makes the dense view tiring to look at.
        let samples_per_pixel = span / pixels as f32;
        // Window in pixels. 0 = no smoothing. Scales with sample density.
        let half_window: usize = if samples_per_pixel < 60.0 {
            0
        } else if samples_per_pixel < 200.0 {
            1
        } else if samples_per_pixel < 600.0 {
            2
        } else if samples_per_pixel < 2000.0 {
            3
        } else {
            4
        };

        // Precompute a Gaussian kernel for the chosen window size. Using
        // a Gaussian (rather than a flat box average) means the center
        // pixel weighs most and weight falls off with distance — so
        // neighboring output pixels never land on the same value the way
        // they do with a box filter. That fixes the staircase / blocky
        // edge artifacts that show up at zoomed-out views.
        let kernel: Vec<f32> = if half_window == 0 {
            vec![1.0]
        } else {
            let sigma = half_window as f32 * 0.6;
            let two_sigma_sq = 2.0 * sigma * sigma;
            let n = half_window * 2 + 1;
            let mut k = Vec::with_capacity(n);
            let mut sum = 0.0;
            for j in 0..n {
                let dx = j as f32 - half_window as f32;
                let w = (-dx * dx / two_sigma_sq).exp();
                k.push(w);
                sum += w;
            }
            for w in k.iter_mut() {
                *w /= sum;
            }
            k
        };
        let smooth = |raw: &Vec<f32>, take_max: bool| -> Vec<f32> {
            if half_window == 0 {
                return raw.clone();
            }
            let mut out = Vec::with_capacity(raw.len());
            for i in 0..raw.len() {
                let mut weighted_sum = 0.0;
                let mut weight_total = 0.0;
                let mut extreme = if take_max { f32::NEG_INFINITY } else { f32::INFINITY };
                for j in 0..kernel.len() {
                    let off = j as i64 - half_window as i64;
                    let idx = i as i64 + off;
                    if idx < 0 || idx as usize >= raw.len() {
                        continue;
                    }
                    let v = raw[idx as usize];
                    let w = kernel[j];
                    weighted_sum += v * w;
                    weight_total += w;
                    if take_max {
                        if v > extreme { extreme = v; }
                    } else if v < extreme {
                        extreme = v;
                    }
                }
                let gaussian_smoothed = if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    raw[i]
                };
                // Blend Gaussian-smoothed value with the local extreme to
                // preserve real peaks. Gaussian gives the soft curve, the
                // extreme blend keeps loud peaks from being washed out.
                out.push(0.5 * gaussian_smoothed + 0.5 * extreme);
            }
            out
        };
        // Only smooth the OUTPUT (body) layer — keep the input min/max
        // raw so the yellow clipping-shadow caps stay sharp and pop
        // through at zoom-out. Smoothing the input too pulls clipped
        // peaks down toward the average, which shrinks (or kills) the
        // yellow caps that signal "this got clipped here."
        let in_max_s = in_max_v;
        let in_min_s = in_min_v;
        let out_max_s = smooth(&out_max_v, true);
        let out_min_s = smooth(&out_min_v, false);
        // Raw (unsmoothed) output extremes for the body TOP/BOTTOM edges.
        // In a clipped region the real output is flat at the ceiling, but
        // Gaussian smoothing pulls the value down between peaks, making
        // the blue/amber boundary a wavy line instead of the clean flat
        // ceiling. Using the raw peak for the edge keeps clipped regions
        // flat-topped (raw wins there) while smoothing still tames jitter
        // elsewhere (raw ≈ smoothed when not clipping).
        let out_max_raw = &out_max_v;
        let out_min_raw = &out_min_v;

        // ---- Pass 3: draw blue body + red shadow from smoothed values. ----
        // Per-pixel "inside a fade" flag, so the clipping caps are
        // SUPPRESSED in fade regions (Model A: a fade is a deliberate
        // volume shape, not clipping — don't draw clip evidence there).
        let px_in_fade: Vec<bool> = {
            let sr = self.sample_rate as f32;
            let total_frames = self.mono_out.len().max(self.mono_in.len());
            let duration = total_frames as f32 / sr.max(1.0);
            // Collect fade ranges in frames, clamped to the trim window
            // (trim acts as a zone boundary for fades, matching the DSP).
            let trim_s = self.trim_start.map(|t| (t * sr) as usize).unwrap_or(0);
            let trim_e = self
                .trim_end
                .map(|t| (t * sr) as usize)
                .unwrap_or(total_frames.max(1));
            let mut ranges: Vec<(usize, usize)> = Vec::new();
            for (s, e, z) in self.zones.iter_zones(duration) {
                let zs = (s * sr) as usize;
                let ze = ((e * sr) as usize).min(total_frames.max(1));
                let eff_start = zs.max(trim_s).min(ze);
                let eff_end = ze.min(trim_e).max(zs);
                let ef = eff_end.saturating_sub(eff_start);
                let fi = ((z.fade_in_secs.max(0.0) * sr) as usize).min(ef / 2);
                let fo = ((z.fade_out_secs.max(0.0) * sr) as usize).min(ef / 2);
                if fi > 0 {
                    ranges.push((eff_start, eff_start + fi));
                }
                if fo > 0 {
                    ranges.push((eff_end.saturating_sub(fo), eff_end));
                }
            }
            let mut v = Vec::with_capacity(pixels);
            // Trimmed-off pixels (outside the trim window) also suppress the
            // clipping cap — that audio is cut on export, so its clipping is
            // irrelevant. Folded into the same per-pixel flag.
            let trim_s_f = self.trim_start.map(|t| (t * sr) as usize).unwrap_or(0);
            let trim_e_f = self
                .trim_end
                .map(|t| (t * sr) as usize)
                .unwrap_or(total_frames.max(1));
            for px in 0..pixels {
                let frame = start_frame + ((px as f32 / pixels as f32) * span) as usize;
                let in_a_fade = ranges.iter().any(|&(a, b)| frame >= a && frame < b);
                let trimmed_off = frame < trim_s_f || frame >= trim_e_f;
                v.push(in_a_fade || trimmed_off);
            }
            v
        };
        for px in 0..pixels {
            let in_max = in_max_s[px];
            let in_min = in_min_s[px];
            let out_max = out_max_s[px];
            let out_min = out_min_s[px];
            let suppress_cap = px_in_fade[px];
            // Body top/bottom edges (blue/amber boundary) use the more
            // extreme of smoothed vs raw, so clipped regions stay flat at
            // the ceiling instead of sagging with the smoothing.
            let out_max_edge = out_max.max(out_max_raw[px]);
            let out_min_edge = out_min.min(out_min_raw[px]);
            let x = ox + px as f32;

            // Blue (output) body with vertical gradient effect.
            // Drawn as N nested layers per pixel column, each insetting
            // further from the top/bottom of the body. The layers'
            // alphas compound through the alpha-blend so the centerline
            // ends up fully opaque while edges fade smoothly. More
            // layers = smoother gradient at higher cost; 7 strikes a
            // good balance between softness and render speed.
            let out_top_y = mid - out_max_edge.clamp(-1.0, 1.0) * half_h;
            let out_bot_y = mid - out_min_edge.clamp(-1.0, 1.0) * half_h;
            let body_h = (out_bot_y - out_top_y).max(1.0);
            const BODY_LAYERS: usize = 11;
            // Per-layer alpha — kept low so even when many stack at
            // center they don't blow out. Alpha blending compounds via
            // 1-(1-a)^n, so 0.11 over 11 layers yields ~72% opacity at
            // the centerline and a smooth falloff to ~11% at the shell.
            let per_layer_alpha = out_color.a * 0.11;
            let layer_color = Color {
                a: per_layer_alpha,
                ..out_color
            };
            for i in 0..BODY_LAYERS {
                // Inset increases linearly from 0 (outermost) to ~45%
                // off each edge (innermost), so the innermost layer
                // hugs the centerline.
                let t = i as f32 / (BODY_LAYERS as f32);
                let inset = body_h * 0.45 * t;
                let y = out_top_y + inset;
                let h_layer = (body_h - 2.0 * inset).max(1.0);
                frame.fill_rectangle(
                    Point::new(x, y),
                    Size::new(1.0, h_layer),
                    layer_color,
                );
            }

            // Red clipping shadow (positive side). Suppressed inside fade
            // regions (Model A). Uses the edge value so the cap boundary
            // matches the (flat-topped) body top.
            if !suppress_cap && in_max > 0.0 && in_max - out_max_edge > threshold {
                let in_top_y_raw = mid - in_max * half_h;
                let canvas_top = oy;
                let red_top_initial = in_top_y_raw.max(canvas_top);
                let mut red_bottom = out_top_y;
                let mut red_top = red_top_initial.min(red_bottom);
                let avail_room = red_bottom - canvas_top;
                if avail_room >= min_indicator_px {
                    if red_bottom - red_top < min_indicator_px {
                        red_top = (red_bottom - min_indicator_px).max(canvas_top);
                    }
                } else {
                    red_top = canvas_top;
                    red_bottom = (canvas_top + avail_room).max(canvas_top);
                }
                if red_bottom > red_top {
                    frame.fill_rectangle(
                        Point::new(x, red_top),
                        Size::new(1.0, red_bottom - red_top),
                        in_color,
                    );
                }
            }
            // Red clipping shadow (negative side, mirror). Suppressed
            // inside fade regions (Model A).
            if !suppress_cap && in_min < 0.0 && out_min_edge - in_min > threshold {
                let in_bot_y_raw = mid - in_min * half_h;
                let canvas_bot = oy + h;
                let mut red_top = out_bot_y;
                let mut red_bot = in_bot_y_raw.min(canvas_bot);
                let avail_room = canvas_bot - red_top;
                if avail_room >= min_indicator_px {
                    if red_bot - red_top < min_indicator_px {
                        red_bot = (red_top + min_indicator_px).min(canvas_bot);
                    }
                } else {
                    red_top = (canvas_bot - avail_room).min(canvas_bot);
                    red_bot = canvas_bot;
                }
                if red_bot > red_top {
                    frame.fill_rectangle(
                        Point::new(x, red_top),
                        Size::new(1.0, red_bot - red_top),
                        in_color,
                    );
                }
            }
        }
    }

    /// Plugin-style "eaten" overlay (legacy two-pass variant — kept for
    /// reference but no longer wired up; `draw_input_with_eaten` replaces
    /// the input-envelope + eaten-overlay pair with a single unified pass).
    #[allow(dead_code)]
    fn draw_eaten_overlay(
        &self,
        frame: &mut Frame,
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
        color: Color,
    ) {
        if self.mono_in.is_empty()
            || self.mono_in.len() != self.mono_out.len()
        {
            return;
        }
        let pixels = w.max(1.0) as usize;
        let mid = oy + h * 0.5;
        let half_h = h * 0.5;
        let start_frame = (self.scroll * self.sample_rate as f32) as usize;
        let end_frame =
            ((self.scroll + self.visible_duration()) * self.sample_rate as f32) as usize;
        let end_frame = end_frame.min(self.mono_in.len()).min(self.mono_out.len());
        if end_frame <= start_frame {
            return;
        }
        let span = (end_frame - start_frame) as f32;
        let threshold = 0.0002;
        for px in 0..pixels {
            let s0 = start_frame + ((px as f32 / pixels as f32) * span) as usize;
            let s1 = start_frame + (((px + 1) as f32 / pixels as f32) * span) as usize;
            let s1 = s1.min(end_frame).max(s0 + 1);
            // Track signed min/max for both signals in this pixel column.
            let mut in_max = f32::NEG_INFINITY;
            let mut in_min = f32::INFINITY;
            let mut out_max = f32::NEG_INFINITY;
            let mut out_min = f32::INFINITY;
            for i in s0..s1 {
                let vi = self.mono_in[i];
                let vo = self.mono_out[i];
                if vi > in_max { in_max = vi; }
                if vi < in_min { in_min = vi; }
                if vo > out_max { out_max = vo; }
                if vo < out_min { out_min = vo; }
            }
            let x = ox + px as f32;
            // Top cap: where input's positive peak rises above output's.
            // Only draw if input actually reaches positive and clipping
            // shaved that positive peak.
            if in_max > 0.0 && in_max - out_max > threshold {
                let in_y = mid - in_max * half_h;
                let out_y = mid - out_max.max(0.0) * half_h;
                frame.fill_rectangle(
                    Point::new(x, in_y),
                    Size::new(1.0, out_y - in_y),
                    color,
                );
            }
            // Bottom cap: where input's negative trough falls below output's.
            if in_min < 0.0 && out_min - in_min > threshold {
                let in_y = mid - in_min * half_h;
                let out_y = mid - out_min.min(0.0) * half_h;
                frame.fill_rectangle(
                    Point::new(x, out_y),
                    Size::new(1.0, in_y - out_y),
                    color,
                );
            }
        }
    }

    fn draw_per_pixel_envelope(
        &self,
        frame: &mut Frame,
        mono: &[f32],
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
        color: Color,
    ) {
        let pixels = w.max(1.0) as usize;
        let mid = oy + h * 0.5;
        let start_frame = (self.scroll * self.sample_rate as f32) as usize;
        let end_frame =
            ((self.scroll + self.visible_duration()) * self.sample_rate as f32) as usize;
        let end_frame = end_frame.min(mono.len());
        if end_frame <= start_frame {
            return;
        }
        let span = (end_frame - start_frame) as f32;
        let half_h = h * 0.5;
        let path = Path::new(|p| {
            for px in 0..pixels {
                let s0 = start_frame + ((px as f32 / pixels as f32) * span) as usize;
                let s1 = start_frame + (((px + 1) as f32 / pixels as f32) * span) as usize;
                let s1 = s1.min(end_frame).max(s0 + 1);
                let mut lo = f32::INFINITY;
                let mut hi = f32::NEG_INFINITY;
                for &s in &mono[s0..s1] {
                    if s < lo { lo = s; }
                    if s > hi { hi = s; }
                }
                let lo = lo.clamp(-1.0, 1.0);
                let hi = hi.clamp(-1.0, 1.0);
                let x = ox + px as f32;
                let y_hi = mid - hi * half_h;
                let y_lo = mid - lo * half_h;
                p.move_to(Point::new(x, y_hi));
                p.line_to(Point::new(x, y_lo));
            }
        });
        frame.stroke(&path, Stroke::default().with_color(color).with_width(1.0));
    }

    /// Sample-accurate "eaten" overlay for use at line-zoom levels. For
    /// each pair of (input, output) samples where they differ enough to
    /// register as clipping, fills the vertical gap between input's
    /// position and output's position with the output color. Combined
    /// with drawing the input as a normal line on top, this gives a
    /// "blue line traces input, red fill shows what the clipper shaved"
    /// look — sample-accurate and visually unambiguous (no two lines
    /// tracing slightly different paths that look phase-shifted).
    /// Sample-level renderer. When zoomed in so far that each sample
    /// gets its own pixel column (or more), draws individual samples as
    /// small dots connected by a Catmull-Rom spline. This is the
    /// OcenAudio-style "audio as connected points" view — the natural
    /// representation for editing operations that work on individual
    /// samples (click removal, manual peak adjustment).
    ///
    /// Catmull-Rom over sinc because:
    ///   - Visually identical at editing zoom levels (you can't see the
    ///     difference between sinc and CR with naked eye at these scales)
    ///   - Much cheaper to compute
    ///   - Doesn't need windowing for finite buffers
    ///
    /// The curve passes exactly through each sample point, so the dots
    /// and the line never disagree about where a sample actually is.
    fn draw_sample_dots_with_spline(
        &self,
        frame: &mut Frame,
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
        out_color: Color,
        in_color: Color,
    ) {
        if self.mono_out.is_empty() {
            return;
        }
        let mid = oy + h * 0.5;
        let half_h = h * 0.5;
        let start_frame = (self.scroll * self.sample_rate as f32) as usize;
        let end_frame =
            ((self.scroll + self.visible_duration()) * self.sample_rate as f32) as usize;
        let end_frame = end_frame.min(self.mono_out.len());
        if end_frame <= start_frame + 1 {
            return;
        }
        let span = (end_frame - start_frame) as f32;

        // Helper closure: convert sample index → (x, y) given a buffer.
        let to_xy = |i: usize, buf: &[f32]| -> (f32, f32) {
            let rel = (i - start_frame) as f32 / span;
            let x = ox + rel * w;
            let v = buf[i].clamp(-1.0, 1.0);
            let y = mid - v * half_h;
            (x, y)
        };

        // Catmull-Rom curve helper. Renders the curve through every
        // sample in `buf` using a thin stroke, no dots.
        let n = end_frame - start_frame;
        let n_steps_per_segment = 8usize;

        let draw_curve = |frame: &mut Frame, buf: &[f32], stroke: Stroke<'_>| {
            let p_at = |idx: i64| -> (f32, f32) {
                let clamped = idx.clamp(start_frame as i64, (end_frame - 1) as i64) as usize;
                to_xy(clamped, buf)
            };
            for i in start_frame..(end_frame - 1) {
                let p0 = p_at(i as i64 - 1);
                let p1 = p_at(i as i64);
                let p2 = p_at(i as i64 + 1);
                let p3 = p_at(i as i64 + 2);
                let mut last = p1;
                for step in 1..=n_steps_per_segment {
                    let t = step as f32 / n_steps_per_segment as f32;
                    let t2 = t * t;
                    let t3 = t2 * t;
                    let x = 0.5
                        * ((2.0 * p1.0)
                            + (-p0.0 + p2.0) * t
                            + (2.0 * p0.0 - 5.0 * p1.0 + 4.0 * p2.0 - p3.0) * t2
                            + (-p0.0 + 3.0 * p1.0 - 3.0 * p2.0 + p3.0) * t3);
                    let y = 0.5
                        * ((2.0 * p1.1)
                            + (-p0.1 + p2.1) * t
                            + (2.0 * p0.1 - 5.0 * p1.1 + 4.0 * p2.1 - p3.1) * t2
                            + (-p0.1 + 3.0 * p1.1 - 3.0 * p2.1 + p3.1) * t3);
                    let next = (x, y);
                    let segment = Path::line(
                        Point::new(last.0, last.1),
                        Point::new(next.0, next.1),
                    );
                    frame.stroke(&segment, stroke);
                    last = next;
                }
            }
        };

        // ---- Layer 1: output curve (blue) drawn first as the base. ----
        let out_stroke = Stroke::default().with_color(out_color).with_width(1.5);
        draw_curve(frame, &self.mono_out, out_stroke);

        // ---- Layer 2: input reference curve. Skipped when the caller
        // passes a transparent color (the dispatch does this so the full
        // input contour isn't painted amber across the whole waveform —
        // clipping evidence comes from the gap-fill overlay instead).
        if in_color.a > 0.0
            && !self.mono_in.is_empty()
            && self.mono_in.len() == self.mono_out.len()
        {
            let in_stroke = Stroke::default().with_color(in_color).with_width(1.2);
            draw_curve(frame, &self.mono_in, in_stroke);
        }

        // ---- Layer 3: output sample dots ON TOP of both curves. ----
        // Dots stay on top because they mark the actual sample positions
        // the user can edit. Yellow line underneath them remains visible
        // around and between the dots.
        let avg_x_step = w / (n.max(1) as f32);
        let dot_radius = (avg_x_step * 0.18).clamp(2.0, 4.0);
        for i in start_frame..end_frame {
            let (x, y) = to_xy(i, &self.mono_out);
            let circle = Path::new(|p| {
                p.circle(Point::new(x, y), dot_radius);
            });
            frame.fill(&circle, out_color);
        }
    }

    fn draw_eaten_overlay_samples(
        &self,
        frame: &mut Frame,
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
        color: Color,
    ) {
        if self.mono_in.is_empty() || self.mono_in.len() != self.mono_out.len() {
            return;
        }
        let mid = oy + h * 0.5;
        let half_h = h * 0.5;
        let start_frame = (self.scroll * self.sample_rate as f32) as usize;
        let end_frame =
            ((self.scroll + self.visible_duration()) * self.sample_rate as f32) as usize;
        let end_frame = end_frame
            .min(self.mono_in.len())
            .min(self.mono_out.len());
        if end_frame <= start_frame + 1 {
            return;
        }
        let span = (end_frame - start_frame) as f32;
        // Skip drawing for differences smaller than this to avoid
        // shimmering noise from quantization round-off.
        let threshold = 0.0002;
        // Fade regions (in FRAME indices) to SUPPRESS amber in. A fade is a
        // deliberate volume shape, not clipping — even where the output
        // momentarily reaches the ceiling near a fade's full-gain end, the
        // user has "handled" that region, so showing clipping evidence
        // there is just noise. We collect each zone's fade-in and fade-out
        // ranges and skip any sample inside one.
        let fade_ranges: Vec<(usize, usize)> = {
            let sr = self.sample_rate as f32;
            let total_frames = self.mono_out.len();
            let duration = total_frames as f32 / sr.max(1.0);
            // Trim acts as a zone boundary for fades, so the suppression
            // region must match: fade-in begins at max(zone_start, trim_start)
            // and fade-out ends at min(zone_end, trim_end).
            let trim_s = self
                .trim_start
                .map(|t| (t * sr) as usize)
                .unwrap_or(0);
            let trim_e = self
                .trim_end
                .map(|t| (t * sr) as usize)
                .unwrap_or(total_frames);
            let mut v: Vec<(usize, usize)> = Vec::new();
            for (s, e, z) in self.zones.iter_zones(duration) {
                let zs = (s * sr) as usize;
                let ze = ((e * sr) as usize).min(total_frames);
                let eff_start = zs.max(trim_s).min(ze);
                let eff_end = ze.min(trim_e).max(zs);
                let eff_frames = eff_end.saturating_sub(eff_start);
                let fi = ((z.fade_in_secs.max(0.0) * sr) as usize).min(eff_frames / 2);
                let fo = ((z.fade_out_secs.max(0.0) * sr) as usize).min(eff_frames / 2);
                if fi > 0 {
                    v.push((eff_start, eff_start + fi));
                }
                if fo > 0 {
                    v.push((eff_end.saturating_sub(fo), eff_end));
                }
            }
            v.sort_unstable();
            v
        };
        let in_fade = |frame: usize| -> bool {
            fade_ranges.iter().any(|&(a, b)| frame >= a && frame < b)
        };
        // Trimmed-off regions (outside [trim_start, trim_end)) are excluded
        // from export, so clipping there is irrelevant — suppress amber.
        let sr_f = self.sample_rate as f32;
        let trim_s_frame = self.trim_start.map(|t| (t * sr_f) as usize).unwrap_or(0);
        let trim_e_frame = self
            .trim_end
            .map(|t| (t * sr_f) as usize)
            .unwrap_or(self.mono_out.len());
        let px_per_sample = w / span.max(1.0);
        let bar_w = (px_per_sample * 0.4).clamp(1.0, 2.0);
        let bar_color = Color { a: color.a * 0.7, ..color };
        for i in start_frame..end_frame {
            // Skip trimmed-off audio and fade regions (Model A: fade wins).
            if i < trim_s_frame || i >= trim_e_frame || in_fade(i) {
                continue;
            }
            let in_v = self.mono_in[i].clamp(-1.0, 1.0);
            let out_v = self.mono_out[i].clamp(-1.0, 1.0);
            if (in_v - out_v).abs() < threshold {
                continue;
            }
            let x = ox + ((i - start_frame) as f32 / span) * w;
            let in_y = mid - in_v * half_h;
            let out_y = mid - out_v * half_h;
            let (top, bot) = if in_y < out_y { (in_y, out_y) } else { (out_y, in_y) };
            frame.fill_rectangle(
                Point::new(x - bar_w * 0.5, top),
                Size::new(bar_w, bot - top),
                bar_color,
            );
        }
    }

    fn draw_waveform_line(
        &self,
        frame: &mut Frame,
        mono: &[f32],
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
        color: Color,
    ) {
        let mid = oy + h * 0.5;
        let half_h = h * 0.5;
        let start_frame = (self.scroll * self.sample_rate as f32) as usize;
        let end_frame =
            ((self.scroll + self.visible_duration()) * self.sample_rate as f32) as usize;
        let end_frame = end_frame.min(mono.len());
        if end_frame <= start_frame + 1 {
            return;
        }
        let span = (end_frame - start_frame) as f32;
        let path = Path::new(|p| {
            let mut first = true;
            for i in start_frame..end_frame {
                let x = ox + ((i - start_frame) as f32 / span) * w;
                let y = mid - mono[i].clamp(-1.0, 1.0) * half_h;
                if first {
                    p.move_to(Point::new(x, y));
                    first = false;
                } else {
                    p.line_to(Point::new(x, y));
                }
            }
        });
        frame.stroke(&path, Stroke::default().with_color(color).with_width(1.2));
    }

    fn draw_reduction_overlay(
        &self,
        frame: &mut Frame,
        ox: f32,
        oy: f32,
        w: f32,
        h: f32,
    ) {
        let pixels = w.max(1.0) as usize;
        let start_frame = (self.scroll * self.sample_rate as f32) as usize;
        let end_frame =
            ((self.scroll + self.visible_duration()) * self.sample_rate as f32) as usize;
        let end_frame = end_frame.min(self.mono_in.len()).min(self.mono_out.len());
        if end_frame <= start_frame {
            return;
        }
        let span = (end_frame - start_frame) as f32;
        let strip_h_max = 14.0;
        let threshold = 0.005;

        // ---- First pass: collect per-pixel reduction amounts ----
        // Reduction = peak_input - peak_output for each pixel column.
        // Stored as a flat buffer so we can smooth across pixels.
        let mut raw: Vec<f32> = Vec::with_capacity(pixels);
        for px in 0..pixels {
            let s0 = start_frame + ((px as f32 / pixels as f32) * span) as usize;
            let s1 = start_frame + (((px + 1) as f32 / pixels as f32) * span) as usize;
            let s1 = s1.min(end_frame).max(s0 + 1);
            let mut peak_in = 0.0f32;
            let mut peak_out = 0.0f32;
            for i in s0..s1 {
                let ai = self.mono_in[i].abs();
                let ao = self.mono_out[i].abs();
                if ai > peak_in { peak_in = ai; }
                if ao > peak_out { peak_out = ao; }
            }
            raw.push((peak_in - peak_out).max(0.0));
        }

        // ---- Second pass: smooth with a moving-average window ----
        // Window of ~21 pixels (10 each side + center). Clamping handles
        // the edges so we don't lose data at canvas boundaries.
        let half_window: usize = 10;
        let mut smoothed: Vec<f32> = Vec::with_capacity(pixels);
        for i in 0..pixels {
            let lo = i.saturating_sub(half_window);
            let hi = (i + half_window + 1).min(pixels);
            let count = (hi - lo) as f32;
            let sum: f32 = raw[lo..hi].iter().sum();
            smoothed.push(sum / count);
        }

        // ---- Third pass: draw smoothed values as continuous bands ----
        // Bars now flow into each other since adjacent pixels have similar
        // (smoothed) values — visually reads as a single red band whose
        // height tracks the amount of reduction in that region of audio,
        // rather than individual spikes for every peak.
        // Red because the reduction overlay shows ACTIVE limiting/clipping
        // happening — the alarm. The clipped-portion display uses amber
        // (less intense) as the passive evidence of what was cut.
        let color = Color::from_rgba(1.0, 0.2, 0.2, 0.85);
        for px in 0..pixels {
            let r = smoothed[px];
            if r <= threshold { continue; }
            let bar_h = (r.sqrt() * strip_h_max).min(strip_h_max);
            let x = ox + px as f32;
            frame.fill_rectangle(Point::new(x, oy), Size::new(1.0, bar_h), color);
            frame.fill_rectangle(Point::new(x, oy + h - bar_h), Size::new(1.0, bar_h), color);
        }
    }

    /// Render the actual wave area: envelopes + zones + splits + axis.
    fn draw_wave_area(&self, frame: &mut Frame, wx: f32, wy: f32, ww: f32, wh: f32) {
        let mid = wy + wh * 0.5;

        // Selected-zone borders only — draw the boundary lines BEFORE
        // the envelopes so they don't sit on top of the waveform. The
        // dim overlay for non-selected zones is applied AFTER the
        // waveform renders, since it needs to dim the rendered colors.
        if self.zones.zones.len() > 1 {
            for (i, (start, end, _z)) in self.zones.iter_zones(self.duration).enumerate() {
                if Some(i) != self.selected_zone {
                    continue;
                }
                let x0 = wx + self.t_to_x(start, ww).max(0.0).min(ww);
                let x1 = wx + self.t_to_x(end, ww).max(0.0).min(ww);
                if x1 <= x0 {
                    continue;
                }
                let border = Color::from_rgba(1.0, 1.0, 1.0, 0.35);
                let stroke = Stroke::default().with_color(border).with_width(1.0);
                let left = Path::line(Point::new(x0, wy), Point::new(x0, wy + wh));
                let right = Path::line(Point::new(x1, wy), Point::new(x1, wy + wh));
                frame.stroke(&left, stroke);
                frame.stroke(&right, stroke);
            }
        }

        // Envelopes — display style depends on zoom level:
        //
        // - Zoomed in (line mode, samples-per-pixel < 3): draw both input
        //   and output as actual sample-accurate waveforms layered on
        //   each other. Lets you see the precise shape of clipping.
        //
        // - Zoomed out (envelope/bucket mode): draw the input envelope as
        //   the dominant body color, and the *gap* between input and
        //   output (the "eaten" portion) as red caps on top. Plugin-style.
        let pixels = ww.max(1.0) as usize;
        let vis_samples = (self.visible_duration() * self.sample_rate as f32) as usize;
        let samples_per_pixel = vis_samples as f32 / pixels as f32;
        let sample_mode =
            samples_per_pixel < 0.5 && !self.mono_out.is_empty() && pixels > 0;
        let line_mode =
            !sample_mode && samples_per_pixel < 6.0 && !self.mono_in.is_empty();

        // "Is processing actually happening?" Compare a sparse sample of
        // input vs output: if all sampled pairs are within float epsilon,
        // the audio is identity (e.g. fresh load with Hard/0dB defaults,
        // or just after Apply). In that case we skip the input overlay
        // entirely — there's no clipping to show, and stochastic
        // threshold-crossing was producing flickering yellow that wasn't
        // actually meaningful.
        //
        // Sparse check: 64 evenly-spaced samples across the buffer is
        // ~enough to catch any real processing without being expensive.
        // If even one pair differs meaningfully, we treat it as
        // "processing active" and render input as usual.
        let processing_active = {
            if self.mono_in.is_empty()
                || self.mono_out.is_empty()
                || self.mono_in.len() != self.mono_out.len()
            {
                false
            } else {
                // Compare the precomputed min/max ENVELOPES rather than a
                // sparse grid of raw samples. The old version probed only
                // 64 evenly-spaced raw samples; with a low ceiling on
                // peaky material clipping touches only the transient peaks
                // (a tiny fraction of samples), and a hard clipper leaves
                // every below-ceiling sample bit-identical to the input —
                // so the 64 probes almost always missed the clipped
                // samples and the whole yellow overlay flipped on/off as
                // the ceiling moved. The envelopes hold per-bucket peaks
                // across the whole file, so any bucket where the input
                // peak exceeds the output peak (exactly where caps draw)
                // is detected. ~2000 buckets, still cheap.
                let env_differ = |a: &[(f32, f32)], b: &[(f32, f32)]| -> bool {
                    let n = a.len().min(b.len());
                    for i in 0..n {
                        if (a[i].0 - b[i].0).abs() > 1e-5
                            || (a[i].1 - b[i].1).abs() > 1e-5
                        {
                            return true;
                        }
                    }
                    false
                };
                if !self.envelope_in.is_empty() && !self.envelope_out.is_empty() {
                    env_differ(self.envelope_in, self.envelope_out)
                } else {
                    self.mono_in
                        .iter()
                        .zip(self.mono_out.iter())
                        .any(|(a, b)| (a - b).abs() > 1e-5)
                }
            }
        };

        if sample_mode {
            // OcenAudio-style spline+dots. Pass TRANSPARENT for the input
            // curve so the spline doesn't paint the full input contour
            // amber (that yellows nearly the whole waveform at this zoom).
            // Clipping evidence comes from the gap-fill below instead.
            self.draw_sample_dots_with_spline(
                frame,
                wx, wy, ww, wh,
                self.envelope_in_color,  // blue (output curve + dots)
                Color::TRANSPARENT,      // skip the amber input curve
            );
            if processing_active {
                self.draw_eaten_overlay_samples(
                    frame,
                    wx, wy, ww, wh,
                    self.envelope_out_color,
                );
            }
        } else if line_mode {
            // Two-layer line view. Draw OUTPUT (blue) as the body, then
            // fill ONLY the in→out gap with amber. We do NOT draw the full
            // input contour in amber: at mid zoom that paints the whole
            // waveform yellow (input == output almost everywhere). The
            // gap-fill marks exactly what the clipper shaved, consistent
            // with envelope mode at every zoom.
            self.draw_adaptive_envelope(
                frame,
                self.envelope_out,
                self.mono_out,
                wx, wy, ww, wh,
                self.envelope_in_color, // output envelope painted in the blue color
            );
            if processing_active {
                self.draw_eaten_overlay_samples(
                    frame,
                    wx, wy, ww, wh,
                    self.envelope_out_color,
                );
            }
        } else {
            // Envelope mode: unified two-layer renderer. Pass colors in
            // the order (out, in) matching the function signature.
            self.draw_input_with_eaten(
                frame,
                wx, wy, ww, wh,
                self.envelope_in_color,  // blue = output body
                if processing_active {
                    self.envelope_out_color // yellow = clipping evidence
                } else {
                    Color::TRANSPARENT
                },
            );
        }

        // Reduction overlay (top + bottom red bars).
        if self.show_reduction
            && !self.mono_in.is_empty()
            && self.mono_in.len() == self.mono_out.len()
        {
            self.draw_reduction_overlay(frame, wx, wy, ww, wh);
        }

        // Ceiling lines, drawn on top of envelopes so they're always visible.
        let half_h = wh * 0.5;
        for (_i, (start, end, z)) in self.zones.iter_zones(self.duration).enumerate() {
            let x0 = wx + self.t_to_x(start, ww).max(0.0).min(ww);
            let x1 = wx + self.t_to_x(end, ww).max(0.0).min(ww);
            if x1 <= x0 { continue; }
            let ceil_amp = 10f32.powf(z.ceiling_db.min(0.0) / 20.0);
            let y_top = mid - ceil_amp * half_h;
            let y_bot = mid + ceil_amp * half_h;
            let line_top = Path::line(Point::new(x0, y_top), Point::new(x1, y_top));
            let line_bot = Path::line(Point::new(x0, y_bot), Point::new(x1, y_bot));
            let stroke = Stroke::default()
                .with_color(Color::from_rgba(0.95, 0.25, 0.25, 0.9))
                .with_width(1.5);
            frame.stroke(&line_top, stroke);
            frame.stroke(&line_bot, stroke);
        }

        // De-emphasize non-selected zones with a dark translucent overlay.
        // Drawn AFTER the waveform + ceiling lines so it dims all of
        // them together, but BEFORE the split markers / playhead / hover
        // which are UI navigation aids and should stay full brightness.
        // This inverts the older "highlight selected" approach: the
        // selected zone now reads as the default visual state and the
        // non-selected zones are temporarily de-emphasized, which keeps
        // the colors inside the selected zone fully readable.
        if self.zones.zones.len() > 1 {
            for (i, (start, end, _z)) in self.zones.iter_zones(self.duration).enumerate() {
                if Some(i) == self.selected_zone {
                    continue;
                }
                let x0 = wx + self.t_to_x(start, ww).max(0.0).min(ww);
                let x1 = wx + self.t_to_x(end, ww).max(0.0).min(ww);
                if x1 <= x0 { continue; }
                frame.fill_rectangle(
                    Point::new(x0, wy),
                    Size::new(x1 - x0, wh),
                    Color::from_rgba(0.0, 0.0, 0.0, 0.45),
                );
            }
        }

        // ---- Trim regions. Dark overlay over the excluded spans
        // (before trim_start, after trim_end), plus a colored handle line
        // with top+bottom nubs at each boundary. Green = start, red =
        // end. Non-destructive: this only previews what export/normalize
        // will use. Drawn after zone dimming so trimmed areas read as the
        // most de-emphasized, but before splits/playhead/hover (UI aids).
        {
            let trim_dim = Color::from_rgba(0.0, 0.0, 0.0, 0.82);
            // Leading excluded region: [0, trim_start].
            if let Some(ts) = self.trim_start {
                if ts > 0.0 {
                    let x0 = wx;
                    let x1 = wx + self.t_to_x(ts, ww).clamp(0.0, ww);
                    if x1 > x0 {
                        frame.fill_rectangle(
                            Point::new(x0, wy),
                            Size::new(x1 - x0, wh),
                            trim_dim,
                        );
                    }
                }
            }
            // Trailing excluded region: [trim_end, duration].
            if let Some(te) = self.trim_end {
                if te < self.duration {
                    let x0 = wx + self.t_to_x(te, ww).clamp(0.0, ww);
                    let x1 = wx + ww;
                    if x1 > x0 {
                        frame.fill_rectangle(
                            Point::new(x0, wy),
                            Size::new(x1 - x0, wh),
                            trim_dim,
                        );
                    }
                }
            }

            // Handle drawing helper: a vertical line across the full height
            // + an inward-pointing triangle "flag" at top and bottom. The
            // start handle points RIGHT (into the kept audio), the end
            // handle points LEFT. The triangles sit on the INSIDE of the
            // line (toward the audio) so that when a handle is parked at the
            // very start (0:00) or end of the file, the flag still protrudes
            // into the canvas and stays grabbable rather than being flush
            // against the edge.
            let flag_w = 9.0;
            let flag_h = 13.0;
            let line_w = 1.5;
            let draw_trim_handle = |frame: &mut Frame, t: f32, color: Color, points_right: bool| {
                let x = wx + self.t_to_x(t, ww);
                if x < wx - 4.0 || x > wx + ww + 4.0 {
                    return;
                }
                // Full-height line.
                frame.fill_rectangle(
                    Point::new(x - line_w * 0.5, wy),
                    Size::new(line_w, wh),
                    color,
                );
                // Triangle tip direction. base sits ON the line, tip points
                // inward by flag_w.
                let dir = if points_right { 1.0 } else { -1.0 };
                let tip_x = x + dir * flag_w;
                // Top flag: base spans [wy, wy+flag_h] on the line, tip at
                // the vertical midpoint of that span.
                let top = Path::new(|b| {
                    b.move_to(Point::new(x, wy));
                    b.line_to(Point::new(x, wy + flag_h));
                    b.line_to(Point::new(tip_x, wy + flag_h * 0.5));
                    b.close();
                });
                frame.fill(&top, color);
                // Bottom flag: mirror at the bottom edge.
                let bottom = Path::new(|b| {
                    b.move_to(Point::new(x, wy + wh));
                    b.line_to(Point::new(x, wy + wh - flag_h));
                    b.line_to(Point::new(tip_x, wy + wh - flag_h * 0.5));
                    b.close();
                });
                frame.fill(&bottom, color);
            };

            // Green start handle (points right), red end handle (points left).
            if let Some(ts) = self.trim_start {
                draw_trim_handle(frame, ts, Color::from_rgba(0.20, 0.80, 0.35, 0.95), true);
            }
            if let Some(te) = self.trim_end {
                draw_trim_handle(frame, te, Color::from_rgba(0.90, 0.25, 0.25, 0.95), false);
            }
        }


        let axis = Path::line(Point::new(wx, mid), Point::new(wx + ww, mid));
        frame.stroke(
            &axis,
            Stroke::default()
                .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.30))
                .with_width(1.0),
        );

        // Split lines + flags.
        for (i, &s) in self.zones.splits.iter().enumerate() {
            let x = wx + self.t_to_x(s, ww);
            if x < wx - 10.0 || x > wx + ww + 10.0 { continue; }
            let selected = Some(i) == self.selected_split;
            let color = if selected {
                Color::from_rgba(0.4, 0.85, 1.0, 1.0)
            } else {
                Color::from_rgba(0.55, 0.75, 1.0, 0.85)
            };
            let line = Path::line(Point::new(x, wy), Point::new(x, wy + wh));
            frame.stroke(
                &line,
                Stroke::default().with_color(color).with_width(1.5),
            );
            let flag_w = 8.0;
            let flag_h = 12.0;
            let top_tri = Path::new(|p| {
                p.move_to(Point::new(x - flag_w, wy));
                p.line_to(Point::new(x + flag_w, wy));
                p.line_to(Point::new(x, wy + flag_h));
                p.close();
            });
            frame.fill(&top_tri, color);
            let bot_tri = Path::new(|p| {
                p.move_to(Point::new(x - flag_w, wy + wh));
                p.line_to(Point::new(x + flag_w, wy + wh));
                p.line_to(Point::new(x, wy + wh - flag_h));
                p.close();
            });
            frame.fill(&bot_tri, color);
        }

        // Draggable guide line (yellow, dashed). Drawn symmetrically: above
        // and below center at the same dB-amplitude offset, like ceiling lines.
        if let Some(db) = self.guide_db {
            let y_top = self.db_to_y_top(db, wy, wh);
            let y_bot = wy + wh - (y_top - wy);
            // White: maximum contrast against any waveform scheme color.
            // The dashed pattern + horizontal orientation keeps it
            // distinguishable from the solid white vertical playhead.
            let color = Color::from_rgba(1.0, 1.0, 1.0, 0.85);
            let stroke = Stroke::default().with_color(color).with_width(1.8);
            // Dashed line: short segments across the wave area.
            let dash = 8.0;
            let gap = 6.0;
            let mut x = wx;
            while x < wx + ww {
                let x2 = (x + dash).min(wx + ww);
                let l1 = Path::line(Point::new(x, y_top), Point::new(x2, y_top));
                let l2 = Path::line(Point::new(x, y_bot), Point::new(x2, y_bot));
                frame.stroke(&l1, stroke);
                frame.stroke(&l2, stroke);
                x += dash + gap;
            }
            // Label on the right edge.
            use iced::widget::canvas::Text;
            frame.fill_text(Text {
                content: format!("{:.1} dB", db),
                position: Point::new(wx + ww - 60.0, y_top - 12.0),
                color,
                size: 10.0.into(),
                ..Text::default()
            });
        }
    }

    /// Time ruler at the top.
    fn draw_time_ruler(&self, frame: &mut Frame, wx: f32, ww: f32) {
        use iced::widget::canvas::Text;
        let h = self.top_inset();
        let baseline = h - 2.0;
        // Ruler background
        frame.fill_rectangle(
            Point::new(wx, 0.0),
            Size::new(ww, h),
            Color::from_rgb(0.06, 0.07, 0.09),
        );

        // Tick mark spacing: pick a "nice" interval based on visible duration.
        let vis = self.visible_duration();
        let target_ticks = (ww / 90.0).max(2.0); // ~one tick per 90 px
        let raw_step = vis / target_ticks;
        // Round to a 1/2/5 × 10^n value.
        let step = nice_time_step(raw_step);

        let start = (self.scroll / step).floor() * step;
        let end = self.scroll + vis;
        let mut t = start;
        while t <= end + step * 0.001 {
            if t >= self.scroll {
                let x = wx + ((t - self.scroll) / vis) * ww;
                let line = Path::line(Point::new(x, h - 6.0), Point::new(x, h));
                frame.stroke(
                    &line,
                    Stroke::default()
                        .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.5))
                        .with_width(1.0),
                );
                let label = format_time(t);
                frame.fill_text(Text {
                    content: label,
                    position: Point::new(x + 3.0, baseline - 12.0),
                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.7),
                    size: 10.0.into(),
                    ..Text::default()
                });
            }
            t += step;
        }
    }

    /// dB ruler at the left.
    fn draw_db_ruler(&self, frame: &mut Frame, wy: f32, wh: f32) {
        use iced::widget::canvas::Text;
        let w = self.left_inset();
        // Background
        frame.fill_rectangle(
            Point::new(0.0, wy),
            Size::new(w, wh),
            Color::from_rgb(0.06, 0.07, 0.09),
        );
        // Marks at 0, -3, -6, -12, -24 dB above and below center.
        let mid = wy + wh * 0.5;
        let half_h = wh * 0.5;
        let ticks = [0.0, -3.0, -6.0, -12.0, -24.0];
        for &db in &ticks {
            let amp = 10f32.powf(db / 20.0);
            let dy = amp * half_h;
            for sign in [-1.0, 1.0].iter() {
                let y = mid + sign * dy;
                let line = Path::line(Point::new(w - 6.0, y), Point::new(w, y));
                frame.stroke(
                    &line,
                    Stroke::default()
                        .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.5))
                        .with_width(1.0),
                );
                let label = if db == 0.0 {
                    "0".to_string()
                } else {
                    format!("{:.0}", db)
                };
                frame.fill_text(Text {
                    content: label,
                    position: Point::new(2.0, y - 6.0),
                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.7),
                    size: 9.0.into(),
                    ..Text::default()
                });
            }
        }
    }

    /// Two thin horizontal peak meters under the wave area.
    fn draw_peak_meters(&self, frame: &mut Frame, wx: f32, y: f32, w: f32) {
        let strip_h = 16.0;
        // Background
        frame.fill_rectangle(
            Point::new(wx, y),
            Size::new(w, strip_h),
            Color::from_rgb(0.06, 0.07, 0.09),
        );
        // Each meter takes half the width minus a small gap.
        let gap = 4.0;
        let half = (w - gap) / 2.0;

        let draw_one = |frame: &mut Frame, ox: f32, label: &str, peak: f32| {
            let bar_h = strip_h - 6.0;
            let by = y + 3.0;
            // Background rail
            frame.fill_rectangle(
                Point::new(ox, by),
                Size::new(half, bar_h),
                Color::from_rgb(0.12, 0.13, 0.16),
            );
            // Filled portion (green→yellow→red)
            let amp = peak.clamp(0.0, 1.5);
            let fill_w = (amp / 1.0).min(1.0) * half;
            let color = if amp >= 1.0 {
                Color::from_rgba(1.0, 0.2, 0.2, 0.95)
            } else if amp >= 0.79 {
                Color::from_rgba(1.0, 0.85, 0.2, 0.95)
            } else {
                Color::from_rgba(0.3, 0.85, 0.3, 0.95)
            };
            frame.fill_rectangle(Point::new(ox, by), Size::new(fill_w, bar_h), color);
            // 0 dB tick
            let tick_x = ox + half;
            let tline = Path::line(Point::new(tick_x, by), Point::new(tick_x, by + bar_h));
            frame.stroke(
                &tline,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.5))
                    .with_width(1.0),
            );
            // Label
            use iced::widget::canvas::Text;
            frame.fill_text(Text {
                content: format!("{}: {:>5.1} dB", label, crate::dsp::amp_to_db(peak)),
                position: Point::new(ox + 4.0, by - 1.0),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.85),
                size: 10.0.into(),
                ..Text::default()
            });
        };
        draw_one(frame, wx, "IN ", self.meter_in);
        draw_one(frame, wx + half + gap, "OUT", self.meter_out);
    }

    /// FFT panel below the wave area (mode 1: spectrum line, mode 2: spectrogram).
    fn draw_fft_panel(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        // Panel background
        frame.fill_rectangle(
            Point::new(wx, y),
            Size::new(w, h),
            Color::from_rgb(0.05, 0.06, 0.08),
        );
        match self.fft_mode {
            1 => self.draw_spectrum_line(frame, wx, y, w, h),
            2 => self.draw_spectrogram(frame, wx, y, w, h),
            3 => self.draw_phase(frame, wx, y, w, h),
            _ => {}
        }
    }

    fn draw_spectrum_line(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        if self.sample_rate == 0 {
            return;
        }
        let Some(spec_out) = self.spectrum_line_out else { return; };
        let bins = spec_out.len();
        if bins == 0 { return; }
        let nyq = self.sample_rate as f32 / 2.0;
        let lo_hz = 20.0_f32.max(nyq / bins as f32);
        let hi_hz = nyq;
        let log_lo = lo_hz.log10();
        let log_hi = hi_hz.log10();

        // dB display range. The old -100..0 squashed all harmonic detail into
        // the top sliver; a tighter top + floor spreads the action across the
        // panel. TOP is a few dB above 0 so the loudest peak isn't clipped at
        // the ceiling; FLOOR cuts the deep noise we don't care about.
        const DB_TOP: f32 = 6.0;
        const DB_FLOOR: f32 = -84.0;
        let db_span = DB_TOP - DB_FLOOR;
        let db_to_y = |db: f32| -> f32 {
            let t = ((db - DB_FLOOR) / db_span).clamp(0.0, 1.0);
            y + (1.0 - t) * h
        };
        // x position for a bin index.
        let bin_x = |i: usize| -> f32 {
            let bin_hz = (i as f32 / bins as f32) * nyq;
            let frac = (bin_hz.log10() - log_lo) / (log_hi - log_lo);
            wx + frac * w
        };

        let build = |spec: &[f32]| -> Path {
            Path::new(|p| {
                let mut first = true;
                for (i, &db) in spec.iter().enumerate().skip(1) {
                    let bin_hz = (i as f32 / bins as f32) * nyq;
                    if bin_hz < lo_hz { continue; }
                    let x = bin_x(i);
                    let yy = db_to_y(db);
                    if first { p.move_to(Point::new(x, yy)); first = false; }
                    else { p.line_to(Point::new(x, yy)); }
                }
            })
        };

        // Horizontal dB grid lines + labels every 12 dB (reference for reading
        // harmonic levels).
        use iced::widget::canvas::Text;
        let mut db = 0.0_f32;
        while db >= DB_FLOOR {
            let yy = db_to_y(db);
            let line = Path::line(Point::new(wx, yy), Point::new(wx + w, yy));
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.06))
                    .with_width(1.0),
            );
            frame.fill_text(Text {
                content: format!("{}", db as i32),
                position: Point::new(wx + 2.0, yy + 1.0),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.3),
                size: 8.0.into(),
                ..Text::default()
            });
            db -= 12.0;
        }

        // Frequency reference grid. A 1-2-5 sequence per decade gives a
        // reference roughly every half-octave without clutter. Major lines
        // (labelled, brighter) at the round decade/half values; minor lines
        // (fainter, unlabelled) fill the gaps so position is easy to read
        // without the grid getting busy with text.
        const GRID_MAJOR: [f32; 6] = [50.0, 100.0, 500.0, 1000.0, 5000.0, 10_000.0];
        const GRID_MINOR: [f32; 8] = [
            20.0, 30.0, 200.0, 300.0, 2000.0, 3000.0, 20_000.0, 15_000.0,
        ];
        let draw_grid_line = |frame: &mut Frame, hz: f32, alpha: f32, labelled: bool| {
            if hz < lo_hz || hz > hi_hz {
                return;
            }
            let frac = (hz.log10() - log_lo) / (log_hi - log_lo);
            let x = wx + frac * w;
            let line = Path::line(Point::new(x, y), Point::new(x, y + h));
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, alpha))
                    .with_width(1.0),
            );
            if labelled {
                let label = if hz >= 1000.0 {
                    format!("{}k", hz as u32 / 1000)
                } else {
                    format!("{}", hz as u32)
                };
                frame.fill_text(Text {
                    content: label,
                    position: Point::new(x + 2.0, y + h - 12.0),
                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.5),
                    size: 9.0.into(),
                    ..Text::default()
                });
            }
        };
        for &hz in GRID_MINOR.iter() {
            draw_grid_line(frame, hz, 0.06, false);
        }
        for &hz in GRID_MAJOR.iter() {
            draw_grid_line(frame, hz, 0.14, true);
        }

        let have_in = self.spectrum_line_in.map(|s| s.len() == bins).unwrap_or(false);

        // DIFFERENCE FILL: where output rose above input (clipper ADDED energy
        // — the harmonics it generated), shade amber; where output dropped
        // below input, shade blue. This isolates exactly what the mode did,
        // which is far more legible than eyeballing the gap between two lines.
        if let Some(spec_in) = self.spectrum_line_in {
            if have_in {
                // Build per-bin filled quads between the two curves. Cheap at
                // this bin count and only the divergence is colored.
                for i in 2..bins {
                    let bin_hz = (i as f32 / bins as f32) * nyq;
                    if bin_hz < lo_hz { continue; }
                    let x0 = bin_x(i - 1);
                    let x1 = bin_x(i);
                    let in0 = spec_in[i - 1];
                    let in1 = spec_in[i];
                    let out0 = spec_out[i - 1];
                    let out1 = spec_out[i];
                    let added = (out0 - in0) + (out1 - in1) >= 0.0;
                    let color = if added {
                        Color::from_rgba(0.95, 0.65, 0.15, 0.28) // amber: added
                    } else {
                        Color::from_rgba(0.30, 0.55, 0.95, 0.22) // blue: removed
                    };
                    let quad = Path::new(|p| {
                        p.move_to(Point::new(x0, db_to_y(in0)));
                        p.line_to(Point::new(x1, db_to_y(in1)));
                        p.line_to(Point::new(x1, db_to_y(out1)));
                        p.line_to(Point::new(x0, db_to_y(out0)));
                        p.close();
                    });
                    frame.fill(&quad, color);
                }
            }
        }

        // Input spectrum line (faint "before" reference).
        if have_in {
            if let Some(spec_in) = self.spectrum_line_in {
                frame.stroke(
                    &build(spec_in),
                    Stroke::default()
                        .with_color(Color::from_rgba(0.62, 0.68, 0.80, 0.7))
                        .with_width(1.0),
                );
            }
        }

        // Output spectrum line (bright foreground).
        frame.stroke(
            &build(spec_out),
            Stroke::default()
                .with_color(Color::from_rgba(0.4, 0.95, 0.7, 0.95))
                .with_width(1.4),
        );
    }

    fn draw_spectrogram(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        // Show the OUTPUT spectrogram by default — that's what reflects clipping.
        let Some(sg) = self.spectrogram_out else {
            // Show "computing…" placeholder.
            use iced::widget::canvas::Text;
            frame.fill_text(Text {
                content: "Computing spectrogram…".to_string(),
                position: Point::new(wx + 8.0, y + 8.0),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.5),
                size: 12.0.into(),
                ..Text::default()
            });
            return;
        };
        let n_slices = sg.time_bins();
        if n_slices == 0 {
            return;
        }

        // Map: x = time (visible window), y = log-frequency, color = dB.
        let nyq = sg.sample_rate as f32 / 2.0;
        let bins = sg.freq_bins;
        if bins == 0 {
            return;
        }
        let lo_hz = 20.0_f32.max(nyq / bins as f32);
        let log_lo = lo_hz.log10();
        let log_span = (nyq.log10() - log_lo).max(1e-6);

        let pixels_w = w as usize;
        let pixels_h = h as usize;
        if pixels_w == 0 || pixels_h == 0 {
            return;
        }
        let scroll = self.scroll;
        let vis = self.visible_duration();

        // Precompute the bin span for each pixel ROW once, instead of per
        // column. `bin_lo`/`bin_hi` is the range of FFT bins that row covers;
        // `frac` is the sub-bin position used when the row covers less than
        // one bin (the low end of a log axis, where several rows share a
        // single bin).
        struct RowMap {
            bin_lo: usize,
            bin_hi: usize,
            frac: f32,
            interp: bool,
        }
        let rows: Vec<RowMap> = (0..pixels_h)
            .map(|py| {
                // Row py covers the band between its top and bottom edges.
                let f_top = 1.0 - (py as f32 / pixels_h as f32);
                let f_bot = 1.0 - ((py + 1) as f32 / pixels_h as f32);
                let hz_hi = 10f32.powf(log_lo + f_top * log_span);
                let hz_lo = 10f32.powf(log_lo + f_bot * log_span);
                let b_exact = (hz_lo / nyq) * bins as f32;
                let b_lo = (b_exact.floor().max(0.0) as usize).min(bins - 1);
                let b_hi = ((((hz_hi / nyq) * bins as f32).ceil() as usize).max(b_lo + 1))
                    .min(bins);
                RowMap {
                    bin_lo: b_lo,
                    bin_hi: b_hi,
                    frac: b_exact - b_exact.floor(),
                    // Fewer than ~1.5 bins in this row means we're in the
                    // sparse low end, where nearest-bin sampling produces
                    // the thick horizontal banding. Interpolate instead.
                    interp: (b_hi - b_lo) <= 1,
                }
            })
            .collect();

        // How many time slices land in one pixel column.
        let slices_per_px =
            (vis * sg.sample_rate as f32 / sg.hop as f32) / pixels_w as f32;
        // Cap how many of them we actually read. Taking every slice at 1x on
        // a long file would be tens of millions of reads per repaint; 8
        // spread across the range captures transients without the cost.
        const MAX_SLICE_SAMPLES: usize = 8;

        // When a single slice covers more than one pixel column (zoomed past
        // the hop rate) we blend between neighbouring slices instead of
        // repeating one. Without this, at high zoom each slice paints a
        // solid block — at 700x that's a ~30px wide stripe.
        //
        // This is smoothing, not new information: the analysis window is
        // 4096 samples (~85 ms), so genuinely finer time detail isn't in the
        // data to recover. Interpolating is the honest way to present that
        // — a gradient reads as "the analysis is coarse here" while a hard
        // block misreads as "the audio changed abruptly here".
        let time_interp = slices_per_px < 1.0;

        // Read one slice's value for a pixel row, resolving the frequency
        // axis (max across bins, or blend when the row covers under a bin).
        let sample_slice = |si: usize, rm: &RowMap| -> f32 {
            let row = sg.row(si);
            if rm.interp {
                // Sparse low end: linear blend between adjacent bins so the
                // bass reads as a gradient instead of blocks.
                let a = row[rm.bin_lo] as f32;
                let b = row[(rm.bin_lo + 1).min(bins - 1)] as f32;
                a + (b - a) * rm.frac
            } else {
                let mut best = 0u8;
                for &v in &row[rm.bin_lo..rm.bin_hi] {
                    if v > best {
                        best = v;
                    }
                }
                best as f32
            }
        };

        let half_fft = sg.fft_size as f32 / 2.0;
        for px in 0..pixels_w {
            let t0 = scroll + (px as f32 / pixels_w as f32) * vis;
            let s_center =
                (t0 * sg.sample_rate as f32 - half_fft) / sg.hop as f32;
            if s_center < 0.0 {
                continue;
            }
            let s0 = s_center as usize;
            if s0 >= n_slices {
                continue;
            }
            let s_frac = s_center - s_center.floor();
            let s_next = (s0 + 1).min(n_slices - 1);
            let s1 = (s0 + slices_per_px.ceil().max(1.0) as usize).min(n_slices);
            let n_read = (s1 - s0).max(1);
            let step = (n_read / MAX_SLICE_SAMPLES).max(1);

            let mut run_start = 0usize;
            let mut run_code: Option<u8> = None;

            for py in 0..pixels_h {
                let rm = &rows[py];
                let best = if time_interp {
                    // Zoomed in past the slice rate: blend the two slices
                    // this pixel falls between.
                    let a = sample_slice(s0, rm);
                    let b = sample_slice(s_next, rm);
                    (a + (b - a) * s_frac).clamp(0.0, 255.0) as u8
                } else {
                    // Zoomed out: many slices per pixel. Reduce by MAX — a
                    // point sample throws away everything between the
                    // sampled slice and the next, so a transient falling in
                    // the gap disappears, and which ones disappear changes
                    // as you scroll, which is what makes it shimmer. Max
                    // keeps every event that occurred in the cell.
                    let mut best = 0u8;
                    let mut si = s0;
                    while si < s1 {
                        let v = sample_slice(si, rm) as u8;
                        if v > best {
                            best = v;
                        }
                        si += step;
                    }
                    best
                };

                // Batch vertical runs of the same value into one rect. Large
                // flat regions (silence, the noise floor) collapse to a
                // handful of draws instead of one per pixel.
                match run_code {
                    Some(c) if c == best => {}
                    Some(c) => {
                        let rgba = sg.code_to_rgba(c);
                        frame.fill_rectangle(
                            Point::new(wx + px as f32, y + run_start as f32),
                            Size::new(1.0, (py - run_start) as f32),
                            Color::from_rgba(rgba[0], rgba[1], rgba[2], rgba[3]),
                        );
                        run_start = py;
                        run_code = Some(best);
                    }
                    None => {
                        run_start = py;
                        run_code = Some(best);
                    }
                }
            }
            if let Some(c) = run_code {
                let rgba = sg.code_to_rgba(c);
                frame.fill_rectangle(
                    Point::new(wx + px as f32, y + run_start as f32),
                    Size::new(1.0, (pixels_h - run_start) as f32),
                    Color::from_rgba(rgba[0], rgba[1], rgba[2], rgba[3]),
                );
            }
        }
    }

    /// Combined PHASE view, laid out side by side:
    ///
    /// ```text
    ///   ┌──────────────────────────────┬────────────┐
    ///   │  correlation vs frequency    │ goniometer │
    ///   └──────────────────────────────┴────────────┘
    /// ```
    ///
    /// The goniometer wants to be square, so it takes a square-ish column on
    /// the right and the frequency curve gets everything left of it — which
    /// is what that curve wants, since it shares the spectrum line's
    /// log-frequency x-mapping. Flipping between Spectrum and Phase keeps
    /// frequencies in the same screen positions, so a dip you spot in one can
    /// be looked up at the same x in the other.
    fn draw_phase(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        // Goniometer column: square where there's room, never more than ~40%
        // of the width (the frequency curve is the more informative panel).
        let (gx, gy, gw, gh) = Self::gonio_rect((wx, y, w, h));
        let band_w = (gx - wx).max(40.0);
        self.draw_band_correlation(frame, wx, y, band_w, h);
        self.draw_goniometer(frame, gx, gy, gw, gh);
        self.draw_mono_badge(frame, gx, gy, gw);
    }

    /// The always-on overview strip: a whole-file map that doubles as the
    /// navigation bar.
    ///
    /// Its CONTENT follows the analysis mode — correlation in Phase view,
    /// a folded waveform everywhere else — but its position, geometry and
    /// interaction never change. That's the point: whatever you're looking
    /// at, the bar under the waveform is always "the whole track, click to
    /// go there".
    fn draw_overview(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        frame.fill_rectangle(
            Point::new(wx, y),
            Size::new(w, h),
            Color::from_rgb(0.05, 0.06, 0.08),
        );
        if self.fft_mode == 3 && self.corr_timeline.is_some() {
            self.draw_corr_timeline(frame, wx, y, w, h);
        } else {
            self.draw_wave_overview(frame, wx, y, w, h);
        }
        // Hairline under the strip, separating it from the peak meters.
        frame.fill_rectangle(
            Point::new(wx, y + h - 1.0),
            Size::new(w, 1.0),
            Color::from_rgba(1.0, 1.0, 1.0, 0.07),
        );
    }

    /// Whole-file waveform map, FOLDED to one side.
    ///
    /// A symmetric mini-waveform in a 22 px strip gives each side 11 px of
    /// dynamic range; folding the peak magnitude upward uses the full 22.
    /// Nothing is lost — audio is near-symmetric at this scale, and the
    /// overview's job is structure (where the drop is, where it breaks
    /// down), not waveshape.
    fn draw_wave_overview(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        let env = if !self.envelope_out.is_empty() {
            self.envelope_out
        } else {
            self.envelope_in
        };
        if env.is_empty() {
            return;
        }
        let n_px = w.floor().max(1.0) as usize;
        let base = y + h - 1.0;
        let usable = h - 2.0;

        for i in 0..n_px {
            let e0 = i * env.len() / n_px;
            let e1 = (((i + 1) * env.len()) / n_px).max(e0 + 1).min(env.len());
            let mut peak = 0.0f32;
            for &(lo, hi) in &env[e0..e1] {
                peak = peak.max(lo.abs()).max(hi.abs());
            }
            if peak <= 1e-5 {
                continue;
            }
            // Mild curve rather than raw amplitude. Linear amplitude in 22 px
            // collapses everything below about -20 dBFS into the bottom two
            // pixels, which erases exactly the quiet sections you'd navigate
            // TO. ^0.6 lifts them without flattening the loud parts into a
            // solid block the way a full dB mapping would.
            let mag = peak.clamp(0.0, 1.0).powf(0.6) * usable;
            frame.fill_rectangle(
                Point::new(wx + i as f32, base - mag),
                Size::new(1.0, mag.max(1.0)),
                Color::from_rgba(0.55, 0.60, 0.70, 0.75),
            );
        }
    }

    /// Correlation over time: a short filled strip sharing the waveform's
    /// time axis (so it follows zoom and scroll).
    ///
    /// Filled area rather than a gradient bar. On real material almost
    /// everything sits between +0.6 and +1.0, and a colour ramp across that
    /// range resolves to maybe four distinguishable steps — you'd get a
    /// uniformly green bar. Height is the far better channel: a notch in a
    /// silhouette is legible at a few pixels, and colour then reinforces it
    /// rather than carrying the whole message alone.
    fn draw_corr_timeline(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        use iced::widget::canvas::Text;
        let Some(tl) = self.corr_timeline else { return };
        if tl.points.is_empty() || tl.hop_secs <= 0.0 {
            return;
        }

        // Non-linear y. A linear −1..+1 axis spends most of its height on a
        // range nothing ever occupies, flattening the variation that matters
        // into a few pixels at the top. Giving the healthy +0.5..+1.0 band
        // 60% of the strip makes normal fluctuation actually visible. The
        // frequency curve below stays linear, so precision is still
        // available — this axis is for navigation, not measurement.
        let corr_to_h = |c: f32| -> f32 {
            let c = c.clamp(-1.0, 1.0);
            let frac = if c >= 0.5 {
                0.40 + (c - 0.5) / 0.5 * 0.60
            } else {
                (c + 1.0) / 1.5 * 0.40
            };
            frac * h
        };

        let n_px = w.floor().max(1.0) as usize;
        // ALWAYS the whole file, independent of zoom and scroll. The strip's
        // job is "where in the track should I look" — if it followed the
        // zoom it would stop answering that the instant you zoomed in to
        // look at something, which is exactly when you still want to know
        // where the other problems are. The viewport box (drawn in the
        // overlay layer) is what ties it back to the waveform above.
        //
        // Being zoom-independent is also what keeps this correct in the
        // meter cache, which isn't invalidated on zoom.
        let span = self.duration.max(1e-6);
        for i in 0..n_px {
            let t0 = (i as f32 / n_px as f32) * span;
            let t1 = ((i + 1) as f32 / n_px as f32) * span;
            // Each pixel may span many windows when zoomed out. Take the
            // WORST correlation in range rather than the average: a two-
            // second phase problem in a minute-wide pixel column has to
            // survive the reduction, or the strip fails at exactly the job
            // it exists for.
            let i0 = (t0 / tl.hop_secs).max(0.0) as usize;
            let i1 = ((t1 / tl.hop_secs).max(0.0) as usize + 1).min(tl.points.len());
            if i0 >= tl.points.len() {
                continue;
            }
            let mut worst = f32::MAX;
            let mut wt = 0.0f32;
            for p in i0..i1.max(i0 + 1).min(tl.points.len()) {
                let (c, pw) = tl.points[p];
                if pw <= 0.05 {
                    continue;
                }
                if c < worst {
                    worst = c;
                }
                if pw > wt {
                    wt = pw;
                }
            }
            if wt <= 0.05 || worst == f32::MAX {
                continue;
            }
            let bar_h = corr_to_h(worst).max(1.0);
            let (r, g, b) = corr_color_ramp(worst);
            frame.fill_rectangle(
                Point::new(wx + i as f32, y + h - bar_h),
                Size::new(1.0, bar_h),
                Color::from_rgba(r, g, b, (0.30 + 0.45 * (1.0 - worst.clamp(0.0, 1.0))) * wt),
            );
        }

        // Reference line at the +0.5 mark — the boundary of the expanded
        // healthy band, so the axis distortion is visible rather than
        // silently misleading.
        let ry = y + h - corr_to_h(0.5);
        frame.stroke(
            &Path::line(Point::new(wx, ry), Point::new(wx + w, ry)),
            Stroke::default()
                .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.12))
                .with_width(1.0),
        );

        frame.fill_text(Text {
            content: "correlation".to_string(),
            position: Point::new(wx + 4.0, y + 1.0),
            color: Color::from_rgba(1.0, 1.0, 1.0, 0.35),
            size: 9.5.into(),
            ..Text::default()
        });
    }

    /// Viewport box + playhead tick over the correlation strip.
    ///
    /// Drawn in the OVERLAY layer, not the meter layer: these two move
    /// constantly (every scroll, zoom and playback frame) while the strip
    /// underneath changes only when the audio does. Splitting them along
    /// that line is what lets the expensive goniometer stay cached while
    /// this stays live.
    fn draw_overview_overlay(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        if self.duration <= 0.0 {
            return;
        }
        let span = self.duration.max(1e-6);
        let to_x = |t: f32| wx + (t / span).clamp(0.0, 1.0) * w;

        // Viewport box: which slice of the file the waveform above is
        // showing. At 1x this is the full width and reads as a plain border;
        // zoomed in it's the link between the two time axes.
        if self.zoom > 1.001 {
            let vx0 = to_x(self.scroll);
            let vx1 = to_x(self.scroll + self.visible_duration());
            // Dim the parts NOT in view rather than outlining the part that
            // is — the eye reads "the bright bit is where I am" faster than
            // it reads a thin rectangle.
            let shade = Color::from_rgba(0.0, 0.0, 0.0, 0.45);
            if vx0 > wx {
                frame.fill_rectangle(Point::new(wx, y), Size::new(vx0 - wx, h), shade);
            }
            if vx1 < wx + w {
                frame.fill_rectangle(
                    Point::new(vx1, y),
                    Size::new(wx + w - vx1, h),
                    shade,
                );
            }
            let box_path = Path::rectangle(
                Point::new(vx0, y + 0.5),
                Size::new((vx1 - vx0).max(2.0), h - 1.0),
            );
            frame.stroke(
                &box_path,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.55))
                    .with_width(1.0),
            );
        }

        // Playhead tick. `None` when nothing is loaded / no position yet.
        if let Some(t) = self.playhead_secs {
            let px = to_x(t);
            frame.fill_rectangle(
                Point::new(px - 0.5, y),
                Size::new(1.5, h),
                Color::from_rgba(1.0, 1.0, 1.0, 0.9),
            );
        }
    }

    /// Mono-fold toggle, drawn as a small badge in the goniometer's
    /// top-right corner.
    ///
    /// It lives here rather than in the toolbar because it's only ever
    /// useful next to a phase display — the curve predicts what will cancel,
    /// this lets you hear whether it actually matters, and having the two a
    /// few pixels apart is what makes that a fast A/B. The keyboard shortcut
    /// (M) still works from anywhere.
    fn draw_mono_badge(&self, frame: &mut Frame, gx: f32, gy: f32, gw: f32) {
        use iced::widget::canvas::Text;
        let (bx, by, bw, bh) = mono_badge_rect(gx, gy, gw);

        let (bg, fg) = if self.mono_fold {
            (
                Color::from_rgba(0.90, 0.62, 0.22, 0.85),
                Color::from_rgb(0.08, 0.08, 0.09),
            )
        } else {
            (
                Color::from_rgba(1.0, 1.0, 1.0, 0.07),
                Color::from_rgba(1.0, 1.0, 1.0, 0.45),
            )
        };

        let pill = Path::rounded_rectangle(
            Point::new(bx, by),
            Size::new(bw, bh),
            3.0.into(),
        );
        frame.fill(&pill, bg);
        if !self.mono_fold {
            frame.stroke(
                &pill,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.15))
                    .with_width(1.0),
            );
        }

        frame.fill_text(Text {
            content: "MONO".to_string(),
            position: Point::new(bx + 4.0, by + 3.5),
            color: fg,
            size: 9.5.into(),
            ..Text::default()
        });
    }

    /// Correlation as a function of frequency.
    ///
    /// y axis is −1..+1 with zero across the middle; the curve is filled
    /// back to that zero line — green above, red below — so negative
    /// (mono-cancelling) regions read as red pockets hanging under the line
    /// rather than as a number you have to interpret. The broadband
    /// correlation is drawn as a dashed reference line across the whole
    /// panel, so the average and the per-band deviation from it are visible
    /// in one glance.
    fn draw_band_correlation(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        use iced::widget::canvas::Text;

        let px0 = wx + CORR_PAD_L;
        let plot_w = (w - CORR_PAD_L - CORR_PAD_R).max(1.0);
        let py0 = y + CORR_PAD_T;
        let plot_h = (h - CORR_PAD_T - CORR_PAD_B).max(1.0);

        let corr_to_y = |c: f32| -> f32 {
            let t = ((c + 1.0) / 2.0).clamp(0.0, 1.0);
            py0 + (1.0 - t) * plot_h
        };
        let zero_y = corr_to_y(0.0);

        // ---- Resolve the curve FIRST -----------------------------------
        // The zone tints below are keyed off how far down the curve actually
        // goes, so the data has to be in hand before anything is drawn.
        //
        // Resolved per PIXEL rather than per bin. At high frequencies many
        // bins land on one pixel (energy-weighted average here is the honest
        // reduction); at low frequencies one bin spans several pixels and
        // simply repeats. Cost is bounded by panel width, not FFT size.
        let n_px = plot_w.floor().max(1.0) as usize;
        let mut curve: Vec<Option<(f32, f32)>> = vec![None; n_px];
        let mut have_data = false;
        let mut min_c = 1.0f32;

        let (nyq, log_lo, log_span) = if self.sample_rate > 0
            && !self.probe_band_corr.is_empty()
        {
            let bins = self.probe_band_corr.len();
            let nyq = self.sample_rate as f32 / 2.0;
            let lo_hz = 20.0_f32.max(nyq / bins as f32);
            let log_lo = lo_hz.log10();
            let log_span = (nyq.log10() - log_lo).max(1e-6);

            for (i, slot) in curve.iter_mut().enumerate() {
                let f_lo = 10f32.powf(log_lo + (i as f32 / n_px as f32) * log_span);
                let f_hi = 10f32.powf(log_lo + ((i + 1) as f32 / n_px as f32) * log_span);
                let b_lo = ((f_lo / nyq) * bins as f32).floor().max(1.0) as usize;
                if b_lo >= bins {
                    continue;
                }
                let b_hi = (((f_hi / nyq) * bins as f32).ceil() as usize)
                    .max(b_lo + 1)
                    .min(bins);
                let mut acc = 0.0f32;
                let mut wsum = 0.0f32;
                for b in b_lo..b_hi {
                    let (c, wt) = self.probe_band_corr[b];
                    acc += c * wt;
                    wsum += wt;
                }
                if wsum > 1e-6 {
                    let c = acc / wsum;
                    let wt = (wsum / (b_hi - b_lo) as f32).clamp(0.0, 1.0);
                    *slot = Some((c, wt));
                    // Only bands with real signal count toward the worst
                    // case — a faded-out dead band shouldn't summon a red
                    // warning zone.
                    if wt > 0.15 {
                        have_data = true;
                        if c < min_c {
                            min_c = c;
                        }
                    }
                }
            }
            (nyq, log_lo, log_span)
        } else {
            (0.0, 0.0, 1.0)
        };
        let _ = nyq;

        // ---- Background -------------------------------------------------
        // Deliberately NO horizontal zone-tint bands. Full-width stripes
        // paint the whole panel regardless of where the data is, so they
        // show through every gap in the fill — a narrow notch ends up
        // looking like a coloured column punching upward, which is
        // background leaking through, not signal. Colour belongs ON the
        // curve (see `corr_color` below), where it only appears if there's
        // something to report. The panel stays quiet on clean material.

        // ---- Grid ------------------------------------------------------
        for (val, label) in [(1.0, "+1"), (0.5, ""), (0.0, "0"), (-0.5, ""), (-1.0, "-1")] {
            let yy = corr_to_y(val);
            let alpha = if label.is_empty() { 0.05 } else { 0.18 };
            let line = Path::line(Point::new(px0, yy), Point::new(px0 + plot_w, yy));
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, alpha))
                    .with_width(1.0),
            );
            if !label.is_empty() {
                frame.fill_text(Text {
                    content: label.to_string(),
                    position: Point::new(wx + 4.0, yy - 6.0),
                    color: Color::from_rgba(1.0, 1.0, 1.0, 0.45),
                    size: 10.0.into(),
                    ..Text::default()
                });
            }
        }

        if !have_data {
            frame.fill_text(Text {
                content: if self.probe_band_corr.is_empty() {
                    "mono / no signal at probe".to_string()
                } else {
                    "no signal at probe".to_string()
                },
                position: Point::new(px0 + 4.0, y + 2.0),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.45),
                size: 12.0.into(),
                ..Text::default()
            });
            return;
        }

        // Vertical frequency grid, matching the spectrum line's decades.
        for (hz, label) in [(100.0f32, "100"), (1000.0, "1k"), (10000.0, "10k")] {
            let frac = (hz.log10() - log_lo) / log_span;
            if !(0.0..=1.0).contains(&frac) {
                continue;
            }
            let fx = px0 + frac * plot_w;
            let line = Path::line(Point::new(fx, py0), Point::new(fx, py0 + plot_h));
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.10))
                    .with_width(1.0),
            );
            frame.fill_text(Text {
                content: label.to_string(),
                position: Point::new(fx + 2.0, py0 + plot_h + 1.0),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.40),
                size: 10.0.into(),
                ..Text::default()
            });
        }

        // ---- Fill between the curve and the zero line ------------------
        // The fill carries the colour: green while healthy, warming through
        // amber as it narrows, red once it goes negative. Because it's keyed
        // to each band's own value, colour shows up exactly where the
        // problem is and nowhere else — a clean track draws an entirely
        // green ribbon and the rest of the panel stays dark.
        //
        // Alpha follows the weight so dead bands fade out instead of
        // drawing noise.
        for (i, slot) in curve.iter().enumerate() {
            let Some((c, wt)) = *slot else { continue };
            if wt <= 0.01 {
                continue;
            }
            let cy = corr_to_y(c);
            let (top, bot) = if cy < zero_y { (cy, zero_y) } else { (zero_y, cy) };
            let (r, g, b) = corr_color_ramp(c);
            // Negative bands get a touch more presence — that's the state
            // worth noticing.
            let alpha = if c < 0.0 { 0.42 } else { 0.26 } * wt;
            frame.fill_rectangle(
                Point::new(px0 + i as f32, top),
                Size::new(1.0, (bot - top).max(0.5)),
                Color::from_rgba(r, g, b, alpha),
            );
        }

        // ---- The curve itself ------------------------------------------
        // Broken into segments so gaps (dead bands) don't get bridged by a
        // straight line that would imply data we don't have.
        let mut seg: Vec<Point> = Vec::new();
        let flush = |frame: &mut Frame, seg: &mut Vec<Point>| {
            if seg.len() > 1 {
                let p = Path::new(|b| {
                    b.move_to(seg[0]);
                    for pt in seg.iter().skip(1) {
                        b.line_to(*pt);
                    }
                });
                frame.stroke(
                    &p,
                    Stroke::default()
                        .with_color(Color::from_rgba(0.55, 0.90, 1.0, 0.85))
                        .with_width(1.4),
                );
            }
            seg.clear();
        };
        for (i, slot) in curve.iter().enumerate() {
            match *slot {
                Some((c, wt)) if wt > 0.15 => {
                    seg.push(Point::new(px0 + i as f32, corr_to_y(c)));
                }
                _ => flush(frame, &mut seg),
            }
        }
        flush(frame, &mut seg);

        // ---- Selected band ----------------------------------------------
        // Dim everything OUTSIDE the selection rather than tinting what's
        // inside: the selected region keeps its true colours (which are the
        // measurement), and the eye still reads the bright part as "this is
        // what I picked".
        if let Some((f_lo, f_hi)) = self.band_sel {
            let to_x = |hz: f32| -> f32 {
                px0 + ((hz.log10() - log_lo) / log_span).clamp(0.0, 1.0) * plot_w
            };
            let x0 = to_x(f_lo);
            let x1 = to_x(f_hi).max(x0 + 1.0);
            let shade = Color::from_rgba(0.0, 0.0, 0.0, 0.55);
            if x0 > px0 {
                frame.fill_rectangle(
                    Point::new(px0, py0),
                    Size::new(x0 - px0, plot_h),
                    shade,
                );
            }
            if x1 < px0 + plot_w {
                frame.fill_rectangle(
                    Point::new(x1, py0),
                    Size::new(px0 + plot_w - x1, plot_h),
                    shade,
                );
            }
            for ex in [x0, x1] {
                frame.stroke(
                    &Path::line(Point::new(ex, py0), Point::new(ex, py0 + plot_h)),
                    Stroke::default()
                        .with_color(Color::from_rgba(0.55, 0.90, 1.0, 0.75))
                        .with_width(1.0),
                );
            }
        }

        // ---- Broadband reference + readout ------------------------------
        // Seeing the average against the curve is the whole point — a +0.76
        // average with a red pocket at 200 Hz is a different situation from
        // a flat +0.76.
        if let Some(corr) = self.probe_correlation {
            // Nudge off the top edge so a +1.00 reading is still a visible
            // line rather than disappearing into the panel border.
            let cy = corr_to_y(corr).max(py0 + 1.0);
            let dash = 6.0;
            let mut x = px0;
            while x < px0 + plot_w {
                let x2 = (x + dash).min(px0 + plot_w);
                let l = Path::line(Point::new(x, cy), Point::new(x2, cy));
                frame.stroke(
                    &l,
                    Stroke::default()
                        .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.35))
                        .with_width(1.0),
                );
                x += dash * 2.0;
            }
            frame.fill_text(Text {
                content: format!("correlation {:+.2}", corr),
                position: Point::new(px0, y + 2.0),
                color: Color::from_rgba(1.0, 1.0, 1.0, 0.85),
                size: 12.0.into(),
                ..Text::default()
            });
        }

        // Worst-band callout. The broadband number can read +1.00 while a
        // band sags to +0.55; naming the worst value makes that legible
        // without having to eyeball the dip against the grid.
        if min_c < 0.85 {
            let col = if min_c < 0.0 {
                Color::from_rgba(0.95, 0.45, 0.45, 0.95)
            } else if min_c < 0.4 {
                Color::from_rgba(0.95, 0.78, 0.35, 0.90)
            } else {
                Color::from_rgba(1.0, 1.0, 1.0, 0.55)
            };
            frame.fill_text(Text {
                content: format!("min {:+.2}", min_c),
                position: Point::new(px0 + 120.0, y + 2.0),
                color: col,
                size: 12.0.into(),
                ..Text::default()
            });
        }

        frame.fill_text(Text {
            content: "correlation / frequency".to_string(),
            position: Point::new(px0 + plot_w - 132.0, y + 2.0),
            color: Color::from_rgba(1.0, 1.0, 1.0, 0.45),
            size: 11.0.into(),
            ..Text::default()
        });
    }

    /// Goniometer: plot the probe window's L/R as a rotated X/Y figure. The
    /// axes are rotated 45° so mono (L==R) draws as a VERTICAL line and a
    /// phase-inverted signal (L==−R) draws HORIZONTAL — the standard studio
    /// vectorscope orientation.
    fn draw_goniometer(&self, frame: &mut Frame, wx: f32, y: f32, w: f32, h: f32) {
        use iced::widget::canvas::Text;
        let cx = wx + w * 0.5;
        let cy = y + h * 0.5;
        let radius = (h * 0.5 - 14.0).min(w * 0.5 - 14.0).max(8.0);

        // Reference circle + crosshair.
        let circle = Path::circle(Point::new(cx, cy), radius);
        frame.stroke(
            &circle,
            Stroke::default()
                .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.12))
                .with_width(1.0),
        );
        for (dx, dy, lbl) in [
            (0.0, -1.0, "M"),  // vertical = mono
            (1.0, 0.0, "+S"),  // right = side
            (-1.0, 0.0, "-S"), // left
        ] {
            let line = Path::line(
                Point::new(cx, cy),
                Point::new(cx + dx * radius, cy + dy * radius),
            );
            frame.stroke(
                &line,
                Stroke::default()
                    .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.08))
                    .with_width(1.0),
            );
            let _ = lbl;
        }

        // Plot points. Rotate (L,R) by −45°: x = (L−R)/√2 (side), y = (L+R)/√2
        // (mid). Screen y is inverted (down = positive), so mid points up.
        let s = std::f32::consts::FRAC_1_SQRT_2;
        for &(l, r) in self.probe_gonio.iter() {
            let side = (l - r) * s;
            let mid = (l + r) * s;
            let px = cx + side * radius;
            let py = cy - mid * radius;
            let dot = Path::circle(Point::new(px, py), 1.2);
            frame.fill(&dot, Color::from_rgba(0.45, 0.85, 0.95, 0.5));
        }

        // Label. With a band selected, say so — an unlabelled scatter that
        // silently means "only 300–800 Hz" would be worse than no feature.
        let (label, col) = match (self.band_sel, self.band_sel_corr) {
            (Some((lo, hi)), corr) => {
                let fmt = |f: f32| {
                    if f >= 1000.0 {
                        format!("{:.1}k", f / 1000.0)
                    } else {
                        format!("{f:.0}")
                    }
                };
                let mut s = format!("{}–{} Hz", fmt(lo), fmt(hi));
                if let Some(c) = corr {
                    s.push_str(&format!("   {c:+.2}"));
                }
                (s, Color::from_rgba(0.55, 0.90, 1.0, 0.9))
            }
            _ => (
                "goniometer".to_string(),
                Color::from_rgba(1.0, 1.0, 1.0, 0.45),
            ),
        };
        frame.fill_text(Text {
            content: label,
            position: Point::new(wx + 6.0, y + 2.0),
            color: col,
            size: 11.0.into(),
            ..Text::default()
        });
    }
}

impl<'a> canvas::Program<CanvasEvent> for Waveform<'a> {
    type State = DragState;

    fn update(
        &self,
        state: &mut DragState,
        event: canvas::Event,
        bounds: Rectangle,
        cursor: mouse::Cursor,
    ) -> (canvas::event::Status, Option<CanvasEvent>) {
        let Some(pos) = cursor.position_in(bounds) else {
            return (canvas::event::Status::Ignored, None);
        };
        let (wx, wy, ww, wh) = self.wave_area(bounds.width, bounds.height);
        // Position relative to the wave area:
        let rx = pos.x - wx;
        let ry = pos.y - wy;
        let in_wave = rx >= 0.0 && ry >= 0.0 && rx <= ww && ry <= wh;
        // The spectrogram shares the waveform's time axis and x range, so
        // the same gestures mean the same thing there. Reaching for them and
        // having nothing happen is the surprising behaviour, not the other
        // way round. Kept separate from `in_wave` because waveform-specific
        // interactions (splits, trim handles, the dB guide) are about the
        // AMPLITUDE axis, which the spectrogram doesn't have.
        let in_sgram = self
            .spectrogram_rect(bounds.width, bounds.height)
            .is_some_and(|(sx, sy, sw, sh)| {
                pos.x >= sx && pos.x <= sx + sw && pos.y >= sy && pos.y <= sy + sh
            });
        let in_timeline = in_wave || in_sgram;
        let t = if in_timeline {
            self.x_to_t(rx, ww)
        } else {
            self.scroll
        };
        let flag_h = 14.0;
        let flag_w = 9.0;
        let in_flag_band = in_wave && (ry < flag_h || ry > wh - flag_h);

        match event {
            canvas::Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                if !in_timeline {
                    return (canvas::event::Status::Ignored, None);
                }
                let (dx, dy) = match delta {
                    mouse::ScrollDelta::Lines { x, y } => (x, y),
                    mouse::ScrollDelta::Pixels { x, y } => (x / 30.0, y / 30.0),
                };
                // Horizontal scroll (two-finger left/right on a trackpad,
                // or shift+wheel) → pan. Use the larger axis to decide
                // intent so a small spurious cross-axis component doesn't
                // both zoom and pan in the same gesture.
                if dx.abs() > dy.abs() && dx.abs() > 0.01 {
                    // Pan by a fraction of the visible window per "line" of
                    // scroll — feels close to the system scroll speed.
                    let pan_secs = -dx * 0.05 * self.visible_duration();
                    return (
                        canvas::event::Status::Captured,
                        Some(CanvasEvent::Pan(pan_secs)),
                    );
                }
                let scroll_amt = dy;
                let factor = if scroll_amt > 0.0 {
                    1.25_f32.powf(scroll_amt.abs())
                } else if scroll_amt < 0.0 {
                    1.0 / 1.25_f32.powf(scroll_amt.abs())
                } else {
                    return (canvas::event::Status::Ignored, None);
                };
                (
                    canvas::event::Status::Captured,
                    Some(CanvasEvent::Zoom(factor, t)),
                )
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Middle)) => {
                if in_timeline {
                    state.panning_from = Some((rx, self.scroll));
                }
                (canvas::event::Status::Captured, None)
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Middle)) => {
                state.panning_from = None;
                (canvas::event::Status::Captured, None)
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Right)) => {
                if !in_wave {
                    return (canvas::event::Status::Ignored, None);
                }
                let zi = self.zones.zone_at(t);
                // Hit-test against splits: same radius as left-click drag.
                let hit_radius = if in_flag_band { 10.0 } else { 5.0 };
                let split_hit = self
                    .zones
                    .splits
                    .iter()
                    .enumerate()
                    .find(|(_, &s)| (rx - self.t_to_x(s, ww)).abs() < hit_radius)
                    .map(|(i, _)| i);
                (
                    canvas::event::Status::Captured,
                    Some(CanvasEvent::RightClickZone(zi, split_hit, pos.x, pos.y)),
                )
            }
            canvas::Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                // The mono badge sits in the FFT panel, BELOW the wave area,
                // so it has to be tested before the `in_wave` gate rejects
                // everything down there.
                // Correlation overview strip: click to navigate. Tested
                // before the `in_wave` gate, since the strip lives below the
                // wave area.
                if let Some((sx, sy, sw, sh)) =
                    self.overview_rect(bounds.width, bounds.height)
                {
                    if pos.x >= sx
                        && pos.x <= sx + sw
                        && pos.y >= sy
                        && pos.y <= sy + sh
                    {
                        state.dragging_overview = true;
                        let frac = ((pos.x - sx) / sw.max(1.0)).clamp(0.0, 1.0);
                        return (
                            canvas::event::Status::Captured,
                            Some(CanvasEvent::NavigateTo(frac * self.duration)),
                        );
                    }
                }
                // Correlation curve: drag horizontally to select a
                // frequency band for the goniometer.
                if let Some((cx, cy, cw, chh)) =
                    self.corr_plot_rect(bounds.width, bounds.height)
                {
                    if pos.x >= cx && pos.x <= cx + cw && pos.y >= cy && pos.y <= cy + chh {
                        state.band_drag_from = Some(((pos.x - cx) / cw.max(1.0)).clamp(0.0, 1.0));
                        // Don't emit yet: a press that turns out to be a
                        // click (no drag) means "clear", and we can't tell
                        // which it is until the button comes back up.
                        return (canvas::event::Status::Captured, None);
                    }
                }
                // Spectrogram: click to seek, drag to scrub — same as the
                // waveform, because it's the same time axis.
                if in_sgram {
                    state.dragging_sgram = true;
                    return (
                        canvas::event::Status::Captured,
                        Some(CanvasEvent::Seek(t.clamp(0.0, self.duration))),
                    );
                }
                if self.mono_badge_hit(bounds.width, bounds.height, pos.x, pos.y) {
                    return (
                        canvas::event::Status::Captured,
                        Some(CanvasEvent::ToggleMonoFold),
                    );
                }
                if !in_wave {
                    return (canvas::event::Status::Ignored, None);
                }
                // 0) Trim handles take priority — they sit at the edges.
                // The whole vertical line is grabbable, and the inward-
                // pointing flag extends the grab zone toward the audio. This
                // matters most when a handle is parked at 0:00 or the file
                // end: the outward half of a symmetric hit zone would be off
                // the canvas, so we bias the zone inward to match the flag.
                let line_slop = 6.0;
                let flag_reach = flag_w + 6.0;
                if let Some(ts) = self.trim_start {
                    let hx = self.t_to_x(ts, ww);
                    // Start flag points RIGHT: extend the grab zone rightward,
                    // especially in the flag band.
                    let right = if in_flag_band { flag_reach } else { line_slop };
                    if rx >= hx - line_slop && rx <= hx + right {
                        state.dragging_trim = Some(TrimHandle::Start);
                        return (canvas::event::Status::Captured, None);
                    }
                }
                if let Some(te) = self.trim_end {
                    let hx = self.t_to_x(te, ww);
                    // End flag points LEFT: extend the grab zone leftward.
                    let left = if in_flag_band { flag_reach } else { line_slop };
                    if rx >= hx - left && rx <= hx + line_slop {
                        state.dragging_trim = Some(TrimHandle::End);
                        return (canvas::event::Status::Captured, None);
                    }
                }
                // 1) Guide line: hit-test ±5 px around the guide's y.
                if let Some(db) = self.guide_db {
                    // Guide is drawn symmetrically — top *and* bottom band.
                    // Either side can be grabbed; we drag whichever the user
                    // clicked.
                    let y_top = self.db_to_y_top(db, wy, wh);
                    let y_bot = wy + wh - (y_top - wy);
                    if (pos.y - y_top).abs() < 5.0 || (pos.y - y_bot).abs() < 5.0 {
                        state.dragging_guide = true;
                        return (canvas::event::Status::Captured, None);
                    }
                }
                // 2) Split line.
                let hit_radius = if in_flag_band { 10.0 } else { 5.0 };
                for (i, &s) in self.zones.splits.iter().enumerate() {
                    let x = self.t_to_x(s, ww);
                    if (rx - x).abs() < hit_radius {
                        state.dragging = Some(i);
                        return (
                            canvas::event::Status::Captured,
                            Some(CanvasEvent::SelectSplit(i)),
                        );
                    }
                }
                // 2.5) Click-repair mode: if active and we're zoomed in far
                // enough that samples are individually resolvable, a click
                // repairs the click at that sample instead of seeking.
                if self.click_repair_mode {
                    let vis = self.visible_duration();
                    let samples_visible = vis * self.sample_rate as f32;
                    // Require enough zoom that each sample is at least ~3 px
                    // wide, so the user is actually pointing at a sample.
                    let px_per_sample = ww / samples_visible.max(1.0);
                    if px_per_sample >= 3.0 {
                        let frame = (t * self.sample_rate as f32).round() as usize;
                        return (
                            canvas::event::Status::Captured,
                            Some(CanvasEvent::RepairClick(frame)),
                        );
                    }
                }
                // 3) Empty space → seek.
                (
                    canvas::event::Status::Captured,
                    Some(CanvasEvent::Seek(t)),
                )
            }
            canvas::Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                if state.dragging_overview {
                    if let Some((sx, _sy, sw, _sh)) =
                        self.overview_rect(bounds.width, bounds.height)
                    {
                        let frac = ((pos.x - sx) / sw.max(1.0)).clamp(0.0, 1.0);
                        return (
                            canvas::event::Status::Captured,
                            Some(CanvasEvent::NavigateTo(frac * self.duration)),
                        );
                    }
                }
                if let Some(handle) = state.dragging_trim {
                    // Clamp to the visible time domain; cross-constraint and
                    // edge clamp to [0, duration] are enforced app-side.
                    let tt = t.clamp(0.0, self.duration);
                    let ev = match handle {
                        TrimHandle::Start => CanvasEvent::MoveTrimStart(tt),
                        TrimHandle::End => CanvasEvent::MoveTrimEnd(tt),
                    };
                    return (canvas::event::Status::Captured, Some(ev));
                }
                if state.dragging_guide {
                    let db = self.y_to_db(ry, wh);
                    return (
                        canvas::event::Status::Captured,
                        Some(CanvasEvent::DragGuide(db)),
                    );
                }
                if let Some((start_rx, start_scroll)) = state.panning_from {
                    let dx_secs = (rx - start_rx) / ww * self.visible_duration();
                    let new_scroll = start_scroll - dx_secs;
                    return (
                        canvas::event::Status::Captured,
                        Some(CanvasEvent::Pan(new_scroll - self.scroll)),
                    );
                }
                if let Some(i) = state.dragging {
                    return (
                        canvas::event::Status::Captured,
                        Some(CanvasEvent::MoveSplit(i, t)),
                    );
                }
                if let Some(anchor) = state.band_drag_from {
                    if let Some((cx, _cy, cw, _ch)) =
                        self.corr_plot_rect(bounds.width, bounds.height)
                    {
                        let here = ((pos.x - cx) / cw.max(1.0)).clamp(0.0, 1.0);
                        if (here - anchor).abs() > 0.004 {
                            let (a, b) = if anchor <= here {
                                (anchor, here)
                            } else {
                                (here, anchor)
                            };
                            if let (Some(f_lo), Some(f_hi)) =
                                (self.corr_x_to_hz(a), self.corr_x_to_hz(b))
                            {
                                return (
                                    canvas::event::Status::Captured,
                                    Some(CanvasEvent::SelectBand(Some((f_lo, f_hi)))),
                                );
                            }
                        }
                    }
                    return (canvas::event::Status::Captured, None);
                }
                if state.dragging_sgram {
                    return (
                        canvas::event::Status::Captured,
                        Some(CanvasEvent::Seek(t.clamp(0.0, self.duration))),
                    );
                }
                if !in_timeline {
                    if state.last_hover_px.is_some() {
                        state.last_hover_px = None;
                        return (
                            canvas::event::Status::Ignored,
                            Some(CanvasEvent::Hover(None)),
                        );
                    }
                    return (canvas::event::Status::Ignored, None);
                }
                let should_emit = match state.last_hover_px {
                    Some(last) => (last - rx).abs() >= 2.0,
                    None => true,
                };
                if should_emit {
                    state.last_hover_px = Some(rx);
                    (
                        canvas::event::Status::Ignored,
                        Some(CanvasEvent::Hover(Some(t))),
                    )
                } else {
                    (canvas::event::Status::Ignored, None)
                }
            }
            canvas::Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                state.dragging = None;
                state.dragging_guide = false;
                state.dragging_trim = None;
                state.dragging_overview = false;
                state.dragging_sgram = false;
                // A press-and-release on the curve without a drag is a
                // click, which means "clear the selection".
                if let Some(anchor) = state.band_drag_from.take() {
                    if let Some((cx, _cy, cw, _ch)) =
                        self.corr_plot_rect(bounds.width, bounds.height)
                    {
                        let here = ((pos.x - cx) / cw.max(1.0)).clamp(0.0, 1.0);
                        if (here - anchor).abs() <= 0.004 {
                            return (
                                canvas::event::Status::Captured,
                                Some(CanvasEvent::SelectBand(None)),
                            );
                        }
                    }
                }
                (canvas::event::Status::Captured, None)
            }
            _ => (canvas::event::Status::Ignored, None),
        }
    }

    fn draw(
        &self,
        _state: &DragState,
        renderer: &Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        // Layer 1: static / expensive content. Re-rendered only when something
        // big changes (file load, zoom, slider release). Includes the rulers,
        // waveform envelopes, zone backgrounds, ceiling lines, splits, AND
        // the spectrogram heatmap (which is precomputed and doesn't move with
        // playback).
        let static_layer = self.cache.draw(renderer, bounds.size(), |frame| {
            let total_w = frame.width();
            let total_h = frame.height();
            let (wx, wy, ww, wh) = self.wave_area(total_w, total_h);

            frame.fill_rectangle(
                Point::ORIGIN,
                Size::new(total_w, total_h),
                Color::from_rgb(0.10, 0.11, 0.13),
            );

            if self.show_time_ruler {
                self.draw_time_ruler(frame, wx, ww);
            }
            if self.show_db_ruler {
                self.draw_db_ruler(frame, wy, wh);
            }

            frame.fill_rectangle(
                Point::new(wx, wy),
                Size::new(ww, wh),
                Color::from_rgb(0.08, 0.09, 0.11),
            );

            self.draw_wave_area(frame, wx, wy, ww, wh);

            // Bottom strip background (under both meters + FFT panel) so the
            // meter_cache layer can paint on top of a stable backdrop.
            let meter_y = self.meter_y(total_h);
            frame.fill_rectangle(
                Point::new(wx, meter_y),
                Size::new(ww, total_h - meter_y),
                Color::from_rgb(0.05, 0.06, 0.08),
            );

            // Overview strip: whole-file, so it belongs in the static layer
            // alongside the spectrogram. Zoom-independent by design, which
            // is what makes it safe here — this cache isn't rebuilt on zoom.
            if let Some((sx, sy, sw, sh)) = self.overview_rect(total_w, total_h) {
                self.draw_overview(frame, sx, sy, sw, sh);
            }

            // Spectrogram is in the static layer because it covers the whole
            // file and only changes when the audio changes.
            if self.fft_mode == 2 {
                let fft_y = meter_y + 18.0;
                let fft_h = total_h - fft_y;
                if fft_h > 10.0 {
                    self.draw_spectrogram(frame, wx, fft_y, ww, fft_h);
                }
            }
        });

        // Layer 2: fast-updating bottom strip — peak meters + real-time
        // spectrum line. Repainted ~30 Hz during playback. Cheap because
        // the area is small.
        let meter_layer = self.meter_cache.draw(renderer, bounds.size(), |frame| {
            let total_w = frame.width();
            let total_h = frame.height();
            let (wx, _wy, ww, _wh) = self.wave_area(total_w, total_h);
            let meter_y = self.meter_y(total_h);
            self.draw_peak_meters(frame, wx, meter_y, ww);
            if self.fft_mode == 1 {
                let fft_y = meter_y + 18.0;
                let fft_h = total_h - fft_y;
                if fft_h > 10.0 {
                    self.draw_spectrum_line(frame, wx, fft_y, ww, fft_h);
                }
            }
            if self.fft_mode == 3 {
                let fft_y = meter_y + 18.0;
                let fft_h = total_h - fft_y;
                if fft_h > 10.0 {
                    self.draw_phase(frame, wx, fft_y, ww, fft_h);
                }
            }
        });

        // Layer 3: overlay (playhead + hover line).
        let overlay = self.overlay_cache.draw(renderer, bounds.size(), |frame| {
            let total_w = frame.width();
            let total_h = frame.height();
            let (wx, wy, ww, wh) = self.wave_area(total_w, total_h);

            if let Some((sx, sy, sw, sh)) = self.overview_rect(total_w, total_h) {
                self.draw_overview_overlay(frame, sx, sy, sw, sh);
            }

            if let Some(t) = self.hover_secs {
                let x = wx + self.t_to_x(t, ww);
                if x >= wx && x <= wx + ww {
                    let dash_len = 6.0;
                    let gap = 4.0;
                    let mut y = wy;
                    while y < wy + wh {
                        let line = Path::line(
                            Point::new(x, y),
                            Point::new(x, (y + dash_len).min(wy + wh)),
                        );
                        frame.stroke(
                            &line,
                            Stroke::default()
                                // White, dashed pattern. The dash + lower
                                // alpha distinguish it from the solid
                                // bright-white vertical playhead line.
                                .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.75))
                                .with_width(1.8),
                        );
                        y += dash_len + gap;
                    }
                }
            }
            // Detected-click markers: short amber ticks at the top of the
            // wave area so the user can spot clicks (even invisible ones).
            // The currently-navigated click is drawn brighter/full-strength.
            if !self.detected_clicks.is_empty() && self.sample_rate > 0 {
                let sr = self.sample_rate as f32;
                for (i, &frame_idx) in self.detected_clicks.iter().enumerate() {
                    let t = frame_idx as f32 / sr;
                    let x = wx + self.t_to_x(t, ww);
                    if x >= wx && x <= wx + ww {
                        let is_current = self.current_click == Some(i);
                        let (tick_w, tick_a, line_a) = if is_current {
                            (2.5, 1.0, 0.5)
                        } else {
                            (2.0, 0.85, 0.22)
                        };
                        let tick = Path::line(
                            Point::new(x, wy),
                            Point::new(x, wy + if is_current { 18.0 } else { 12.0 }),
                        );
                        frame.stroke(
                            &tick,
                            Stroke::default()
                                .with_color(Color::from_rgba(1.0, 0.65, 0.15, tick_a))
                                .with_width(tick_w),
                        );
                        let full = Path::line(
                            Point::new(x, wy),
                            Point::new(x, wy + wh),
                        );
                        frame.stroke(
                            &full,
                            Stroke::default()
                                .with_color(Color::from_rgba(1.0, 0.65, 0.15, line_a))
                                .with_width(1.0),
                        );
                    }
                }
            }

            if let Some(t) = self.playhead_secs {
                let x = wx + self.t_to_x(t, ww);
                if x >= wx && x <= wx + ww {
                    let line = Path::line(Point::new(x, wy), Point::new(x, wy + wh));
                    frame.stroke(
                        &line,
                        Stroke::default()
                            .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.95))
                            .with_width(1.5),
                    );
                }
            }

            // Mirror the playhead and hover line onto the spectrogram.
            // Without them the panel looks interactive but reads as inert —
            // and there's no way to line a moment in the image up with a
            // moment in the waveform, which is most of why you'd have both
            // on screen at once.
            if let Some((sx, sy, sw, sh)) = self.spectrogram_rect(total_w, total_h) {
                if let Some(t) = self.playhead_secs {
                    let x = sx + self.t_to_x(t, sw);
                    if x >= sx && x <= sx + sw {
                        frame.stroke(
                            &Path::line(Point::new(x, sy), Point::new(x, sy + sh)),
                            Stroke::default()
                                .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.95))
                                .with_width(1.5),
                        );
                    }
                }
                if let Some(t) = self.hover_secs {
                    let x = sx + self.t_to_x(t, sw);
                    if x >= sx && x <= sx + sw {
                        frame.stroke(
                            &Path::line(Point::new(x, sy), Point::new(x, sy + sh)),
                            Stroke::default()
                                .with_color(Color::from_rgba(1.0, 1.0, 1.0, 0.30))
                                .with_width(1.0),
                        );
                    }
                }
            }
        });

        vec![static_layer, meter_layer, overlay]
    }
}

fn draw_envelope(
    frame: &mut Frame,
    env: &[(f32, f32)],
    ox: f32,
    oy: f32,
    w: f32,
    h: f32,
    color: Color,
) {
    if env.is_empty() {
        return;
    }
    let mid = oy + h * 0.5;
    let half_h = h * 0.5;
    let n = env.len() as f32;
    let path = Path::new(|p| {
        for (i, &(lo, hi)) in env.iter().enumerate() {
            let x = ox + (i as f32 / n) * w;
            let y_hi = mid - hi * half_h;
            let y_lo = mid - lo * half_h;
            p.move_to(Point::new(x, y_hi));
            p.line_to(Point::new(x, y_lo));
        }
    });
    frame.stroke(&path, Stroke::default().with_color(color).with_width(1.0));
}

/// Pick a "nice" tick interval for the time ruler (1, 2, 5, 10, 20, 50…).
fn nice_time_step(raw: f32) -> f32 {
    if raw <= 0.0 {
        return 1.0;
    }
    let exp = raw.log10().floor();
    let base = 10f32.powf(exp);
    let mantissa = raw / base;
    let nice = if mantissa < 1.5 {
        1.0
    } else if mantissa < 3.5 {
        2.0
    } else if mantissa < 7.5 {
        5.0
    } else {
        10.0
    };
    nice * base
}

/// Format a time in seconds with appropriate precision for ruler labels.
fn format_time(t: f32) -> String {
    if t < 1.0 {
        format!("{:.0}ms", t * 1000.0)
    } else if t < 60.0 {
        format!("{:.2}s", t)
    } else {
        let m = (t / 60.0) as u32;
        let s = t - (m * 60) as f32;
        format!("{}:{:05.2}", m, s)
    }
}

#[derive(Debug, Default)]
pub struct DragState {
    pub dragging: Option<usize>,
    pub last_hover_px: Option<f32>,
    pub panning_from: Option<(f32, f32)>,
    /// True while the user is dragging the horizontal guide line.
    pub dragging_guide: bool,
    /// Which trim handle is being dragged, if any.
    pub dragging_trim: Option<TrimHandle>,
    /// True while scrubbing the correlation overview strip.
    pub dragging_overview: bool,
    /// True while scrubbing on the spectrogram panel.
    pub dragging_sgram: bool,
    /// Anchor x-fraction of an in-progress frequency-band drag on the
    /// correlation curve. `None` when not dragging a band.
    pub band_drag_from: Option<f32>,
}

/// Which trim boundary a drag is moving.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrimHandle {
    Start,
    End,
}

/// Downsample a mono audio buffer into `width` (min,max) buckets for drawing.
/// Parallelized via rayon.
pub fn build_envelope(mono: &[f32], width: usize) -> Vec<(f32, f32)> {
    use rayon::prelude::*;
    if mono.is_empty() || width == 0 {
        return Vec::new();
    }
    let bucket = (mono.len() as f32 / width as f32).max(1.0);
    (0..width)
        .into_par_iter()
        .map(|i| {
            let start = (i as f32 * bucket) as usize;
            let end = ((i as f32 + 1.0) * bucket) as usize;
            let end = end.min(mono.len()).max(start + 1);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            for &s in &mono[start..end] {
                if s < lo {
                    lo = s;
                }
                if s > hi {
                    hi = s;
                }
            }
            (lo.clamp(-1.0, 1.0), hi.clamp(-1.0, 1.0))
        })
        .collect()
}

/// Snap a target time (seconds) to the nearest zero crossing in `mono`.
/// Searches ±`window_ms` around the target. Returns the snapped time.
///
/// "Zero crossing" = a sample where the sign changes between consecutive
/// samples. Among candidates we pick the one closest to the target time.
/// Falls back to the original target if no crossing is found in window.
pub fn snap_to_zero_crossing(
    mono: &[f32],
    sample_rate: u32,
    target_secs: f32,
    window_ms: f32,
) -> f32 {
    if mono.is_empty() || sample_rate == 0 {
        return target_secs;
    }
    let target_frame = (target_secs * sample_rate as f32) as isize;
    let half_window = ((window_ms / 1000.0) * sample_rate as f32) as isize;
    let lo = (target_frame - half_window).max(1) as usize;
    let hi = ((target_frame + half_window) as usize).min(mono.len().saturating_sub(1));
    if hi <= lo {
        return target_secs;
    }
    let mut best: Option<(usize, isize)> = None; // (frame, distance_in_frames)
    for i in lo..hi {
        let prev = mono[i - 1];
        let cur = mono[i];
        // Sign change OR exactly zero on this sample.
        let crosses = (prev <= 0.0 && cur >= 0.0) || (prev >= 0.0 && cur <= 0.0);
        if crosses {
            let d = (i as isize - target_frame).abs();
            match best {
                Some((_, bd)) if d >= bd => {}
                _ => best = Some((i, d)),
            }
        }
    }
    match best {
        Some((frame, _)) => frame as f32 / sample_rate as f32,
        None => target_secs,
    }
}
