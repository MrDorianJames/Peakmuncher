//! FFT-based spectrum analysis.
//!
//! Two outputs:
//!   1. `spectrum_at`: a single FFT slice at a given time (for the real-time
//!      "right now" line).
//!   2. `compute_spectrogram`: a 2D dB heatmap of the whole file, computed
//!      once after load (or after a render) and cached.
//!
//! Uses Hann windowing, 50% overlap, and post-FFT log-magnitude in dB.

use rustfft::{num_complex::Complex32, FftPlanner};
use std::sync::Arc;

/// Default spectrogram FFT size. This is a TRADE, not a quality setting:
/// window length buys frequency resolution and spends time resolution, and
/// the product is fixed. At 48 kHz:
///
/// ```text
///   1024 → 21 ms / 47 Hz    transients sharp, bass unusable
///   2048 → 43 ms / 23 Hz    balanced
///   4096 → 85 ms / 12 Hz    harmonics sharp, transients smeared
///   8192 → 171 ms / 6 Hz    tonal analysis only
/// ```
///
/// 2048 is the default because most looking-at-a-spectrogram is looking at
/// rhythm and transients; 4096 smears a kick across ~85 ms, which reads as
/// a blob rather than a hit.
pub const FFT_SIZE: usize = 2048;

/// Overlap divisor: hop = fft_size / 8, i.e. 87.5% overlap. Heavy overlap is
/// what makes the image continuous rather than striped when zoomed in.
///
/// Note this makes total memory INDEPENDENT of FFT size — halving the window
/// halves the bins but doubles the slice count.
const HOP_DIV: usize = 8;

/// Selectable spectrogram window sizes.
pub const FFT_SIZES: [usize; 4] = [1024, 2048, 4096, 8192];

/// Human-readable trade-off for each window size.
pub fn fft_size_label(n: usize, sample_rate: u32) -> String {
    let sr = if sample_rate == 0 { 48000 } else { sample_rate };
    let ms = n as f32 / sr as f32 * 1000.0;
    let hz = sr as f32 / n as f32;
    format!("{n}  ({ms:.0} ms / {hz:.0} Hz)")
}

/// Larger FFT for the single-frame spectrum LINE (not the spectrogram).
/// The line is a frozen, averaged analysis at a probe point, so we trade
/// time resolution (irrelevant when frozen) for frequency resolution. At
/// 48 kHz, 16384 ≈ 2.9 Hz/bin — enough to resolve discrete bass harmonics
/// (a 60 Hz fundamental and its 120/180 Hz harmonics land ~20+ bins apart
/// instead of 2-3 with the 2048 spectrogram FFT).
pub const SPECTRUM_FFT_SIZE: usize = 16384;

/// Number of overlapping frames averaged around the probe point for the
/// single-frame line. Averaging reinforces stable harmonic peaks and pulls
/// the noise floor down, so the harmonic structure reads cleanly instead of
/// flickering frame-to-frame. Frames are spaced SPECTRUM_FFT_SIZE/4 apart.
const SPECTRUM_AVG_FRAMES: usize = 4;



/// Blackman-Harris window of length `n`. Used by the single-frame spectrum
/// line. Its sidelobes are ~-92 dB (vs Hann's ~-31 dB), so a strong harmonic
/// no longer leaks enough energy into neighboring bins to bury a weaker
/// harmonic beside it — which is exactly what makes discrete harmonics
/// visible. The tradeoff (a slightly wider main lobe) is fine here because
/// the large FFT size more than makes up the resolution.
fn blackman_harris_window(n: usize) -> Vec<f32> {
    const A0: f32 = 0.35875;
    const A1: f32 = 0.48829;
    const A2: f32 = 0.14128;
    const A3: f32 = 0.01168;
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| {
            let x = (i as f32) / denom;
            let t = 2.0 * std::f32::consts::PI * x;
            A0 - A1 * t.cos() + A2 * (2.0 * t).cos() - A3 * (3.0 * t).cos()
        })
        .collect()
}

/// Compute one averaged FFT spectrum centered at `center_frame`, for the
/// frozen spectrum LINE. Returns `SPECTRUM_FFT_SIZE/2` dB magnitudes in
/// [-120, 0] (0 = DC, last = Nyquist). Returns `None` if there isn't enough
/// audio.
///
/// Uses SPECTRUM_FFT_SIZE-point FFTs, a Blackman-Harris window, and averages
/// SPECTRUM_AVG_FRAMES overlapping frames around the probe point. Averaging
/// is done in the POWER domain (mag²), not dB — averaging dB is a geometric
/// mean that under-weights peaks and would distort the harmonic picture.
pub fn spectrum_at(mono: &[f32], center_frame: usize) -> Option<Vec<f32>> {
    let n = SPECTRUM_FFT_SIZE;
    if mono.len() < n {
        return None;
    }
    let half = n / 2;
    let bins = n / 2;
    let window = blackman_harris_window(n);
    // Window power gain, for normalizing magnitude consistently across windows.
    let win_sum: f32 = window.iter().sum();
    let scale = 2.0 / win_sum.max(1e-9);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    // Frame start offsets, centered on `center_frame`, spaced n/4 apart, all
    // clamped into range. Duplicate/clamped offsets just average toward the
    // same frame near the file edges — harmless.
    let spacing = n / 4;
    let max_start = mono.len() - n;
    let mut power_sum = vec![0.0f32; bins];
    let mut frames_used = 0u32;

    for k in 0..SPECTRUM_AVG_FRAMES {
        // Spread frames symmetrically around the center.
        let offset = (k as isize - (SPECTRUM_AVG_FRAMES as isize - 1) / 2) * spacing as isize;
        let center = center_frame as isize + offset;
        let start = (center - half as isize).clamp(0, max_start as isize) as usize;

        let mut buf: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(mono[start + i] * window[i], 0.0))
            .collect();
        fft.process(&mut buf);
        for (acc, c) in power_sum.iter_mut().zip(buf[..bins].iter()) {
            let mag = (c.re * c.re + c.im * c.im).sqrt() * scale;
            *acc += mag * mag; // accumulate power
        }
        frames_used += 1;
    }

    let inv = 1.0 / frames_used.max(1) as f32;
    Some(
        power_sum
            .iter()
            .map(|&p| {
                let mag = (p * inv).sqrt(); // mean power → magnitude
                if mag <= 1e-6 {
                    -120.0
                } else {
                    20.0 * mag.log10()
                }
            })
            .collect(),
    )
}

/// FFT size for the per-frequency correlation curve. Deliberately MUCH
/// smaller than `SPECTRUM_FFT_SIZE`: correlation asks "which region of the
/// spectrum is out of phase", not "which harmonic", so frequency resolution
/// past ~1/6 octave is wasted — and a smaller FFT buys more averaging frames
/// for the same cost, which is what this measurement actually needs.
pub const CORR_FFT_SIZE: usize = 4096;

/// Overlapping frames averaged into the cross-spectrum. This averaging is
/// NOT optional: for a single frame, L and R are one complex number per bin
/// and their "correlation" is always ±1 — the measurement only becomes
/// meaningful as an expectation over several frames.
const CORR_AVG_FRAMES: usize = 12;

/// Smoothing width, as a fraction of an octave. Per-bin correlation is
/// jittery even after frame averaging; smoothing across neighboring bins is
/// what turns it into a readable curve. 1/6 octave ≈ 60 independent points
/// across the audible range, which is plenty for "where is the problem".
const CORR_SMOOTH_OCT: f32 = 1.0 / 6.0;

/// Bins this far below the window's peak energy are treated as "no signal"
/// and faded out. Where there's no energy the denominator collapses and the
/// ratio flies around wildly — drawing that would be pure noise.
const CORR_FADE_FLOOR_DB: f32 = -70.0;
/// dB range over which the fade ramps from invisible to fully drawn.
const CORR_FADE_RANGE_DB: f32 = 20.0;

/// Hann window of arbitrary length (the fixed-size `hann_window` above is
/// kept for the spectrogram's hot path).
fn hann_n(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| {
            let x = (i as f32) / denom;
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * x).cos())
        })
        .collect()
}

/// Per-frequency stereo correlation at a probe point.
///
/// Returns one `(correlation, weight)` pair per bin, where bin `i` is at
/// `i / bins * nyquist` Hz (same mapping as the spectrum line, so the two
/// views line up on screen). `correlation` is −1..+1; `weight` is 0..1 and
/// says how much signal there is at that frequency — draw faint or not at
/// all where it's low.
///
/// The math is the broadband Pearson correlation lifted into the frequency
/// domain. Per bin, averaged over frames:
///
/// ```text
///   Sxy = Σ Re(L · conj(R))     cross-spectrum (real part)
///   Sxx = Σ |L|²                auto-spectra
///   Syy = Σ |R|²
///   corr = Sxy / sqrt(Sxx · Syy)
/// ```
///
/// Taking `Re(Sxy)` rather than `|Sxy|` is the important choice: the
/// magnitude version is *coherence*, which is always positive and says
/// whether the channels are related at all. The real part is signed, so
/// phase-inverted content — the thing that cancels in mono — reads negative.
///
/// `samples` is interleaved; `channels` must be ≥ 2 (returns `None` for mono).
pub fn band_correlation(
    samples: &[f32],
    channels: usize,
    center_frame: usize,
) -> Option<Vec<(f32, f32)>> {
    if channels < 2 {
        return None;
    }
    let n = CORR_FFT_SIZE;
    let frames = samples.len() / channels;
    if frames < n {
        return None;
    }
    let bins = n / 2;
    let half = n / 2;
    let window = hann_n(n);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);

    // 50% overlap, frames spread symmetrically around the probe. Clamping at
    // the file edges just re-averages the same frame — harmless.
    let spacing = n / 2;
    let max_start = frames - n;

    let mut sxx = vec![0.0f32; bins];
    let mut syy = vec![0.0f32; bins];
    let mut sxy = vec![0.0f32; bins];

    for k in 0..CORR_AVG_FRAMES {
        let offset = (k as isize - (CORR_AVG_FRAMES as isize - 1) / 2) * spacing as isize;
        let center = center_frame as isize + offset;
        let start = (center - half as isize).clamp(0, max_start as isize) as usize;

        let mut bl: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(samples[(start + i) * channels] * window[i], 0.0))
            .collect();
        let mut br: Vec<Complex32> = (0..n)
            .map(|i| Complex32::new(samples[(start + i) * channels + 1] * window[i], 0.0))
            .collect();
        fft.process(&mut bl);
        fft.process(&mut br);

        for i in 0..bins {
            let l = bl[i];
            let r = br[i];
            sxx[i] += l.re * l.re + l.im * l.im;
            syy[i] += r.re * r.re + r.im * r.im;
            // Re(L · conj(R)) = Lre·Rre + Lim·Rim
            sxy[i] += l.re * r.re + l.im * r.im;
        }
    }

    // Octave-fraction smoothing. Smooth the SPECTRA, then take the ratio —
    // not the other way round. Averaging a ratio across bins weights quiet
    // bins the same as loud ones and biases the result; averaging the
    // cross- and auto-spectra first is the properly energy-weighted estimate.
    let lo_mul = (-CORR_SMOOTH_OCT * 0.5).exp2();
    let hi_mul = (CORR_SMOOTH_OCT * 0.5).exp2();

    let mut out = vec![(0.0f32, 0.0f32); bins];
    let mut energy = vec![0.0f32; bins];

    for i in 1..bins {
        let lo = ((i as f32 * lo_mul).floor() as usize).max(1);
        let hi = (((i as f32 * hi_mul).ceil() as usize) + 1).min(bins);
        let (lo, hi) = if hi > lo { (lo, hi) } else { (i, (i + 1).min(bins)) };

        let mut axx = 0.0f32;
        let mut ayy = 0.0f32;
        let mut axy = 0.0f32;
        for b in lo..hi {
            axx += sxx[b];
            ayy += syy[b];
            axy += sxy[b];
        }
        let denom = (axx * ayy).sqrt();
        let corr = if denom > 1e-20 {
            (axy / denom).clamp(-1.0, 1.0)
        } else {
            0.0
        };
        out[i].0 = corr;
        energy[i] = denom;
    }

    // Weight each bin by how far its energy sits above the noise floor,
    // relative to the loudest bin in this window.
    let peak = energy.iter().cloned().fold(0.0f32, f32::max).max(1e-20);
    for i in 1..bins {
        let db_rel = 10.0 * (energy[i] / peak).max(1e-20).log10();
        out[i].1 = ((db_rel - CORR_FADE_FLOOR_DB) / CORR_FADE_RANGE_DB).clamp(0.0, 1.0);
    }

    Some(out)
}

/// Whole-file broadband correlation over time.
///
/// One correlation value per short window, so the Phase view can show
/// *where* in the file the phase trouble is — the per-frequency curve
/// answers "what's wrong here", this answers "where should I look". No FFT
/// involved: it's the same Pearson correlation the probe readout uses, just
/// evaluated on a sliding window across the whole file, which makes it a
/// single cheap pass rather than a transform.
#[derive(Debug)]
pub struct CorrTimeline {
    /// `(correlation, weight)` per window. Weight is 0..1 and reflects how
    /// much signal the window had — silence gets weight 0 so it can be
    /// drawn as a gap instead of a meaningless value.
    pub points: Vec<(f32, f32)>,
    /// Seconds between successive points.
    pub hop_secs: f32,
}

impl CorrTimeline {
    /// Sample the timeline at a time in seconds. Returns `None` outside the
    /// analyzed range.
    pub fn at(&self, secs: f32) -> Option<(f32, f32)> {
        if self.hop_secs <= 0.0 || self.points.is_empty() {
            return None;
        }
        let i = (secs / self.hop_secs) as usize;
        self.points.get(i).copied()
    }
}

/// Window length for the correlation timeline. Matches the probe readout's
/// 100 ms so the strip and the numeric readout agree when you park on a spot.
const CORR_TL_WINDOW_MS: f32 = 100.0;
/// 50% overlap — smooths the curve without doubling cost twice over.
const CORR_TL_OVERLAP: usize = 2;
/// Windows quieter than this (relative to the file's loudest window) are
/// treated as silence and given zero weight.
const CORR_TL_FLOOR_DB: f32 = -60.0;
const CORR_TL_FADE_DB: f32 = 15.0;

/// Compute the correlation timeline for interleaved `samples`.
/// Returns an empty timeline for mono or empty input.
pub fn correlation_timeline(samples: &[f32], channels: usize, sample_rate: u32) -> CorrTimeline {
    let empty = CorrTimeline {
        points: Vec::new(),
        hop_secs: 0.0,
    };
    if channels < 2 || sample_rate == 0 || samples.is_empty() {
        return empty;
    }
    let frames = samples.len() / channels;
    let win = (((CORR_TL_WINDOW_MS / 1000.0) * sample_rate as f32) as usize).max(256);
    if frames < win {
        return empty;
    }
    let hop = (win / CORR_TL_OVERLAP).max(1);
    let n_points = (frames - win) / hop + 1;

    // f64 accumulators: a 100 ms window at 96 kHz is ~10k terms, and f32
    // summation of squares drifts enough over that to visibly wobble the
    // result on quiet material.
    let mut raw: Vec<(f32, f64)> = Vec::with_capacity(n_points);
    for p in 0..n_points {
        let lo = p * hop;
        let hi = (lo + win).min(frames);
        let mut sum_l = 0.0f64;
        let mut sum_r = 0.0f64;
        let mut sum_ll = 0.0f64;
        let mut sum_rr = 0.0f64;
        let mut sum_lr = 0.0f64;
        let n = (hi - lo).max(1) as f64;
        for f in lo..hi {
            let l = samples[f * channels] as f64;
            let r = samples[f * channels + 1] as f64;
            sum_l += l;
            sum_r += r;
            sum_ll += l * l;
            sum_rr += r * r;
            sum_lr += l * r;
        }
        // Covariance / (σL · σR), mean-subtracted so a DC offset in one
        // channel doesn't masquerade as correlation.
        let cov = sum_lr / n - (sum_l / n) * (sum_r / n);
        let var_l = (sum_ll / n - (sum_l / n).powi(2)).max(0.0);
        let var_r = (sum_rr / n - (sum_r / n).powi(2)).max(0.0);
        let denom = (var_l * var_r).sqrt();
        let corr = if denom > 1e-18 {
            (cov / denom).clamp(-1.0, 1.0) as f32
        } else {
            0.0
        };
        // Window energy, for the silence gate.
        let energy = (sum_ll + sum_rr) / (2.0 * n);
        raw.push((corr, energy));
    }

    let peak = raw.iter().map(|(_, e)| *e).fold(0.0f64, f64::max).max(1e-20);
    let points = raw
        .into_iter()
        .map(|(c, e)| {
            let db = 10.0 * (e / peak).max(1e-20).log10();
            let w = (((db as f32) - CORR_TL_FLOOR_DB) / CORR_TL_FADE_DB).clamp(0.0, 1.0);
            (c, w)
        })
        .collect();

    CorrTimeline {
        points,
        hop_secs: hop as f32 / sample_rate as f32,
    }
}

/// dB value represented by u8 code 0 (see `Spectrogram::data`).
const SG_DB_FLOOR: f32 = -120.0;

/// Band-limit a stereo window and return goniometer points plus the band's
/// own correlation.
///
/// Used by the frequency selection on the correlation curve: the broadband
/// goniometer is energy-weighted, so a narrow band that's badly out of phase
/// contributes only its share of the dots and disappears under a loud
/// near-mono kick. Restricting to the selected band is what makes the
/// scatter actually show the problem the curve is reporting.
///
/// Method: FFT both channels, zero every bin outside `[lo_hz, hi_hz]`, then
/// inverse FFT. Brick-wall filtering rings in the time domain, which would
/// be unacceptable for audio you listen to — here it's fine, because the
/// goniometer only cares about the L/R *relationship*, and the ringing is
/// identical in both channels so it doesn't bias the phase picture.
///
/// Returns `(points, correlation)`, or `None` for mono / too-short input.
pub fn band_limited_gonio(
    samples: &[f32],
    channels: usize,
    center_frame: usize,
    sample_rate: u32,
    lo_hz: f32,
    hi_hz: f32,
    max_points: usize,
) -> Option<(Vec<(f32, f32)>, f32)> {
    if channels < 2 || sample_rate == 0 {
        return None;
    }
    let n = CORR_FFT_SIZE;
    let frames = samples.len() / channels;
    if frames < n {
        return None;
    }
    let half = n / 2;
    let start = (center_frame as isize - half as isize).clamp(0, (frames - n) as isize) as usize;

    let mut planner = FftPlanner::<f32>::new();
    let fwd = planner.plan_fft_forward(n);
    let inv = planner.plan_fft_inverse(n);

    // No analysis window here: a Hann taper would fade the edges of the
    // block toward zero, piling up dots at the origin and squashing the
    // apparent width. The block is rectangular and we accept the spectral
    // leakage, since we're throwing away magnitudes anyway.
    let mut bl: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(samples[(start + i) * channels], 0.0))
        .collect();
    let mut br: Vec<Complex32> = (0..n)
        .map(|i| Complex32::new(samples[(start + i) * channels + 1], 0.0))
        .collect();
    fwd.process(&mut bl);
    fwd.process(&mut br);

    let nyq = sample_rate as f32 / 2.0;
    let bins = n / 2;
    let b_lo = ((lo_hz / nyq) * bins as f32).floor().max(0.0) as usize;
    let b_hi = (((hi_hz / nyq) * bins as f32).ceil() as usize).clamp(b_lo + 1, bins);

    // Zero everything outside the band, in BOTH halves of the spectrum —
    // the upper half is the conjugate mirror, and leaving it populated
    // would produce a complex (non-real) inverse transform.
    let zero = Complex32::new(0.0, 0.0);
    for k in 0..n {
        let mirror = if k <= half { k } else { n - k };
        if mirror < b_lo || mirror >= b_hi {
            bl[k] = zero;
            br[k] = zero;
        }
    }
    inv.process(&mut bl);
    inv.process(&mut br);

    let scale = 1.0 / n as f32;
    // Correlation of the band-limited signals, so the readout beside the
    // scatter is measured on exactly what's being drawn.
    let mut sum_ll = 0.0f64;
    let mut sum_rr = 0.0f64;
    let mut sum_lr = 0.0f64;
    for i in 0..n {
        let l = (bl[i].re * scale) as f64;
        let r = (br[i].re * scale) as f64;
        sum_ll += l * l;
        sum_rr += r * r;
        sum_lr += l * r;
    }
    let denom = (sum_ll * sum_rr).sqrt();
    let corr = if denom > 1e-18 {
        (sum_lr / denom).clamp(-1.0, 1.0) as f32
    } else {
        0.0
    };

    // Normalize the scatter so a quiet band still fills the scope. Without
    // this, selecting a band 40 dB down draws a dot at the origin — the
    // shape is the information here, not the level.
    let peak = (0..n)
        .map(|i| bl[i].re.abs().max(br[i].re.abs()) * scale)
        .fold(0.0f32, f32::max);
    let gain = if peak > 1e-9 { 0.9 / peak } else { 0.0 };

    let step = (n / max_points.max(1)).max(1);
    let points: Vec<(f32, f32)> = (0..n)
        .step_by(step)
        .map(|i| (bl[i].re * scale * gain, br[i].re * scale * gain))
        .collect();

    Some((points, corr))
}

/// A whole-file spectrogram: a flat `time × freq` grid of dB magnitudes.
///
/// Stored as **u8**, not f32. Raising the resolution to 4096/512 multiplies
/// the cell count by 8, which at f32 would be ~250 MB for a 5-minute file —
/// enough to matter. Quantizing to 0.47 dB steps over a −120..0 dB range
/// costs nothing visible (the colour ramp resolves far less than that) and
/// brings a higher-resolution spectrogram in at the same memory as the old
/// low-resolution one. Flat rather than `Vec<Vec<_>>` so a column walk stays
/// in cache.
#[derive(Debug)]
pub struct Spectrogram {
    /// `data[t * freq_bins + f]`, dB encoded as u8.
    pub data: Vec<u8>,
    pub freq_bins: usize,
    pub hop: usize,
    pub fft_size: usize,
    pub sample_rate: u32,
    /// Auto-fitted display range, as u8 codes. See `fit_contrast`.
    pub floor_code: u8,
    pub ceil_code: u8,
}

impl Spectrogram {
    /// Number of time bins.
    pub fn time_bins(&self) -> usize {
        if self.freq_bins == 0 {
            0
        } else {
            self.data.len() / self.freq_bins
        }
    }
    /// One time slice as a u8 row.
    #[inline]
    pub fn row(&self, t: usize) -> &[u8] {
        let fb = self.freq_bins;
        &self.data[t * fb..(t + 1) * fb]
    }
    pub fn empty(sample_rate: u32) -> Self {
        Self {
            data: Vec::new(),
            freq_bins: 0,
            hop: FFT_SIZE / HOP_DIV,
            fft_size: FFT_SIZE,
            sample_rate,
            floor_code: 0,
            ceil_code: 255,
        }
    }

    /// Map a stored code to a colour using this spectrogram's fitted range.
    #[inline]
    pub fn code_to_rgba(&self, code: u8) -> [f32; 4] {
        let lo = self.floor_code as f32;
        let hi = (self.ceil_code as f32).max(lo + 1.0);
        let t = ((code as f32 - lo) / (hi - lo)).clamp(0.0, 1.0);
        magma(t)
    }
}

/// Choose the displayed dB window by looking at what's actually in the file.
///
/// A fixed −100..0 dB range spends most of the colour ramp on the noise
/// floor: a −70 dB floor lands 30% up the ramp, which in magma is a solid
/// purple. Every quiet region then glows, and real content has to compete
/// with it — which is what makes a spectrogram look washed out and
/// detail-free even at high resolution.
///
/// Fitting to percentiles of the actual data puts black at the file's own
/// noise floor and full brightness at its peaks, so the ramp is spent
/// entirely on content. Percentiles rather than min/max because a single
/// outlier bin would otherwise set the whole scale.
fn fit_contrast(data: &[u8]) -> (u8, u8) {
    if data.is_empty() {
        return (0, 255);
    }
    let mut hist = [0u32; 256];
    for &v in data {
        hist[v as usize] += 1;
    }
    let total = data.len() as u64;
    let pick = |frac: f64| -> u8 {
        let target = (total as f64 * frac) as u64;
        let mut acc = 0u64;
        for (i, &c) in hist.iter().enumerate() {
            acc += c as u64;
            if acc >= target {
                return i as u8;
            }
        }
        255
    };
    // P70 as black: most cells in any spectrogram are floor, so the
    // majority-percentile IS the floor. P99.9 as white, leaving a little
    // headroom so transients still read as brighter than sustained tones.
    let lo = pick(0.70);
    let hi = pick(0.999).max(lo.saturating_add(20));
    (lo, hi)
}

/// Compute a spectrogram from `mono` audio. Parallelized across slices.
pub fn compute_spectrogram(mono: &[f32], sample_rate: u32, fft_size: usize) -> Spectrogram {
    use rayon::prelude::*;
    let fft_size = if FFT_SIZES.contains(&fft_size) {
        fft_size
    } else {
        FFT_SIZE
    };
    let hop = fft_size / HOP_DIV;
    if mono.len() < fft_size {
        return Spectrogram::empty(sample_rate);
    }
    let n_slices = (mono.len() - fft_size) / hop + 1;
    let bins = fft_size / 2;
    let window: Arc<Vec<f32>> = Arc::new(hann_n(fft_size));

    // Plan the FFT ONCE and share it. The old code planned a fresh 2048-point
    // FFT inside every slice; at 8x the slice count that setup cost would
    // have dominated. `Arc<dyn Fft>` is Send+Sync, so rayon is still happy.
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut data = vec![0u8; n_slices * bins];
    let scale = 2.0 / fft_size as f32;
    let inv_step = 255.0 / -SG_DB_FLOOR;

    data.par_chunks_mut(bins)
        .enumerate()
        .for_each(|(si, out)| {
            let start = si * hop;
            let mut buf: Vec<Complex32> = (0..fft_size)
                .map(|i| Complex32::new(mono[start + i] * window[i], 0.0))
                .collect();
            fft.process(&mut buf);
            for (o, c) in out.iter_mut().zip(buf[..bins].iter()) {
                let power = (c.re * c.re + c.im * c.im) * (scale * scale);
                // 10·log10(power) == 20·log10(mag), one log instead of a
                // sqrt plus a log.
                let db = if power <= 1e-12 {
                    SG_DB_FLOOR
                } else {
                    10.0 * power.log10()
                };
                *o = (((db - SG_DB_FLOOR) * inv_step).clamp(0.0, 255.0)) as u8;
            }
        });

    let (floor_code, ceil_code) = fit_contrast(&data);
    Spectrogram {
        data,
        freq_bins: bins,
        hop,
        fft_size,
        sample_rate,
        floor_code,
        ceil_code,
    }
}

/// Map a dB value in [-100, 0] to an RGBA color for spectrogram rendering.
/// Uses the MAGMA colormap: near-black → deep purple → magenta → orange →
/// pale cream. Perceptually uniform (equal dB steps look like equal visual
/// steps, unlike a rainbow/jet ramp), and its black low end reads cleanly as
/// silence against the dark theme. Control points sampled from matplotlib's
/// magma.
pub fn magma(t: f32) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    // Magma control points across the 0..1 range.
    let stops: [(f32, [f32; 3]); 8] = [
        (0.00, [0.001, 0.000, 0.014]), // near black
        (0.14, [0.078, 0.043, 0.206]), // deep indigo
        (0.29, [0.232, 0.060, 0.438]), // purple
        (0.43, [0.400, 0.086, 0.500]), // magenta-purple
        (0.57, [0.588, 0.149, 0.474]), // magenta
        (0.71, [0.867, 0.288, 0.376]), // red-orange
        (0.86, [0.988, 0.553, 0.349]), // orange
        (1.00, [0.987, 0.909, 0.749]), // pale cream
    ];
    let mut a = stops[0];
    let mut b = stops[1];
    for i in 0..stops.len() - 1 {
        if t >= stops[i].0 && t <= stops[i + 1].0 {
            a = stops[i];
            b = stops[i + 1];
            break;
        }
    }
    let span = (b.0 - a.0).max(1e-6);
    let f = ((t - a.0) / span).clamp(0.0, 1.0);
    [
        a.1[0] + (b.1[0] - a.1[0]) * f,
        a.1[1] + (b.1[1] - a.1[1]) * f,
        a.1[2] + (b.1[2] - a.1[2]) * f,
        1.0,
    ]
}
