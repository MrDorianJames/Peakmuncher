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

pub const FFT_SIZE: usize = 2048;
const HOP: usize = FFT_SIZE / 2;

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

/// Precomputed Hann window of length FFT_SIZE (used by the spectrogram).
fn hann_window() -> Vec<f32> {
    (0..FFT_SIZE)
        .map(|n| {
            let x = (n as f32) / (FFT_SIZE as f32 - 1.0);
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * x).cos())
        })
        .collect()
}

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

/// A whole-file spectrogram: rows are FFT slices in time, each row is
/// `FFT_SIZE/2` dB magnitudes. Stored row-major (one Vec per time slice).
#[derive(Debug)]
pub struct Spectrogram {
    pub slices: Vec<Vec<f32>>,
    pub hop: usize,
    pub sample_rate: u32,
}

impl Spectrogram {
    /// Number of time bins.
    pub fn time_bins(&self) -> usize {
        self.slices.len()
    }
    /// Number of frequency bins (FFT_SIZE / 2).
    pub fn freq_bins(&self) -> usize {
        self.slices.first().map(|s| s.len()).unwrap_or(0)
    }
    /// Time in seconds for the center of slice `i`.
    pub fn time_secs(&self, i: usize) -> f32 {
        (i * self.hop + FFT_SIZE / 2) as f32 / self.sample_rate as f32
    }
}

/// Compute a spectrogram from `mono` audio. Parallelized across slices.
pub fn compute_spectrogram(mono: &[f32], sample_rate: u32) -> Spectrogram {
    use rayon::prelude::*;
    if mono.len() < FFT_SIZE {
        return Spectrogram {
            slices: Vec::new(),
            hop: HOP,
            sample_rate,
        };
    }
    let n_slices = (mono.len() - FFT_SIZE) / HOP + 1;
    let window: Arc<Vec<f32>> = Arc::new(hann_window());

    // Each slice plans its own FFT — small cost, lets us parallelize cleanly.
    let slices: Vec<Vec<f32>> = (0..n_slices)
        .into_par_iter()
        .map(|si| {
            let start = si * HOP;
            let mut buf: Vec<Complex32> = (0..FFT_SIZE)
                .map(|i| Complex32::new(mono[start + i] * window[i], 0.0))
                .collect();
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);
            fft.process(&mut buf);
            let scale = 2.0 / FFT_SIZE as f32;
            buf[..FFT_SIZE / 2]
                .iter()
                .map(|c| {
                    let mag = (c.re * c.re + c.im * c.im).sqrt() * scale;
                    if mag <= 1e-6 {
                        -120.0
                    } else {
                        20.0 * mag.log10()
                    }
                })
                .collect()
        })
        .collect();

    Spectrogram {
        slices,
        hop: HOP,
        sample_rate,
    }
}

/// Map a dB value in [-100, 0] to an RGBA color for spectrogram rendering.
/// Uses the MAGMA colormap: near-black → deep purple → magenta → orange →
/// pale cream. Perceptually uniform (equal dB steps look like equal visual
/// steps, unlike a rainbow/jet ramp), and its black low end reads cleanly as
/// silence against the dark theme. Control points sampled from matplotlib's
/// magma.
pub fn db_to_rgba(db: f32) -> [f32; 4] {
    let t = ((db + 100.0) / 100.0).clamp(0.0, 1.0);
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
