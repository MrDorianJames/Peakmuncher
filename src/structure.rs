//! Chroma-based structural segmentation.
//!
//! Implements Foote's checkerboard novelty algorithm on a chroma
//! self-similarity matrix. The pipeline:
//!
//! 1. Compute a chromagram: 12 pitch-class bins per time frame.
//! 2. Build an N×N self-similarity matrix (SSM) by cosine-similarity of
//!    every pair of chroma vectors. Repeating sections show as bright
//!    off-diagonals; section boundaries show as discontinuities along
//!    the main diagonal.
//! 3. Convolve a "checkerboard kernel" along the diagonal — at each
//!    frame, the kernel asks: how similar is the past N frames to itself,
//!    how similar is the future N frames to itself, and how *different*
//!    are past and future from each other? Peaks in this novelty curve
//!    are structural boundaries.
//!
//! This is signal-processing (no neural net), but captures harmonic
//! transitions that pure dynamics analysis misses (e.g., a quiet pad
//! intro into a same-loudness drum entry where the chord progression
//! changes).
//!
//! Reference: Foote, J. (2000). "Automatic audio segmentation using a
//! measure of audio novelty."

use rustfft::{num_complex::Complex32, FftPlanner};
use std::sync::Arc;

const FFT_SIZE: usize = 4096; // longer than `fft.rs` for finer pitch resolution
const HOP: usize = 1024;
const N_CHROMA: usize = 12;

/// Compute novelty curve from `mono` audio, sampled at `hop_secs` per step.
///
/// Returns `(novelty, hop_secs)` where `novelty[i]` is the structural-novelty
/// score at time `i * hop_secs`. Values are normalized to roughly [0..1].
///
/// Returns an empty vec if the audio is too short.
pub fn novelty_curve(mono: &[f32], sample_rate: u32) -> (Vec<f32>, f32) {
    if mono.len() < FFT_SIZE * 4 || sample_rate == 0 {
        return (Vec::new(), 0.0);
    }
    let hop_secs = HOP as f32 / sample_rate as f32;

    // 1) Chromagram: one 12-vector per hop.
    let chroma = chromagram(mono, sample_rate);
    let n_frames = chroma.len();
    if n_frames < 16 {
        return (Vec::new(), hop_secs);
    }

    // 2) SSM via cosine similarity. We don't actually need the full matrix
    //    in memory if we only want the diagonal-band that the checkerboard
    //    kernel touches — but for simplicity (and N up to ~3000 is fine
    //    memory-wise: 36 MB for f32) we compute the full thing.
    //
    //    For very long audio we might switch to a banded representation,
    //    but a 5-minute song at 1024-sample hops at 48kHz is ~14k frames,
    //    which would be ~800 MB. So we cap and downsample.
    let target_max_frames = 3000;
    let stride = (n_frames as f32 / target_max_frames as f32).ceil() as usize;
    let stride = stride.max(1);
    let downsampled: Vec<[f32; N_CHROMA]> = (0..n_frames)
        .step_by(stride)
        .map(|i| chroma[i])
        .collect();
    let n = downsampled.len();
    let effective_hop = hop_secs * stride as f32;

    // Pre-normalize each chroma vector so cosine similarity is just a dot.
    let normalized: Vec<[f32; N_CHROMA]> = downsampled
        .iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            let mut out = [0.0; N_CHROMA];
            for k in 0..N_CHROMA {
                out[k] = v[k] / norm;
            }
            out
        })
        .collect();

    // 3) Foote checkerboard novelty along the diagonal.
    //    Kernel size: should span ~3-6 seconds of audio for boundary detection.
    let target_kernel_secs = 4.0;
    let half_kernel = ((target_kernel_secs / effective_hop) as usize / 2).max(8);

    // Build a Gaussian-tapered checkerboard kernel of size (2L) × (2L):
    //   +1 in top-left and bottom-right quadrants
    //   -1 in top-right and bottom-left
    //   tapered by a 2D Gaussian centered at the kernel midpoint
    let l = half_kernel;
    let kernel_size = 2 * l;
    let mut kernel = vec![vec![0.0f32; kernel_size]; kernel_size];
    let sigma = l as f32 / 2.0;
    for i in 0..kernel_size {
        for j in 0..kernel_size {
            let di = i as f32 - l as f32 + 0.5;
            let dj = j as f32 - l as f32 + 0.5;
            let gauss = (-(di * di + dj * dj) / (2.0 * sigma * sigma)).exp();
            let sign = if (di < 0.0) == (dj < 0.0) { 1.0 } else { -1.0 };
            kernel[i][j] = sign * gauss;
        }
    }

    // Compute novelty[t] = sum over kernel of SSM[t-l..t+l, t-l..t+l] * kernel
    let mut novelty = vec![0.0f32; n];
    for t in l..(n - l) {
        let mut sum = 0.0f32;
        for ki in 0..kernel_size {
            let row_idx = t + ki - l;
            for kj in 0..kernel_size {
                let col_idx = t + kj - l;
                // Inline cosine similarity = dot(normalized[row_idx], normalized[col_idx]).
                let mut dot = 0.0f32;
                let a = &normalized[row_idx];
                let b = &normalized[col_idx];
                for c in 0..N_CHROMA {
                    dot += a[c] * b[c];
                }
                sum += dot * kernel[ki][kj];
            }
        }
        novelty[t] = sum;
    }

    // Normalize the novelty curve to roughly [0..1] for easier thresholding,
    // using a robust max (P95) instead of true max.
    let mut sorted = novelty.iter().copied().filter(|x| *x > 0.0).collect::<Vec<_>>();
    if !sorted.is_empty() {
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95 = sorted[sorted.len() * 95 / 100].max(1e-6);
        for v in novelty.iter_mut() {
            *v = (*v / p95).max(0.0);
        }
    }

    // The novelty curve is sampled at `effective_hop` (not `hop_secs`) due
    // to downsampling. Caller needs that to map back to seconds.
    (novelty, effective_hop)
}

/// Compute a chromagram from mono audio. Returns one 12-vector per hop.
/// Each bin is the linear magnitude sum of all FFT bins falling into that
/// pitch class (with octave folding).
fn chromagram(mono: &[f32], sample_rate: u32) -> Vec<[f32; N_CHROMA]> {
    use rayon::prelude::*;

    let n_slices = (mono.len() - FFT_SIZE) / HOP + 1;
    let window: Arc<Vec<f32>> = Arc::new(hann_window(FFT_SIZE));

    // Precompute which chroma bin each FFT bin maps to.
    // Frequency of bin k at FFT_SIZE = k * sample_rate / FFT_SIZE.
    // MIDI note number from frequency: 69 + 12 * log2(f / 440).
    // Chroma class = midi % 12.
    let n_bins = FFT_SIZE / 2;
    let bin_to_chroma: Vec<Option<usize>> = (0..n_bins)
        .map(|k| {
            let freq = k as f32 * sample_rate as f32 / FFT_SIZE as f32;
            if freq < 80.0 || freq > 8000.0 {
                // Skip sub-bass and very high frequencies — they're either
                // noise floor or harmonics that confuse pitch class.
                None
            } else {
                let midi = 69.0 + 12.0 * (freq / 440.0).log2();
                Some((midi.round() as i32).rem_euclid(12) as usize)
            }
        })
        .collect();

    (0..n_slices)
        .into_par_iter()
        .map(|si| {
            let start = si * HOP;
            let mut buf: Vec<Complex32> = (0..FFT_SIZE)
                .map(|i| Complex32::new(mono[start + i] * window[i], 0.0))
                .collect();
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);
            fft.process(&mut buf);
            let mut chroma = [0.0f32; N_CHROMA];
            for k in 0..n_bins {
                if let Some(c) = bin_to_chroma[k] {
                    let mag = (buf[k].re * buf[k].re + buf[k].im * buf[k].im).sqrt();
                    chroma[c] += mag;
                }
            }
            chroma
        })
        .collect()
}

fn hann_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = (i as f32) / (n as f32 - 1.0);
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * x).cos())
        })
        .collect()
}

/// Compute an onset-strength curve via spectral flux.
///
/// At each FFT hop, sum the positive magnitude differences across all bins.
/// Sharp rises (drum hits, vocal entries, instrument starts) produce big
/// peaks. Used to snap detected zone boundaries to the nearest musical
/// onset within a small search window — section boundaries set by the
/// detector are usually within a couple of seconds of a sharp onset, but
/// the smoothed-RMS-derivative peak isn't always co-located with the
/// onset itself.
///
/// Returns `(onset, hop_secs)` aligned to the same grid as `novelty_curve`.
pub fn onset_strength(mono: &[f32], sample_rate: u32) -> (Vec<f32>, f32) {
    use rayon::prelude::*;
    if mono.len() < FFT_SIZE * 2 || sample_rate == 0 {
        return (Vec::new(), 0.0);
    }
    let hop_secs = HOP as f32 / sample_rate as f32;
    let n_slices = (mono.len() - FFT_SIZE) / HOP + 1;
    let window: Arc<Vec<f32>> = Arc::new(hann_window(FFT_SIZE));
    let n_bins = FFT_SIZE / 2;

    // Compute magnitudes per slice.
    let mags: Vec<Vec<f32>> = (0..n_slices)
        .into_par_iter()
        .map(|si| {
            let start = si * HOP;
            let mut buf: Vec<Complex32> = (0..FFT_SIZE)
                .map(|i| Complex32::new(mono[start + i] * window[i], 0.0))
                .collect();
            let mut planner = FftPlanner::<f32>::new();
            let fft = planner.plan_fft_forward(FFT_SIZE);
            fft.process(&mut buf);
            (0..n_bins)
                .map(|k| (buf[k].re * buf[k].re + buf[k].im * buf[k].im).sqrt())
                .collect()
        })
        .collect();

    // Spectral flux: sum of positive bin-to-bin diffs.
    let mut flux = vec![0.0f32; n_slices];
    for i in 1..n_slices {
        let mut sum = 0.0f32;
        for k in 0..n_bins {
            let d = mags[i][k] - mags[i - 1][k];
            if d > 0.0 {
                sum += d;
            }
        }
        flux[i] = sum;
    }
    // Normalize to [0..1] by P95 for consistent search behavior across files.
    let mut sorted: Vec<f32> = flux.iter().copied().filter(|x| *x > 0.0).collect();
    if !sorted.is_empty() {
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let p95 = sorted[sorted.len() * 95 / 100].max(1e-9);
        for v in flux.iter_mut() {
            *v = (*v / p95).clamp(0.0, 2.0);
        }
    }
    (flux, hop_secs)
}
