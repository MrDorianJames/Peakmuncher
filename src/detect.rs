//! Automatic zone detection from dynamic transitions + harmonic structure.
//!
//! Two signals are combined:
//!
//! 1. **Dynamics** — per-window RMS (loudness) and crest factor (peak/RMS).
//!    Smoothed and tracked frame-to-frame; transitions show as peaks in
//!    the rate of change.
//! 2. **Harmonic novelty** (chroma-SSM) — Foote checkerboard novelty on a
//!    chromagram self-similarity matrix. Captures structural transitions
//!    where the chord progression / harmonic content changes, even when
//!    loudness doesn't.
//!
//! Both signals are normalized to [0..1], resampled onto a shared timeline,
//! and combined by taking the per-frame maximum. Peak-find on the combined
//! signal, suppress close-spaced detections, cap total count.
//!
//! Output: a sorted list of split times in seconds.

const WINDOW_MS: f32 = 100.0;
const SMOOTH_WINDOW_S: f32 = 2.0;
const MIN_ZONE_SECS: f32 = 3.0;
/// Hard ceiling on splits — `(duration_secs / SECONDS_PER_SPLIT_CAP)`.
/// Acts as a safety net so the detector never produces an absurd number
/// of splits on noisy material.
const SECONDS_PER_SPLIT_CAP: f32 = 10.0;

/// Detect zone splits.
///
/// `mono`        — mono-mixed samples
/// `sample_rate` — Hz
/// `sensitivity` — 0.0..=1.0. Higher → more splits. 0.55 is the empirical
///                 sweet spot from real-song tuning.
pub fn detect_splits(mono: &[f32], sample_rate: u32, sensitivity: f32) -> Vec<f32> {
    if mono.is_empty() || sample_rate == 0 {
        return Vec::new();
    }
    let duration = mono.len() as f32 / sample_rate as f32;
    if duration < MIN_ZONE_SECS * 2.0 {
        return Vec::new();
    }
    let sensitivity = sensitivity.clamp(0.0, 1.0);

    let win_samples = ((WINDOW_MS / 1000.0) * sample_rate as f32) as usize;
    let win_samples = win_samples.max(64);
    let n_windows = mono.len() / win_samples;
    if n_windows < 4 {
        return Vec::new();
    }

    // === 1) Dynamics signal ===========================================
    let mut rms = Vec::with_capacity(n_windows);
    let mut peak = Vec::with_capacity(n_windows);
    for w in 0..n_windows {
        let start = w * win_samples;
        let end = (start + win_samples).min(mono.len());
        let mut sumsq = 0.0f32;
        let mut p = 0.0f32;
        for &s in &mono[start..end] {
            sumsq += s * s;
            let a = s.abs();
            if a > p {
                p = a;
            }
        }
        let r = (sumsq / (end - start) as f32).sqrt().max(1e-6);
        rms.push(r);
        peak.push(p.max(1e-6));
    }
    let rms_db: Vec<f32> = rms.iter().map(|r| 20.0 * r.log10()).collect();
    let crest_db: Vec<f32> = rms
        .iter()
        .zip(peak.iter())
        .map(|(r, p)| 20.0 * (p / r).log10())
        .collect();
    let smooth_n = ((SMOOTH_WINDOW_S * 1000.0 / WINDOW_MS) as usize).max(1);
    let rms_smoothed = moving_average(&rms_db, smooth_n);
    let crest_smoothed = moving_average(&crest_db, smooth_n);
    let rms_scale = robust_range(&rms_smoothed).max(1e-3);
    let crest_scale = robust_range(&crest_smoothed).max(1e-3);

    let mut dynamics_delta = vec![0.0f32; n_windows];
    for i in 1..n_windows {
        let dr = (rms_smoothed[i] - rms_smoothed[i - 1]) / rms_scale;
        let dc = (crest_smoothed[i] - crest_smoothed[i - 1]) / crest_scale;
        dynamics_delta[i] = (dr * dr + dc * dc).sqrt();
    }
    // Normalize to [0..1] by P95 so it's comparable with chroma novelty.
    normalize_p95(&mut dynamics_delta);

    // === 2) Harmonic novelty (chroma-SSM) ===============================
    let (chroma_novelty, chroma_hop_secs) =
        crate::structure::novelty_curve(mono, sample_rate);
    // Resample chroma novelty onto the dynamics timeline (n_windows points
    // at WINDOW_MS spacing).
    let mut chroma_aligned = vec![0.0f32; n_windows];
    if !chroma_novelty.is_empty() && chroma_hop_secs > 0.0 {
        for i in 0..n_windows {
            let t = i as f32 * (WINDOW_MS / 1000.0);
            let src = (t / chroma_hop_secs) as usize;
            if src < chroma_novelty.len() {
                chroma_aligned[i] = chroma_novelty[src];
            }
        }
    }

    // === 3) Combine: per-frame max =======================================
    // Max (rather than sum) keeps each signal's confidence interpretable —
    // a strong dynamic transition with weak chroma still scores high, and
    // vice versa, instead of being attenuated by averaging.
    let mut combined = vec![0.0f32; n_windows];
    for i in 0..n_windows {
        combined[i] = dynamics_delta[i].max(chroma_aligned[i]);
    }

    // === 4) Threshold + peak-find ========================================
    // Sensitivity curve is non-linear: shallow at low end, steeper at high
    // end. This makes 55% give meaningfully fewer results than 100% (the
    // sweet spot was empirically near the middle of the slider).
    //   sensitivity 0.0 → multiplier 6.0 (very strict, few splits)
    //   sensitivity 0.55 → multiplier ~2.7
    //   sensitivity 1.0 → multiplier 1.2 (loose, many splits)
    let s_curve = sensitivity.powf(1.5);
    let threshold_mult = 6.0 - s_curve * 4.8;

    let mut sorted: Vec<f32> = combined.iter().copied().filter(|x| *x > 0.0).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.is_empty() {
        return Vec::new();
    }
    let median = sorted[sorted.len() / 2].max(1e-6);
    let threshold = median * threshold_mult;

    let neighborhood = (smooth_n / 2).max(2);
    let mut candidates: Vec<(usize, f32)> = Vec::new();
    for i in neighborhood..(n_windows - neighborhood) {
        let v = combined[i];
        if v < threshold {
            continue;
        }
        let lo = i.saturating_sub(neighborhood);
        let hi = (i + neighborhood + 1).min(n_windows);
        let is_peak = combined[lo..hi].iter().all(|&x| x <= v);
        if is_peak {
            candidates.push((i, v));
        }
    }

    // Sort candidates by strength, then greedily accept them, skipping any
    // that fall too close to an already-accepted split.
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let min_zone_windows = (MIN_ZONE_SECS * 1000.0 / WINDOW_MS) as usize;

    // Hard cap on total splits: never produce more than `duration / 8s`.
    let max_splits = ((duration / SECONDS_PER_SPLIT_CAP) as usize).max(1);

    let mut accepted: Vec<usize> = Vec::new();
    for (idx, _) in candidates {
        if accepted.len() >= max_splits {
            break;
        }
        let too_close = accepted
            .iter()
            .any(|&a| idx.abs_diff(a) < min_zone_windows);
        let near_edge = idx < min_zone_windows || idx + min_zone_windows > n_windows;
        if !too_close && !near_edge {
            accepted.push(idx);
        }
    }
    accepted.sort_unstable();

    let secs_per_window = WINDOW_MS / 1000.0;
    let raw_splits: Vec<f32> = accepted
        .into_iter()
        .map(|i| (i as f32 + 0.5) * secs_per_window)
        .filter(|&t| t > 0.1 && t < duration - 0.1)
        .collect();

    // Snap each detected split to the nearest strong onset within ±1.5s.
    // The smoothed-RMS-derivative peak is often a couple of seconds away
    // from the actual transient that *defines* the section start. Snapping
    // pulls splits onto the musical event itself.
    let (onset, onset_hop) = crate::structure::onset_strength(mono, sample_rate);
    if onset.is_empty() || onset_hop <= 0.0 {
        return raw_splits;
    }
    const SNAP_WINDOW_S: f32 = 1.5;
    /// A snap is only applied if the candidate onset is at least this strong.
    /// Below this, the surrounding signal is too noisy to be a clear musical
    /// event and we leave the original detector position alone.
    const ONSET_MIN_STRENGTH: f32 = 0.20;
    let snap_radius = (SNAP_WINDOW_S / onset_hop) as usize;

    raw_splits
        .into_iter()
        .map(|t| {
            let center = (t / onset_hop) as usize;
            let lo = center.saturating_sub(snap_radius);
            let hi = (center + snap_radius + 1).min(onset.len());
            if lo >= hi {
                return t;
            }
            // Find the strongest onset in the window.
            let (best_i, best_v) = onset[lo..hi]
                .iter()
                .enumerate()
                .map(|(i, v)| (lo + i, *v))
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((center, 0.0));
            if best_v < ONSET_MIN_STRENGTH {
                t // No clear onset to snap to.
            } else {
                best_i as f32 * onset_hop
            }
        })
        .collect()
}

/// Centered moving average. `n` is the window size (will be clamped to odd).
fn moving_average(input: &[f32], n: usize) -> Vec<f32> {
    if input.is_empty() {
        return Vec::new();
    }
    let n = if n % 2 == 0 { n + 1 } else { n }.max(1);
    let half = n / 2;
    let mut out = Vec::with_capacity(input.len());
    for i in 0..input.len() {
        let lo = i.saturating_sub(half);
        let hi = (i + half + 1).min(input.len());
        let slice = &input[lo..hi];
        let avg = slice.iter().sum::<f32>() / slice.len() as f32;
        out.push(avg);
    }
    out
}

/// Robust spread estimator: P90 - P10. Less sensitive to outliers than
/// max - min.
fn robust_range(input: &[f32]) -> f32 {
    if input.len() < 2 {
        return 0.0;
    }
    let mut sorted = input.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p10 = sorted[sorted.len() / 10];
    let p90 = sorted[sorted.len() * 9 / 10];
    (p90 - p10).abs()
}

/// In-place normalize to [0..1] using P95 as the upper reference.
fn normalize_p95(buf: &mut [f32]) {
    let mut sorted: Vec<f32> = buf.iter().copied().filter(|x| x.is_finite()).collect();
    if sorted.is_empty() {
        return;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p95 = sorted[sorted.len() * 95 / 100].max(1e-9);
    for v in buf.iter_mut() {
        *v = (*v / p95).clamp(0.0, 2.0);
    }
}
