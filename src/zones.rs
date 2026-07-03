//! Zone-based automation.
//!
//! The audio file is divided by user-placed split points into N+1 zones.
//! Each zone gets its own ceiling (in dB), input-gain (dB), output-gain (dB),
//! and clipper type. Inside a zone, parameters are constant — no ramps —
//! producing the "stepped" horizontal-line look the user asked for.

use crate::dsp::{self, ClipperType};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ZoneParams {
    pub ceiling_db: f32,
    pub input_gain_db: f32,
    pub output_gain_db: f32,
    pub clipper: ClipperType,
    /// Oversampling factor used when *rendering to file*: 1 (off), 2, 4, or 8.
    /// Live preview never oversamples (would be too slow on slider drags).
    #[serde(default = "default_oversampling")]
    pub oversampling: u8,
    /// DC offset correction applied per-zone, in linear amplitude (range
    /// -0.5 to +0.5). Subtracted from every sample BEFORE input gain, so
    /// a recording with +0.02 DC bias becomes correctly centered when the
    /// user sets dc_offset to -0.02 (or hits Auto-detect). Defaults to 0
    /// for back-compat with existing saved projects.
    #[serde(default)]
    pub dc_offset: f32,
    /// Per-zone fade-in length in seconds (0 = no fade). Linear ramp from
    /// silence to full level starting at the zone's left boundary. Capped
    /// at the zone's own length — fades never cross a zone boundary, so
    /// the rest of the file is preserved from being affected.
    #[serde(default)]
    pub fade_in_secs: f32,
    /// Per-zone fade-out length in seconds (0 = no fade). Linear ramp from
    /// full level to silence ending at the zone's right boundary.
    #[serde(default)]
    pub fade_out_secs: f32,
    /// Per-zone DC blocker enable. A one-pole high-pass filter that
    /// continuously removes any DC bias and slow drift below the cutoff.
    /// Useful when the audio's DC offset wanders over time (the constant
    /// `dc_offset` field can only correct uniform bias).
    #[serde(default)]
    pub dc_blocker_enabled: bool,
    /// Cutoff frequency for the DC blocker in Hz. Default 20 Hz — below
    /// audible bass content for most music but well above DC. Range
    /// 5-60 Hz; higher values chase down drift faster but cut more
    /// sub-bass.
    #[serde(default = "default_dc_blocker_hz")]
    pub dc_blocker_hz: f32,
}

fn default_dc_blocker_hz() -> f32 {
    20.0
}

fn default_oversampling() -> u8 {
    1
}

impl Default for ZoneParams {
    fn default() -> Self {
        Self {
            // Default to TRUE IDENTITY: Hard clipper at 0 dB ceiling
            // passes any normal audio sample unchanged (clamp -1..1 on
            // a sample already within ±1.0 returns it as-is). This
            // means a freshly-loaded file and a freshly-Applied state
            // both show their input unaltered until the user chooses
            // to process. Pick Tangent + lower ceiling to engage soft
            // clipping.
            ceiling_db: 0.0,
            input_gain_db: 0.0,
            output_gain_db: 0.0,
            clipper: ClipperType::Hard,
            oversampling: 1,
            dc_offset: 0.0,
            fade_in_secs: 0.0,
            fade_out_secs: 0.0,
            dc_blocker_enabled: false,
            dc_blocker_hz: 20.0,
        }
    }
}

/// Sorted list of split points (in seconds, strictly between 0 and duration)
/// plus per-zone parameters (length = splits.len() + 1).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZoneMap {
    pub splits: Vec<f32>,
    pub zones: Vec<ZoneParams>,
}

impl ZoneMap {
    pub fn new() -> Self {
        Self {
            splits: Vec::new(),
            zones: vec![ZoneParams::default()],
        }
    }

    /// Insert a split, copying the params of the zone being split.
    pub fn add_split(&mut self, t_secs: f32) {
        // Find which existing zone this falls in; insert keeps splits sorted.
        let idx = self.splits.partition_point(|&s| s < t_secs);
        // Refuse near-duplicates.
        if let Some(&prev) = self.splits.get(idx.wrapping_sub(1)) {
            if (prev - t_secs).abs() < 1e-3 {
                return;
            }
        }
        if let Some(&next) = self.splits.get(idx) {
            if (next - t_secs).abs() < 1e-3 {
                return;
            }
        }
        self.splits.insert(idx, t_secs);
        let new_zone = self.zones[idx]; // duplicate the zone we just split
        self.zones.insert(idx + 1, new_zone);
    }

    pub fn remove_split(&mut self, idx: usize) {
        if idx >= self.splits.len() {
            return;
        }
        self.splits.remove(idx);
        // Merge: keep the left zone's params, drop the right.
        self.zones.remove(idx + 1);
    }

    /// Which zone is sample-time `t` in?
    pub fn zone_at(&self, t_secs: f32) -> usize {
        self.splits.partition_point(|&s| s <= t_secs)
    }

    /// Walk through all zones with their (start, end) times in seconds.
    pub fn iter_zones<'a>(
        &'a self,
        duration: f32,
    ) -> impl Iterator<Item = (f32, f32, &'a ZoneParams)> + 'a {
        let mut starts = Vec::with_capacity(self.zones.len());
        starts.push(0.0);
        starts.extend(self.splits.iter().copied());
        let mut ends: Vec<f32> = self.splits.iter().copied().collect();
        ends.push(duration);
        starts
            .into_iter()
            .zip(ends.into_iter())
            .zip(self.zones.iter())
            .map(|((s, e), z)| (s, e, z))
    }
}

/// Apply all zones' processing to a buffer, returning the new samples.
/// Per-sample: pick the zone for this frame's time, apply input gain →
/// waveshape against the zone ceiling → apply output gain.
pub fn render(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    zones: &ZoneMap,
    // Trim window in FRAME indices [start, end). Fades anchor to the trim
    // edges as if trim were a zone boundary: the fade-in of a zone begins at
    // max(zone_start, trim_start) and the fade-out ends at min(zone_end,
    // trim_end). None = no trim (fades anchor to the raw zone edges).
    trim_window: Option<(usize, usize)>,
) -> Vec<f32> {
    use rayon::prelude::*;

    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let duration = frames as f32 / sample_rate.max(1) as f32;
    let (trim_s, trim_e) = trim_window.unwrap_or((0, frames));

    // Precompute zone frame ranges + gain/ceiling values.
    struct Z {
        start: usize,
        end: usize,
        in_gain: f32,
        out_gain: f32,
        ceiling: f32,
        clipper: ClipperType,
        dc_offset: f32,
        fade_in_frames: usize,
        fade_out_frames: usize,
        /// Zone-relative frame where the fade-in begins, = max(zone_start,
        /// trim_start) - zone_start. Normally 0; nonzero only for the zone
        /// containing the trim start.
        fade_in_anchor: usize,
        /// Zone-relative frame where the fade-out must reach zero, =
        /// min(zone_end, trim_end) - zone_start. Normally zone_frames;
        /// smaller only for the zone containing the trim end.
        fade_out_end: usize,
        /// DC blocker enabled for this zone.
        dc_blocker: bool,
        /// Pole coefficient R for one-pole HP. R = exp(-2π·fc/fs).
        /// 0 disables (mathematically becomes a fixed-zero filter — no
        /// memory — which is wrong, so we only use R when dc_blocker
        /// is true).
        dc_blocker_r: f32,
    }
    let zlist: Vec<Z> = zones
        .iter_zones(duration)
        .map(|(s, e, z)| {
            let start = ((s * sample_rate as f32) as usize).min(frames);
            let end = ((e * sample_rate as f32) as usize).min(frames);
            let zone_frames = end.saturating_sub(start);
            // Cap each fade at half the zone so they can't overlap weirdly
            // when the zone is short and both fades are requested.
            let max_per_fade_frames = zone_frames / 2;
            let fi = (z.fade_in_secs.max(0.0) * sample_rate as f32) as usize;
            let fo = (z.fade_out_secs.max(0.0) * sample_rate as f32) as usize;
            // Trim acts as a zone boundary for fades: the effective region of
            // this zone is the zone clamped to the trim window. The fade-in
            // begins at the effective start, the fade-out ends at the
            // effective end.
            let eff_start = start.max(trim_s).min(end);
            let eff_end = end.min(trim_e).max(start);
            let fade_in_anchor = eff_start.saturating_sub(start);
            let fade_out_end = eff_end.saturating_sub(start);
            Z {
                start,
                end,
                in_gain: dsp::db_to_amp(z.input_gain_db),
                out_gain: dsp::db_to_amp(z.output_gain_db),
                ceiling: dsp::db_to_amp(z.ceiling_db.min(0.0)),
                clipper: z.clipper,
                dc_offset: z.dc_offset,
                fade_in_frames: fi.min(max_per_fade_frames),
                fade_out_frames: fo.min(max_per_fade_frames),
                fade_in_anchor,
                fade_out_end,
                dc_blocker: z.dc_blocker_enabled,
                dc_blocker_r: {
                    // R = exp(-2π·fc/fs). Standard one-pole HP coeff.
                    let fs = sample_rate.max(1) as f32;
                    let fc = z.dc_blocker_hz.clamp(1.0, 200.0);
                    (-2.0 * std::f32::consts::PI * fc / fs).exp()
                },
            }
        })
        .collect();

    // Allocate output, then split into per-zone mutable slices and process
    // zones in parallel.
    let mut out = vec![0.0f32; samples.len()];
    {
        let mut chunks: Vec<&mut [f32]> = Vec::with_capacity(zlist.len());
        let mut rest = &mut out[..];
        for z in &zlist {
            let len = (z.end - z.start) * ch;
            let (this, tail) = rest.split_at_mut(len);
            chunks.push(this);
            rest = tail;
        }
        let in_slices: Vec<&[f32]> = zlist
            .iter()
            .map(|z| &samples[z.start * ch..z.end * ch])
            .collect();

        chunks
            .into_par_iter()
            .zip(in_slices.into_par_iter())
            .zip(zlist.par_iter())
            .for_each(|((out_slice, in_slice), z)| {
                let _zone_frames = (z.end - z.start) as usize;
                let fi = z.fade_in_frames;
                let fo = z.fade_out_frames;
                // Fades anchor to the trim-clamped region (trim acts as a
                // zone boundary): fade-in begins at fade_in_anchor, fade-out
                // reaches zero at fade_out_end.
                let fade_in_start = z.fade_in_anchor;
                let fo_start_frame = z.fade_out_end.saturating_sub(fo);

                // The per-sample stateless path (DC offset → input gain →
                // clipper → output gain → fade) has no cross-sample
                // dependency, so when the DC blocker is OFF we split the
                // zone into chunks and process them across cores. This is
                // the key win for long single-zone files, which otherwise
                // run the whole file on ONE core (zone-level parallelism
                // gives nothing when there's one zone). The DC blocker is a
                // stateful one-pole IIR (y[n] depends on y[n-1]) that can't
                // be naively chunked, so that case keeps the serial path.
                if z.dc_blocker {
                    let mut dc_x_prev = vec![0.0f32; ch];
                    let mut dc_y_prev = vec![0.0f32; ch];
                    for i in 0..in_slice.len() {
                        let mut s = in_slice[i] + z.dc_offset;
                        let c = i % ch;
                        let y = s - dc_x_prev[c] + z.dc_blocker_r * dc_y_prev[c];
                        dc_x_prev[c] = s;
                        dc_y_prev[c] = y;
                        s = y;
                        s *= z.in_gain;
                        let shaped = dsp::shape(s, z.ceiling, z.clipper);
                        let mut v = shaped * z.out_gain;
                        if fi > 0 || fo > 0 {
                            let frame_in_zone = i / ch;
                            let mut g: f32 = 1.0;
                            // Fade-in ramps from fade_in_start over fi frames.
                            // Audio before fade_in_start is left at full gain
                            // (it's the trimmed-off lead-in, cut on export).
                            if fi > 0 && frame_in_zone >= fade_in_start {
                                let into = frame_in_zone - fade_in_start;
                                if into < fi {
                                    g *= into as f32 / fi as f32;
                                }
                            }
                            // Fade-out ramps to zero, reaching 0 at
                            // fade_out_end. Only WITHIN the fade region —
                            // frames past fade_out_end are the trimmed-off
                            // tail and stay at full gain (cut on export, but
                            // shown normally in the editor, like the pre-
                            // start lead-in).
                            if fo > 0
                                && frame_in_zone >= fo_start_frame
                                && frame_in_zone < z.fade_out_end
                            {
                                let from_end =
                                    z.fade_out_end.saturating_sub(frame_in_zone);
                                g *= (from_end as f32 / fo as f32).min(1.0);
                            }
                            v *= g;
                        }
                        out_slice[i] = v;
                    }
                } else {
                    // Stateless — parallelize across interleaved-frame
                    // chunks. Chunk on frame boundaries (multiples of ch)
                    // so channel indexing and fade frame math stay correct.
                    // ~64k frames/chunk balances parallelism vs overhead.
                    const CHUNK_FRAMES: usize = 65536;
                    let chunk_samples = CHUNK_FRAMES * ch;
                    out_slice
                        .par_chunks_mut(chunk_samples)
                        .enumerate()
                        .for_each(|(ci, out_chunk)| {
                            let base = ci * chunk_samples;
                            for j in 0..out_chunk.len() {
                                let i = base + j;
                                // DC offset → input gain → clipper → output
                                // gain → fade. Frame index is global to the
                                // zone for correct fade ramps.
                                let mut s = in_slice[i] + z.dc_offset;
                                s *= z.in_gain;
                                let shaped = dsp::shape(s, z.ceiling, z.clipper);
                                let mut v = shaped * z.out_gain;
                                if fi > 0 || fo > 0 {
                                    let frame_in_zone = i / ch;
                                    let mut g: f32 = 1.0;
                                    if fi > 0 && frame_in_zone >= fade_in_start {
                                        let into = frame_in_zone - fade_in_start;
                                        if into < fi {
                                            g *= into as f32 / fi as f32;
                                        }
                                    }
                                    if fo > 0
                                        && frame_in_zone >= fo_start_frame
                                        && frame_in_zone < z.fade_out_end
                                    {
                                        let from_end = z
                                            .fade_out_end
                                            .saturating_sub(frame_in_zone);
                                        g *= (from_end as f32 / fo as f32).min(1.0);
                                    }
                                    v *= g;
                                }
                                out_chunk[j] = v;
                            }
                        });
                }
            });
    }
    out
}

/// Fast preview path: compute the OUTPUT display envelope directly, sampling
/// only every `decim`th frame, without materializing the full processed or
/// mono buffers. Returns `(envelope, norm_gain)` — the gain is so the caller
/// can scale the input envelope to match (preview push/pull). Used during
/// slider drags; a full `render` follows on release.
///
/// Approximations (acceptable because preview snaps to the exact full render
/// on release): the DC blocker (a stateful IIR that can't be decimated) is
/// skipped, and normalize uses a simple peak scalar measured on the
/// decimated output rather than full LUFS.
#[allow(clippy::too_many_arguments)]
pub fn render_envelope_decimated(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    zones: &ZoneMap,
    normalize: bool,
    normalize_mode: crate::NormalizeMode,
    normalize_target_db: f32,
    trim_window: Option<(usize, usize)>,
    width: usize,
    decim: usize,
) -> (Vec<(f32, f32)>, f32) {
    use rayon::prelude::*;

    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    if frames == 0 || width == 0 {
        return (Vec::new(), 1.0);
    }
    let duration = frames as f32 / sample_rate.max(1) as f32;
    let decim = decim.max(1);

    // Per-zone params as frame ranges (mirrors `render`, minus DC blocker).
    struct ZD {
        start: usize,
        end: usize,
        in_gain: f32,
        out_gain: f32,
        ceiling: f32,
        clipper: ClipperType,
        dc_offset: f32,
        fade_in_frames: usize,
        fade_out_frames: usize,
        fade_in_anchor: usize,
        fade_out_end: usize,
    }
    let (trim_s, trim_e) = trim_window.unwrap_or((0, frames));
    let zlist: Vec<ZD> = zones
        .iter_zones(duration)
        .map(|(s, e, z)| {
            let start = ((s * sample_rate as f32) as usize).min(frames);
            let end = ((e * sample_rate as f32) as usize).min(frames);
            let zone_frames = end.saturating_sub(start);
            let max_per_fade_frames = zone_frames / 2;
            let fi = (z.fade_in_secs.max(0.0) * sample_rate as f32) as usize;
            let fo = (z.fade_out_secs.max(0.0) * sample_rate as f32) as usize;
            let eff_start = start.max(trim_s).min(end);
            let eff_end = end.min(trim_e).max(start);
            ZD {
                start,
                end,
                in_gain: dsp::db_to_amp(z.input_gain_db),
                out_gain: dsp::db_to_amp(z.output_gain_db),
                ceiling: dsp::db_to_amp(z.ceiling_db.min(0.0)),
                clipper: z.clipper,
                dc_offset: z.dc_offset,
                fade_in_frames: fi.min(max_per_fade_frames),
                fade_out_frames: fo.min(max_per_fade_frames),
                fade_in_anchor: eff_start.saturating_sub(start),
                fade_out_end: eff_end.saturating_sub(start),
            }
        })
        .collect();

    // Resolve which zone owns a given frame (zones are contiguous, sorted).
    let zone_for = |f: usize| -> Option<&ZD> {
        zlist.iter().find(|z| f >= z.start && f < z.end)
    };

    // Compute the processed MONO value at frame f (decimated; no DC blocker).
    let mono_out_at = |f: usize| -> f32 {
        let Some(z) = zone_for(f) else { return 0.0 };
        let base = f * ch;
        // Mix channels to mono first, then run the per-sample chain once on
        // the mono value (preview-grade; the full render processes per
        // channel, but for a mono display envelope this is visually fine).
        let mut acc = 0.0;
        for c in 0..ch {
            acc += samples[base + c];
        }
        let mut s = acc / ch as f32 + z.dc_offset;
        s *= z.in_gain;
        let shaped = dsp::shape(s, z.ceiling, z.clipper);
        let mut v = shaped * z.out_gain;
        let fi = z.fade_in_frames;
        let fo = z.fade_out_frames;
        if fi > 0 || fo > 0 {
            let frame_in_zone = f - z.start;
            let fo_start = z.fade_out_end.saturating_sub(fo);
            let mut g = 1.0f32;
            if fi > 0 && frame_in_zone >= z.fade_in_anchor {
                let into = frame_in_zone - z.fade_in_anchor;
                if into < fi {
                    g *= into as f32 / fi as f32;
                }
            }
            if fo > 0 && frame_in_zone >= fo_start && frame_in_zone < z.fade_out_end {
                let from_end = z.fade_out_end.saturating_sub(frame_in_zone);
                g *= (from_end as f32 / fo as f32).min(1.0);
            }
            v *= g;
        }
        v
    };

    // Build the envelope in parallel over buckets, sampling every `decim`th
    // frame. Each bucket tracks min/max of the decimated mono output.
    let bucket = (frames as f32 / width as f32).max(1.0);
    let raw_env: Vec<(f32, f32)> = (0..width)
        .into_par_iter()
        .map(|i| {
            let start_f = (i as f32 * bucket) as usize;
            let end_f = (((i as f32 + 1.0) * bucket) as usize)
                .min(frames)
                .max(start_f + 1);
            let mut lo = f32::INFINITY;
            let mut hi = f32::NEG_INFINITY;
            let mut f = start_f;
            while f < end_f {
                let v = mono_out_at(f);
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
                f += decim;
            }
            if !lo.is_finite() {
                let v = mono_out_at(start_f);
                lo = v;
                hi = v;
            }
            (lo, hi)
        })
        .collect();

    // Normalize gain (preview-grade). Peak mode: scale so the decimated
    // peak hits the target. LUFS mode: skip in preview (gain 1.0) — the
    // release render computes it exactly.
    let norm_gain = if normalize {
        match normalize_mode {
            crate::NormalizeMode::Peak => {
                // Peak over the measurement window (or whole file), measured
                // on the decimated envelope extremes.
                let (lo_f, hi_f) = match trim_window {
                    Some((f0, f1)) => {
                        let b0 = ((f0 as f32 / bucket) as usize).min(width);
                        let b1 = ((f1 as f32 / bucket) as usize).min(width).max(b0 + 1);
                        (b0, b1)
                    }
                    None => (0, width),
                };
                let mut peak = 0.0f32;
                for b in lo_f..hi_f.min(raw_env.len()) {
                    peak = peak.max(raw_env[b].0.abs()).max(raw_env[b].1.abs());
                }
                if peak < 1e-6 {
                    1.0
                } else {
                    dsp::db_to_amp(normalize_target_db) / peak
                }
            }
            crate::NormalizeMode::Lufs => 1.0,
        }
    } else {
        1.0
    };

    // Apply the gain and clamp.
    let envelope: Vec<(f32, f32)> = raw_env
        .into_iter()
        .map(|(lo, hi)| {
            (
                (lo * norm_gain).clamp(-1.0, 1.0),
                (hi * norm_gain).clamp(-1.0, 1.0),
            )
        })
        .collect();

    (envelope, norm_gain)
}

/// Like `render`, but applies per-zone oversampling for higher-quality
/// clipping. Used at file-save time, where we can afford the CPU.
///
/// For each zone whose `oversampling > 1`, the audio chunk is upsampled
/// (separately per channel via `Oversampler::process_block`), the clipper
/// runs at the high rate inside `process_block`, then the result is
/// downsampled back to the original rate. Zones with `oversampling == 1`
/// fall through to the same fast path as `render`.
pub fn render_with_oversampling(
    samples: &[f32],
    channels: u16,
    sample_rate: u32,
    zones: &ZoneMap,
    trim_window: Option<(usize, usize)>,
) -> Vec<f32> {
    use rayon::prelude::*;
    use crate::oversample::Oversampler;

    let ch = channels.max(1) as usize;
    let frames = samples.len() / ch;
    let duration = frames as f32 / sample_rate.max(1) as f32;
    let (trim_s, trim_e) = trim_window.unwrap_or((0, frames));

    struct Z {
        start: usize,
        end: usize,
        in_gain: f32,
        out_gain: f32,
        ceiling: f32,
        clipper: ClipperType,
        oversampling: u8,
        dc_offset: f32,
        fade_in_frames: usize,
        fade_out_frames: usize,
        fade_in_anchor: usize,
        fade_out_end: usize,
        dc_blocker: bool,
        dc_blocker_r: f32,
    }
    let zlist: Vec<Z> = zones
        .iter_zones(duration)
        .map(|(s, e, z)| {
            let start = ((s * sample_rate as f32) as usize).min(frames);
            let end = ((e * sample_rate as f32) as usize).min(frames);
            let zone_frames = end.saturating_sub(start);
            let max_per_fade_frames = zone_frames / 2;
            let fi = (z.fade_in_secs.max(0.0) * sample_rate as f32) as usize;
            let fo = (z.fade_out_secs.max(0.0) * sample_rate as f32) as usize;
            let eff_start = start.max(trim_s).min(end);
            let eff_end = end.min(trim_e).max(start);
            let fade_in_anchor = eff_start.saturating_sub(start);
            let fade_out_end = eff_end.saturating_sub(start);
            Z {
                start,
                end,
                in_gain: dsp::db_to_amp(z.input_gain_db),
                out_gain: dsp::db_to_amp(z.output_gain_db),
                ceiling: dsp::db_to_amp(z.ceiling_db.min(0.0)),
                clipper: z.clipper,
                oversampling: z.oversampling.max(1),
                dc_offset: z.dc_offset,
                fade_in_frames: fi.min(max_per_fade_frames),
                fade_out_frames: fo.min(max_per_fade_frames),
                fade_in_anchor,
                fade_out_end,
                dc_blocker: z.dc_blocker_enabled,
                dc_blocker_r: {
                    let fs = sample_rate.max(1) as f32;
                    let fc = z.dc_blocker_hz.clamp(1.0, 200.0);
                    (-2.0 * std::f32::consts::PI * fc / fs).exp()
                },
            }
        })
        .collect();

    let mut out = vec![0.0f32; samples.len()];
    let mut chunks: Vec<&mut [f32]> = Vec::with_capacity(zlist.len());
    {
        let mut rest = &mut out[..];
        for z in &zlist {
            let len = (z.end - z.start) * ch;
            let (this, tail) = rest.split_at_mut(len);
            chunks.push(this);
            rest = tail;
        }
    }
    let in_slices: Vec<&[f32]> = zlist
        .iter()
        .map(|z| &samples[z.start * ch..z.end * ch])
        .collect();

    chunks
        .into_par_iter()
        .zip(in_slices.into_par_iter())
        .zip(zlist.par_iter())
        .for_each(|((out_slice, in_slice), z)| {
            let _zone_frames = (z.end - z.start) as usize;
            let fi = z.fade_in_frames;
            let fo = z.fade_out_frames;
            let fade_in_start = z.fade_in_anchor;
            let fo_start_frame = z.fade_out_end.saturating_sub(fo);
            // Helper to compute per-frame fade gain. Fades anchor to the
            // trim-clamped region (trim acts as a zone boundary).
            let fade_gain = |frame_in_zone: usize| -> f32 {
                let mut g: f32 = 1.0;
                if fi > 0 && frame_in_zone >= fade_in_start {
                    let into = frame_in_zone - fade_in_start;
                    if into < fi {
                        g *= into as f32 / fi as f32;
                    }
                }
                if fo > 0 && frame_in_zone >= fo_start_frame && frame_in_zone < z.fade_out_end {
                    let from_end = z.fade_out_end.saturating_sub(frame_in_zone);
                    g *= (from_end as f32 / fo as f32).min(1.0);
                }
                g
            };

            if z.oversampling == 1 {
                // Fast path: same as `render`. Per-channel DC blocker state.
                let mut dc_x_prev = vec![0.0f32; ch];
                let mut dc_y_prev = vec![0.0f32; ch];
                for i in 0..in_slice.len() {
                    let mut s = in_slice[i] + z.dc_offset;
                    if z.dc_blocker {
                        let c = i % ch;
                        let y = s - dc_x_prev[c] + z.dc_blocker_r * dc_y_prev[c];
                        dc_x_prev[c] = s;
                        dc_y_prev[c] = y;
                        s = y;
                    }
                    s *= z.in_gain;
                    let shaped = dsp::shape(s, z.ceiling, z.clipper);
                    let mut v = shaped * z.out_gain;
                    if fi > 0 || fo > 0 {
                        v *= fade_gain(i / ch);
                    }
                    out_slice[i] = v;
                }
                return;
            }
            // Oversampled path: process each channel independently. The
            // input/output per channel is interleaved with stride `ch`.
            let os = Oversampler::new(z.oversampling as usize);
            for c in 0..ch {
                let nframes = in_slice.len() / ch;
                let mut mono_in = Vec::with_capacity(nframes);
                // DC blocker runs at native rate, before upsampling. The
                // upsampler's FIR LPF passes DC (DC is at 0 Hz) so doing
                // it post-upsample would be wasteful.
                let mut dc_x_prev = 0.0f32;
                let mut dc_y_prev = 0.0f32;
                for n in 0..nframes {
                    let mut s = in_slice[n * ch + c] + z.dc_offset;
                    if z.dc_blocker {
                        let y = s - dc_x_prev + z.dc_blocker_r * dc_y_prev;
                        dc_x_prev = s;
                        dc_y_prev = y;
                        s = y;
                    }
                    mono_in.push(s);
                }
                let in_gain = z.in_gain;
                let out_gain = z.out_gain;
                let ceiling = z.ceiling;
                let clipper = z.clipper;
                let mono_out = os.process_block(&mono_in, |s| {
                    let s = s * in_gain;
                    let shaped = dsp::shape(s, ceiling, clipper);
                    shaped * out_gain
                });
                for n in 0..nframes {
                    // Fade applied AFTER oversampling so it operates on the
                    // final per-channel samples at original sample rate.
                    let g = if fi > 0 || fo > 0 { fade_gain(n) } else { 1.0 };
                    out_slice[n * ch + c] = mono_out[n] * g;
                }
            }
        });
    out
}
