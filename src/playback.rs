//! Audio playback via cpal.
//!
//! The processed audio buffer lives in an `Arc<Mutex<Option<PlaySource>>>`.
//! When the user clicks Play, we spin up a cpal output stream whose callback
//! pulls samples from the current source. The playhead position is shared
//! via an `Arc<AtomicU64>` (frame index) so the UI thread can read it cheaply
//! and draw the playhead line.
//!
//! "A/B" toggle is just swapping which buffer (input vs processed) the source
//! points at — mid-playback if you want.

use anyhow::{anyhow, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Stream, StreamConfig};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Which buffer the player should be reading right now.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Input,
    Processed,
}

/// Shared, lock-free state the audio callback reads on every tick.
#[derive(Debug)]
pub struct PlaybackState {
    pub frame: AtomicU64, // current playhead, in source-frame units
    pub playing: AtomicBool,
    pub which: Mutex<Source>,
    /// Peak |amplitude| seen since last UI read (×1e6 for atomic storage).
    /// We track input and output separately so the meter can show pre/post.
    pub peak_in: AtomicU32,
    pub peak_out: AtomicU32,
    /// Monitor the L+R mono sum instead of stereo. A MONITORING control
    /// only — it changes what comes out of the speakers, never the audio
    /// that gets analyzed, rendered or exported.
    pub mono_fold: AtomicBool,
}

impl PlaybackState {
    pub fn new() -> Self {
        Self {
            frame: AtomicU64::new(0),
            playing: AtomicBool::new(false),
            which: Mutex::new(Source::Processed),
            peak_in: AtomicU32::new(0),
            peak_out: AtomicU32::new(0),
            mono_fold: AtomicBool::new(false),
        }
    }
}

/// Holds both buffers (interleaved f32) plus stream metadata.
#[derive(Clone)]
pub struct Buffers {
    pub input: Arc<Vec<f32>>,
    pub processed: Arc<Vec<f32>>,
    pub channels: u16,
    pub sample_rate: u32,
}

/// Owns the live cpal stream. Drop = stop. Kept in App state.
pub struct Player {
    _stream: Stream,
    pub state: Arc<PlaybackState>,
    pub buffers: Arc<Mutex<Option<Buffers>>>,
    pub sample_rate: u32,
}

impl Player {
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default audio output device"))?;
        let supported = device
            .default_output_config()
            .map_err(|e| anyhow!("default output config failed: {e}"))?;
        let sample_rate = supported.sample_rate().0;
        let out_channels = supported.channels();
        let config = StreamConfig {
            channels: out_channels,
            sample_rate: supported.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let state = Arc::new(PlaybackState::new());
        let buffers: Arc<Mutex<Option<Buffers>>> = Arc::new(Mutex::new(None));

        let cb_state = state.clone();
        let cb_buffers = buffers.clone();
        let device_sr = sample_rate;
        let stream = device
            .build_output_stream(
                &config,
                move |out: &mut [f32], _info| {
                    fill_callback(out, out_channels, device_sr, &cb_state, &cb_buffers);
                },
                |err| eprintln!("audio stream error: {err}"),
                None,
            )
            .map_err(|e| anyhow!("build_output_stream: {e}"))?;
        stream.play().map_err(|e| anyhow!("stream.play: {e}"))?;

        Ok(Self {
            _stream: stream,
            state,
            buffers,
            sample_rate,
        })
    }

    pub fn load(
        &self,
        input: Arc<Vec<f32>>,
        processed: Arc<Vec<f32>>,
        channels: u16,
        sample_rate: u32,
    ) {
        *self.buffers.lock().unwrap() = Some(Buffers {
            input,
            processed,
            channels,
            sample_rate,
        });
        self.state.frame.store(0, Ordering::Relaxed);
        self.state.playing.store(false, Ordering::Relaxed);
    }

    /// Replace ONLY the processed buffer (e.g. after the user moved a slider).
    /// Does not touch the playhead — the user can keep listening uninterrupted.
    pub fn update_processed(&self, processed: Arc<Vec<f32>>) {
        if let Some(b) = self.buffers.lock().unwrap().as_mut() {
            b.processed = processed;
        }
    }

    pub fn play(&self) {
        self.state.playing.store(true, Ordering::Relaxed);
    }

    pub fn pause(&self) {
        self.state.playing.store(false, Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.state.playing.store(false, Ordering::Relaxed);
        self.state.frame.store(0, Ordering::Relaxed);
    }

    pub fn seek(&self, frame: u64) {
        self.state.frame.store(frame, Ordering::Relaxed);
    }

    pub fn set_source(&self, src: Source) {
        *self.state.which.lock().unwrap() = src;
    }

    pub fn current_source(&self) -> Source {
        *self.state.which.lock().unwrap()
    }

    /// Toggle mono-fold monitoring. Takes effect on the next audio callback,
    /// so it's safe to flip mid-playback — which is the point: the useful
    /// comparison is A/B'ing stereo against mono on the same passage.
    pub fn set_mono_fold(&self, on: bool) {
        self.state.mono_fold.store(on, Ordering::Relaxed);
    }

    pub fn is_playing(&self) -> bool {
        self.state.playing.load(Ordering::Relaxed)
    }

    /// Read peak |amplitude| since the last call, then reset both meters.
    /// Returns `(peak_input, peak_processed)` as linear amplitudes [0..1+].
    pub fn take_peaks(&self) -> (f32, f32) {
        let pi = self.state.peak_in.swap(0, Ordering::Relaxed) as f32 / 1_000_000.0;
        let po = self.state.peak_out.swap(0, Ordering::Relaxed) as f32 / 1_000_000.0;
        (pi, po)
    }

    /// Current playhead in seconds (uses the loaded buffer's sample rate).
    pub fn position_secs(&self) -> f32 {
        let f = self.state.frame.load(Ordering::Relaxed) as f32;
        let sr = self
            .buffers
            .lock()
            .unwrap()
            .as_ref()
            .map(|b| b.sample_rate)
            .unwrap_or(self.sample_rate);
        f / sr as f32
    }
}

/// Audio thread callback. Pulls source samples and writes them to the cpal
/// output. Handles channel-count mismatch (mono/stereo) by duplication or
/// averaging, and sample-rate mismatch via simple linear interpolation —
/// fine for monitoring; swap in `rubato` later if you need pristine quality.
fn fill_callback(
    out: &mut [f32],
    out_channels: u16,
    device_sr: u32,
    state: &PlaybackState,
    buffers: &Mutex<Option<Buffers>>,
) {
    out.fill(0.0);
    if !state.playing.load(Ordering::Relaxed) {
        return;
    }

    let guard = buffers.lock().unwrap();
    let Some(buf) = guard.as_ref() else { return };
    let which = *state.which.lock().unwrap();
    let src: &[f32] = match which {
        Source::Input => &buf.input,
        Source::Processed => &buf.processed,
    };
    let in_ch = buf.channels.max(1) as usize;
    let in_frames = src.len() / in_ch;
    if in_frames == 0 {
        return;
    }

    // Step the playhead by (file_sr / device_sr) per output frame so we play
    // back at the correct pitch regardless of device rate.
    let ratio = buf.sample_rate as f64 / device_sr as f64;
    let out_frames = out.len() / out_channels.max(1) as usize;
    let mut pos = state.frame.load(Ordering::Relaxed) as f64;

    let mono = state.mono_fold.load(Ordering::Relaxed);

    let mut peak_in_local = 0.0f32;
    let mut peak_out_local = 0.0f32;

    for f in 0..out_frames {
        let i0 = pos.floor() as usize;
        let i1 = i0 + 1;
        if i0 >= in_frames {
            state.playing.store(false, Ordering::Relaxed);
            state.frame.store(in_frames as u64, Ordering::Relaxed);
            // Still publish whatever peaks we collected this callback.
            update_peak(&state.peak_in, peak_in_local);
            update_peak(&state.peak_out, peak_out_local);
            return;
        }
        let frac = (pos - i0 as f64) as f32;

        // Mono fold, or a stereo file going to a mono device, both need the
        // same thing: the average of all input channels. Compute it once
        // per frame rather than once per output channel.
        //
        // The average (not the sum) is the correct fold: content that's
        // already mono keeps its level, and anything out of phase between
        // the channels cancels — which is exactly what a mono PA does, and
        // exactly what this toggle exists to let you hear.
        let downmix_needed = mono || (in_ch > 1 && (out_channels as usize) == 1);
        let folded = if downmix_needed && in_ch > 1 {
            let mut acc = 0.0;
            for c in 0..in_ch {
                let a = src[i0 * in_ch + c];
                let b = if i1 < in_frames { src[i1 * in_ch + c] } else { a };
                acc += a + (b - a) * frac;
            }
            acc / in_ch as f32
        } else {
            0.0
        };

        for oc in 0..out_channels as usize {
            let s_final = if downmix_needed && in_ch > 1 {
                // Same folded signal to every output channel — mono in the
                // centre of the image, not collapsed into the left speaker.
                folded
            } else {
                let ic = if in_ch == 1 { 0 } else { oc.min(in_ch - 1) };
                let s0 = src[i0 * in_ch + ic];
                let s1 = if i1 < in_frames {
                    src[i1 * in_ch + ic]
                } else {
                    s0
                };
                s0 + (s1 - s0) * frac
            };
            out[f * out_channels as usize + oc] = s_final;
        }

        // Peak tracking — sample both buffers at this position for A/B-aware
        // metering. Use channel 0 to keep it cheap.
        let bi = buf.input[i0 * in_ch].abs();
        let bo = buf.processed[i0 * in_ch].abs();
        if bi > peak_in_local {
            peak_in_local = bi;
        }
        if bo > peak_out_local {
            peak_out_local = bo;
        }

        pos += ratio;
    }
    state.frame.store(pos as u64, Ordering::Relaxed);
    update_peak(&state.peak_in, peak_in_local);
    update_peak(&state.peak_out, peak_out_local);
}

/// Atomically max-update a peak value (stored as fixed-point ×1e6).
fn update_peak(slot: &AtomicU32, val: f32) {
    let new = (val * 1_000_000.0).clamp(0.0, u32::MAX as f32) as u32;
    let mut cur = slot.load(Ordering::Relaxed);
    while new > cur {
        match slot.compare_exchange_weak(cur, new, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}
