//! Oversampling for clipping.
//!
//! Why oversample? Hard clipping creates harmonics far above the original
//! signal's spectrum. At a normal sample rate (44.1 / 48 kHz), those harmonics
//! exceed Nyquist and alias back into the audible band as nasty digital grit.
//! Working at a higher rate gives the clipper room to produce harmonics
//! without aliasing, and the brick-wall low-pass during downsampling removes
//! whatever lands above the original Nyquist.
//!
//! Technique: zero-stuff upsample → FIR low-pass → clip at high rate →
//! FIR low-pass → decimate. The two FIR passes share the same kernel
//! (a windowed-sinc with Kaiser window), implemented as polyphase to avoid
//! computing samples that get thrown away.
//!
//! NOTE on UX: oversampling is heavy (FIR convolution on a long block).
//! PeakMuncher applies it only when **saving** the processed file — the
//! interactive preview/live editing shows the un-oversampled clipping so
//! sliders stay responsive. The user picks the OS factor per zone; on Save,
//! `process_block` runs over each zone with its chosen factor.

/// Build a windowed-sinc low-pass filter kernel.
///
/// `taps`     — total number of taps. More taps = sharper transition but
///              higher CPU and longer latency. 64 is a good balance.
/// `cutoff`   — cutoff as a fraction of the *high-rate* sample rate
///              (e.g. for 4x oversampling and a 22.05 kHz cutoff,
///              cutoff = 22050 / (4 * 44100) = 0.125).
/// `beta`     — Kaiser window beta. ~8.0 ≈ ~80 dB stopband attenuation.
///
/// Returns coefficients normalized so the DC gain is 1.0.
pub fn kaiser_lowpass(taps: usize, cutoff: f32, beta: f32) -> Vec<f32> {
    let n = taps as i32;
    let mid = (n - 1) as f32 / 2.0;
    let mut h = Vec::with_capacity(taps);
    let i0_beta = bessel_i0(beta);
    for k in 0..n {
        let x = k as f32 - mid;
        // Sinc, with the cutoff scaled by 2 because sinc(0) = 1 corresponds
        // to a Nyquist-frequency low-pass; we want cutoff as a fraction of Fs.
        let sinc = if x.abs() < 1e-9 {
            2.0 * cutoff
        } else {
            let arg = 2.0 * std::f32::consts::PI * cutoff * x;
            arg.sin() / (std::f32::consts::PI * x)
        };
        // Kaiser window
        let w_arg = 2.0 * (k as f32) / (n as f32 - 1.0) - 1.0;
        let w = bessel_i0(beta * (1.0 - w_arg * w_arg).sqrt()) / i0_beta;
        h.push(sinc * w);
    }
    // Normalize to unity DC gain.
    let sum: f32 = h.iter().sum();
    if sum.abs() > 1e-9 {
        for c in h.iter_mut() {
            *c /= sum;
        }
    }
    h
}

/// Modified Bessel function of the first kind, order 0. Series expansion
/// converges quickly for the small arguments we use (β ≤ ~10).
fn bessel_i0(x: f32) -> f32 {
    let x = x as f64;
    let mut sum = 1.0_f64;
    let mut term = 1.0_f64;
    for k in 1..50 {
        term *= (x / (2.0 * k as f64)).powi(2);
        sum += term;
        if term < 1e-12 * sum {
            break;
        }
    }
    sum as f32
}

/// Up/down-sampler pair sharing a single FIR kernel. Construct once per
/// oversampling factor and reuse across blocks.
pub struct Oversampler {
    factor: usize,
    /// Polyphase decomposition of the kernel. `phases[i][k]` is the k-th tap
    /// of the i-th phase. Each phase has `taps_per_phase` coefficients.
    phases: Vec<Vec<f32>>,
    taps_per_phase: usize,
}

impl Oversampler {
    /// `factor` must be 2, 4, or 8. Other values panic; 1x doesn't need this.
    pub fn new(factor: usize) -> Self {
        assert!(matches!(factor, 2 | 4 | 8), "oversampling factor must be 2/4/8");
        // 64 taps total → good stopband, ~1.5 ms latency at 48 kHz × factor.
        let total_taps = 64 * factor;
        // Cutoff at 0.45 * Fs (high-rate) leaves a ~10% guard band, which
        // a 64-tap-per-phase Kaiser handles cleanly.
        let cutoff = 0.45 / factor as f32;
        let kernel = kaiser_lowpass(total_taps, cutoff, 8.0);
        // Polyphase decomposition: split the kernel into `factor` phases by
        // taking every factor-th tap, offset by phase index.
        let taps_per_phase = total_taps / factor;
        let mut phases = vec![Vec::with_capacity(taps_per_phase); factor];
        for (i, &c) in kernel.iter().enumerate() {
            phases[i % factor].push(c * factor as f32); // gain compensation
        }
        Self {
            factor,
            phases,
            taps_per_phase,
        }
    }

    /// Process a block of mono input through:
    ///   upsample → clip(high-rate) → downsample
    ///
    /// `clip_fn` is called for every high-rate sample and should apply the
    /// clipper non-linearity (input gain + clipper curve + output gain).
    /// Returns a new `Vec<f32>` the same length as `input`.
    pub fn process_block<F: Fn(f32) -> f32>(&self, input: &[f32], clip_fn: F) -> Vec<f32> {
        if input.is_empty() {
            return Vec::new();
        }
        // 1) Upsample with polyphase. For each input sample x[n], emit
        //    `factor` output samples y[n*factor + i] = sum_k phases[i][k] * x[n - k].
        let n_in = input.len();
        let n_high = n_in * self.factor;
        let mut hi = vec![0.0f32; n_high];
        let tpp = self.taps_per_phase;
        for n in 0..n_in {
            for i in 0..self.factor {
                let phase = &self.phases[i];
                let mut acc = 0.0f32;
                for k in 0..tpp {
                    if n >= k {
                        acc += phase[k] * input[n - k];
                    }
                }
                hi[n * self.factor + i] = acc;
            }
        }
        // 2) Clip at high rate.
        for s in hi.iter_mut() {
            *s = clip_fn(*s);
        }
        // 3) Downsample with same polyphase kernel. We only need every
        //    factor-th output, so we can skip computation for the rest.
        //    For output sample y[n], y[n] = sum_i sum_k phases[i][k] * hi[n*factor - i - k*factor]
        //    Equivalent to a single FIR convolution then decimate, but the
        //    polyphase form lets us reuse the same coefficients.
        let mut out = vec![0.0f32; n_in];
        for n in 0..n_in {
            let center = n * self.factor;
            let mut acc = 0.0f32;
            for i in 0..self.factor {
                let phase = &self.phases[i];
                for k in 0..tpp {
                    let idx = center as isize - i as isize - (k * self.factor) as isize;
                    if idx >= 0 && (idx as usize) < n_high {
                        acc += phase[k] * hi[idx as usize];
                    }
                }
            }
            // Polyphase decomposition gave us a 1/factor scale on the
            // upsample (compensated above) and another 1/factor here for
            // the decimation. The net is correct because we already scaled
            // each phase by `factor` in `new()`.
            out[n] = acc / self.factor as f32;
        }
        out
    }
}
