//! Waveshaping clipper functions, modeled after PeakEater.
//!
//! Each function takes a normalized input sample `x` (typically in roughly
//! [-G, G] where G is the input gain) and a `ceiling` value in linear
//! amplitude (0..1, where 1.0 = 0 dBFS). Output is bounded by ±ceiling.
//!
//! These are the standard waveshaping curves used in soft-clippers; PeakEater
//! uses essentially the same set. Re-implemented here from first principles
//! (no PeakEater code copied) so this app has no licensing entanglement.

use std::f32::consts::FRAC_PI_2;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ClipperType {
    Hard,
    Quintic,
    Cubic,
    Tangent,
    Algebraic,
    Arctangent,
    /// Wavefolder-style: folds back past the threshold instead of
    /// saturating flat. Bright, aggressive harmonics — an effect, not just
    /// a limiter. (DnB/dub texture.)
    SineFold,
    /// Drive-scaled tanh: warmer, tape-like saturation that rounds off
    /// more as it's pushed. Distinct from plain Tangent (k=1).
    TanhDrive,
    /// Logistic S-curve: smooth everywhere with a gentle approach to the
    /// rails. Clean, polished — sits between Cubic and Arctangent.
    Sigmoid,
}

impl ClipperType {
    pub const ALL: [ClipperType; 9] = [
        ClipperType::Hard,
        ClipperType::Quintic,
        ClipperType::Cubic,
        ClipperType::Tangent,
        ClipperType::Algebraic,
        ClipperType::Arctangent,
        ClipperType::SineFold,
        ClipperType::TanhDrive,
        ClipperType::Sigmoid,
    ];

    pub fn label(&self) -> &'static str {
        match self {
            ClipperType::Hard => "Hard",
            ClipperType::Quintic => "Quintic",
            ClipperType::Cubic => "Cubic",
            ClipperType::Tangent => "Tangent",
            ClipperType::Algebraic => "Algebraic",
            ClipperType::Arctangent => "Arctangent",
            ClipperType::SineFold => "Sine Fold",
            ClipperType::TanhDrive => "Tanh Drive",
            ClipperType::Sigmoid => "Sigmoid",
        }
    }
}

impl std::fmt::Display for ClipperType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Apply the chosen waveshaper. `ceiling` is linear amplitude (0..=1).
#[inline]
pub fn shape(sample: f32, ceiling: f32, kind: ClipperType) -> f32 {
    if ceiling <= 1e-6 {
        return 0.0;
    }
    // Normalize sample into ceiling space so each shaper sees [-1, 1]-ish input
    let n = sample / ceiling;
    let y = match kind {
        ClipperType::Hard => n.clamp(-1.0, 1.0),

        // Quintic soft-clip: smooth up to ±1, flat beyond.
        // y = (15 x - 10 x^3 + 3 x^5) / 8 inside [-1,1], saturating outside.
        ClipperType::Quintic => {
            if n.abs() >= 1.0 {
                n.signum()
            } else {
                let x = n;
                let x3 = x * x * x;
                let x5 = x3 * x * x;
                (15.0 * x - 10.0 * x3 + 3.0 * x5) / 8.0
            }
        }

        // Classic cubic soft-clip: y = 1.5 x - 0.5 x^3 inside, saturate outside.
        ClipperType::Cubic => {
            if n.abs() >= 1.0 {
                n.signum()
            } else {
                1.5 * n - 0.5 * n * n * n
            }
        }

        // Tangent shaper: tanh-style smooth saturation.
        ClipperType::Tangent => n.tanh(),

        // Algebraic shaper: y = x / sqrt(1 + x^2). Smooth, never quite reaches 1.
        ClipperType::Algebraic => n / (1.0 + n * n).sqrt(),

        // Arctangent shaper: y = (2/π) * atan(x). Asymptotes to ±1.
        ClipperType::Arctangent => (n.atan()) / FRAC_PI_2,

        // Sine fold (wavefolder): within [-1,1], y = sin(n·π/2) — a smooth
        // saturation reaching ±1 at n=±1. Beyond ±1 the sine continues and
        // FOLDS the signal back, generating bright odd/even harmonics
        // instead of a flat clip. `sin` is inherently bounded to [-1,1],
        // so the output never exceeds the ceiling. The further past ±1 the
        // input goes, the more folds — aggressive, effect-like.
        ClipperType::SineFold => (n * FRAC_PI_2).sin(),

        // Drive-scaled tanh: y = tanh(k·n) / tanh(k), normalized so n=±1
        // maps to ±1 exactly. k>1 pushes harder into saturation than plain
        // Tangent (which is k=1), for a warmer, more compressed/tape-like
        // curve. k = 2.5 chosen as a musical default.
        ClipperType::TanhDrive => {
            const K: f32 = 2.5;
            // The tanh(K·n)/tanh(K) normalization slightly exceeds ±1 for
            // |n| > 1 (e.g. ~1.01 at n=2), so clamp to keep output strictly
            // within the ceiling.
            ((K * n).tanh() / K.tanh()).clamp(-1.0, 1.0)
        }

        // Logistic sigmoid: a smooth S-curve. Raw logistic is
        // 1/(1+e^-x); we center it to ±1 and normalize by its value at
        // n=1 so the curve passes through ±1 at n=±1 (continuous with the
        // other shapers' endpoints). k = 3.0 sets the steepness. Clean,
        // gentle approach to the rails — "polished".
        ClipperType::Sigmoid => {
            const K: f32 = 3.0;
            let logistic = |x: f32| 2.0 / (1.0 + (-K * x).exp()) - 1.0;
            // Normalize so |y| = 1 at |n| = 1.
            let norm = logistic(1.0).max(1e-6);
            (logistic(n) / norm).clamp(-1.0, 1.0)
        }
    };
    y * ceiling
}

/// dB → linear amplitude.
#[inline]
pub fn db_to_amp(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

/// linear amplitude → dB (clamped at -120 dB for display).
#[inline]
pub fn amp_to_db(amp: f32) -> f32 {
    if amp <= 1e-6 {
        -120.0
    } else {
        20.0 * amp.log10()
    }
}
