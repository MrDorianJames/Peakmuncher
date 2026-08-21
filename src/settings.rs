//! User-facing settings persisted across sessions.
//!
//! Stored as JSON at `~/.config/peakmuncher/settings.json` (or the platform
//! equivalent via the `dirs` crate). Loaded on startup; saved on every
//! change. Failures to read or write are silently ignored — settings are
//! a polish layer, not core functionality.

use iced::{theme::Palette, Color, Theme};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThemeChoice {
    Dark,
    Light,
}

impl Default for ThemeChoice {
    fn default() -> Self {
        Self::Dark
    }
}

impl ThemeChoice {
    pub const ALL: [Self; 2] = [Self::Dark, Self::Light];
    pub fn label(self) -> &'static str {
        match self {
            Self::Dark => "Dark",
            Self::Light => "Light",
        }
    }
}

/// Named accent color presets, plus a "follow system" option that reads
/// the desktop environment's accent color (KDE only at the moment).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AccentColor {
    Blue,
    Green,
    Orange,
    Purple,
    Red,
    Teal,
    /// Read from desktop environment at startup (KDE Plasma supported).
    /// Falls back to Blue if detection fails.
    System,
}

impl Default for AccentColor {
    fn default() -> Self {
        Self::Blue
    }
}

impl AccentColor {
    pub const ALL: [Self; 7] = [
        Self::Blue,
        Self::Green,
        Self::Orange,
        Self::Purple,
        Self::Red,
        Self::Teal,
        Self::System,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Blue => "Blue",
            Self::Green => "Green",
            Self::Orange => "Orange",
            Self::Purple => "Purple",
            Self::Red => "Red",
            Self::Teal => "Teal",
            Self::System => "System",
        }
    }
    /// Resting / base color used for buttons and slider tracks.
    pub fn color(self) -> Color {
        match self {
            Self::Blue => Color::from_rgb(0.16, 0.38, 0.78),
            Self::Green => Color::from_rgb(0.20, 0.60, 0.35),
            Self::Orange => Color::from_rgb(0.85, 0.50, 0.15),
            Self::Purple => Color::from_rgb(0.55, 0.30, 0.75),
            Self::Red => Color::from_rgb(0.78, 0.25, 0.25),
            Self::Teal => Color::from_rgb(0.10, 0.55, 0.60),
            Self::System => system_accent().unwrap_or(Self::Blue.color()),
        }
    }
    /// Slightly brighter variant used on hover/press.
    pub fn color_hover(self) -> Color {
        let c = self.color();
        Color::from_rgb(
            (c.r + 0.08).min(1.0),
            (c.g + 0.08).min(1.0),
            (c.b + 0.08).min(1.0),
        )
    }
}

/// Read the desktop environment's accent color, currently KDE Plasma only.
///
/// KDE stores it in `~/.config/kdeglobals` under `[General]` as
/// `AccentColor=R,G,B`. Returns None when not on KDE / file missing /
/// no accent set / parse failure — caller should fall back to a default.
pub fn system_accent() -> Option<Color> {
    let cfg_dir = dirs::config_dir()?;
    let path = cfg_dir.join("kdeglobals");
    let bytes = std::fs::read(&path).ok()?;
    let text = std::str::from_utf8(&bytes).ok()?;

    // Naive INI parse: walk lines, track current [section], pick AccentColor
    // when we're under [General].
    let mut in_general = false;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') && line.ends_with(']') {
            in_general = line.eq_ignore_ascii_case("[General]");
            continue;
        }
        if !in_general {
            continue;
        }
        if let Some(rest) = line
            .strip_prefix("AccentColor=")
            .or_else(|| line.strip_prefix("AccentColor ="))
        {
            let parts: Vec<&str> = rest.split(',').map(|s| s.trim()).collect();
            if parts.len() == 3 {
                let r = parts[0].parse::<f32>().ok()? / 255.0;
                let g = parts[1].parse::<f32>().ok()? / 255.0;
                let b = parts[2].parse::<f32>().ok()? / 255.0;
                return Some(Color::from_rgb(r, g, b));
            }
        }
    }
    None
}

/// Colour scheme for the waveform's input + processed envelopes. Structural
/// elements (splits, ceilings, meters, reduction overlay) keep their
/// semantic colors regardless of scheme — those are about meaning, not
/// aesthetics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WaveformScheme {
    Classic,    // teal-gray in, orange out (default)
    Mono,       // light gray in, white out
    Warm,       // amber in, red out
    Cool,       // blue in, cyan out
    AccentTied, // derived from current accent (darker = in, lighter = out)
    /// Signature PeakMuncher look: bright steel/sky blue input, deep
    /// crimson output. (Previously named "PeakEater" — kept the alias
    /// so old settings.json files still load.)
    #[serde(alias = "PeakEater")]
    PeakMuncher,
    /// Use the user's `custom_input_color` / `custom_output_color` from
    /// settings.json. Lets people dial in any envelope colors without
    /// editing source code.
    Custom,
}

impl Default for WaveformScheme {
    fn default() -> Self {
        Self::Classic
    }
}

impl WaveformScheme {
    pub const ALL: [Self; 7] = [
        Self::Classic,
        Self::Mono,
        Self::Warm,
        Self::Cool,
        Self::AccentTied,
        Self::PeakMuncher,
        Self::Custom,
    ];
    pub fn label(self) -> &'static str {
        match self {
            Self::Classic => "Classic",
            Self::Mono => "Mono",
            Self::Warm => "Warm",
            Self::Cool => "Cool",
            Self::AccentTied => "Accent-tied",
            Self::PeakMuncher => "PeakMuncher",
            Self::Custom => "Custom",
        }
    }

    /// Resolve the (input_color, output_color) pair. `accent` is used by
    /// `AccentTied`; `custom_in`/`custom_out` are used by `Custom`. All
    /// are ignored otherwise.
    pub fn colors(self, accent: Color, custom_in: Color, custom_out: Color) -> (Color, Color) {
        match self {
            Self::Classic => (
                Color::from_rgba(0.70, 0.70, 0.75, 0.55),
                Color::from_rgba(1.00, 0.55, 0.18, 0.85),
            ),
            Self::Mono => (
                Color::from_rgba(0.55, 0.55, 0.55, 0.50),
                Color::from_rgba(0.95, 0.95, 0.95, 0.85),
            ),
            Self::Warm => (
                Color::from_rgba(0.90, 0.65, 0.20, 0.55),
                Color::from_rgba(0.85, 0.20, 0.20, 0.85),
            ),
            Self::Cool => (
                Color::from_rgba(0.30, 0.50, 0.85, 0.55),
                Color::from_rgba(0.30, 0.85, 0.95, 0.85),
            ),
            Self::AccentTied => {
                let dim = Color::from_rgba(
                    accent.r * 0.55,
                    accent.g * 0.55,
                    accent.b * 0.55,
                    0.55,
                );
                let bright = Color::from_rgba(
                    (accent.r + 0.15).min(1.0),
                    (accent.g + 0.15).min(1.0),
                    (accent.b + 0.15).min(1.0),
                    0.85,
                );
                (dim, bright)
            }
            Self::PeakMuncher => (
                // Deep indigo (#232394) — distinct, sits well against the
                // amber clipped portion above it.
                Color::from_rgba(0.137, 0.137, 0.580, 1.00),
                // Amber for the clipped portion — passive evidence of
                // limiting. Red is reserved for the reduction overlay
                // (active alarm).
                Color::from_rgba(1.00, 0.65, 0.15, 0.95),
            ),
            Self::Custom => (custom_in, custom_out),
        }
    }
}

impl std::fmt::Display for WaveformScheme {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Hex-string serialized color. Accepts `#RRGGBB` or `#RRGGBBAA` on
/// deserialize; serializes as `#RRGGBBAA`. Wrapped in a newtype so we
/// can give it custom serde and a sensible Default.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HexColor(pub Color);

impl HexColor {
    pub fn new(c: Color) -> Self {
        Self(c)
    }
}

impl From<HexColor> for Color {
    fn from(h: HexColor) -> Color {
        h.0
    }
}

/// Default custom input color: a moderate gray, intentionally bland so
/// the user notices it's the placeholder until they edit it.
fn default_custom_in() -> HexColor {
    HexColor(Color::from_rgba(0.50, 0.50, 0.55, 0.65))
}

/// Default custom output color: a moderate orange, distinguishable from
/// the input default so the user can tell layers apart immediately.
fn default_custom_out() -> HexColor {
    HexColor(Color::from_rgba(0.95, 0.50, 0.20, 0.85))
}

impl Serialize for HexColor {
    fn serialize<S: serde::Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        let r = (self.0.r * 255.0).round() as u8;
        let g = (self.0.g * 255.0).round() as u8;
        let b = (self.0.b * 255.0).round() as u8;
        let a = (self.0.a * 255.0).round() as u8;
        ser.serialize_str(&format!("#{r:02X}{g:02X}{b:02X}{a:02X}"))
    }
}

impl<'de> Deserialize<'de> for HexColor {
    fn deserialize<D: serde::Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        use serde::de::Error;
        let s = String::deserialize(de)?;
        parse_hex_color(&s).ok_or_else(|| D::Error::custom(format!("invalid hex color: {s}")))
    }
}

/// Parse `#RRGGBB` (alpha defaults to FF) or `#RRGGBBAA`. Case-insensitive.
/// Tolerant of missing leading `#`.
fn parse_hex_color(s: &str) -> Option<HexColor> {
    let s = s.trim().trim_start_matches('#');
    let (r, g, b, a) = match s.len() {
        6 => (
            u8::from_str_radix(&s[0..2], 16).ok()?,
            u8::from_str_radix(&s[2..4], 16).ok()?,
            u8::from_str_radix(&s[4..6], 16).ok()?,
            255u8,
        ),
        8 => (
            u8::from_str_radix(&s[0..2], 16).ok()?,
            u8::from_str_radix(&s[2..4], 16).ok()?,
            u8::from_str_radix(&s[4..6], 16).ok()?,
            u8::from_str_radix(&s[6..8], 16).ok()?,
        ),
        _ => return None,
    };
    Some(HexColor(Color::from_rgba(
        r as f32 / 255.0,
        g as f32 / 255.0,
        b as f32 / 255.0,
        a as f32 / 255.0,
    )))
}

/// A user-configured external program that an exported file can be handed
/// off to. `command` is a template: the token `%f` is replaced with the
/// exported file's full path (as a single argument, so paths with spaces
/// are safe). The first whitespace-separated token is the executable; the
/// rest are its arguments.
///
/// Examples:
///   name = "CDJ Prep",  command = "cdj-prep %f"
///   name = "Queue",      command = "some-tool --input %f --queue"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExternalTool {
    pub name: String,
    pub command: String,
}

impl ExternalTool {
    /// Split the command template into (program, args), substituting `%f`
    /// with `file_path`. The `%f` token is replaced as a WHOLE argument so
    /// spaces in the path don't split it. Returns None if the template has
    /// no program token. Naive whitespace tokenization otherwise (no shell
    /// quoting) — fine for the simple `prog --flag %f` templates intended.
    pub fn resolve(&self, file_path: &str) -> Option<(String, Vec<String>)> {
        let mut tokens = self.command.split_whitespace();
        let program = tokens.next()?.to_string();
        let args: Vec<String> = tokens
            .map(|tok| tok.replace("%f", file_path))
            .collect();
        Some((program, args))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    #[serde(default)]
    pub theme: ThemeChoice,
    #[serde(default)]
    pub accent: AccentColor,
    #[serde(default)]
    pub waveform_scheme: WaveformScheme,
    /// Used when `waveform_scheme == Custom`. Hex format: "#RRGGBB" or
    /// "#RRGGBBAA". Edit by hand in settings.json.
    #[serde(default = "default_custom_in")]
    pub custom_input_color: HexColor,
    #[serde(default = "default_custom_out")]
    pub custom_output_color: HexColor,
    #[serde(default)]
    pub default_open_dir: Option<PathBuf>,
    #[serde(default)]
    pub default_export_dir: Option<PathBuf>,
    /// User-configured external programs an exported file can be sent to
    /// (e.g. a CDJ-prep batch tool). Empty by default. `%f` in a tool's
    /// command is replaced with the exported file path.
    #[serde(default)]
    pub external_tools: Vec<ExternalTool>,
    #[serde(default)]
    pub keybindings: crate::keybindings::Bindings,
    /// Spectrogram window size. Longer = better frequency resolution and
    /// worse time resolution; the product is fixed, so this is a trade to
    /// suit what you're looking for, not a quality dial.
    #[serde(default = "default_sg_fft")]
    pub spectrogram_fft_size: usize,
}

fn default_sg_fft() -> usize {
    crate::fft::FFT_SIZE
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            theme: ThemeChoice::default(),
            accent: AccentColor::default(),
            waveform_scheme: WaveformScheme::default(),
            custom_input_color: default_custom_in(),
            custom_output_color: default_custom_out(),
            default_open_dir: None,
            default_export_dir: None,
            external_tools: Vec::new(),
            keybindings: crate::keybindings::Bindings::default(),
            spectrogram_fft_size: default_sg_fft(),
        }
    }
}

impl Settings {
    pub fn load() -> Self {
        let Some(path) = config_path() else { return Self::default(); };
        let Ok(bytes) = std::fs::read(&path) else { return Self::default(); };
        serde_json::from_slice(&bytes).unwrap_or_default()
    }

    pub fn save(&self) {
        let Some(path) = config_path() else { return; };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_vec_pretty(self) {
            let _ = std::fs::write(&path, json);
        }
    }

    /// Build the iced `Theme` derived from these settings: the chosen base
    /// theme (Dark/Light) with its `primary` color overridden by the
    /// chosen accent. Using a custom theme means *every* widget that uses
    /// theme defaults (buttons, sliders, pick-lists, scrollbars) picks up
    /// the accent automatically — without per-widget styling overrides.
    pub fn build_theme(&self) -> Theme {
        let base = match self.theme {
            ThemeChoice::Dark => Palette::DARK,
            ThemeChoice::Light => Palette::LIGHT,
        };
        let accent = self.accent.color();
        let palette = Palette {
            primary: accent,
            ..base
        };
        let name = format!(
            "PeakMuncher {}",
            match self.theme {
                ThemeChoice::Dark => "Dark",
                ThemeChoice::Light => "Light",
            }
        );
        Theme::custom(name, palette)
    }
}

fn config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("peakmuncher").join("settings.json"))
}
