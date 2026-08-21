//! User-customizable keyboard bindings.
//!
//! The map is `HashMap<KeyAction, String>` where the value is a key combo
//! string parseable by `parse_combo`. Defaults match the currently-shipped
//! shortcuts.
//!
//! Shipping format (in `settings.json`):
//! ```json
//! "keybindings": {
//!   "toggle_play": "Space",
//!   "stop": "Shift+Space",
//!   "undo": "Ctrl+Z",
//!   ...
//! }
//! ```
//!
//! Stable string identifiers are used for action keys so renaming an action
//! variant doesn't silently break a user's config. If a binding is missing
//! or malformed, we fall back to the default for that action.

use iced::keyboard::{key::Key, key::Named, Modifiers};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Every action that can be bound to a key. The serde rename uses
/// snake_case so JSON looks natural.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KeyAction {
    TogglePlay,
    Stop,
    AddSplit,
    ToggleSnap,
    ToggleReduction,
    ToggleGuideLine,
    ToggleMonoFold,
    PrevCorrDip,
    NextCorrDip,
    CycleFft,
    ZoomIn,
    ZoomOut,
    ZoomReset,
    PrevZone,
    NextZone,
    DeleteSplit,
    Undo,
    Redo,
}

impl KeyAction {
    pub const ALL: [KeyAction; 18] = [
        KeyAction::TogglePlay,
        KeyAction::Stop,
        KeyAction::AddSplit,
        KeyAction::ToggleSnap,
        KeyAction::ToggleReduction,
        KeyAction::ToggleGuideLine,
        KeyAction::ToggleMonoFold,
        KeyAction::PrevCorrDip,
        KeyAction::NextCorrDip,
        KeyAction::CycleFft,
        KeyAction::ZoomIn,
        KeyAction::ZoomOut,
        KeyAction::ZoomReset,
        KeyAction::PrevZone,
        KeyAction::NextZone,
        KeyAction::DeleteSplit,
        KeyAction::Undo,
        KeyAction::Redo,
    ];

    /// Human-readable label for the read-only display in Settings.
    pub fn label(self) -> &'static str {
        match self {
            KeyAction::TogglePlay => "Play / pause",
            KeyAction::Stop => "Rewind to start",
            KeyAction::AddSplit => "Add split at playhead",
            KeyAction::ToggleSnap => "Toggle zero-crossing snap",
            KeyAction::ToggleReduction => "Toggle reduction overlay",
            KeyAction::ToggleGuideLine => "Toggle guide line",
            KeyAction::ToggleMonoFold => "Monitor mono sum",
            KeyAction::PrevCorrDip => "Previous correlation dip",
            KeyAction::NextCorrDip => "Next correlation dip",
            KeyAction::CycleFft => "Cycle FFT view",
            KeyAction::ZoomIn => "Zoom in",
            KeyAction::ZoomOut => "Zoom out",
            KeyAction::ZoomReset => "Reset zoom",
            KeyAction::PrevZone => "Previous zone",
            KeyAction::NextZone => "Next zone",
            KeyAction::DeleteSplit => "Delete selected split",
            KeyAction::Undo => "Undo",
            KeyAction::Redo => "Redo",
        }
    }

    /// Default key combo string (matches what shipped before this feature).
    pub fn default_combo(self) -> &'static str {
        match self {
            KeyAction::TogglePlay => "Space",
            KeyAction::Stop => "Shift+Space",
            KeyAction::AddSplit => "S",
            KeyAction::ToggleSnap => "Z",
            KeyAction::ToggleReduction => "R",
            KeyAction::ToggleGuideLine => "G",
            KeyAction::ToggleMonoFold => "M",
            KeyAction::PrevCorrDip => "[",
            KeyAction::NextCorrDip => "]",
            KeyAction::CycleFft => "F",
            KeyAction::ZoomIn => "=",
            KeyAction::ZoomOut => "-",
            KeyAction::ZoomReset => "0",
            KeyAction::PrevZone => "ArrowLeft",
            KeyAction::NextZone => "ArrowRight",
            KeyAction::DeleteSplit => "Delete",
            KeyAction::Undo => "Ctrl+Z",
            KeyAction::Redo => "Ctrl+Shift+Z",
        }
    }
}

/// Bindings storage. Wraps a HashMap so we can give it sensible Default
/// behavior (= all defaults populated).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Bindings(pub HashMap<KeyAction, String>);

impl Default for Bindings {
    fn default() -> Self {
        let mut m = HashMap::new();
        for a in KeyAction::ALL {
            m.insert(a, a.default_combo().to_string());
        }
        Bindings(m)
    }
}

impl Bindings {
    /// Look up the action for a given (modifiers, key) pair. Iterates the
    /// map and returns the first match. With ~14 entries this is O(N) but
    /// trivially cheap (called once per keypress).
    pub fn lookup(&self, modifiers: Modifiers, key: &Key) -> Option<KeyAction> {
        let normalized = modifiers_normalized(modifiers);
        // For printable characters, the shift state is already baked into
        // the character itself (e.g. pressing "+" delivers '+' with SHIFT
        // held). So when the incoming key is a Character + Shift, we also
        // try the lookup with Shift cleared — that lets the user write
        // `"zoom_in": "+"` instead of having to write `"Shift++"`.
        let alt_mods = if matches!(key.as_ref(), Key::Character(_))
            && normalized.contains(Modifiers::SHIFT)
        {
            Some(normalized - Modifiers::SHIFT)
        } else {
            None
        };

        let try_match = |target_mods: Modifiers| -> Option<KeyAction> {
            for (action, combo) in &self.0 {
                if let Some((cm, ck)) = parse_combo(combo) {
                    if cm == target_mods && key_matches(&ck, key) {
                        return Some(*action);
                    }
                }
            }
            // Fall back to defaults if the user removed an action's binding.
            for action in KeyAction::ALL {
                if self.0.contains_key(&action) {
                    continue;
                }
                if let Some((cm, ck)) = parse_combo(action.default_combo()) {
                    if cm == target_mods && key_matches(&ck, key) {
                        return Some(action);
                    }
                }
            }
            None
        };

        // Try literal modifiers first; if no match and the key was a
        // shifted character, also try without the shift.
        try_match(normalized).or_else(|| alt_mods.and_then(try_match))
    }

    /// Get the displayed combo string for an action, falling back to the
    /// default if the user hasn't overridden it.
    pub fn display(&self, action: KeyAction) -> String {
        self.0
            .get(&action)
            .cloned()
            .unwrap_or_else(|| action.default_combo().to_string())
    }
}

/// Compare ignoring lock keys (caps lock, num lock) which iced reports
/// alongside command/shift/alt; we only care about the modifier side of
/// the keyboard.
fn modifiers_normalized(m: Modifiers) -> Modifiers {
    let mut out = Modifiers::empty();
    if m.shift() { out |= Modifiers::SHIFT; }
    if m.command() { out |= Modifiers::COMMAND; }
    if m.alt() { out |= Modifiers::ALT; }
    out
}

/// Parse a combo string like `"Ctrl+Shift+Z"` into modifiers + key. Returns
/// None on malformed input — caller should fall back to defaults.
fn parse_combo(s: &str) -> Option<(Modifiers, ParsedKey)> {
    let mut mods = Modifiers::empty();
    let mut key_part: Option<&str> = None;
    for piece in s.split('+').map(str::trim) {
        match piece.to_ascii_lowercase().as_str() {
            "ctrl" | "control" | "cmd" | "command" => mods |= Modifiers::COMMAND,
            "shift" => mods |= Modifiers::SHIFT,
            "alt" | "option" => mods |= Modifiers::ALT,
            _ => {
                // The actual key. If we already saw one, the combo is invalid.
                if key_part.is_some() {
                    return None;
                }
                key_part = Some(piece);
            }
        }
    }
    let key_part = key_part?;
    let key = parse_key(key_part)?;
    Some((mods, key))
}

/// Parse the non-modifier portion of a combo. Recognized named keys come
/// from `iced::keyboard::key::Named`. Anything else is treated as a literal
/// character (single-char preferred but multi-char allowed for things like
/// `"+"`).
#[derive(Debug, Clone)]
enum ParsedKey {
    Named(Named),
    Character(String),
}

fn parse_key(s: &str) -> Option<ParsedKey> {
    let n = match s.to_ascii_lowercase().as_str() {
        "space" => Named::Space,
        "enter" | "return" => Named::Enter,
        "escape" | "esc" => Named::Escape,
        "tab" => Named::Tab,
        "backspace" => Named::Backspace,
        "delete" | "del" => Named::Delete,
        "home" => Named::Home,
        "end" => Named::End,
        "pageup" => Named::PageUp,
        "pagedown" => Named::PageDown,
        "arrowleft" | "left" => Named::ArrowLeft,
        "arrowright" | "right" => Named::ArrowRight,
        "arrowup" | "up" => Named::ArrowUp,
        "arrowdown" | "down" => Named::ArrowDown,
        _ => {
            // Not a named key — treat as a character.
            return Some(ParsedKey::Character(s.to_string()));
        }
    };
    Some(ParsedKey::Named(n))
}

fn key_matches(parsed: &ParsedKey, actual: &Key) -> bool {
    match (parsed, actual.as_ref()) {
        (ParsedKey::Named(p), Key::Named(a)) => *p == a,
        (ParsedKey::Character(p), Key::Character(a)) => p.eq_ignore_ascii_case(a),
        _ => false,
    }
}
