//! Persistence for the "Recent files" list.
//!
//! Stored as JSON at `~/.config/peakmuncher/recent.json` (or the platform
//! equivalent via the `dirs` crate). Capped at 8 entries, most-recent-first,
//! deduplicated so opening the same file twice doesn't make two entries.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

const MAX_ENTRIES: usize = 8;

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RecentFiles {
    pub paths: Vec<PathBuf>,
}

impl RecentFiles {
    /// Load from the config file. Returns an empty list if the file doesn't
    /// exist or can't be parsed (we never error out the app over this).
    pub fn load() -> Self {
        let Some(path) = config_path() else { return Self::default(); };
        let Ok(bytes) = std::fs::read(&path) else { return Self::default(); };
        serde_json::from_slice(&bytes).unwrap_or_default()
    }

    /// Save to the config file. Best-effort; failures are silently ignored.
    pub fn save(&self) {
        let Some(path) = config_path() else { return; };
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        if let Ok(json) = serde_json::to_vec_pretty(self) {
            let _ = std::fs::write(&path, json);
        }
    }

    /// Add (or move-to-front) a path. Bumps to position 0, dedupes, caps at MAX_ENTRIES.
    pub fn push(&mut self, path: &Path) {
        // Canonicalize when possible so different ways of spelling the same
        // path (relative vs absolute, symlinks) don't create duplicates.
        let canonical = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
        self.paths.retain(|p| p != &canonical);
        self.paths.insert(0, canonical);
        if self.paths.len() > MAX_ENTRIES {
            self.paths.truncate(MAX_ENTRIES);
        }
    }
}

fn config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("peakmuncher").join("recent.json"))
}
