//! Project files (.pmproj).
//!
//! A project bundles an audio file path with the zone map and view state,
//! so you can save your work, close PeakMuncher, and pick up exactly where
//! you left off. Stored as JSON with a version field so the schema can
//! evolve without breaking older projects.
//!
//! The audio path is stored as both an absolute path and a relative-to-the-
//! project-file path. On load, we try the absolute path first, then fall back
//! to the relative one resolved against the project file's directory — this
//! makes projects portable when you move the audio + project together.

use crate::zones::ZoneMap;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Bump this when the schema changes in a backward-incompatible way.
pub const PROJECT_FORMAT_VERSION: u32 = 1;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    /// Schema version, for forward-compat.
    #[serde(default = "default_version")]
    pub version: u32,
    /// Absolute path to the source audio file when the project was saved.
    pub audio_path: PathBuf,
    /// Path relative to the project file's directory (used as fallback if
    /// the absolute path no longer exists, e.g. user moved the folder).
    #[serde(default)]
    pub audio_path_relative: Option<PathBuf>,
    pub zones: ZoneMap,
    /// View / global state worth preserving across sessions.
    #[serde(default)]
    pub view: ViewState,
}

fn default_version() -> u32 {
    PROJECT_FORMAT_VERSION
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViewState {
    #[serde(default = "default_zoom")]
    pub zoom: f32,
    #[serde(default)]
    pub scroll: f32,
    #[serde(default)]
    pub selected_zone: usize,
    #[serde(default = "default_snap")]
    pub snap_enabled: bool,
    #[serde(default)]
    pub normalize_enabled: bool,
    #[serde(default = "default_norm_target")]
    pub normalize_target_db: f32,
    /// Export trim boundaries in seconds. `None` (the default for older
    /// projects without these fields) means "no trim" — on load the app
    /// re-parks them at the file edges.
    #[serde(default)]
    pub trim_start: Option<f32>,
    #[serde(default)]
    pub trim_end: Option<f32>,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            zoom: 1.0,
            scroll: 0.0,
            selected_zone: 0,
            snap_enabled: true,
            normalize_enabled: false,
            normalize_target_db: -0.3,
            trim_start: None,
            trim_end: None,
        }
    }
}

fn default_zoom() -> f32 {
    1.0
}
fn default_snap() -> bool {
    true
}
fn default_norm_target() -> f32 {
    -0.3
}

/// Resolve the audio path: try absolute first, then relative-to-project-dir.
/// Returns the path that actually exists on disk, or None if neither does.
pub fn resolve_audio_path(project: &Project, project_path: &Path) -> Option<PathBuf> {
    if project.audio_path.exists() {
        return Some(project.audio_path.clone());
    }
    if let Some(rel) = &project.audio_path_relative {
        let project_dir = project_path.parent()?;
        let candidate = project_dir.join(rel);
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

/// Build a relative path from `from_dir` to `to_path`. Falls back to just
/// the filename if a meaningful relative path can't be made (e.g. they're
/// on different drives on Windows).
pub fn make_relative(from_dir: &Path, to_path: &Path) -> PathBuf {
    // Try a naive same-prefix relative. We don't need full pathdiff
    // sophistication for this — most projects will have the audio next to
    // the project file or one level up.
    if let (Ok(from), Ok(to)) = (from_dir.canonicalize(), to_path.canonicalize()) {
        if let Ok(rel) = to.strip_prefix(&from) {
            return rel.to_path_buf();
        }
        // Try walking up from `from` looking for a common ancestor.
        let mut cur = from.as_path();
        let mut up = PathBuf::new();
        while let Some(parent) = cur.parent() {
            if let Ok(rel) = to.strip_prefix(parent) {
                return up.join(rel);
            }
            up.push("..");
            cur = parent;
        }
    }
    // Fallback: just the filename. Works if user keeps the audio next to
    // the project file.
    to_path
        .file_name()
        .map(PathBuf::from)
        .unwrap_or_else(|| to_path.to_path_buf())
}
