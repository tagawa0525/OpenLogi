//! User configuration, persisted as TOML at the platform-standard config
//! path.
//!
//! Per-device state (button bindings, …) lives under the
//! [`Config::devices`] map, keyed by a stable physical-device identifier such
//! as `"receiver:abc123:slot:2"`. Schema migrations branch on
//! [`Config::schema_version`].

use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
};

use atomic_write_file::AtomicWriteFile;
use serde::{Deserialize, Serialize};
use thiserror::Error;

mod device;
mod settings;

pub use device::{DeviceConfig, DeviceIdentity};
pub use settings::{
    AppSettings, Appearance, AssetSourcePreference, DEFAULT_THUMBWHEEL_SENSITIVITY, GestureOwner,
    Lighting, MAX_THUMBWHEEL_SENSITIVITY, MIN_THUMBWHEEL_SENSITIVITY,
    SMARTSHIFT_AUTO_DISENGAGE_DEFAULT, SMARTSHIFT_MIN_AUTO_DISENGAGE, ScrollResolution, SmartShift,
    WheelMode,
};

use crate::binding::{
    Action, Binding, ButtonId, GestureDirection, default_binding, default_binding_for,
};
use crate::paths::{self, PathsError};

/// The schema version the current build produces. Bumped on breaking layout
/// changes; readers branch on the parsed value before consuming the rest of
/// the file.
///
/// v4 removes the one-gesture-button-per-device owner lock: gesture mode is a
/// per-button fact read from the binding shape, so `gesture_owner` no longer
/// serializes. Loading a v3-or-older file resolves the old owner and rewrites
/// the shapes to dispatch identically
/// (see `Config::migrate_owner_locked_gestures`); the version gate is what
/// keeps that pass off v4 files, where several gesture-shaped buttons are a
/// deliberate state, not a dormant leftover.
///
/// v3 changes the device map from model keys to physical-device keys. No v2
/// device entries are migrated because model-scoped settings cannot be assigned
/// safely when two identical devices exist.
///
/// v2 merged the per-device `button_bindings` + `gesture_bindings` maps into a
/// single `bindings: BTreeMap<ButtonId, Binding>`. A v1 file still loads (the
/// `RawDeviceConfig` shim folds the legacy fields) and self-heals to v2 on the
/// next save; [`Config::load_from_path`] rejects only versions *newer* than this
/// so a forward file fails loudly instead of silently losing bindings.
pub const SCHEMA_VERSION: u32 = 4;

/// Top-level config document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Schema version the file was written with. Compared against
    /// [`SCHEMA_VERSION`] on load: older layouts migrate, newer ones are
    /// rejected loudly rather than silently losing settings.
    pub schema_version: u32,
    /// Non-device-scoped preferences (autostart, tray, language, …).
    #[serde(default, skip_serializing_if = "AppSettings::is_default")]
    pub app_settings: AppSettings,
    /// Physical config key of the carousel-selected device, persisted so a
    /// restart restores the last view rather than always landing on the
    /// first paired device. `None` means "fall back to the first device".
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub selected_device: Option<String>,
    /// Per-device state, keyed by the stable physical-device identifier
    /// (e.g. `"receiver:abc123:slot:2"`) so two identical models never share
    /// an entry.
    #[serde(default)]
    pub devices: BTreeMap<String, DeviceConfig>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            schema_version: SCHEMA_VERSION,
            app_settings: AppSettings::default(),
            selected_device: None,
            devices: BTreeMap::new(),
        }
    }
}

/// Failure loading or persisting `config.toml`. The file-scoped variants
/// carry the offending path so callers can surface an actionable message.
#[derive(Debug, Error)]
pub enum ConfigError {
    /// The platform config directory could not be resolved (no home
    /// directory for the current user).
    #[error("could not resolve config path")]
    Path(#[from] PathsError),
    /// Reading the config file from disk failed.
    #[error("could not read config at {path}")]
    Read {
        /// The config file the read targeted.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// The file was read but is not valid TOML for this schema.
    #[error("could not parse config at {path}")]
    Parse {
        /// The config file that failed to parse.
        path: PathBuf,
        /// The underlying TOML deserialization error.
        #[source]
        source: toml::de::Error,
    },
    /// Writing the updated config back to disk failed.
    #[error("could not write config at {path}")]
    Write {
        /// The config file the write targeted.
        path: PathBuf,
        /// The underlying I/O error.
        #[source]
        source: io::Error,
    },
    /// The in-memory config could not be serialized to TOML — a bug in the
    /// config types rather than user error, since [`Config`] always
    /// serializes cleanly.
    #[error("could not serialize config")]
    Serialize(#[from] toml::ser::Error),
    /// The file declares a `schema_version` newer than this build
    /// understands; failing loudly avoids silently dropping settings a newer
    /// build wrote.
    #[error("config at {path} has unsupported schema_version {found}")]
    UnsupportedSchemaVersion {
        /// The config file carrying the unsupported version.
        path: PathBuf,
        /// The `schema_version` the file declared.
        found: u32,
    },
}

#[allow(
    clippy::result_large_err,
    reason = "Config I/O keeps rich parse/write context and is not a hot path"
)]
impl Config {
    /// Loads the config from the default user path, returning
    /// [`Config::default`] if the file does not exist yet.
    pub fn load_or_default() -> Result<Self, ConfigError> {
        Self::load_from_path(&paths::config_path()?)
    }

    /// Same as [`Self::load_or_default`] but reads from `path`. Used by tests
    /// to avoid touching the real user config.
    pub fn load_from_path(path: &Path) -> Result<Self, ConfigError> {
        match fs::read_to_string(path) {
            Ok(text) => {
                let mut config: Self =
                    toml::from_str(&text).map_err(|source| ConfigError::Parse {
                        path: path.to_path_buf(),
                        source,
                    })?;
                // Accept any version up to the current one: older files migrate
                // through the per-device [`RawDeviceConfig`] shim and self-heal on
                // the next save. Only a *newer* file is rejected — loudly, so a
                // downgraded binary refuses to load (and silently wipe) a config
                // it can't represent.
                if config.schema_version > SCHEMA_VERSION {
                    return Err(ConfigError::UnsupportedSchemaVersion {
                        path: path.to_path_buf(),
                        found: config.schema_version,
                    });
                }
                // An owner-locked file (v3 and older) rewrites its gesture
                // shapes to shape-driven form. Version-gated: on a v4 file
                // several gesture-shaped buttons are a deliberate state that
                // must round-trip untouched.
                if config.schema_version <= 3 {
                    config.migrate_owner_locked_gestures();
                }
                // Stamp the in-memory doc to the current version so a re-save
                // writes the migrated shape (the device shim already folded
                // the legacy fields during deserialize).
                config.schema_version = SCHEMA_VERSION;
                Ok(config)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(Self::default()),
            Err(source) => Err(ConfigError::Read {
                path: path.to_path_buf(),
                source,
            }),
        }
    }

    /// Writes the config atomically to the default user path: serialize to a
    /// sibling temp file, then rename over the target. On Unix the temp file
    /// is created with mode 0600.
    pub fn save_atomic(&self) -> Result<(), ConfigError> {
        self.save_to_path(&paths::config_path()?)
    }

    /// Same as [`Self::save_atomic`] but writes to `path`. Used by tests.
    pub fn save_to_path(&self, path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|source| ConfigError::Write {
                path: path.to_path_buf(),
                source,
            })?;
        }
        let body = toml::to_string_pretty(self)?;
        write_atomic(path, body.as_bytes()).map_err(|source| ConfigError::Write {
            path: path.to_path_buf(),
            source,
        })
    }

    /// Returns the bindings stored for `device_key`, or an empty map if the
    /// device has no committed bindings yet.
    #[must_use]
    pub fn bindings_for(&self, device_key: &str) -> BTreeMap<ButtonId, Binding> {
        self.devices
            .get(device_key)
            .map(|d| d.bindings.clone())
            .unwrap_or_default()
    }

    /// Records `binding` for `button` on `device_key`, creating the device
    /// entry if needed. Replaces the whole binding (use
    /// [`Self::set_gesture_direction`] to edit one direction of a gesture
    /// binding in place).
    pub fn set_binding(&mut self, device_key: &str, button: ButtonId, binding: Binding) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .bindings
            .insert(button, binding);
    }

    /// Returns the gesture sub-bindings for `device_key`'s gesture button, or an
    /// empty map if it isn't in gesture mode. Derived from the unified
    /// [`DeviceConfig::bindings`]; kept as a convenience for the agent-side
    /// per-direction adapter.
    #[must_use]
    pub fn gesture_bindings_for(&self, device_key: &str) -> BTreeMap<GestureDirection, Action> {
        // Read the *owner's* stored map: the gesture role can sit on the
        // dedicated gesture button or the MX Master 4 haptic panel (or an
        // OS-hook button — callers gate on the owner kind), and each button
        // keeps its own per-direction map.
        let Some(owner) = self.gesture_owner(device_key) else {
            return BTreeMap::new();
        };
        match self
            .devices
            .get(device_key)
            .and_then(|d| d.bindings.get(&owner))
        {
            Some(Binding::Gesture(map)) => map.clone(),
            _ => BTreeMap::new(),
        }
    }

    /// Records `action` for one `direction` of `button`'s gesture binding,
    /// creating the device entry if needed.
    ///
    /// A button with no binding yet is seeded from its canonical
    /// [`default_binding_for`] — for [`ButtonId::GestureButton`] that is the full
    /// default direction map (including a [`GestureDirection::Click`]), so the
    /// merged map never persists a gesture binding whose click projection is a
    /// no-op. A prior [`Binding::Single`] is upgraded to [`Binding::Gesture`],
    /// preserving its action as the `Click` entry.
    pub fn set_gesture_direction(
        &mut self,
        device_key: &str,
        button: ButtonId,
        direction: GestureDirection,
        action: Action,
    ) {
        if let Binding::Gesture(map) = self.ensure_gesture_binding(device_key, button) {
            map.insert(direction, action);
        }
    }

    /// Ensure `button` on `device_key` is a [`Binding::Gesture`], creating the
    /// device + a default binding if needed and upgrading a [`Binding::Single`]
    /// in place (its action kept as the [`GestureDirection::Click`]). Returns the
    /// entry so the caller can finish it — seed every direction
    /// ([`Binding::fill_gesture_defaults`]) or set just one. Shared by
    /// [`Self::set_gesture_mode`] and [`Self::set_gesture_direction`] so the two
    /// promote a button into gesture mode identically.
    fn ensure_gesture_binding(&mut self, device_key: &str, button: ButtonId) -> &mut Binding {
        let entry = self
            .devices
            .entry(device_key.to_string())
            .or_default()
            .bindings
            .entry(button)
            .or_insert_with(|| default_binding_for(button));
        // An explicit `Single` at the button's canonical single default carries
        // no user customization — it is the "pinned off" marker
        // [`Self::set_gesture_mode`] stamps on a gesture-shaped-by-default
        // button. Re-promoting it restores the canonical default shape (the
        // full default direction map for the dedicated gesture button) rather
        // than freezing the pin action as a Click choice the user never made.
        // For single-shaped-by-default buttons this replaces the value with
        // itself, so the rule is uniform.
        if *entry == Binding::Single(default_binding(button)) {
            *entry = default_binding_for(button);
        }
        entry.upgrade_to_gesture();
        entry
    }

    /// The single button the pre-v4 owner-locked runtime would have dispatched
    /// gestures from, inferred from the binding shapes — the owner-lock-era
    /// resolution rule, retained for [`Self::migrate_owner_locked_gestures`]
    /// and the transition shims. `None` means gestures were off.
    fn infer_gesture_owner(bindings: &BTreeMap<ButtonId, Binding>) -> Option<ButtonId> {
        // An OS-hook button left in gesture mode took the role over.
        if let Some((id, _)) = bindings
            .iter()
            .find(|(id, b)| **id != ButtonId::GestureButton && b.is_gesture())
        {
            return Some(*id);
        }
        // A dedicated HID++ gesture button explicitly demoted to a single action means gestures off.
        if matches!(
            bindings.get(&ButtonId::GestureButton),
            Some(Binding::Single(_))
        ) {
            return None;
        }
        // Default: the dedicated HID++ gesture button owns the gesture role.
        Some(ButtonId::GestureButton)
    }

    /// One gesture-mode button of `device_key`, `None` when nothing gestures.
    ///
    /// Transition shim over the shape-driven model for callers still built
    /// around the retired one-owner lock: when several buttons gesture at once
    /// (representable since v4) it reports just one, by the owner-lock-era
    /// preference order. New code reads [`Self::gesture_mode_buttons`].
    #[must_use]
    pub fn gesture_owner(&self, device_key: &str) -> Option<ButtonId> {
        self.devices.get(device_key).map_or(
            // No config yet → the dedicated HID++ gesture button gestures by default.
            Some(ButtonId::GestureButton),
            |device| Self::infer_gesture_owner(&device.bindings),
        )
    }

    /// Make `button` the device's sole gesture button.
    ///
    /// Transition shim over the shape-driven model preserving the retired
    /// selector's exclusive semantics: promotes `button` (see
    /// [`Self::set_gesture_mode`]) and demotes every other gesture-mode button
    /// to a [`Binding::Single`] of its `Click`. New code sets each button's
    /// mode independently with [`Self::set_gesture_mode`].
    pub fn set_gesture_owner(&mut self, device_key: &str, button: ButtonId) {
        for other in self.gesture_mode_buttons(device_key) {
            if other != button {
                self.set_gesture_mode(device_key, other, false);
            }
        }
        self.set_gesture_mode(device_key, button, true);
    }

    /// Turn every gesture-mode button of `device_key` off.
    ///
    /// Transition shim over the shape-driven model: demotes each one via
    /// [`Self::set_gesture_mode`], which pins the gesture-shaped-by-default
    /// dedicated button with an explicit `Single`.
    pub fn disable_gestures(&mut self, device_key: &str) {
        for button in self.gesture_mode_buttons(device_key) {
            self.set_gesture_mode(device_key, button, false);
        }
    }

    /// Whether `button` on `device_key` is in gesture mode — a per-button fact
    /// read straight from the binding shape: a stored [`Binding::Gesture`], or
    /// no stored binding on a button whose canonical default
    /// ([`default_binding_for`]) is gesture-shaped (the dedicated HID++ gesture
    /// button starts in gesture mode).
    ///
    /// Gesture mode is not exclusive: any number of buttons may gesture at
    /// once, each with its own direction map. This replaces the former
    /// one-gesture-button-per-device owner lock — see [`Self::set_gesture_mode`].
    #[must_use]
    pub fn is_gesture_mode(&self, device_key: &str, button: ButtonId) -> bool {
        self.devices
            .get(device_key)
            .and_then(|d| d.bindings.get(&button))
            .map_or_else(
                || default_binding_for(button).is_gesture(),
                Binding::is_gesture,
            )
    }

    /// Every button of `device_key` currently in gesture mode, in [`ButtonId`]
    /// declaration order. Purely config-derived: callers cross it with the
    /// device's actual controls (a model without the dedicated gesture button
    /// simply never captures it).
    #[must_use]
    pub fn gesture_mode_buttons(&self, device_key: &str) -> Vec<ButtonId> {
        ButtonId::ALL
            .iter()
            .copied()
            .filter(|b| self.is_gesture_mode(device_key, *b))
            .collect()
    }

    /// Turn gesture mode on or off for one button, independently of every
    /// other button.
    ///
    /// On: promote the stored binding in place ([`Binding::upgrade_to_gesture`]
    /// keeps a prior single action as the [`GestureDirection::Click`] entry)
    /// and seed unbound directions from
    /// [`default_gesture_binding`](crate::binding::default_gesture_binding).
    /// Off: demote to a [`Binding::Single`] of the map's `Click` action,
    /// falling back to the button's canonical
    /// [`default_binding`](crate::binding::default_binding) when the map has no
    /// explicit `Click` — a demoted button always keeps a meaningful press. A
    /// button gesturing only by default (no stored binding) is pinned off with
    /// an explicit `Single` at its canonical default, which the capture layer
    /// leaves native.
    pub fn set_gesture_mode(&mut self, device_key: &str, button: ButtonId, enabled: bool) {
        if enabled {
            self.ensure_gesture_binding(device_key, button)
                .fill_gesture_defaults();
        } else if let Some(binding) = self
            .devices
            .get_mut(device_key)
            .and_then(|d| d.bindings.get_mut(&button))
        {
            binding.demote_to_single(default_binding(button));
        } else if default_binding_for(button).is_gesture() {
            self.devices
                .entry(device_key.to_string())
                .or_default()
                .bindings
                .insert(button, Binding::Single(default_binding(button)));
        }
    }

    /// One-time load migration for owner-locked files (`schema_version <= 3`).
    ///
    /// Under the owner lock at most one button dispatched gestures; every other
    /// gesture-capable button could keep a dormant direction map awaiting
    /// re-selection, with [`DeviceConfig::gesture_owner`] recording the choice
    /// (absent = infer). The shape-driven model has no dormant state — a stored
    /// [`Binding::Gesture`] IS gesture mode — so this resolves the old owner
    /// and rewrites the shapes to dispatch exactly what the old config did:
    ///
    /// - the owner keeps its gesture map;
    /// - every other gesture-shaped binding demotes to a [`Binding::Single`] of
    ///   its `Click` — the only part of a dormant map the old runtime
    ///   dispatched;
    /// - a non-owner dedicated gesture button with no stored binding is pinned
    ///   with an explicit `Single` at its canonical default (absence would
    ///   re-enter gesture mode under the gesture-shaped default), which the
    ///   capture layer leaves native;
    /// - the consumed `gesture_owner` never serializes again — the shape is
    ///   the whole truth from here on.
    fn migrate_owner_locked_gestures(&mut self) {
        for device in self.devices.values_mut() {
            let owner = match device.gesture_owner.take() {
                Some(GestureOwner::Off) => None,
                Some(GestureOwner::Button(id)) => Some(id),
                None => Self::infer_gesture_owner(&device.bindings),
            };
            for (id, binding) in &mut device.bindings {
                if Some(*id) != owner {
                    binding.demote_to_single(default_binding(*id));
                }
            }
            if owner != Some(ButtonId::GestureButton) {
                device
                    .bindings
                    .entry(ButtonId::GestureButton)
                    .or_insert_with(|| Binding::Single(default_binding(ButtonId::GestureButton)));
            }
        }
    }

    /// Resolve the effective binding map for `device_key`, overlaying the
    /// per-app entry for `bundle_id` (if any) on top of the global per-device
    /// `bindings`. A per-app override replaces the whole button with a
    /// [`Binding::Single`]; everything else falls through.
    ///
    /// Returns an empty map when the device has no recorded bindings yet.
    /// Callers (the GUI / hook) layer their own defaults on top.
    #[must_use]
    pub fn effective_bindings(
        &self,
        device_key: &str,
        bundle_id: Option<&str>,
    ) -> BTreeMap<ButtonId, Binding> {
        let Some(device) = self.devices.get(device_key) else {
            return BTreeMap::new();
        };
        let mut out = device.bindings.clone();
        if let Some(bid) = bundle_id
            && let Some(overlay) = device.per_app_bindings.get(bid)
        {
            for (k, v) in overlay {
                out.insert(*k, Binding::Single(v.clone()));
            }
        }
        out
    }

    /// Records a per-app override. Creates the device + app entries as
    /// needed; passing an action of `None` removes the override and prunes
    /// the empty app map.
    pub fn set_per_app_binding(
        &mut self,
        device_key: &str,
        bundle_id: &str,
        button: ButtonId,
        action: Option<Action>,
    ) {
        let entry = self
            .devices
            .entry(device_key.to_string())
            .or_default()
            .per_app_bindings
            .entry(bundle_id.to_string())
            .or_default();
        match action {
            Some(a) => {
                entry.insert(button, a);
            }
            None => {
                entry.remove(&button);
            }
        }
        if let Some(d) = self.devices.get_mut(device_key) {
            d.per_app_bindings.retain(|_, m| !m.is_empty());
        }
    }

    /// HID++ config key of the carousel-selected device, if any.
    #[must_use]
    pub fn selected_device(&self) -> Option<&str> {
        self.selected_device.as_deref()
    }

    /// Update the carousel-selected device. Pass `None` to clear the
    /// selection (e.g. when the previously-selected device disappears).
    pub fn set_selected_device(&mut self, key: Option<String>) {
        self.selected_device = key;
    }

    /// The ordered DPI preset list for `device_key`, or an empty `Vec` if the
    /// device has none configured yet.
    #[must_use]
    pub fn dpi_presets(&self, device_key: &str) -> Vec<u32> {
        self.devices
            .get(device_key)
            .map(|d| d.dpi_presets.clone())
            .unwrap_or_default()
    }

    /// Replace the DPI preset list for `device_key`. Pass an empty `Vec` to
    /// clear (the device block is kept; the field is just omitted on save
    /// thanks to `skip_serializing_if`).
    pub fn set_dpi_presets(&mut self, device_key: &str, presets: Vec<u32>) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .dpi_presets = presets;
    }

    /// The last-known [`DeviceIdentity`] for `device_key`, or `None` if the
    /// device has never been seen online (or was configured before identities
    /// were recorded).
    #[must_use]
    pub fn device_identity(&self, device_key: &str) -> Option<&DeviceIdentity> {
        self.devices
            .get(device_key)
            .and_then(|d| d.identity.as_ref())
    }

    /// Record (or refresh) the identity captured for `device_key` while it was
    /// online, creating the device entry if needed.
    pub fn set_device_identity(&mut self, device_key: &str, identity: DeviceIdentity) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .identity = Some(identity);
    }

    /// Whether `device_key` has a non-empty per-app binding overlay for the
    /// foreground app `app` (bundle id). Drives the menu-bar popover's "override
    /// active" badge — when the current app has its own bindings for this
    /// device, the global bindings are (partly) overridden.
    #[must_use]
    pub fn has_app_override(&self, device_key: &str, app: &str) -> bool {
        self.devices.get(device_key).is_some_and(|d| {
            d.per_app_bindings
                .get(app)
                .is_some_and(|overlay| !overlay.is_empty())
        })
    }

    /// Iterate every device we've recorded an identity for, as
    /// `(config_key, identity)`. Used to seed offline placeholder cards so a
    /// known device stays visible (with its panels) before any live probe.
    pub fn known_identities(&self) -> impl Iterator<Item = (&str, &DeviceIdentity)> {
        self.devices
            .iter()
            .filter_map(|(k, d)| d.identity.as_ref().map(|i| (k.as_str(), i)))
    }

    /// The lighting config for `device_key`, or `None` if unset.
    #[must_use]
    pub fn lighting(&self, device_key: &str) -> Option<Lighting> {
        self.devices
            .get(device_key)
            .and_then(|d| d.lighting.clone())
    }

    /// Replace the lighting config for `device_key`.
    pub fn set_lighting(&mut self, device_key: &str, lighting: Lighting) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .lighting = Some(lighting);
    }

    /// The committed sensor DPI for `device_key`, or `None` if never set.
    #[must_use]
    pub fn dpi(&self, device_key: &str) -> Option<u32> {
        self.devices.get(device_key).and_then(|d| d.dpi)
    }

    /// Record the committed sensor DPI for `device_key`, so the agent can
    /// re-apply it when the device reconnects (#189).
    pub fn set_dpi(&mut self, device_key: &str, dpi: u32) {
        self.devices.entry(device_key.to_string()).or_default().dpi = Some(dpi);
    }

    /// The SmartShift wheel config for `device_key`, or `None` if never set.
    #[must_use]
    pub fn smartshift(&self, device_key: &str) -> Option<SmartShift> {
        self.devices.get(device_key).and_then(|d| d.smartshift)
    }

    /// Record the SmartShift wheel config for `device_key`, so the agent can
    /// re-apply it when the device reconnects (#189).
    pub fn set_smartshift(&mut self, device_key: &str, smartshift: SmartShift) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .smartshift = Some(smartshift);
    }

    /// Whether `device_key`'s scroll wheel is inverted (issue #126). `false`
    /// (the native direction) for an unconfigured or absent device.
    #[must_use]
    pub fn invert_scroll(&self, device_key: &str) -> bool {
        self.devices
            .get(device_key)
            .is_some_and(|d| d.invert_scroll)
    }

    /// Set whether `device_key`'s scroll wheel is inverted. The agent reads this
    /// on the next `ReloadConfig` and applies it in the OS hook.
    pub fn set_invert_scroll(&mut self, device_key: &str, invert: bool) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .invert_scroll = invert;
    }

    /// The configured wheel resolution for `device_key`, or `None` when
    /// OpenLogi should leave the device's current resolution unchanged.
    #[must_use]
    pub fn scroll_resolution(&self, device_key: &str) -> Option<ScrollResolution> {
        self.devices
            .get(device_key)
            .and_then(|device| device.scroll_resolution)
    }

    /// Set the wheel resolution OpenLogi should restore for `device_key`.
    /// Passing `None` returns the device to its unmanaged default state.
    pub fn set_scroll_resolution(
        &mut self,
        device_key: &str,
        resolution: Option<ScrollResolution>,
    ) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .scroll_resolution = resolution;
    }

    /// Whether OpenLogi manages `device_key` at all (capture + volatile
    /// re-apply). Unconfigured devices are managed.
    #[must_use]
    pub fn device_enabled(&self, device_key: &str) -> bool {
        self.devices.get(device_key).is_none_or(|d| d.enabled)
    }

    /// Enable or disable OpenLogi's management of `device_key`.
    pub fn set_device_enabled(&mut self, device_key: &str, enabled: bool) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .enabled = enabled;
    }

    /// The effective thumb-wheel sensitivity for `device_key`: the device's
    /// override when set, else the app-wide default.
    #[must_use]
    pub fn thumbwheel_sensitivity(&self, device_key: &str) -> i32 {
        self.devices
            .get(device_key)
            .and_then(|d| d.thumbwheel_sensitivity)
            .unwrap_or(self.app_settings.thumbwheel_sensitivity)
    }

    /// Set (or clear, with `None`) `device_key`'s thumb-wheel sensitivity
    /// override.
    pub fn set_device_thumbwheel_sensitivity(
        &mut self,
        device_key: &str,
        sensitivity: Option<i32>,
    ) {
        self.devices
            .entry(device_key.to_string())
            .or_default()
            .thumbwheel_sensitivity = sensitivity;
    }
}

/// Write `bytes` to `path` atomically via a randomized temp file + rename,
/// with the directory fsync the old hand-rolled writer lacked.
fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    #[cfg_attr(
        not(unix),
        expect(unused_mut, reason = "only the unix path mutates the options")
    )]
    let mut options = AtomicWriteFile::options();
    #[cfg(unix)]
    {
        use atomic_write_file::unix::OpenOptionsExt as _;
        use std::os::unix::fs::OpenOptionsExt as _;
        // Force 0600 on every save, matching the previous writer.
        options.preserve_mode(false).mode(0o600);
    }
    let mut file = options.open(path)?;
    io::Write::write_all(&mut file, bytes)?;
    file.commit()
}

#[cfg(test)]
#[allow(clippy::expect_used, reason = "expect/unwrap are idiomatic in tests")]
mod tests {
    use std::assert_matches;

    use super::*;
    use crate::binding::{default_binding, default_gesture_binding};

    fn write_and_read(config: &Config) -> Config {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        config.save_to_path(&path).expect("save");
        Config::load_from_path(&path).expect("load")
    }

    #[test]
    fn missing_file_yields_default() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("nonexistent.toml");
        let cfg = Config::load_from_path(&path).expect("load");
        assert_eq!(cfg.schema_version, SCHEMA_VERSION);
        assert!(cfg.devices.is_empty());
    }

    #[test]
    fn lighting_roundtrips_per_device() {
        let mut cfg = Config::default();
        cfg.set_lighting(
            "g513",
            Lighting {
                enabled: true,
                color: "00aabb".parse().expect("valid hex"),
                brightness: 75,
            },
        );
        let restored = write_and_read(&cfg);
        assert_eq!(
            restored.lighting("g513"),
            Some(Lighting {
                enabled: true,
                color: "00aabb".parse().expect("valid hex"),
                brightness: 75,
            })
        );
        assert_eq!(restored.lighting("absent"), None);
    }

    #[test]
    fn unparseable_lighting_color_falls_back_to_white() {
        let cfg: Config = toml::from_str(
            r#"
                schema_version = 3
                [devices.g513.lighting]
                enabled = true
                color = "red"
                brightness = 50
            "#,
        )
        .expect("config with a bad color still loads");
        assert_eq!(
            cfg.lighting("g513").map(|l| l.color),
            Some(crate::color::Rgb::WHITE)
        );
    }

    #[test]
    fn hash_prefixed_lighting_color_migrates_to_canonical_hex() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r##"
                schema_version = 3
                [devices.g513.lighting]
                enabled = true
                color = "#ff0000"
                brightness = 50
            "##,
        )
        .expect("write config");

        let cfg = Config::load_from_path(&path).expect("load hash-prefixed color");
        assert_eq!(
            cfg.lighting("g513").map(|lighting| lighting.color),
            Some(crate::color::Rgb::new(0xff, 0x00, 0x00))
        );

        cfg.save_to_path(&path).expect("save canonical color");
        let saved = fs::read_to_string(path).expect("read saved config");
        assert!(saved.contains("color = \"ff0000\""));
        assert!(!saved.contains("color = \"#"));
    }

    #[test]
    fn dpi_roundtrips_per_device() {
        let mut cfg = Config::default();
        cfg.set_dpi("2b042", 1600);
        let restored = write_and_read(&cfg);
        assert_eq!(restored.dpi("2b042"), Some(1600));
        assert_eq!(restored.dpi("absent"), None);
    }

    #[test]
    fn smartshift_roundtrips_per_device() {
        let mut cfg = Config::default();
        cfg.set_smartshift(
            "2b042",
            SmartShift {
                mode: WheelMode::Ratchet,
                auto_disengage: 16,
                tunable_torque: 30,
            },
        );
        let restored = write_and_read(&cfg);
        assert_eq!(
            restored.smartshift("2b042"),
            Some(SmartShift {
                mode: WheelMode::Ratchet,
                auto_disengage: 16,
                tunable_torque: 30,
            })
        );
        assert_eq!(restored.smartshift("absent"), None);
    }

    #[test]
    fn invert_scroll_roundtrips_per_device() {
        let mut cfg = Config::default();
        // Default is the native direction for any device, present or not.
        assert!(!cfg.invert_scroll("2b042"));
        cfg.set_invert_scroll("2b042", true);
        let restored = write_and_read(&cfg);
        assert!(restored.invert_scroll("2b042"));
        assert!(!restored.invert_scroll("absent"));
    }

    #[test]
    fn default_invert_scroll_is_omitted_from_toml() {
        // A device block with only the default (false) invert_scroll must not
        // emit the field — `skip_serializing_if` keeps configs clean.
        let mut cfg = Config::default();
        cfg.set_binding("2b042", ButtonId::Back, Binding::Single(Action::Copy));
        cfg.set_invert_scroll("2b042", false);
        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(
            !body.contains("invert_scroll"),
            "default invert_scroll should be omitted: {body}"
        );
    }

    #[test]
    fn scroll_resolution_roundtrips_all_three_states() {
        let mut cfg = Config::default();
        assert_eq!(cfg.scroll_resolution("mouse"), None);

        cfg.set_scroll_resolution("mouse", Some(ScrollResolution::Low));
        let low = write_and_read(&cfg);
        assert_eq!(low.scroll_resolution("mouse"), Some(ScrollResolution::Low));

        cfg.set_scroll_resolution("mouse", Some(ScrollResolution::High));
        let high = write_and_read(&cfg);
        assert_eq!(
            high.scroll_resolution("mouse"),
            Some(ScrollResolution::High)
        );

        cfg.set_scroll_resolution("mouse", None);
        let unmanaged = write_and_read(&cfg);
        assert_eq!(unmanaged.scroll_resolution("mouse"), None);
    }

    #[test]
    fn unset_scroll_resolution_is_omitted_from_toml() {
        let mut cfg = Config::default();
        cfg.set_binding("mouse", ButtonId::Back, Binding::Single(Action::Copy));
        cfg.set_scroll_resolution("mouse", Some(ScrollResolution::Low));
        cfg.set_scroll_resolution("mouse", None);

        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(
            !body.contains("scroll_resolution"),
            "unset scroll resolution should be omitted: {body}"
        );
    }

    #[test]
    fn config_without_scroll_resolution_loads_as_unmanaged() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(
            &path,
            r"
                schema_version = 3
                [devices.mouse]
                invert_scroll = true
            ",
        )
        .expect("write config");

        let cfg = Config::load_from_path(&path).expect("load existing config");
        assert_eq!(cfg.scroll_resolution("mouse"), None);
        assert!(cfg.invert_scroll("mouse"));
    }

    #[test]
    fn bindings_roundtrip_per_device() {
        let mut cfg = Config::default();
        cfg.set_binding("2b042", ButtonId::Back, Binding::Single(Action::Copy));
        cfg.set_binding(
            "2b042",
            ButtonId::DpiToggle,
            Binding::Single(Action::CustomShortcut(crate::binding::KeyCombo {
                modifiers: crate::binding::KeyCombo::MOD_CMD,
                key_code: 0x23, // kVK_ANSI_P
                display: "⌘P".into(),
            })),
        );
        cfg.set_binding("4082d", ButtonId::Back, Binding::Single(Action::Paste));

        let parsed = write_and_read(&cfg);

        // Per-device isolation.
        let a = parsed.bindings_for("2b042");
        assert_eq!(a.get(&ButtonId::Back), Some(&Binding::Single(Action::Copy)));
        assert_eq!(
            a.get(&ButtonId::DpiToggle),
            Some(&Binding::Single(Action::CustomShortcut(
                crate::binding::KeyCombo {
                    modifiers: crate::binding::KeyCombo::MOD_CMD,
                    key_code: 0x23,
                    display: "⌘P".into(),
                }
            )))
        );

        let b = parsed.bindings_for("4082d");
        assert_eq!(
            b.get(&ButtonId::Back),
            Some(&Binding::Single(Action::Paste))
        );
        assert_eq!(b.len(), 1, "device b should only see its own bindings");

        // Unknown device returns empty map without panic.
        assert!(parsed.bindings_for("deadbeef").is_empty());
    }

    #[test]
    fn human_readable_toml_layout() {
        let mut cfg = Config::default();
        cfg.set_binding(
            "2b042",
            ButtonId::Back,
            Binding::Single(Action::BrowserBack),
        );
        let body = toml::to_string_pretty(&cfg).expect("serialize");

        // The key only contains [A-Za-z0-9_], so TOML emits it as a bare-word
        // table key (no surrounding quotes). The test asserts the observable
        // structure rather than locking in a specific quoting.
        assert!(body.contains("schema_version = 4"), "got: {body}");
        assert!(body.contains("[devices.2b042.bindings]"), "got: {body}");
        // A `Single` binding serializes byte-identically to the pre-v2 bare
        // `Action`, so the leaf line is unchanged.
        assert!(body.contains("Back = \"BrowserBack\""), "got: {body}");
    }

    #[test]
    fn dpi_presets_roundtrip_per_device() {
        let mut cfg = Config::default();
        cfg.set_dpi_presets("2b042", vec![800, 1600, 3200]);
        cfg.set_dpi_presets("4082d", vec![400, 1600]);

        let parsed = write_and_read(&cfg);

        assert_eq!(parsed.dpi_presets("2b042"), vec![800, 1600, 3200]);
        assert_eq!(parsed.dpi_presets("4082d"), vec![400, 1600]);
        assert!(parsed.dpi_presets("unknown").is_empty());
    }

    #[test]
    fn empty_dpi_presets_skip_serialization() {
        let mut cfg = Config::default();
        // Add a binding so the device block exists.
        cfg.set_binding("2b042", ButtonId::Back, Binding::Single(Action::Copy));
        cfg.set_dpi_presets("2b042", vec![800]);
        cfg.set_dpi_presets("2b042", vec![]); // clear

        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(
            !body.contains("dpi_presets"),
            "empty dpi_presets should be omitted: {body}"
        );
    }

    #[test]
    fn device_identity_roundtrips_and_is_iterable() {
        use crate::device::{Capabilities, DeviceKind};

        let mut cfg = Config::default();
        let mouse = DeviceIdentity {
            display_name: "MX Master 3S".to_string(),
            model_info: None,
            codename: None,
            kind: DeviceKind::Mouse,
            capabilities: Capabilities {
                buttons: true,
                pointer: true,
                lighting: false,
                scroll_inversion: false,
                hires_wheel: true,
            },
        };
        cfg.set_device_identity("2b034", mouse.clone());
        // Recording an identity must not disturb unrelated per-device state.
        cfg.set_binding(
            "2b034",
            ButtonId::Back,
            Binding::Single(Action::BrowserBack),
        );

        let parsed = write_and_read(&cfg);
        assert_eq!(parsed.device_identity("2b034"), Some(&mouse));
        assert_eq!(parsed.device_identity("absent"), None);
        assert_eq!(
            parsed.bindings_for("2b034").get(&ButtonId::Back),
            Some(&Binding::Single(Action::BrowserBack)),
            "identity must coexist with bindings on the same device block"
        );
        assert_eq!(
            parsed.known_identities().collect::<Vec<_>>(),
            vec![("2b034", &mouse)]
        );
    }

    #[test]
    fn selected_device_roundtrips() {
        let mut cfg = Config::default();
        assert_eq!(cfg.selected_device(), None);
        cfg.set_selected_device(Some("2b042".into()));
        let parsed = write_and_read(&cfg);
        assert_eq!(parsed.selected_device(), Some("2b042"));
    }

    #[test]
    fn per_app_overlay_takes_precedence() {
        let mut cfg = Config::default();
        cfg.set_binding(
            "2b042",
            ButtonId::Back,
            Binding::Single(Action::BrowserBack),
        );
        cfg.set_binding(
            "2b042",
            ButtonId::Forward,
            Binding::Single(Action::BrowserForward),
        );
        cfg.set_per_app_binding(
            "2b042",
            "com.microsoft.VSCode",
            ButtonId::Back,
            Some(Action::Undo),
        );

        // Global: both buttons are browser nav.
        let global = cfg.effective_bindings("2b042", None);
        assert_eq!(
            global.get(&ButtonId::Back),
            Some(&Binding::Single(Action::BrowserBack))
        );
        assert_eq!(
            global.get(&ButtonId::Forward),
            Some(&Binding::Single(Action::BrowserForward))
        );

        // VSCode: Back overridden (wrapped as Single), Forward inherits.
        let vscode = cfg.effective_bindings("2b042", Some("com.microsoft.VSCode"));
        assert_eq!(
            vscode.get(&ButtonId::Back),
            Some(&Binding::Single(Action::Undo))
        );
        assert_eq!(
            vscode.get(&ButtonId::Forward),
            Some(&Binding::Single(Action::BrowserForward))
        );

        // Unrelated app falls through.
        let other = cfg.effective_bindings("2b042", Some("com.apple.Safari"));
        assert_eq!(
            other.get(&ButtonId::Back),
            Some(&Binding::Single(Action::BrowserBack))
        );
    }

    #[test]
    fn per_app_binding_removal_prunes_empty_app() {
        let mut cfg = Config::default();
        cfg.set_per_app_binding(
            "2b042",
            "com.example.App",
            ButtonId::Back,
            Some(Action::Copy),
        );
        cfg.set_per_app_binding("2b042", "com.example.App", ButtonId::Back, None);
        assert!(
            cfg.devices["2b042"].per_app_bindings.is_empty(),
            "removing last override should prune the app entry"
        );
    }

    #[test]
    fn app_settings_default_omits_block() {
        let cfg = Config::default();
        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(
            !body.contains("app_settings"),
            "default app_settings should be omitted: {body}"
        );
    }

    #[test]
    fn app_settings_launch_at_login_roundtrips() {
        let mut cfg = Config::default();
        cfg.app_settings.launch_at_login = true;
        let parsed = write_and_read(&cfg);
        assert!(parsed.app_settings.launch_at_login);
    }

    #[test]
    fn asset_source_preference_roundtrips() {
        let mut cfg = Config::default();
        cfg.app_settings.asset_source = AssetSourcePreference::OpenLogi;

        let body = toml::to_string_pretty(&cfg).expect("serialize");
        let parsed = write_and_read(&cfg);

        assert!(body.contains("asset_source = \"openlogi\""));
        assert_eq!(
            parsed.app_settings.asset_source,
            AssetSourcePreference::OpenLogi
        );
    }

    #[test]
    fn config_without_asset_source_keeps_automatic_selection() {
        let parsed: Config = toml::from_str(
            r"
                schema_version = 3
                [app_settings]
                auto_download_assets = false
            ",
        )
        .expect("config predating the asset-source setting loads");

        assert_eq!(
            parsed.app_settings.asset_source,
            AssetSourcePreference::Automatic
        );
    }

    #[test]
    fn cleared_selected_device_omits_field() {
        let mut cfg = Config::default();
        cfg.set_selected_device(Some("2b042".into()));
        cfg.set_selected_device(None);
        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(
            !body.contains("selected_device"),
            "cleared selection should not appear: {body}"
        );
    }

    #[test]
    fn empty_device_block_is_skipped_in_output() {
        // Inserting then clearing should not leave a [devices."x"] header
        // with no bindings under it (skip_serializing_if on bindings).
        let mut cfg = Config::default();
        cfg.set_binding("2b042", ButtonId::Back, Binding::Single(Action::Copy));
        cfg.devices
            .get_mut("2b042")
            .expect("entry")
            .bindings
            .clear();
        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(
            !body.contains("Back"),
            "cleared bindings should not appear: {body}"
        );
    }

    #[test]
    fn migrates_v1_button_and_gesture_bindings() {
        // A pre-v2 file: split button_bindings + a flat gesture_bindings map.
        let v1 = "\
schema_version = 1

[devices.2b042.button_bindings]
Back = \"BrowserBack\"

[devices.2b042.gesture_bindings]
Up = \"Copy\"
Click = \"Paste\"
";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, v1).expect("write");

        // v1 still loads (version <= current) and folds into the merged map.
        let cfg = Config::load_from_path(&path).expect("load v1");
        let bindings = cfg.bindings_for("2b042");
        assert_eq!(
            bindings.get(&ButtonId::Back),
            Some(&Binding::Single(Action::BrowserBack))
        );
        let mut gesture = BTreeMap::new();
        gesture.insert(GestureDirection::Up, Action::Copy);
        gesture.insert(GestureDirection::Click, Action::Paste);
        assert_eq!(
            bindings.get(&ButtonId::GestureButton),
            Some(&Binding::Gesture(gesture))
        );

        // Saving self-heals to the current shape: stamped version + merged table,
        // legacy field names gone.
        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(body.contains("schema_version = 4"), "got: {body}");
        assert!(body.contains("[devices.2b042.bindings]"), "got: {body}");
        assert!(!body.contains("button_bindings"), "got: {body}");
        assert!(!body.contains("gesture_bindings"), "got: {body}");
    }

    #[test]
    fn migration_gesture_map_wins_over_legacy_single_gesture_button_entry() {
        // The data-loss guard: when a legacy single button_bindings[GestureButton]
        // entry coexists with a gesture_bindings map (reachable via hand-edited
        // or very old configs), the gesture map must survive — not be shadowed by
        // the single entry. Mirrors the pre-v2 "gesture entries win" rule.
        let v1 = "\
schema_version = 1

[devices.2b042.button_bindings]
GestureButton = \"MissionControl\"

[devices.2b042.gesture_bindings]
Up = \"Copy\"
Down = \"Paste\"
";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, v1).expect("write");

        let cfg = Config::load_from_path(&path).expect("load v1");
        let mut gesture = BTreeMap::new();
        gesture.insert(GestureDirection::Up, Action::Copy);
        gesture.insert(GestureDirection::Down, Action::Paste);
        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::GestureButton),
            Some(&Binding::Gesture(gesture)),
            "gesture map must win over the legacy single GestureButton entry"
        );
    }

    #[test]
    fn migration_drops_vestigial_lone_gesture_button_single() {
        // A v1 file with only `button_bindings[GestureButton]` and no
        // `gesture_bindings` (the pre-gesture-picker shape). That entry never
        // dispatched in v1 — the gesture button's plain press routes through the
        // gesture `Click` slot, not the per-button map — so migrating it to a
        // `Binding::Single` would leave an unreachable entry the GUI hides and the
        // runtime ignores. It must be dropped, not shadow the gesture path.
        let v1 = "\
schema_version = 1

[devices.2b042.button_bindings]
GestureButton = \"MissionControl\"
Back = \"BrowserBack\"
";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, v1).expect("write");

        let bindings = Config::load_from_path(&path)
            .expect("load v1")
            .bindings_for("2b042");
        // An ordinary button still migrates to a `Single`...
        assert_eq!(
            bindings.get(&ButtonId::Back),
            Some(&Binding::Single(Action::BrowserBack))
        );
        // ...but the vestigial gesture-button single is gone, leaving the button
        // to fall back to its canonical default rather than an unreachable entry.
        assert_eq!(bindings.get(&ButtonId::GestureButton), None);
    }

    #[test]
    fn rejects_newer_schema_version_but_accepts_v1() {
        // A future version is rejected loudly; the current and older versions
        // load (older ones migrate through the shim).
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, "schema_version = 99\n").expect("write");
        assert_matches!(
            Config::load_from_path(&path).expect_err("v99 should fail"),
            ConfigError::UnsupportedSchemaVersion { found: 99, .. }
        );

        fs::write(&path, "schema_version = 1\n").expect("write");
        assert!(
            Config::load_from_path(&path).is_ok(),
            "v1 should still load"
        );
    }

    #[test]
    fn set_gesture_direction_upgrades_single_to_gesture() {
        let mut cfg = Config::default();
        // Start from a Single binding, then bind a swipe direction.
        cfg.set_binding(
            "2b042",
            ButtonId::Back,
            Binding::Single(Action::BrowserBack),
        );
        cfg.set_gesture_direction("2b042", ButtonId::Back, GestureDirection::Up, Action::Copy);

        match cfg.bindings_for("2b042").get(&ButtonId::Back) {
            Some(Binding::Gesture(map)) => {
                // The prior single action is preserved as the Click entry.
                assert_eq!(
                    map.get(&GestureDirection::Click),
                    Some(&Action::BrowserBack)
                );
                assert_eq!(map.get(&GestureDirection::Up), Some(&Action::Copy));
            }
            other => panic!("expected Gesture after upgrade, got {other:?}"),
        }
    }

    #[test]
    fn set_gesture_direction_on_fresh_gesture_button_seeds_click() {
        // Binding one direction on a never-configured gesture button must still
        // persist a `Click`, so the click projection is the canonical default
        // rather than `Action::None` (which reads as a no-op press).
        let mut cfg = Config::default();
        cfg.set_gesture_direction(
            "2b042",
            ButtonId::GestureButton,
            GestureDirection::Up,
            Action::Copy,
        );

        match cfg.bindings_for("2b042").get(&ButtonId::GestureButton) {
            Some(Binding::Gesture(map)) => {
                assert_eq!(map.get(&GestureDirection::Up), Some(&Action::Copy));
                assert_eq!(
                    map.get(&GestureDirection::Click),
                    Some(&crate::binding::default_gesture_binding(
                        GestureDirection::Click
                    )),
                    "a fresh gesture button must seed a Click from its default"
                );
            }
            other => panic!("expected Gesture, got {other:?}"),
        }
    }

    #[test]
    fn gesture_owner_defaults_to_hidpp_button_yields_to_oshook_and_can_be_off() {
        let mut cfg = Config::default();
        // Default: the dedicated HID++ gesture button owns the gesture role even with no config.
        assert_eq!(cfg.gesture_owner("2b042"), Some(ButtonId::GestureButton));

        // A dedicated HID++ gesture binding keeps it the owner.
        cfg.set_gesture_direction(
            "2b042",
            ButtonId::GestureButton,
            GestureDirection::Up,
            Action::MissionControl,
        );
        assert_eq!(cfg.gesture_owner("2b042"), Some(ButtonId::GestureButton));

        // An explicit OS-hook gesture button takes the role over.
        cfg.set_binding(
            "2b042",
            ButtonId::Forward,
            Binding::Gesture(BTreeMap::from([(GestureDirection::Up, Action::Copy)])),
        );
        assert_eq!(cfg.gesture_owner("2b042"), Some(ButtonId::Forward));

        // Turning gestures off explicitly yields `None` (not the HID++ button default).
        let mut off = Config::default();
        off.disable_gestures("2b042");
        assert_eq!(off.gesture_owner("2b042"), None);
    }

    #[test]
    fn set_gesture_owner_shim_switches_by_demoting_the_previous_button() {
        // The transition shim keeps the retired selector's exclusive semantics
        // on top of the shape-driven model: switching demotes the previous
        // gesture button to a Single of its Click (no dormant maps — a stored
        // gesture map always MEANS gesture mode).
        let mut cfg = Config::default();
        cfg.set_gesture_direction(
            "2b042",
            ButtonId::GestureButton,
            GestureDirection::Up,
            Action::Copy,
        );
        assert_eq!(cfg.gesture_owner("2b042"), Some(ButtonId::GestureButton));

        cfg.set_binding("2b042", ButtonId::Back, Action::BrowserBack.into());
        cfg.set_gesture_owner("2b042", ButtonId::Back);
        assert_eq!(cfg.gesture_owner("2b042"), Some(ButtonId::Back));

        let bindings = cfg.bindings_for("2b042");
        // Back is a full five-direction gesture button: its prior single action
        // stays as Click, and the swipe arms are seeded from defaults.
        match bindings.get(&ButtonId::Back) {
            Some(Binding::Gesture(map)) => {
                assert_eq!(
                    map.get(&GestureDirection::Click),
                    Some(&Action::BrowserBack)
                );
                assert_eq!(
                    map.get(&GestureDirection::Up),
                    Some(&default_gesture_binding(GestureDirection::Up)),
                    "a promoted button gets full default arms"
                );
            }
            other => panic!("expected Back to be a gesture binding, got {other:?}"),
        }
        // The dedicated button demoted to its Click choice (seeded AppExpose).
        assert_eq!(
            bindings.get(&ButtonId::GestureButton),
            Some(&Binding::Single(default_gesture_binding(
                GestureDirection::Click
            ))),
            "the previous gesture button demotes to its Click"
        );
        assert!(!cfg.is_gesture_mode("2b042", ButtonId::GestureButton));
    }

    #[test]
    fn set_gesture_owner_seeds_a_fresh_button_with_full_directions() {
        let mut cfg = Config::default();
        // The dedicated HID++ gesture button gets the full default direction map.
        cfg.set_gesture_owner("2b042", ButtonId::GestureButton);
        match cfg.bindings_for("2b042").get(&ButtonId::GestureButton) {
            Some(Binding::Gesture(map)) => {
                for dir in GestureDirection::ALL {
                    assert_eq!(map.get(&dir), Some(&default_gesture_binding(dir)));
                }
            }
            other => panic!("expected full default gesture map, got {other:?}"),
        }

        // A fresh OS-hook button also gets all five directions, not just a Click:
        // its native action stays as Click, and the swipe arms are defaults — so
        // the GUI's shown defaults are exactly what the runtime dispatches.
        cfg.set_gesture_owner("2b042", ButtonId::Forward);
        match cfg.bindings_for("2b042").get(&ButtonId::Forward) {
            Some(Binding::Gesture(map)) => {
                assert_eq!(
                    map.get(&GestureDirection::Click),
                    Some(&default_binding(ButtonId::Forward))
                );
                for dir in [
                    GestureDirection::Up,
                    GestureDirection::Down,
                    GestureDirection::Left,
                    GestureDirection::Right,
                ] {
                    assert_eq!(map.get(&dir), Some(&default_gesture_binding(dir)));
                }
            }
            other => panic!("expected full gesture map for Forward, got {other:?}"),
        }
    }

    #[test]
    fn disable_gestures_shim_demotes_every_gesture_button() {
        // The transition shim's "off" flattens each gesture-mode button to a
        // Single of its Click — the shape IS the state, so nothing dormant
        // survives to resurrect unexpectedly.
        let mut cfg = Config::default();
        cfg.set_gesture_direction(
            "2b042",
            ButtonId::GestureButton,
            GestureDirection::Up,
            Action::Copy,
        );
        cfg.disable_gestures("2b042");

        assert_eq!(cfg.gesture_owner("2b042"), None);
        assert!(cfg.gesture_mode_buttons("2b042").is_empty());
        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::GestureButton),
            Some(&Binding::Single(default_gesture_binding(
                GestureDirection::Click
            ))),
            "off keeps the map's Click as the button's press"
        );
    }

    #[test]
    fn gesture_state_roundtrips_through_shapes_without_the_owner_field() {
        // Since v4 the binding shape is the whole persisted truth: gesture
        // state survives a save/load cycle with no `gesture_owner` scalar in
        // the document.
        let mut cfg = Config::default();
        cfg.set_gesture_mode("2b042", ButtonId::Back, true);
        cfg.disable_gestures("4082d");

        let parsed = write_and_read(&cfg);
        assert!(parsed.is_gesture_mode("2b042", ButtonId::Back));
        assert!(
            parsed.is_gesture_mode("2b042", ButtonId::GestureButton),
            "the dedicated button's default gesture mode is untouched by Back's promotion"
        );
        assert!(parsed.gesture_mode_buttons("4082d").is_empty());

        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(!body.contains("gesture_owner"), "got: {body}");
    }

    #[test]
    fn invalid_gesture_owner_string_is_tolerated_not_fatal() {
        // A hand-edit typo in gesture_owner must NOT fail the whole-document parse
        // (which would revert every device's settings to defaults). It degrades
        // to "infer" while the rest of the device config survives.
        let toml = "\
schema_version = 2

[devices.2b042]
gesture_owner = \"bogus\"

[devices.2b042.bindings]
Back = \"Copy\"
";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, toml).expect("write");

        let cfg =
            Config::load_from_path(&path).expect("an invalid gesture_owner must not fail the load");
        // The rest of the device config survived...
        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::Back),
            Some(&Binding::Single(Action::Copy))
        );
        // ...and the bad owner degraded to inference (HID++ button default here).
        assert_eq!(cfg.gesture_owner("2b042"), Some(ButtonId::GestureButton));
    }

    // ── Shape-driven gesture mode (the owner lock removed) ──

    #[test]
    fn gesture_mode_is_per_button_and_not_exclusive() {
        // The owner lock is gone: promoting a second button must not demote the
        // dedicated gesture button's default gesture mode.
        let mut cfg = Config::default();
        cfg.set_gesture_mode("2b042", ButtonId::MiddleClick, true);

        assert!(cfg.is_gesture_mode("2b042", ButtonId::GestureButton));
        assert!(cfg.is_gesture_mode("2b042", ButtonId::MiddleClick));
        let buttons = cfg.gesture_mode_buttons("2b042");
        assert!(
            buttons.contains(&ButtonId::GestureButton),
            "got: {buttons:?}"
        );
        assert!(buttons.contains(&ButtonId::MiddleClick), "got: {buttons:?}");
    }

    #[test]
    fn set_gesture_mode_on_keeps_click_and_seeds_directions() {
        // Promoting a single-bound button keeps its action as the Click entry
        // and seeds every swipe arm, so the button exposes the full five-way set.
        let mut cfg = Config::default();
        cfg.set_binding("2b042", ButtonId::Back, Binding::Single(Action::Copy));
        cfg.set_gesture_mode("2b042", ButtonId::Back, true);

        let bindings = cfg.bindings_for("2b042");
        let Some(Binding::Gesture(map)) = bindings.get(&ButtonId::Back) else {
            panic!(
                "expected a gesture binding, got {:?}",
                bindings.get(&ButtonId::Back)
            );
        };
        assert_eq!(map.get(&GestureDirection::Click), Some(&Action::Copy));
        for dir in [
            GestureDirection::Up,
            GestureDirection::Down,
            GestureDirection::Left,
            GestureDirection::Right,
        ] {
            assert_eq!(
                map.get(&dir),
                Some(&default_gesture_binding(dir)),
                "unseeded arm {dir:?}"
            );
        }
    }

    #[test]
    fn set_gesture_mode_off_demotes_to_the_click_action() {
        let mut cfg = Config::default();
        cfg.set_gesture_direction(
            "2b042",
            ButtonId::GestureButton,
            GestureDirection::Click,
            Action::Paste,
        );
        cfg.set_gesture_mode("2b042", ButtonId::GestureButton, false);

        assert!(!cfg.is_gesture_mode("2b042", ButtonId::GestureButton));
        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::GestureButton),
            Some(&Binding::Single(Action::Paste))
        );
    }

    #[test]
    fn set_gesture_mode_off_without_click_falls_back_to_the_default() {
        // A sparse hand-edited map with no Click must not demote to a dead
        // Single(None) — the button falls back to its canonical single default.
        let mut cfg = Config::default();
        let mut map = BTreeMap::new();
        map.insert(GestureDirection::Up, Action::Copy);
        cfg.set_binding("2b042", ButtonId::Back, Binding::Gesture(map));
        cfg.set_gesture_mode("2b042", ButtonId::Back, false);

        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::Back),
            Some(&Binding::Single(Action::BrowserBack))
        );
    }

    #[test]
    fn migration_demotes_the_dormant_non_owner_gesture_maps() {
        // An owner-locked config: Middle owns gestures; the dedicated button
        // keeps a dormant map from an earlier reign. Shape-driven load keeps the
        // owner gesturing and flattens the dormant map to its Click, so a stored
        // map's presence always MEANS gesture mode.
        let toml = "\
schema_version = 3

[devices.2b042]
gesture_owner = \"MiddleClick\"

[devices.2b042.bindings]
GestureButton = { Up = \"Copy\", Click = \"Paste\" }
MiddleClick = { Up = \"MissionControl\", Click = \"MiddleClick\" }
";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, toml).expect("write");

        let cfg = Config::load_from_path(&path).expect("load");
        assert!(cfg.is_gesture_mode("2b042", ButtonId::MiddleClick));
        assert!(!cfg.is_gesture_mode("2b042", ButtonId::GestureButton));
        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::GestureButton),
            Some(&Binding::Single(Action::Paste)),
            "the dormant map demotes to its Click choice"
        );
        // The legacy field is gone on save — the binding shape is the whole truth.
        let body = toml::to_string_pretty(&cfg).expect("serialize");
        assert!(!body.contains("gesture_owner"), "got: {body}");
    }

    #[test]
    fn migration_off_pins_the_dedicated_button_out_of_gesture_mode() {
        // gesture_owner = "Off" with no stored bindings: absence would re-enter
        // default gesture mode under shape rules, so the load pins the dedicated
        // button with an explicit Single at its canonical default (which the
        // capture layer treats as native/undiverted).
        let toml = "\
schema_version = 3

[devices.2b042]
gesture_owner = \"Off\"
";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, toml).expect("write");

        let cfg = Config::load_from_path(&path).expect("load");
        assert!(!cfg.is_gesture_mode("2b042", ButtonId::GestureButton));
        assert!(cfg.gesture_mode_buttons("2b042").is_empty());
        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::GestureButton),
            Some(&Binding::Single(default_binding(ButtonId::GestureButton)))
        );
    }

    #[test]
    fn migration_infers_the_owner_for_pre_field_configs() {
        // Pre-owner-field file: a gesture-shaped OS-hook button was the owner by
        // inference, silencing the dedicated button. The load preserves exactly
        // that: Back keeps gesturing, the dedicated button is pinned off.
        let toml = "\
schema_version = 2

[devices.2b042.bindings]
Back = { Up = \"Copy\" }
";
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("config.toml");
        fs::write(&path, toml).expect("write");

        let cfg = Config::load_from_path(&path).expect("load");
        assert!(cfg.is_gesture_mode("2b042", ButtonId::Back));
        assert!(!cfg.is_gesture_mode("2b042", ButtonId::GestureButton));
        assert_eq!(
            cfg.bindings_for("2b042").get(&ButtonId::GestureButton),
            Some(&Binding::Single(default_binding(ButtonId::GestureButton)))
        );
    }
}
