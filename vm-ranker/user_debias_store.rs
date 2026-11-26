use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwapOption;
use log::{error, info, warn};
use xai_user_debias::UserDebiasModel;

use crate::metrics::{
    USER_DEBIAS_MODEL_LOADED, USER_DEBIAS_MODEL_RELOADS, USER_DEBIAS_MODEL_TRAINED_AT_MS,
    USER_DEBIAS_MODEL_TRAIN_RECORDS,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReloadOutcome {
    Loaded,
    Unchanged,
}

pub struct UserDebiasStore {
    path: PathBuf,
    model: ArcSwapOption<UserDebiasModel>,
    loaded_fingerprint: AtomicU64,
    changed_at_ms: AtomicU64,
}

impl UserDebiasStore {
    pub fn load(path: PathBuf) -> Arc<Self> {
        let store = Arc::new(Self {
            path,
            model: ArcSwapOption::empty(),
            loaded_fingerprint: AtomicU64::new(0),
            changed_at_ms: AtomicU64::new(0),
        });
        USER_DEBIAS_MODEL_LOADED.set(0);
        USER_DEBIAS_MODEL_TRAINED_AT_MS.set(0);
        USER_DEBIAS_MODEL_TRAIN_RECORDS.set(0);
        match store.reload() {
            Ok(_) => info!("user-debias model loaded from {}", store.path.display()),
            Err(e) => error!(
                "user-debias model unavailable at startup ({}): {e}",
                store.path.display()
            ),
        }
        store
    }

    pub fn model(&self) -> Option<Arc<UserDebiasModel>> {
        self.model.load_full()
    }

    pub fn changed_at_ms(&self) -> u64 {
        self.changed_at_ms.load(Ordering::Acquire)
    }

    pub fn spawn_reloader(self: &Arc<Self>, every: Duration) {
        let store = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticks = tokio::time::interval(every);
            ticks.tick().await;
            loop {
                ticks.tick().await;
                let store = Arc::clone(&store);
                let result = tokio::task::spawn_blocking(move || store.reload_and_record()).await;
                if let Err(e) = result {
                    error!("user-debias reload task failed: {e}");
                }
            }
        });
    }

    fn reload_and_record(&self) {
        match self.reload() {
            Ok(ReloadOutcome::Loaded) => {
                USER_DEBIAS_MODEL_RELOADS.with_label_values(&["ok"]).inc();
                info!("user-debias model reloaded from {}", self.path.display());
            }
            Ok(ReloadOutcome::Unchanged) => {
                USER_DEBIAS_MODEL_RELOADS
                    .with_label_values(&["unchanged"])
                    .inc();
            }
            Err(e) => {
                USER_DEBIAS_MODEL_RELOADS
                    .with_label_values(&["error"])
                    .inc();
                warn!(
                    "user-debias model reload failed, keeping current model ({}): {e}",
                    self.path.display()
                );
            }
        }
    }

    fn reload(&self) -> Result<ReloadOutcome, String> {
        let bytes = std::fs::read(&self.path).map_err(|e| format!("read: {e}"))?;
        let fingerprint = fingerprint(&bytes);
        if self.model.load().is_some()
            && fingerprint == self.loaded_fingerprint.load(Ordering::Acquire)
        {
            return Ok(ReloadOutcome::Unchanged);
        }
        let model = UserDebiasModel::from_bytes(&bytes).map_err(|e| format!("parse: {e}"))?;
        USER_DEBIAS_MODEL_TRAINED_AT_MS.set(model.trained_at_ms());
        USER_DEBIAS_MODEL_TRAIN_RECORDS.set(model.train_records());
        self.model.store(Some(Arc::new(model)));
        self.loaded_fingerprint
            .store(fingerprint, Ordering::Release);
        self.changed_at_ms.store(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            Ordering::Release,
        );
        USER_DEBIAS_MODEL_LOADED.set(1);
        Ok(ReloadOutcome::Loaded)
    }
}

fn fingerprint(bytes: &[u8]) -> u64 {
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}
