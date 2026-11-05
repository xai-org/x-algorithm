use crate::metrics::{CAROUSEL_MODELS_LOADED, CAROUSEL_MODEL_RELOADS};
use arc_swap::ArcSwap;
use bytes::Bytes;
use log::{info, warn};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

pub struct CarouselModelStore {
    dir: PathBuf,
    files: ArcSwap<HashMap<String, Bytes>>,
}

impl CarouselModelStore {
    pub fn load(dir: PathBuf) -> Arc<Self> {
        let files = read_dir(&dir, &HashMap::new());
        CAROUSEL_MODELS_LOADED.set(files.len() as i64);
        info!(
            "carousel models loaded from {}: {:?}",
            dir.display(),
            files.keys().collect::<Vec<_>>()
        );
        Arc::new(Self {
            dir,
            files: ArcSwap::from_pointee(files),
        })
    }

    pub fn get(&self, name: &str) -> Option<Bytes> {
        self.files.load().get(name).cloned()
    }

    pub fn spawn_reloader(self: &Arc<Self>, every: Duration) {
        let store = Arc::clone(self);
        tokio::spawn(async move {
            let mut ticks = tokio::time::interval(every);
            ticks.tick().await;
            loop {
                ticks.tick().await;
                let store = Arc::clone(&store);
                let reloaded = tokio::task::spawn_blocking(move || {
                    let files = read_dir(&store.dir, &store.files.load());
                    CAROUSEL_MODELS_LOADED.set(files.len() as i64);
                    store.files.store(Arc::new(files));
                })
                .await;
                if let Err(e) = reloaded {
                    warn!("carousel model reload task failed: {e}");
                }
            }
        });
    }
}

fn read_dir(dir: &Path, previous: &HashMap<String, Bytes>) -> HashMap<String, Bytes> {
    let entries = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) => {
            warn!("carousel models: cannot read {}: {e}", dir.display());
            CAROUSEL_MODEL_RELOADS
                .with_label_values(&["dir_error"])
                .inc();
            return previous.clone();
        }
    };
    let mut files = HashMap::new();
    let mut outcome = "ok";
    for path in entries.flatten().map(|entry| entry.path()) {
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|stem| stem.to_str()) else {
            continue;
        };
        match std::fs::read(&path) {
            Ok(bytes) => {
                files.insert(name.to_string(), Bytes::from(bytes));
            }
            Err(e) => {
                warn!("carousel model {}: {e}", path.display());
                outcome = "file_error";
                if let Some(bytes) = previous.get(name) {
                    files.insert(name.to_string(), bytes.clone());
                }
            }
        }
    }
    CAROUSEL_MODEL_RELOADS.with_label_values(&[outcome]).inc();
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_json_files_by_name_and_keeps_previous_copies_when_the_dir_is_missing() {
        let dir = std::env::temp_dir().join(format!("carousel_models_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("lr_v2.json"), b"{\"a\":1}").unwrap();
        std::fs::write(dir.join("notes.txt"), b"ignored").unwrap();

        let files = read_dir(&dir, &HashMap::new());
        assert_eq!(files.keys().collect::<Vec<_>>(), vec!["lr_v2"]);
        assert_eq!(&files["lr_v2"][..], b"{\"a\":1}");

        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(read_dir(&dir, &files), files);
    }
}
