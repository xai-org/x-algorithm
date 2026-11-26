use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use deadpool_postgres::{Manager, ManagerConfig, Pool, RecyclingMethod, Runtime};
use tracing::info;

use crate::facts::EntityType;

pub const STALE_AFTER: Duration = Duration::from_secs(600);

const STATEMENT_TIMEOUT: Duration = Duration::from_secs(5);
const POOL_MAX: usize = 8;
const POOL_WAIT_TIMEOUT: Duration = Duration::from_secs(5);

pub fn new_claim_nonce() -> i64 {
    use std::hash::{BuildHasher, Hasher};
    use std::sync::atomic::{AtomicU64, Ordering};
    static N: AtomicU64 = AtomicU64::new(0);
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u64(N.fetch_add(1, Ordering::Relaxed));
    h.write_u128(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos(),
    );
    (h.finish() >> 1) as i64
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ClaimKey {
    pub run_id: String,
    pub entity: EntityType,
    pub entity_id: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemClaim {
    NotInList,
    Free,
    Handed { nonce: i64, age: Duration },
    Finished { status: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Take {
    Taken,
    Held,
    BudgetReached,
    NotInList,
}

#[async_trait]
pub trait ClaimStore: Send + Sync {
    async fn state(&self, key: &ClaimKey) -> anyhow::Result<ItemClaim>;
    async fn take(
        &self,
        key: &ClaimKey,
        nonce: i64,
        budget: u64,
        pod: &str,
    ) -> anyhow::Result<Take>;
    async fn finish(&self, key: &ClaimKey, nonce: i64, status: &str) -> anyhow::Result<()>;
}

pub struct PgClaimStore {
    pool: Pool,
}

impl PgClaimStore {
    pub fn new(dsn: &str, env: &str) -> anyhow::Result<Self> {
        let app_name = format!("xai-abuse-enforcement-service-backfill/{env}");
        let config = xai_abuse_ledger_service::pg::parse_dsn(dsn, &app_name, STATEMENT_TIMEOUT)
            .map_err(|e| anyhow::anyhow!("BACKFILL_CLAIMS_PG_DSN {e}"))?;
        let manager = Manager::from_config(
            config,
            xai_abuse_ledger_service::pg::make_tls(),
            ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            },
        );
        let pool = Pool::builder(manager)
            .max_size(POOL_MAX)
            .runtime(Runtime::Tokio1)
            .wait_timeout(Some(POOL_WAIT_TIMEOUT))
            .create_timeout(Some(POOL_WAIT_TIMEOUT))
            .build()
            .map_err(|e| anyhow::anyhow!("backfill claims pool: {e}"))?;
        info!(
            application_name = app_name,
            max_size = POOL_MAX,
            "backfill claims pool built (lazy)"
        );
        Ok(Self { pool })
    }

    #[cfg(test)]
    pub(crate) fn new_plaintext_for_test(dsn: &str) -> anyhow::Result<Self> {
        let config: tokio_postgres::Config = dsn.parse()?;
        let manager = Manager::from_config(
            config,
            tokio_postgres::NoTls,
            ManagerConfig {
                recycling_method: RecyclingMethod::Fast,
            },
        );
        let pool = Pool::builder(manager)
            .max_size(POOL_MAX)
            .runtime(Runtime::Tokio1)
            .build()?;
        Ok(Self { pool })
    }
}

const STATE_SQL: &str = "SELECT claim_state, claim_nonce, \
     (extract(epoch FROM (now() - claimed_at)) * 1000)::bigint, claim_status \
     FROM log_only.backfill_items WHERE run_id = $1 AND entity_type = $2 AND entity_id = $3";

fn row_claim(row: Option<tokio_postgres::Row>) -> anyhow::Result<ItemClaim> {
    let Some(row) = row else {
        return Ok(ItemClaim::NotInList);
    };
    let state: Option<String> = row.try_get(0)?;
    Ok(match state.as_deref() {
        None => ItemClaim::Free,
        Some("finished") => ItemClaim::Finished {
            status: row.try_get::<_, Option<String>>(3)?.unwrap_or_default(),
        },
        Some(_) => ItemClaim::Handed {
            nonce: row.try_get::<_, Option<i64>>(1)?.unwrap_or_default(),
            age: Duration::from_millis(row.try_get::<_, Option<i64>>(2)?.unwrap_or(0).max(0) as u64),
        },
    })
}

#[async_trait]
impl ClaimStore for PgClaimStore {
    async fn state(&self, key: &ClaimKey) -> anyhow::Result<ItemClaim> {
        let client = self.pool.get().await?;
        let row = client
            .query_opt(
                STATE_SQL,
                &[&key.run_id, &key.entity.as_str(), &key.entity_id],
            )
            .await?;
        row_claim(row)
    }

    async fn take(
        &self,
        key: &ClaimKey,
        nonce: i64,
        budget: u64,
        pod: &str,
    ) -> anyhow::Result<Take> {
        let mut client = self.pool.get().await?;
        let tx = client.transaction().await?;
        let row = tx
            .query_opt(
                &format!("{STATE_SQL} FOR UPDATE"),
                &[&key.run_id, &key.entity.as_str(), &key.entity_id],
            )
            .await?;
        let needs_ticket = match row_claim(row)? {
            ItemClaim::NotInList => {
                tx.rollback().await?;
                return Ok(Take::NotInList);
            }
            ItemClaim::Finished { .. } => {
                tx.rollback().await?;
                return Ok(Take::Held);
            }
            ItemClaim::Handed { nonce: holder, .. } if holder == nonce => false,
            ItemClaim::Handed { age, .. } if age < STALE_AFTER => {
                tx.rollback().await?;
                return Ok(Take::Held);
            }
            ItemClaim::Handed { .. } | ItemClaim::Free => true,
        };
        if needs_ticket {
            let budget = i64::try_from(budget).unwrap_or(i64::MAX);
            let ok: bool = tx
                .query_one(
                    "SELECT log_only.backfill_take_ticket($1, $2)",
                    &[&key.entity.as_str(), &budget],
                )
                .await?
                .try_get(0)?;
            if !ok {
                tx.rollback().await?;
                return Ok(Take::BudgetReached);
            }
        }
        tx.execute(
            "UPDATE log_only.backfill_items SET claim_state = 'handed_to_ais', claim_nonce = $4, \
             claimed_at = now(), claim_pod = $5 \
             WHERE run_id = $1 AND entity_type = $2 AND entity_id = $3",
            &[
                &key.run_id,
                &key.entity.as_str(),
                &key.entity_id,
                &nonce,
                &pod,
            ],
        )
        .await?;
        tx.commit().await?;
        Ok(Take::Taken)
    }

    async fn finish(&self, key: &ClaimKey, nonce: i64, status: &str) -> anyhow::Result<()> {
        let client = self.pool.get().await?;
        client
            .execute(
                "UPDATE log_only.backfill_items SET claim_state = 'finished', claim_status = $5 \
                 WHERE run_id = $1 AND entity_type = $2 AND entity_id = $3 \
                 AND (claim_state IS NULL OR (claim_state = 'handed_to_ais' \
                 AND (claim_nonce = $4 OR claimed_at < now() - make_interval(secs => $6))))",
                &[
                    &key.run_id,
                    &key.entity.as_str(),
                    &key.entity_id,
                    &nonce,
                    &status,
                    &(STALE_AFTER.as_secs() as f64),
                ],
            )
            .await?;
        Ok(())
    }
}

#[derive(Default)]
pub struct KeyLocks {
    locks: Mutex<HashMap<ClaimKey, std::sync::Weak<tokio::sync::Mutex<()>>>>,
}

impl KeyLocks {
    pub async fn lock(&self, key: &ClaimKey) -> tokio::sync::OwnedMutexGuard<()> {
        let lock = {
            let mut locks = self.locks.lock().unwrap_or_else(|p| p.into_inner());
            if locks.len() > 10_000 {
                locks.retain(|_, w| w.strong_count() > 0);
            }
            match locks.get(key).and_then(std::sync::Weak::upgrade) {
                Some(l) => l,
                None => {
                    let l = Arc::new(tokio::sync::Mutex::new(()));
                    locks.insert(key.clone(), Arc::downgrade(&l));
                    l
                }
            }
        };
        lock.lock_owned().await
    }
}

#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    #[derive(Debug, Clone)]
    struct Row {
        state: Option<&'static str>,
        nonce: i64,
        at: std::time::Instant,
        status: Option<String>,
    }

    #[derive(Default)]
    pub(crate) struct MemClaimStore {
        rows: Mutex<HashMap<ClaimKey, Row>>,
        pub all_listed: bool,
        pub tickets: Mutex<HashMap<&'static str, u64>>,
        pub state_barrier: Mutex<Option<Arc<tokio::sync::Barrier>>>,
        pub fail_reads: std::sync::atomic::AtomicBool,
    }

    impl MemClaimStore {
        pub(crate) fn listing_all() -> Self {
            Self {
                all_listed: true,
                ..Default::default()
            }
        }

        pub(crate) fn add(&self, key: ClaimKey) {
            self.rows.lock().unwrap().insert(
                key,
                Row {
                    state: None,
                    nonce: 0,
                    at: std::time::Instant::now(),
                    status: None,
                },
            );
        }

        pub(crate) fn set_handed(&self, key: ClaimKey, nonce: i64, age: Duration) {
            self.rows.lock().unwrap().insert(
                key,
                Row {
                    state: Some("handed_to_ais"),
                    nonce,
                    at: std::time::Instant::now() - age,
                    status: None,
                },
            );
        }

        pub(crate) fn peek(&self, key: &ClaimKey) -> ItemClaim {
            let rows = self.rows.lock().unwrap();
            self.claim_of(&rows, key)
        }

        pub(crate) fn clear(&self) {
            self.rows.lock().unwrap().clear();
        }

        pub(crate) fn tickets(&self, entity: EntityType) -> u64 {
            self.tickets
                .lock()
                .unwrap()
                .get(entity.as_str())
                .copied()
                .unwrap_or(0)
        }

        fn claim_of(&self, rows: &HashMap<ClaimKey, Row>, key: &ClaimKey) -> ItemClaim {
            match rows.get(key) {
                None if self.all_listed => ItemClaim::Free,
                None => ItemClaim::NotInList,
                Some(r) => match r.state {
                    None => ItemClaim::Free,
                    Some("finished") => ItemClaim::Finished {
                        status: r.status.clone().unwrap_or_default(),
                    },
                    Some(_) => ItemClaim::Handed {
                        nonce: r.nonce,
                        age: r.at.elapsed(),
                    },
                },
            }
        }
    }

    #[async_trait]
    impl ClaimStore for MemClaimStore {
        async fn state(&self, key: &ClaimKey) -> anyhow::Result<ItemClaim> {
            if self.fail_reads.load(std::sync::atomic::Ordering::Relaxed) {
                anyhow::bail!("claim store unreachable (test)");
            }
            let claim = {
                let rows = self.rows.lock().unwrap();
                self.claim_of(&rows, key)
            };
            let barrier = self.state_barrier.lock().unwrap().clone();
            if let Some(b) = barrier {
                b.wait().await;
            }
            Ok(claim)
        }

        async fn take(
            &self,
            key: &ClaimKey,
            nonce: i64,
            budget: u64,
            _pod: &str,
        ) -> anyhow::Result<Take> {
            let mut rows = self.rows.lock().unwrap();
            let needs_ticket = match self.claim_of(&rows, key) {
                ItemClaim::NotInList => return Ok(Take::NotInList),
                ItemClaim::Finished { .. } => return Ok(Take::Held),
                ItemClaim::Handed { nonce: holder, .. } if holder == nonce => false,
                ItemClaim::Handed { age, .. } if age < STALE_AFTER => return Ok(Take::Held),
                _ => true,
            };
            if needs_ticket {
                let mut tickets = self.tickets.lock().unwrap();
                let used = tickets.entry(key.entity.as_str()).or_default();
                if *used >= budget {
                    return Ok(Take::BudgetReached);
                }
                *used += 1;
            }
            rows.insert(
                key.clone(),
                Row {
                    state: Some("handed_to_ais"),
                    nonce,
                    at: std::time::Instant::now(),
                    status: None,
                },
            );
            Ok(Take::Taken)
        }

        async fn finish(&self, key: &ClaimKey, nonce: i64, status: &str) -> anyhow::Result<()> {
            let mut rows = self.rows.lock().unwrap();
            let free_or_ours = match self.claim_of(&rows, key) {
                ItemClaim::Free => true,
                ItemClaim::Handed { nonce: holder, age } => holder == nonce || age >= STALE_AFTER,
                _ => false,
            };
            if free_or_ours {
                rows.insert(
                    key.clone(),
                    Row {
                        state: Some("finished"),
                        nonce,
                        at: std::time::Instant::now(),
                        status: Some(status.to_owned()),
                    },
                );
            }
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::MemClaimStore;
    use super::*;

    fn key(id: i64) -> ClaimKey {
        ClaimKey {
            run_id: "run-00000001".into(),
            entity: EntityType::User,
            entity_id: id,
        }
    }

    #[tokio::test]
    async fn take_is_once_per_pair_and_costs_one_ticket() {
        let s = MemClaimStore::default();
        s.add(key(1));
        assert_eq!(s.take(&key(1), 7, 10, "p").await.unwrap(), Take::Taken);
        assert_eq!(s.take(&key(1), 8, 10, "p").await.unwrap(), Take::Held);
        assert_eq!(s.take(&key(1), 7, 10, "p").await.unwrap(), Take::Taken);
        assert_eq!(s.tickets(EntityType::User), 1);
        s.finish(&key(1), 8, "success").await.unwrap();
        assert!(
            matches!(s.state(&key(1)).await.unwrap(), ItemClaim::Handed { .. }),
            "not the holder"
        );
        s.finish(&key(1), 7, "success").await.unwrap();
        assert_eq!(
            s.take(&key(1), 7, 10, "p").await.unwrap(),
            Take::Held,
            "finished stays finished"
        );
        assert_eq!(s.take(&key(2), 9, 10, "p").await.unwrap(), Take::NotInList);
    }

    #[tokio::test]
    async fn a_stale_claim_is_taken_over_and_the_budget_is_exact() {
        let s = MemClaimStore::default();
        s.set_handed(key(1), 5, STALE_AFTER + Duration::from_secs(1));
        s.add(key(2));
        s.add(key(3));
        assert_eq!(s.take(&key(1), 6, 2, "p").await.unwrap(), Take::Taken);
        assert_eq!(s.take(&key(2), 6, 2, "p").await.unwrap(), Take::Taken);
        assert_eq!(
            s.take(&key(3), 6, 2, "p").await.unwrap(),
            Take::BudgetReached
        );
        assert_eq!(
            s.state(&key(3)).await.unwrap(),
            ItemClaim::Free,
            "nothing claimed"
        );
    }

    #[test]
    fn claim_nonces_differ() {
        let n: std::collections::HashSet<i64> = (0..10_000).map(|_| new_claim_nonce()).collect();
        assert_eq!(n.len(), 10_000);
        assert!(n.iter().all(|v| *v >= 0));
    }

    #[tokio::test]
    async fn key_locks_serialize_one_pair() {
        let locks = Arc::new(KeyLocks::default());
        let g = locks.lock(&key(1)).await;
        let other = locks.lock(&key(2)).await;
        drop(other);
        let l2 = locks.clone();
        let waiter = tokio::spawn(async move {
            let _g = l2.lock(&key(1)).await;
        });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!waiter.is_finished());
        drop(g);
        waiter.await.unwrap();
    }

    #[tokio::test]
    #[ignore = "needs BACKFILL_CLAIMS_TEST_DSN"]
    async fn pg_take_is_atomic_and_the_budget_exact() {
        let Ok(dsn) = std::env::var("BACKFILL_CLAIMS_TEST_DSN") else {
            return;
        };
        let store = Arc::new(PgClaimStore::new_plaintext_for_test(&dsn).unwrap());
        let k = ClaimKey {
            run_id: "run-pgtest01".into(),
            entity: EntityType::User,
            entity_id: 42,
        };
        assert_eq!(store.state(&k).await.unwrap(), ItemClaim::Free);
        let (a, b) = tokio::join!(
            store.take(&k, 1, 1_000_000, "a"),
            store.take(&k, 2, 1_000_000, "b")
        );
        let won: Vec<Take> = vec![a.unwrap(), b.unwrap()];
        assert_eq!(
            won.iter().filter(|t| **t == Take::Taken).count(),
            1,
            "{won:?}"
        );
        assert_eq!(
            won.iter().filter(|t| **t == Take::Held).count(),
            1,
            "{won:?}"
        );
        let ItemClaim::Handed { nonce, .. } = store.state(&k).await.unwrap() else {
            panic!("handed");
        };
        store.finish(&k, nonce, "success").await.unwrap();
        assert_eq!(
            store.state(&k).await.unwrap(),
            ItemClaim::Finished {
                status: "success".into()
            }
        );
        assert_eq!(
            store.take(&k, nonce, 1_000_000, "a").await.unwrap(),
            Take::Held
        );
        let missing = ClaimKey {
            entity_id: 43_000,
            ..k.clone()
        };
        assert_eq!(
            store.take(&missing, 3, 10, "a").await.unwrap(),
            Take::NotInList
        );
        let mut tasks = Vec::new();
        for id in 100..120 {
            let s = store.clone();
            tasks.push(tokio::spawn(async move {
                let k = ClaimKey {
                    run_id: "run-pgtest02".into(),
                    entity: EntityType::Post,
                    entity_id: id,
                };
                s.take(&k, id, 2, "x").await.unwrap()
            }));
        }
        let mut taken = 0;
        for t in tasks {
            taken += usize::from(t.await.unwrap() == Take::Taken);
        }
        assert_eq!(taken, 2);
        let client = store.pool.get().await.unwrap();
        let open: i64 = client
            .query_one(
                "SELECT count(*) FROM pg_stat_activity WHERE usename = current_user \
                 AND pid <> pg_backend_pid() AND state LIKE 'idle in transaction%'",
                &[],
            )
            .await
            .unwrap()
            .get(0);
        assert_eq!(open, 0);
        for sql in [
            "UPDATE log_only.backfill_budget SET n = 0",
            "DELETE FROM log_only.backfill_budget",
            "INSERT INTO log_only.backfill_budget (entity_type, minute, n) VALUES ('post', now(), 1)",
            "SELECT n FROM log_only.backfill_budget",
        ] {
            let err = client.execute(sql, &[]).await.expect_err(sql);
            assert_eq!(
                err.code(),
                Some(&tokio_postgres::error::SqlState::INSUFFICIENT_PRIVILEGE),
                "{sql}: {err}"
            );
        }
    }
}
