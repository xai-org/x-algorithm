//! Deterministic storage/scheduling model around the real reader, writer helpers
//! and scorer. Visibility delays and last-write-wins are MODEL assumptions, not
//! claims about the behavior of Manhattan. No wall-clock sleeps are used.
use super::*;
use std::collections::BTreeMap;
use std::sync::Barrier;

const START: i64 = 1_000_000;

fn query(now: i64) -> ScoredPostsQuery {
    let mut q = ScoredPostsQuery {
        user_id: 7,
        request_time_ms: now,
        ..Default::default()
    };
    q.params.set(EnableCrossRequestAuthorDiversity, true);
    q.params.set(AuthorDiversityHistoryWeight, 1.0);
    q
}

fn batch(request: i64, time: i64, author: u64) -> ServedHistory {
    ServedHistory {
        request_type: served_history::RequestType::INITIAL,
        served_id: Some(request),
        served_time_ms: Some(time),
        entries: build_post_entries(
            &ScoredPost {
                tweet_id: request as u64 * 10 + if author == 1 { 1 } else { 2 },
                author_id: author,
                ..Default::default()
            },
            0,
        ),
    }
}

fn hydrate(now: i64, history: Vec<ServedHistory>, fail: bool) -> ScoredPostsQuery {
    let h = ServedHistoryQueryHydrator {
        client: Arc::new(MockHistory {
            batches: Mutex::new(HashMap::from([(7, history)])),
            fail,
        }),
    };
    let mut q = query(now);
    if let Ok(result) = ready(h.hydrate(&q)) {
        h.update(&mut q, result);
    }
    q
}

fn candidates(request: i64) -> [PostCandidate; 2] {
    [
        PostCandidate {
            tweet_id: request as u64 * 10 + 1,
            author_id: 1,
            base_positive: 100.0,
            in_network: Some(true),
            ..Default::default()
        },
        PostCandidate {
            tweet_id: request as u64 * 10 + 2,
            author_id: request as u64 + 1000,
            base_positive: 70.0,
            in_network: Some(true),
            ..Default::default()
        },
    ]
}

fn coefficient(q: &ScoredPostsQuery) -> f64 {
    RankingScorer::author_diversity_multipliers(q, &candidates(1), &[100.0, 70.0])[0]
}

fn chosen_author(q: &ScoredPostsQuery, request: i64) -> u64 {
    let c = candidates(request);
    let output = ready(
        RankingScorer {
            author_cold_start: AuthorColdStart,
        }
        .score(q, &c),
    );
    let scores: Vec<_> = output
        .into_iter()
        .map(|r| r.unwrap().score.unwrap())
        .collect();
    c[usize::from(scores[1] > scores[0])].author_id
}

// Key bytes are injected verbatim from the production put/delete expression.
/* STORAGE_KEY */

#[derive(Default)]
struct Store {
    // A single viewer/timeline/platform partition. The pkey is held constant.
    committed: BTreeMap<Vec<Vec<u8>>, ServedHistory>,
    pending: Vec<(i64, usize, ServedHistory)>,
    sequence: usize,
    collisions: usize,
}

impl Store {
    fn schedule(&mut self, value: ServedHistory, visible_at: i64, drop_write: bool) {
        if !drop_write {
            self.pending.push((visible_at, self.sequence, value));
            self.sequence += 1;
        }
    }
    fn commit(&mut self, value: ServedHistory) {
        let key = storage_lkey(value.served_time_ms.unwrap());
        if self.committed.insert(key, value).is_some() {
            self.collisions += 1;
        }
    }
    fn advance(&mut self, now: i64) {
        self.pending
            .sort_by_key(|(visible, sequence, _)| (*visible, *sequence));
        let mut waiting = Vec::new();
        for (visible, sequence, value) in std::mem::take(&mut self.pending) {
            if visible <= now {
                self.commit(value);
            } else {
                waiting.push((visible, sequence, value));
            }
        }
        self.pending = waiting;
    }
    fn snapshot(&self) -> Vec<ServedHistory> {
        self.committed.values().cloned().collect()
    }
}

#[test]
fn enumerate_every_three_request_read_write_interleaving() {
    // R_i means hydrate + rank; W_i means the write becomes visible. Each W_i
    // depends on its R_i. Enumerate all 6! / 2^3 = 90 linear extensions.
    fn enumerate(
        path: &mut Vec<(bool, usize)>,
        read: u8,
        written: u8,
        out: &mut Vec<Vec<(bool, usize)>>,
    ) {
        if written == 7 {
            out.push(path.clone());
            return;
        }
        for request in 0..3 {
            let mask = 1 << request;
            if read & mask == 0 {
                path.push((false, request));
                enumerate(path, read | mask, written, out);
                path.pop();
            } else if written & mask == 0 {
                path.push((true, request));
                enumerate(path, read, written | mask, out);
                path.pop();
            }
        }
    }
    let mut schedules = Vec::new();
    enumerate(&mut Vec::new(), 0, 0, &mut schedules);
    assert_eq!(schedules.len(), 90);
    let mut blind = 0;
    for (number, path) in schedules.iter().enumerate() {
        let mut store = Store::default();
        let mut prepared: Vec<Option<ServedHistory>> = vec![None, None, None];
        let mut coefficients = Vec::new();
        let mut authors = Vec::new();
        let mut events = Vec::new();
        for (step, &(write, request)) in path.iter().enumerate() {
            let now = START + step as i64;
            if write {
                store.commit(prepared[request].take().unwrap());
                events.push(format!("W{request}"));
            } else {
                let q = hydrate(now, store.snapshot(), false);
                let m = coefficient(&q);
                assert!(m.is_finite() && (0.25..=1.0).contains(&m));
                if store.committed.is_empty() {
                    assert_eq!(m, 1.0);
                } else {
                    assert!(
                        m < 1.0,
                        "a committed first A exposure must carry across requests"
                    );
                }
                coefficients.push(m);
                let author = chosen_author(&q, request as i64 + 1);
                authors.push(author);
                prepared[request] = Some(batch(request as i64 + 1, now, author));
                events.push(format!("R{request}"));
            }
        }
        let all_blind = coefficients.iter().all(|&m| m == 1.0);
        blind += usize::from(all_blind);
        assert_eq!(all_blind, path[..3].iter().all(|(write, _)| !write));
        let after = coefficient(&hydrate(START + 10, store.snapshot(), false));
        assert!(after < 1.0);
        assert_eq!(store.collisions, 0);
        println!("\nSIMULATION {{\"kind\":\"interleaving\",\"case\":{number},\"schedule\":\"{}\",\"multipliers\":{:?},\"selected_authors\":{:?},\"all_reads_blind\":{all_blind},\"after_commits_multiplier\":{after}}}", events.join(" "), coefficients, authors);
    }
    assert_eq!(blind, 36);
    println!("\nSIMULATION {{\"kind\":\"interleaving_summary\",\"schedules\":90,\"all_reads_blind\":{blind},\"some_read_saw_history\":54}}");
}

#[test]
fn delay_matrix_finds_the_visibility_boundary() {
    for interval_ms in [10, 50, 200, 1000] {
        for delay_ms in [0, 5, 20, 100, 500] {
            let mut store = Store::default();
            let mut full = 0;
            let mut a_slots = 0;
            for request in 0..20 {
                let now = START + request * interval_ms;
                store.advance(now); // Writes visible at the exact read time are included.
                let q = hydrate(now, store.snapshot(), false);
                full += usize::from(coefficient(&q) == 1.0);
                let author = chosen_author(&q, request + 1);
                a_slots += usize::from(author == 1);
                store.schedule(batch(request + 1, now, author), now + delay_ms, false);
            }
            let expected = ((delay_ms + interval_ms - 1) / interval_ms).clamp(1, 20) as usize;
            assert_eq!(full, expected);
            assert_eq!(store.collisions, 0);
            store.advance(START + 20 * interval_ms + delay_ms);
            assert!(
                coefficient(&hydrate(
                    START + 20 * interval_ms + delay_ms,
                    store.snapshot(),
                    false
                )) < 1.0
            );
            println!("\nSIMULATION {{\"kind\":\"delay_matrix\",\"interval_ms\":{interval_ms},\"visibility_delay_ms\":{delay_ms},\"requests\":20,\"full_multiplier_requests\":{full},\"author_a_slots\":{a_slots}}}");
        }
    }
}

#[test]
fn eight_os_threads_can_read_the_same_snapshot_before_any_write() {
    let client = Arc::new(MockHistory::default());
    let barrier = Arc::new(Barrier::new(8));
    let values = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for worker in 0..8 {
            let client = client.clone();
            let barrier = barrier.clone();
            handles.push(scope.spawn(move || {
                let hydrator = ServedHistoryQueryHydrator {
                    client: client.clone(),
                };
                let mut q = query(START + worker);
                let h = ready(hydrator.hydrate(&q)).unwrap();
                hydrator.update(&mut q, h);
                let m = coefficient(&q);
                barrier.wait(); // Every read finished before any commit is allowed.
                client
                    .batches
                    .lock()
                    .unwrap()
                    .entry(7)
                    .or_default()
                    .push(batch(worker + 1, START + worker, 1));
                m
            }));
        }
        handles
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect::<Vec<_>>()
    });
    assert_eq!(values, vec![1.0; 8]);
    let q = hydrate(
        START + 10,
        client.batches.lock().unwrap()[&7].clone(),
        false,
    );
    let recovered = coefficient(&q);
    assert!((recovered - 0.2734375).abs() < 1e-12); // 5-count cap, weight 1
    println!("\nSIMULATION {{\"kind\":\"thread_barrier\",\"threads\":8,\"full_multiplier_reads\":8,\"after_commit_multiplier\":{recovered}}}");
}

#[test]
fn committed_write_does_not_retroactively_change_an_already_hydrated_query() {
    let mut store = Store::default();
    let snapshot = hydrate(START + 10, store.snapshot(), false);
    store.commit(batch(1, START, 1));
    assert_eq!(coefficient(&snapshot), 1.0);
    assert!(coefficient(&hydrate(START + 10, store.snapshot(), false)) < 1.0);
}

#[test]
fn equal_time_read_write_tie_depends_on_operation_order() {
    let mut store = Store::default();
    store.schedule(batch(1, START, 1), START + 10, false);
    let read_before_commit = coefficient(&hydrate(START + 10, store.snapshot(), false));
    store.advance(START + 10);
    let read_after_commit = coefficient(&hydrate(START + 10, store.snapshot(), false));
    assert_eq!(read_before_commit, 1.0);
    assert!(read_after_commit < 1.0);
    println!("\nSIMULATION {{\"kind\":\"equal_time_boundary\",\"before_commit_multiplier\":{read_before_commit},\"after_commit_multiplier\":{read_after_commit}}}");
}

#[test]
fn failed_writes_and_reads_have_distinct_recovery_paths() {
    let mut store = Store::default();
    for request in 1..=5 {
        let now = START + request;
        store.advance(now);
        assert_eq!(coefficient(&hydrate(now, store.snapshot(), false)), 1.0);
        store.schedule(batch(request, now, 1), now, true);
    }
    assert!(store.committed.is_empty());
    store.schedule(batch(6, START + 6, 1), START + 6, false);
    store.advance(START + 6);
    let failed_read = hydrate(START + 7, store.snapshot(), true);
    assert!(!failed_read.served_history_loaded);
    assert_eq!(coefficient(&failed_read), 1.0);
    assert!(coefficient(&hydrate(START + 7, store.snapshot(), false)) < 1.0);
    println!("\nSIMULATION {{\"kind\":\"fault_recovery\",\"dropped_writes\":5,\"blind_requests\":5,\"read_failure_falls_back\":true,\"recovery_after_successful_write_and_read\":true}}");
}

#[test]
fn out_of_order_completion_and_duplicate_reads_preserve_event_time_counts() {
    let mut store = Store::default();
    store.schedule(batch(1, START, 1), START + 100, false);
    store.schedule(batch(2, START + 80, 1), START + 90, false);
    store.advance(START + 95);
    assert_eq!(store.committed.len(), 1);
    store.advance(START + 200);
    assert_eq!(store.committed.len(), 2);
    let mut history = store.snapshot();
    let normal = coefficient(&hydrate(START + 200, history.clone(), false));
    let exponent = 2.0f64.powf(-200.0 / 60_000.0) + 2.0f64.powf(-120.0 / 60_000.0);
    let expected = 0.25 + 0.75 * 0.5f64.powf(exponent);
    assert!((normal - expected).abs() < 1e-12);
    history.reverse();
    history.extend(history.clone());
    assert_eq!(coefficient(&hydrate(START + 200, history, false)), normal);
    println!("\nSIMULATION {{\"kind\":\"out_of_order\",\"counted_batches\":2,\"duplicate_and_reverse_invariant\":true,\"multiplier\":{normal}}}");
}

#[test]
fn timestamp_only_storage_key_can_collide_for_distinct_requests() {
    // The same lkey is a code fact. Overwrite is specifically the LWW model.
    let mut store = Store::default();
    store.commit(batch(1, START, 1));
    store.commit(batch(2, START, 2));
    assert_eq!(store.collisions, 1);
    assert_eq!(store.committed.len(), 1);
    let actual = coefficient(&hydrate(START + 1, store.snapshot(), false));
    assert_eq!(actual, 1.0); // A's batch was overwritten by B in this model.
    let ideal = coefficient(&hydrate(
        START + 1,
        vec![batch(1, START, 1), batch(2, START, 2)],
        false,
    ));
    assert!(ideal < 1.0);
    println!("\nSIMULATION {{\"kind\":\"timestamp_key_collision\",\"distinct_requests\":2,\"stored_rows_lww_model\":1,\"author_a_multiplier_lww_model\":{actual},\"author_a_multiplier_if_both_retained\":{ideal}}}");
}

#[test]
fn retrying_same_batch_is_idempotent_in_the_storage_model() {
    let mut store = Store::default();
    let value = batch(1, START, 1);
    store.commit(value.clone());
    let before = coefficient(&hydrate(START + 1, store.snapshot(), false));
    store.commit(value);
    let after = coefficient(&hydrate(START + 1, store.snapshot(), false));
    assert_eq!(before, after);
    assert_eq!(store.committed.len(), 1);
}

#[test]
fn seeded_property_checks_cover_history_order_replays_and_caps() {
    use util::author_diversity::*;
    let initial_seed = 0x202_2026u64;
    let mut seed = initial_seed;
    fn random(seed: &mut u64) -> u64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *seed
    }
    for trial in 0..1000 {
        let config = HistoryConfig {
            window_ms: 300_000,
            half_life_ms: 60_000.0,
            weight: 0.25,
            max_count: 5.0,
            max_items: 40,
        };
        let mut events = Vec::new();
        for _ in 0..100 {
            events.push(AuthorExposure {
                served_at_ms: START + 1000 - (random(&mut seed) % 400_000) as i64,
                request_id: Some((random(&mut seed) % 30) as i64),
                item: ExposureItem::Position((random(&mut seed) % 5) as i64),
                author_id: random(&mut seed) % 9,
            });
        }
        let expected = recent_author_counts(&events, START, config).unwrap();
        for i in (1..events.len()).rev() {
            let j = random(&mut seed) as usize % (i + 1);
            events.swap(i, j);
        }
        events.extend(events.clone());
        assert_eq!(
            recent_author_counts(&events, START, config).unwrap(),
            expected,
            "trial {trial}"
        );
        for (&author, &value) in &expected {
            assert_ne!(author, 0);
            assert!(value.is_finite() && (0.0..=1.25).contains(&value));
        }
    }
    println!("\nSIMULATION {{\"kind\":\"property_checks\",\"seed\":{initial_seed},\"trials\":1000,\"invariants\":[\"permutation\",\"replay\",\"finite\",\"cap\",\"unknown_author_excluded\"]}}");
}
