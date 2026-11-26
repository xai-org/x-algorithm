// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
pub const FEATURE_DIM: usize = 1310;
pub const FEATURE_VERSION: &str = "serving_v1";

pub const VOCAB: usize = 64;
const HOME_SURFACES: [i32; 2] = [1, 15];
const GALLERY_SURFACE: i32 = 10;
const RECENT_N: usize = 128;
const RECENT_NS: [usize; 3] = [32, 128, 512];
const EWMA_HALFLIFE_HOURS: [f32; 2] = [24.0, 168.0];
const WINDOWS_H: [f32; 4] = [1.0, 6.0, 24.0, 168.0];
const KEY_ACTIONS: [usize; 10] = [1, 4, 6, 13, 17, 29, 30, 36, 37, 47];
const ENGAGE_MASK: u64 = !((1u64 << 11) | (1u64 << 12));
const MAX_AGE_H: f64 = 24.0 * 365.0;
const MAX_ACTIVE_DAY_BUCKET: f32 = 399.0;
const MAX_POSITION: f32 = 500.0;
const MAX_DWELL_SEC: f32 = 3600.0;
const MAX_VIDEO_DUR_MS: f64 = 3.6e6;
const N_HOURS: usize = 24;
const N_MEDIA_TYPES: usize = 16;
const N_APP_BUCKETS: u64 = 32;
const TZ_ENUM_TO_UTC_OFFSET: [f64; 32] = [
    0.0, -5.0, -6.0, -8.0, -7.0, -3.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 9.0, 5.5, 8.0, 9.0,
    7.0, 7.0, 4.0, 3.0, 8.0, 8.0, 11.0, -10.0, -7.0, -5.0, -6.0, -3.0, 1.0, 0.0,
];

#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize)]
pub struct SequenceRows {
    pub acts: Vec<u64>,
    pub ts_ms: Vec<i64>,
    pub surface: Vec<i32>,
    pub tz: Vec<i32>,
    pub dwell_sec: Vec<f32>,
    pub has_media: Vec<bool>,
    pub media_type: Vec<i32>,
    pub video_dur_ms: Vec<i64>,
    pub in_network: Vec<bool>,
    pub position: Vec<i32>,
}

impl SequenceRows {
    pub fn len(&self) -> usize {
        self.acts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.acts.is_empty()
    }

    pub fn retain_until(&mut self, cutoff_ms: i64) {
        let keep: Vec<bool> = self.ts_ms.iter().map(|&t| t <= cutoff_ms).collect();
        if keep.iter().all(|&k| k) {
            return;
        }
        self.retain(&keep);
    }

    pub(crate) fn retain(&mut self, keep: &[bool]) {
        fn filter<T: Copy>(col: &mut Vec<T>, keep: &[bool]) {
            let mut i = 0;
            col.retain(|_| {
                let k = keep[i];
                i += 1;
                k
            });
        }
        filter(&mut self.acts, keep);
        filter(&mut self.ts_ms, keep);
        filter(&mut self.surface, keep);
        filter(&mut self.tz, keep);
        filter(&mut self.dwell_sec, keep);
        filter(&mut self.has_media, keep);
        filter(&mut self.media_type, keep);
        filter(&mut self.video_dur_ms, keep);
        filter(&mut self.in_network, keep);
        filter(&mut self.position, keep);
    }

    pub fn columns(&self) -> SequenceColumns<'_> {
        SequenceColumns {
            acts: &self.acts,
            ts_ms: &self.ts_ms,
            surface: &self.surface,
            tz: &self.tz,
            dwell_sec: &self.dwell_sec,
            has_media: &self.has_media,
            media_type: &self.media_type,
            video_dur_ms: &self.video_dur_ms,
            in_network: &self.in_network,
            position: &self.position,
        }
    }

    fn column_lengths(&self) -> [usize; 10] {
        [
            self.acts.len(),
            self.ts_ms.len(),
            self.surface.len(),
            self.tz.len(),
            self.dwell_sec.len(),
            self.has_media.len(),
            self.media_type.len(),
            self.video_dur_ms.len(),
            self.in_network.len(),
            self.position.len(),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RequestContext {
    pub surface_id: usize,
    pub hour: usize,
    pub dow: usize,
    pub app: usize,
}

pub fn surface_id(surface: i32) -> usize {
    if HOME_SURFACES.contains(&surface) {
        1
    } else if surface == GALLERY_SURFACE {
        2
    } else {
        0
    }
}

pub fn request_context(req_ms: i64, tz: i32, client_app: i64, surface: i32) -> RequestContext {
    let local = req_ms as f64 / 1000.0 + utc_offset_hours(tz) * 3600.0;
    let hour = (py_floordiv(py_mod(local, 86400.0), 3600.0) as i64).rem_euclid(N_HOURS as i64);
    let dow = (py_floordiv(local, 86400.0) as i64 + 3).rem_euclid(7);
    RequestContext {
        surface_id: surface_id(surface),
        hour: hour as usize,
        dow: dow as usize,
        app: (client_app.unsigned_abs() % N_APP_BUCKETS) as usize,
    }
}

fn utc_offset_hours(tz: i32) -> f64 {
    let last = TZ_ENUM_TO_UTC_OFFSET.len() as i32 - 1;
    TZ_ENUM_TO_UTC_OFFSET[tz.clamp(0, last) as usize]
}

fn py_mod(a: f64, b: f64) -> f64 {
    let m = a % b;
    if m == 0.0 {
        0.0f64.copysign(b)
    } else if (b < 0.0) != (m < 0.0) {
        m + b
    } else {
        m
    }
}

fn py_floordiv(a: f64, b: f64) -> f64 {
    let m = a % b;
    let mut div = (a - m) / b;
    if m != 0.0 && (b < 0.0) != (m < 0.0) {
        div -= 1.0;
    }
    if div == 0.0 {
        return 0.0f64.copysign(a / b);
    }
    let mut floored = div.floor();
    if div - floored > 0.5 {
        floored += 1.0;
    }
    floored
}

fn local_hour(ts_ms: i64, tz: i32) -> usize {
    let local = ts_ms as f64 / 1000.0 + utc_offset_hours(tz) * 3600.0;
    (py_floordiv(py_mod(local, 86400.0), 3600.0) as i64).rem_euclid(N_HOURS as i64) as usize
}

fn hours_since(req_ms: i64, ts_ms: i64) -> f64 {
    ((req_ms as f64 - ts_ms as f64) / 3.6e6).clamp(0.0, MAX_AGE_H)
}

fn pairwise_sum_f32(a: &[f32]) -> f32 {
    let n = a.len();
    if n < 8 {
        return a.iter().fold(0.0f32, |acc, &v| acc + v);
    }
    if n <= 128 {
        let mut r = [a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7]];
        let mut i = 8;
        while i < n - (n % 8) {
            for (k, acc) in r.iter_mut().enumerate() {
                *acc += a[i + k];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum_f32(&a[..n2]) + pairwise_sum_f32(&a[n2..])
}

fn pairwise_mean_or_zero(values: &[f32]) -> f32 {
    if values.is_empty() {
        0.0
    } else {
        pairwise_sum_f32(values) / values.len() as f32
    }
}

fn masked_bit_rates(acts: &[u64], mask: &[bool]) -> [f32; VOCAB] {
    let mut counts = [0u32; VOCAB];
    let mut n = 0u32;
    for (&a, &m) in acts.iter().zip(mask) {
        if !m {
            continue;
        }
        n += 1;
        let mut bits = a;
        while bits != 0 {
            let i = bits.trailing_zeros() as usize;
            counts[i] += 1;
            bits &= bits - 1;
        }
    }
    let mut out = [0.0f32; VOCAB];
    if n > 0 {
        for (o, &c) in out.iter_mut().zip(&counts) {
            *o = c as f32 / n as f32;
        }
    }
    out
}

fn sequential_mean_cols<const K: usize>(rows: impl Iterator<Item = [f32; K]>) -> [f32; K] {
    let mut acc = [0.0f32; K];
    let mut n = 0u32;
    for row in rows {
        n += 1;
        for (a, v) in acc.iter_mut().zip(row) {
            *a += v;
        }
    }
    if n > 0 {
        for a in acc.iter_mut() {
            *a /= n as f32;
        }
    }
    acc
}

fn logit_rate_f32(k: f32, n: f32) -> f32 {
    let ratio = (k + 0.5) / ((n - k) + 0.5);
    (ratio as f64).ln() as f32
}

fn log1p_f32(x: f32) -> f32 {
    (x as f64).ln_1p() as f32
}

fn count_true(mask: &[bool]) -> usize {
    mask.iter().filter(|&&m| m).count()
}

fn and_masks(a: &[bool], b: &[bool]) -> Vec<bool> {
    a.iter().zip(b).map(|(&x, &y)| x && y).collect()
}

fn logit_rates_for_mask(acts: &[u64], mask: &[bool]) -> [f32; VOCAB] {
    let rates = masked_bit_rates(acts, mask);
    let cnt = count_true(mask) as f32;
    rates.map(|r| logit_rate_f32(r * cnt, cnt))
}

pub fn featurize_sequence(rows: &SequenceRows, req_ms: i64) -> Vec<f32> {
    let n = rows.len();
    assert!(
        rows.column_lengths().iter().all(|&len| len == n),
        "SequenceRows columns have inconsistent lengths: {:?}",
        rows.column_lengths()
    );
    let cnt = n as f64;
    let cnt_f32 = n as f32;
    let acts = &rows.acts;

    let age_h: Vec<f32> = rows
        .ts_ms
        .iter()
        .map(|&t| hours_since(req_ms, t) as f32)
        .collect();
    let dwell: Vec<f32> = rows
        .dwell_sec
        .iter()
        .map(|&d| d.clamp(0.0, MAX_DWELL_SEC))
        .collect();
    let log_dwell: Vec<f32> = dwell.iter().map(|&d| log1p_f32(d)).collect();
    let video_dur: Vec<f64> = rows.video_dur_ms.iter().map(|&v| v as f64).collect();
    let is_video: Vec<bool> = video_dur.iter().map(|&v| v > 0.0).collect();
    let dwell_frac: Vec<f32> = dwell
        .iter()
        .zip(&video_dur)
        .map(|(&d, &v)| {
            if v > 0.0 {
                (((d * 1000.0f32) as f64) / v.max(1.0)).clamp(0.0, 1.0) as f32
            } else {
                0.0
            }
        })
        .collect();
    let is_home: Vec<bool> = rows
        .surface
        .iter()
        .map(|s| HOME_SURFACES.contains(s))
        .collect();
    let is_gal: Vec<bool> = rows.surface.iter().map(|&s| s == GALLERY_SURFACE).collect();
    let engaged: Vec<bool> = acts.iter().map(|&a| a & ENGAGE_MASK != 0).collect();
    let recent_mask = |k: usize| -> Vec<bool> { (0..n).map(|i| (n - 1 - i) < k).collect() };
    let recent = recent_mask(RECENT_N);
    let all_rows = vec![true; n];
    let pos_c: Vec<f32> = rows
        .position
        .iter()
        .map(|&p| (p as f32).clamp(0.0, MAX_POSITION))
        .collect();
    let log_pos: Vec<f32> = pos_c.iter().map(|&p| log1p_f32(p)).collect();
    let has_media_f: Vec<f32> = rows
        .has_media
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
    let in_network_f: Vec<f32> = rows
        .in_network
        .iter()
        .map(|&b| if b { 1.0 } else { 0.0 })
        .collect();
    let bool_f = |b: bool| if b { 1.0f32 } else { 0.0f32 };

    let mut f: Vec<f32> = Vec::with_capacity(FEATURE_DIM);

    let rates_all = masked_bit_rates(acts, &all_rows);
    f.extend_from_slice(&rates_all);
    f.extend_from_slice(&masked_bit_rates(acts, &recent));

    for hl in EWMA_HALFLIFE_HOURS {
        let w: Vec<f32> = age_h
            .iter()
            .map(|&a| (-((a / hl) as f64)).exp2() as f32)
            .collect();
        let mut num = [0.0f32; VOCAB];
        for (&a, &wi) in acts.iter().zip(&w) {
            let mut bits = a;
            while bits != 0 {
                let i = bits.trailing_zeros() as usize;
                num[i] += wi;
                bits &= bits - 1;
            }
        }
        let w_sum = pairwise_sum_f32(&w);
        let den = if w_sum > 1e-9f32 { w_sum } else { 1e-9f32 };
        if n > 0 {
            f.extend(num.iter().map(|&v| v / den));
        } else {
            f.extend_from_slice(&[0.0f32; VOCAB]);
        }
    }

    f.extend_from_slice(&masked_bit_rates(acts, &is_home));
    f.extend_from_slice(&masked_bit_rates(acts, &is_gal));

    f.extend_from_slice(&sequential_mean_cols((0..n).map(|i| {
        [
            log_dwell[i],
            bool_f(dwell[i] > 0.0),
            bool_f(is_video[i]),
            dwell_frac[i],
            has_media_f[i],
            in_network_f[i],
        ]
    })));
    f.extend_from_slice(&sequential_mean_cols(
        (0..n)
            .filter(|&i| is_gal[i])
            .map(|i| [log_dwell[i], dwell_frac[i]]),
    ));
    let recent_log_dwell: Vec<f32> = (0..n)
        .filter(|&i| recent[i])
        .map(|i| log_dwell[i])
        .collect();
    f.push(pairwise_mean_or_zero(&recent_log_dwell));

    let mut hour_counts = [0u32; N_HOURS];
    for (&t, &z) in rows.ts_ms.iter().zip(&rows.tz) {
        hour_counts[local_hour(t, z)] += 1;
    }
    let denom = cnt_f32.max(1.0);
    f.extend(hour_counts.iter().map(|&c| c as f32 / denom));

    let mut media_counts = [0u32; N_MEDIA_TYPES];
    for &m in &rows.media_type {
        media_counts[m.clamp(0, N_MEDIA_TYPES as i32 - 1) as usize] += 1;
    }
    f.extend(media_counts.iter().map(|&c| c as f32 / denom));

    let span_h = rows
        .ts_ms
        .iter()
        .min()
        .map_or(0.0, |&t| hours_since(req_ms, t));
    let last24 = age_h.iter().filter(|&&a| a <= 24.0).count() as f64;
    let last1 = age_h.iter().filter(|&&a| a <= 1.0).count() as f64;
    let mut day_buckets: Vec<i64> = age_h
        .iter()
        .map(|&a| (a / 24.0f32).clamp(0.0, MAX_ACTIVE_DAY_BUCKET) as i64)
        .collect();
    day_buckets.sort_unstable();
    day_buckets.dedup();
    let active_days = day_buckets.len() as f64;
    let cnt_home = count_true(&is_home) as f64;
    let cnt_gal = count_true(&is_gal) as f64;
    let cnt_recent = count_true(&recent) as f64;
    f.extend(
        [
            cnt.ln_1p(),
            span_h.ln_1p(),
            (cnt / span_h.max(1.0)).ln_1p(),
            last24.ln_1p(),
            last1.ln_1p(),
            active_days.ln_1p(),
            cnt_home / cnt.max(1.0),
            cnt_gal / cnt.max(1.0),
            cnt_recent.ln_1p(),
        ]
        .map(|v| v as f32),
    );

    for k in RECENT_NS {
        if k != RECENT_N {
            f.extend_from_slice(&masked_bit_rates(acts, &recent_mask(k)));
        }
    }

    for w_h in WINDOWS_H {
        let mut c = 0.0f64;
        let mut e = 0.0f64;
        for (&a, &eng) in age_h.iter().zip(&engaged) {
            if a <= w_h {
                c += 1.0;
                if eng {
                    e += 1.0;
                }
            }
        }
        f.extend([c.ln_1p(), e.ln_1p(), e / c.max(1.0)].map(|v| v as f32));
    }

    let since_any_h = rows
        .ts_ms
        .iter()
        .max()
        .map_or(MAX_AGE_H, |&t| hours_since(req_ms, t));
    f.push(since_any_h.ln_1p() as f32);
    for bit in KEY_ACTIONS {
        let last_ts = acts
            .iter()
            .zip(&rows.ts_ms)
            .filter(|(a, _)| (**a >> bit) & 1 == 1)
            .map(|(_, &t)| t)
            .max();
        let since_h = last_ts.map_or(MAX_AGE_H, |t| hours_since(req_ms, t));
        f.push(since_h.ln_1p() as f32);
    }

    f.extend_from_slice(&sequential_mean_cols((0..n).map(|i| {
        [
            log_pos[i],
            bool_f(pos_c[i] >= 20.0),
            bool_f(pos_c[i] >= 50.0),
        ]
    })));
    let engaged_log_pos: Vec<f32> = (0..n).filter(|&i| engaged[i]).map(|i| log_pos[i]).collect();
    f.push(pairwise_mean_or_zero(&engaged_log_pos));
    let max_pos = pos_c.iter().copied().fold(0.0f32, f32::max);
    f.push(if n == 0 { 0.0 } else { log1p_f32(max_pos) });

    let is_text: Vec<bool> = rows
        .has_media
        .iter()
        .zip(&is_video)
        .map(|(&m, &v)| !m && !v)
        .collect();
    f.extend_from_slice(&masked_bit_rates(acts, &is_video));
    f.extend_from_slice(&masked_bit_rates(acts, &is_text));
    f.push((count_true(&is_video) as f64 / cnt.max(1.0)) as f32);
    f.push((count_true(&is_text) as f64 / cnt.max(1.0)) as f32);
    f.extend_from_slice(&sequential_mean_cols((0..n).map(|i| {
        [
            bool_f(dwell[i] >= 5.0),
            bool_f(dwell[i] >= 30.0),
            (video_dur[i].clamp(0.0, MAX_VIDEO_DUR_MS) / 1000.0).ln_1p() as f32,
        ]
    })));
    f.extend_from_slice(&masked_bit_rates(acts, &engaged));

    let counts = rates_all.map(|r| r * cnt_f32);
    f.extend(counts.iter().map(|&c| log1p_f32(c)));
    f.extend(counts.iter().map(|&c| logit_rate_f32(c, cnt_f32)));

    let m24: Vec<bool> = age_h.iter().map(|&a| a <= 24.0).collect();
    let c24 = count_true(&m24) as f64;
    f.extend_from_slice(&masked_bit_rates(acts, &m24));
    f.extend_from_slice(&logit_rates_for_mask(acts, &m24));
    f.extend_from_slice(&masked_bit_rates(acts, &and_masks(&is_home, &recent)));
    f.extend_from_slice(&masked_bit_rates(acts, &and_masks(&is_gal, &recent)));
    f.extend_from_slice(&logit_rates_for_mask(acts, &is_home));
    f.extend_from_slice(&logit_rates_for_mask(acts, &is_gal));
    f.extend([cnt_home.ln_1p(), cnt_gal.ln_1p(), c24.ln_1p()].map(|v| v as f32));

    debug_assert_eq!(f.len(), FEATURE_DIM);
    f
}

#[derive(Clone, Copy, Debug)]
pub struct SequenceColumns<'a> {
    pub acts: &'a [u64],
    pub ts_ms: &'a [i64],
    pub surface: &'a [i32],
    pub tz: &'a [i32],
    pub dwell_sec: &'a [f32],
    pub has_media: &'a [bool],
    pub media_type: &'a [i32],
    pub video_dur_ms: &'a [i64],
    pub in_network: &'a [bool],
    pub position: &'a [i32],
}

impl SequenceColumns<'_> {
    fn lengths(&self) -> [usize; 10] {
        [
            self.acts.len(),
            self.ts_ms.len(),
            self.surface.len(),
            self.tz.len(),
            self.dwell_sec.len(),
            self.has_media.len(),
            self.media_type.len(),
            self.video_dur_ms.len(),
            self.in_network.len(),
            self.position.len(),
        ]
    }

    fn rows(&self, range: std::ops::Range<usize>) -> SequenceRows {
        SequenceRows {
            acts: self.acts[range.clone()].to_vec(),
            ts_ms: self.ts_ms[range.clone()].to_vec(),
            surface: self.surface[range.clone()].to_vec(),
            tz: self.tz[range.clone()].to_vec(),
            dwell_sec: self.dwell_sec[range.clone()].to_vec(),
            has_media: self.has_media[range.clone()].to_vec(),
            media_type: self.media_type[range.clone()].to_vec(),
            video_dur_ms: self.video_dur_ms[range.clone()].to_vec(),
            in_network: self.in_network[range.clone()].to_vec(),
            position: self.position[range].to_vec(),
        }
    }
}

const GALLERY_REPEATED: [(usize, usize); 9] = [
    (0, 64),
    (256, 128),
    (384, 1),
    (387, 1),
    (390, 2),
    (439, 2),
    (859, 64),
    (1051, 256),
    (1307, 2),
];

const GALLERY_REPEATED_DIM: usize = 520;

pub const HISTORY_STATS_DIM: usize = 2 * FEATURE_DIM - GALLERY_REPEATED_DIM;
pub const HISTORY_STATS_MAX_ROWS: usize = 1023;

pub fn history_stats(rows: &SequenceRows, req_ms: i64) -> Vec<f32> {
    let skip = rows.len().saturating_sub(HISTORY_STATS_MAX_ROWS);
    let newest;
    let rows = if skip > 0 {
        newest = rows.columns().rows(skip..rows.len());
        &newest
    } else {
        rows
    };
    let mut out = featurize_sequence(rows, req_ms);
    let mut gallery = rows.clone();
    let is_gallery: Vec<bool> = rows.surface.iter().map(|&s| s == GALLERY_SURFACE).collect();
    gallery.retain(&is_gallery);
    let gallery_features = featurize_sequence(&gallery, req_ms);
    let mut start = 0;
    for (repeated_start, repeated_len) in GALLERY_REPEATED {
        out.extend_from_slice(&gallery_features[start..repeated_start]);
        start = repeated_start + repeated_len;
    }
    out.extend_from_slice(&gallery_features[start..]);
    debug_assert_eq!(out.len(), HISTORY_STATS_DIM);
    out
}

pub fn featurize_batch(
    columns: &SequenceColumns<'_>,
    offsets: &[usize],
    req_ms: &[i64],
) -> Result<Vec<f32>, crate::Error> {
    map_records(columns, offsets, req_ms, FEATURE_DIM, featurize_sequence)
}

pub fn history_stats_batch(
    columns: &SequenceColumns<'_>,
    offsets: &[usize],
    req_ms: &[i64],
) -> Result<Vec<f32>, crate::Error> {
    map_records(columns, offsets, req_ms, HISTORY_STATS_DIM, history_stats)
}

fn map_records(
    columns: &SequenceColumns<'_>,
    offsets: &[usize],
    req_ms: &[i64],
    dim: usize,
    featurize: fn(&SequenceRows, i64) -> Vec<f32>,
) -> Result<Vec<f32>, crate::Error> {
    use rayon::prelude::*;

    let lengths = columns.lengths();
    if lengths.iter().any(|&len| len != lengths[0]) {
        return Err(crate::Error::Invalid(format!(
            "sequence columns have inconsistent lengths: {lengths:?}"
        )));
    }
    if offsets.len() != req_ms.len() + 1 {
        return Err(crate::Error::Invalid(format!(
            "offsets has {} entries for {} records",
            offsets.len(),
            req_ms.len()
        )));
    }
    if offsets.first() != Some(&0)
        || offsets.last() != Some(&lengths[0])
        || offsets.windows(2).any(|w| w[0] > w[1])
    {
        return Err(crate::Error::Invalid(
            "offsets must start at 0, be non-decreasing and end at the column length".to_string(),
        ));
    }
    let mut out = vec![0.0f32; req_ms.len() * dim];
    out.par_chunks_mut(dim).enumerate().for_each(|(i, row)| {
        let rows = columns.rows(offsets[i]..offsets[i + 1]);
        row.copy_from_slice(&featurize(&rows, req_ms[i]));
    });
    Ok(out)
}
