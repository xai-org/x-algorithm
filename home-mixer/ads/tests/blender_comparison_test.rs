use crate::ads::{
    least_movement_blender, multi_risk_blender, partition_organic_blender, safe_gap_blender,
    slot_spread_blender,
    util::{should_drop_handle, should_drop_keyword},
};
use crate::params::RESULT_SIZE;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use xai_ads_injection_proto::ads_injected_timeline::DroppedAd;
use xai_home_mixer_proto::{feed_item, BrandSafetyVerdict, FeedItem, ScoredPost};
use xai_recsys_proto::{AdAdjacencyControl, AdIndexInfo, BrandSafetyRiskLevel};

const HIGH_RISK_LABEL_POST_RATE: f64 = 0.045;
const NSFW_AUTHOR_POST_RATE: f64 = 0.117;
const MEDIUM_RISK_POST_RATE: f64 = 0.411;
const LOW_RISK_POST_RATE: f64 = 0.093;

const NSFW_AUTHOR_POOL: &[u64] = &[2001, 2002, 2003, 2004, 2005];

const BSR_LOW_AD_RATE: f64 = 0.214;
const BSR_IAS_AD_RATE: f64 = 0.003;
const BSR_NO_RISK_AD_RATE: f64 = 0.013;
const BSR_HIGH_AD_RATE: f64 = 0.10;

const KNOWN_HANDLE_POST_RATE: f64 = 0.10;
const AD_EXCLUDED_HANDLES_RATE: f64 = 0.10;
const AD_EXCLUDED_KEYWORDS_RATE: f64 = 0.10;

const HANDLE_POOL: &[i64] = &[1001, 1002, 1003, 1004, 1005];

const SCREEN_NAME_POOL: &[&str] = &["user_a", "user_b", "user_c", "user_d", "user_e"];

const KEYWORD_POOL: &[&str] = &["alpha", "bravo", "charlie", "delta", "echo"];

const KEYWORD_IN_TEXT_RATE: f64 = 0.10;

const SAFE_TEXT_POOL: &[&str] = &[
    "what a beautiful day outside",
    "just finished a great workout",
    "loving this new recipe I found",
    "anyone watching the game tonight",
    "happy birthday to my best friend",
];

const IDEAL_GAPS: &[usize] = &[1, 5, 9, 13, 17, 21, 25];

const NUM_SCENARIOS: usize = 1000;

const RNG_SEED: u64 = 412;

struct Scenario {
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
}

fn expected_high_risk_post_rate() -> f64 {
    1.0 - (1.0 - HIGH_RISK_LABEL_POST_RATE) * (1.0 - NSFW_AUTHOR_POST_RATE)
}

fn is_avoid_verdict(verdict: BrandSafetyVerdict) -> bool {
    matches!(
        verdict,
        BrandSafetyVerdict::MediumRisk | BrandSafetyVerdict::HighRisk
    )
}

fn block_ideal_gaps(posts: &mut [ScoredPost]) {
    let n = posts.len();
    for &gap in IDEAL_GAPS {
        if gap >= n {
            continue;
        }
        let left = gap - 1;
        let right = gap;
        if is_avoid_verdict(posts[left].brand_safety_verdict())
            || is_avoid_verdict(posts[right].brand_safety_verdict())
        {
            continue;
        }
        if let Some(j) = (right + 1..n).find(|&j| is_avoid_verdict(posts[j].brand_safety_verdict()))
        {
            posts.swap(right, j);
        }
    }
}

fn generate_scenarios(rng: &mut StdRng) -> Vec<Scenario> {
    (0..NUM_SCENARIOS)
        .map(|_| {
            let num_posts = rng.random_range(31..=35);
            let num_ads = rng.random_range(3..=15);

            let mut posts: Vec<ScoredPost> = (1..=num_posts)
                .map(|id| {
                    let roll: f64 = rng.random();
                    let tweet_verdict = if roll < HIGH_RISK_LABEL_POST_RATE {
                        BrandSafetyVerdict::HighRisk
                    } else if roll < HIGH_RISK_LABEL_POST_RATE + MEDIUM_RISK_POST_RATE {
                        BrandSafetyVerdict::MediumRisk
                    } else if roll
                        < HIGH_RISK_LABEL_POST_RATE + MEDIUM_RISK_POST_RATE + LOW_RISK_POST_RATE
                    {
                        BrandSafetyVerdict::LowRisk
                    } else {
                        BrandSafetyVerdict::Safe
                    };
                    let nsfw_author = rng.random_bool(NSFW_AUTHOR_POST_RATE);
                    let verdict = if nsfw_author {
                        BrandSafetyVerdict::HighRisk
                    } else {
                        tweet_verdict
                    };

                    let (author_id, screen_names) = if rng.random_bool(KNOWN_HANDLE_POST_RATE) {
                        let idx = rng.random_range(0..HANDLE_POOL.len());
                        let aid = HANDLE_POOL[idx] as u64;
                        let mut map = HashMap::new();
                        map.insert(aid, SCREEN_NAME_POOL[idx].to_string());
                        (aid, map)
                    } else if nsfw_author {
                        let aid = NSFW_AUTHOR_POOL[rng.random_range(0..NSFW_AUTHOR_POOL.len())];
                        (aid, HashMap::new())
                    } else {
                        (id as u64, HashMap::new())
                    };

                    let tweet_text = if rng.random_bool(KEYWORD_IN_TEXT_RATE) {
                        let kw = KEYWORD_POOL[rng.random_range(0..KEYWORD_POOL.len())];
                        format!("This post is about {kw} and other topics")
                    } else {
                        SAFE_TEXT_POOL[rng.random_range(0..SAFE_TEXT_POOL.len())].to_string()
                    };

                    ScoredPost {
                        tweet_id: id as u64,
                        author_id,
                        brand_safety_verdict: verdict.into(),
                        screen_names,
                        tweet_text,
                        ..Default::default()
                    }
                })
                .collect();
            block_ideal_gaps(&mut posts);

            let ads: Vec<AdIndexInfo> = (0..num_ads)
                .map(|i| {
                    let insert_position = i * 8 + 1;
                    let roll: f64 = rng.random();
                    let risk = if roll < BSR_LOW_AD_RATE {
                        BrandSafetyRiskLevel::BsrLow
                    } else if roll < BSR_LOW_AD_RATE + BSR_IAS_AD_RATE {
                        BrandSafetyRiskLevel::BsrIas
                    } else if roll < BSR_LOW_AD_RATE + BSR_IAS_AD_RATE + BSR_NO_RISK_AD_RATE {
                        BrandSafetyRiskLevel::BsrNoRisk
                    } else if roll
                        < BSR_LOW_AD_RATE + BSR_IAS_AD_RATE + BSR_NO_RISK_AD_RATE + BSR_HIGH_AD_RATE
                    {
                        BrandSafetyRiskLevel::BsrHigh
                    } else {
                        BrandSafetyRiskLevel::BsrNormal
                    };

                    let handles = if rng.random_bool(AD_EXCLUDED_HANDLES_RATE) {
                        let count = rng.random_range(1..=3);
                        (0..count)
                            .map(|_| HANDLE_POOL[rng.random_range(0..HANDLE_POOL.len())])
                            .collect()
                    } else {
                        vec![]
                    };

                    let keywords = if rng.random_bool(AD_EXCLUDED_KEYWORDS_RATE) {
                        let count = rng.random_range(1..=2);
                        (0..count)
                            .map(|_| {
                                KEYWORD_POOL[rng.random_range(0..KEYWORD_POOL.len())].to_string()
                            })
                            .collect()
                    } else {
                        vec![]
                    };

                    AdIndexInfo {
                        post_id: 1000 + i as i64,
                        insert_position,
                        ad_adjacency_control: Some(AdAdjacencyControl {
                            brand_safety_risk: risk.into(),
                            keywords,
                            handles,
                        }),
                        ..Default::default()
                    }
                })
                .collect();

            Scenario { posts, ads }
        })
        .collect()
}

fn ad_count(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .count()
}

fn post_count(items: &[FeedItem]) -> usize {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Post(_))))
        .count()
}

fn ad_positions(items: &[FeedItem]) -> Vec<i32> {
    items
        .iter()
        .filter(|i| matches!(i.item, Some(feed_item::Item::Ad(_))))
        .map(|i| i.position)
        .collect()
}

fn ad_bsr(ad: &AdIndexInfo) -> BrandSafetyRiskLevel {
    ad.ad_adjacency_control
        .as_ref()
        .map(|c| c.brand_safety_risk())
        .unwrap_or(BrandSafetyRiskLevel::BsrUnknown)
}

fn is_bsr_low_or_ias(risk: BrandSafetyRiskLevel) -> bool {
    matches!(
        risk,
        BrandSafetyRiskLevel::BsrLow | BrandSafetyRiskLevel::BsrIas
    )
}

struct AdjacencyViolations {
    med_risk_adj: usize,
    non_high_med_risk_adj: usize,
    bsr_low_low_risk_adj: usize,
    bsr_low_placed: usize,
    bsr_low_available: usize,
    bsr_high_placed: usize,
    bsr_high_available: usize,
    bsr_low_only_placed: usize,
    bsr_ias_placed: usize,
    bsr_normal_placed: usize,
    bsr_no_risk_placed: usize,
    handle_adj: usize,
    keyword_adj: usize,
}

fn count_adjacency_violations(
    items: &[FeedItem],
    input_ads: &[AdIndexInfo],
) -> AdjacencyViolations {
    let mut v = AdjacencyViolations {
        med_risk_adj: 0,
        non_high_med_risk_adj: 0,
        bsr_low_low_risk_adj: 0,
        bsr_low_placed: 0,
        handle_adj: 0,
        keyword_adj: 0,
        bsr_high_placed: 0,
        bsr_low_only_placed: 0,
        bsr_ias_placed: 0,
        bsr_normal_placed: 0,
        bsr_no_risk_placed: 0,
        bsr_low_available: input_ads
            .iter()
            .filter(|a| is_bsr_low_or_ias(ad_bsr(a)))
            .count(),
        bsr_high_available: input_ads
            .iter()
            .filter(|a| ad_bsr(a) == BrandSafetyRiskLevel::BsrHigh)
            .count(),
    };

    for (i, item) in items.iter().enumerate() {
        let ad = match &item.item {
            Some(feed_item::Item::Ad(ad)) => ad,
            _ => continue,
        };

        let risk = ad_bsr(ad);
        let is_sensitive = is_bsr_low_or_ias(risk);
        if is_sensitive {
            v.bsr_low_placed += 1;
        }
        if risk == BrandSafetyRiskLevel::BsrHigh {
            v.bsr_high_placed += 1;
        }
        match risk {
            BrandSafetyRiskLevel::BsrLow => v.bsr_low_only_placed += 1,
            BrandSafetyRiskLevel::BsrIas => v.bsr_ias_placed += 1,
            BrandSafetyRiskLevel::BsrNormal => v.bsr_normal_placed += 1,
            BrandSafetyRiskLevel::BsrNoRisk => v.bsr_no_risk_placed += 1,
            _ => {}
        }

        let above_post = if i > 0 {
            match &items[i - 1].item {
                Some(feed_item::Item::Post(p)) => Some(p),
                _ => None,
            }
        } else {
            None
        };
        let below_post = if i + 1 < items.len() {
            match &items[i + 1].item {
                Some(feed_item::Item::Post(p)) => Some(p),
                _ => None,
            }
        } else {
            None
        };

        let above_verdict = above_post.map(|p| p.brand_safety_verdict());
        let below_verdict = below_post.map(|p| p.brand_safety_verdict());

        if above_verdict == Some(BrandSafetyVerdict::MediumRisk)
            || below_verdict == Some(BrandSafetyVerdict::MediumRisk)
        {
            v.med_risk_adj += 1;
            if risk != BrandSafetyRiskLevel::BsrHigh {
                v.non_high_med_risk_adj += 1;
            }
        }

        if is_sensitive
            && (above_verdict == Some(BrandSafetyVerdict::LowRisk)
                || below_verdict == Some(BrandSafetyVerdict::LowRisk))
        {
            v.bsr_low_low_risk_adj += 1;
        }

        if should_drop_handle(ad, above_post, below_post) {
            v.handle_adj += 1;
        }

        if should_drop_keyword(ad, above_post, below_post) {
            v.keyword_adj += 1;
        }
    }

    v
}

fn check_invariants(items: &[FeedItem], blender_name: &str, scenario_idx: usize) -> bool {
    if items.is_empty() {
        return true;
    }

    if matches!(items.first(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        eprintln!("[{blender_name}] scenario {scenario_idx}: ad at position 0!");
        return false;
    }
    if matches!(items.last(), Some(item) if matches!(item.item, Some(feed_item::Item::Ad(_)))) {
        eprintln!("[{blender_name}] scenario {scenario_idx}: ad at last position!");
        return false;
    }

    let positions = ad_positions(items);
    for w in positions.windows(2) {
        let between = (w[1] - w[0] - 1) as usize;
        if between < 2 {
            eprintln!(
                "[{blender_name}] scenario {scenario_idx}: only {between} items between ads at {} and {}",
                w[0], w[1]
            );
            return false;
        }
    }

    true
}

struct BlenderStats {
    name: &'static str,
    total_ads_placed: usize,
    total_ads_available: usize,
    total_posts: usize,
    total_items: usize,
    invariant_violations: usize,
    total_spacing_sum: f64,
    total_spacing_count: usize,
    ads_with_medrisk_adj: usize,
    bsr_low_with_lowrisk_adj: usize,
    bsr_low_available: usize,
    bsr_low_placed: usize,
    non_high_med_risk_adj: usize,
    bsr_high_available: usize,
    bsr_high_placed: usize,
    bsr_low_only_placed: usize,
    bsr_ias_placed: usize,
    bsr_normal_placed: usize,
    bsr_no_risk_placed: usize,
    handle_adj: usize,
    keyword_adj: usize,
}

impl BlenderStats {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            total_ads_placed: 0,
            total_ads_available: 0,
            total_posts: 0,
            total_items: 0,
            invariant_violations: 0,
            total_spacing_sum: 0.0,
            total_spacing_count: 0,
            ads_with_medrisk_adj: 0,
            bsr_low_with_lowrisk_adj: 0,
            bsr_low_available: 0,
            bsr_low_placed: 0,
            non_high_med_risk_adj: 0,
            bsr_high_available: 0,
            bsr_high_placed: 0,
            bsr_low_only_placed: 0,
            bsr_ias_placed: 0,
            bsr_normal_placed: 0,
            bsr_no_risk_placed: 0,
            handle_adj: 0,
            keyword_adj: 0,
        }
    }

    fn ad_fill_rate(&self) -> f64 {
        if self.total_ads_available == 0 {
            return 0.0;
        }
        self.total_ads_placed as f64 / self.total_ads_available as f64 * 100.0
    }

    fn ad_load_pct(&self) -> f64 {
        if self.total_items == 0 {
            return 0.0;
        }
        self.total_ads_placed as f64 / self.total_items as f64 * 100.0
    }

    fn avg_spacing(&self) -> f64 {
        if self.total_spacing_count == 0 {
            return 0.0;
        }
        self.total_spacing_sum / self.total_spacing_count as f64
    }

    fn medrisk_adj_pct(&self) -> f64 {
        if self.total_ads_placed == 0 {
            return 0.0;
        }
        self.ads_with_medrisk_adj as f64 / self.total_ads_placed as f64 * 100.0
    }

    fn bsr_low_lowrisk_adj_pct(&self) -> f64 {
        if self.bsr_low_placed == 0 {
            return 0.0;
        }
        self.bsr_low_with_lowrisk_adj as f64 / self.bsr_low_placed as f64 * 100.0
    }

    fn bsr_low_drop_pct(&self) -> f64 {
        if self.bsr_low_available == 0 {
            return 0.0;
        }
        (1.0 - self.bsr_low_placed as f64 / self.bsr_low_available as f64) * 100.0
    }

    fn bsr_high_drop_pct(&self) -> f64 {
        if self.bsr_high_available == 0 {
            return 0.0;
        }
        (1.0 - self.bsr_high_placed as f64 / self.bsr_high_available as f64) * 100.0
    }
}

type BlendFn = fn(Vec<ScoredPost>, Vec<AdIndexInfo>, usize) -> Vec<FeedItem>;

fn safe_gap_blend(
    posts: Vec<ScoredPost>,
    ads: Vec<AdIndexInfo>,
    min_posts: usize,
) -> Vec<FeedItem> {
    safe_gap_blender::blend_impl(posts, ads, min_posts, RESULT_SIZE)
}

#[test]
fn blender_comparison_report() {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let scenarios = generate_scenarios(&mut rng);

    let blenders: Vec<(&str, BlendFn)> = vec![
        ("safe_gap", safe_gap_blend),
        ("partition_organic", partition_organic_blender::blend_impl),
        ("multi_risk", multi_risk_blender::blend_impl),
        ("slot_spread", |posts, ads, min_posts| {
            slot_spread_blender::blend_with_drops(posts, ads, min_posts, &mut Vec::new())
        }),
        ("least_movement", |posts, ads, min_posts| {
            least_movement_blender::blend_with_drops(posts, ads, min_posts, &mut Vec::new())
        }),
    ];

    let mut all_stats: Vec<BlenderStats> = blenders
        .iter()
        .map(|(name, _)| BlenderStats::new(name))
        .collect();

    for (scenario_idx, scenario) in scenarios.iter().enumerate() {
        for (blender_idx, (_name, blend_fn)) in blenders.iter().enumerate() {
            let posts = scenario.posts.clone();
            let ads = scenario.ads.clone();
            let num_ads_available = ads.len();

            let result = blend_fn(posts, ads, 0);

            let ads_placed = ad_count(&result);
            let posts_in_result = post_count(&result);

            if !check_invariants(&result, all_stats[blender_idx].name, scenario_idx) {
                all_stats[blender_idx].invariant_violations += 1;
            }

            let positions = ad_positions(&result);
            for w in positions.windows(2) {
                let spacing = (w[1] - w[0] - 1) as f64;
                all_stats[blender_idx].total_spacing_sum += spacing;
                all_stats[blender_idx].total_spacing_count += 1;
            }

            let v = count_adjacency_violations(&result, &scenario.ads);
            all_stats[blender_idx].ads_with_medrisk_adj += v.med_risk_adj;
            all_stats[blender_idx].non_high_med_risk_adj += v.non_high_med_risk_adj;
            all_stats[blender_idx].bsr_low_with_lowrisk_adj += v.bsr_low_low_risk_adj;
            all_stats[blender_idx].bsr_low_available += v.bsr_low_available;
            all_stats[blender_idx].bsr_low_placed += v.bsr_low_placed;
            all_stats[blender_idx].bsr_high_available += v.bsr_high_available;
            all_stats[blender_idx].bsr_high_placed += v.bsr_high_placed;
            all_stats[blender_idx].bsr_low_only_placed += v.bsr_low_only_placed;
            all_stats[blender_idx].bsr_ias_placed += v.bsr_ias_placed;
            all_stats[blender_idx].bsr_normal_placed += v.bsr_normal_placed;
            all_stats[blender_idx].bsr_no_risk_placed += v.bsr_no_risk_placed;
            all_stats[blender_idx].handle_adj += v.handle_adj;
            all_stats[blender_idx].keyword_adj += v.keyword_adj;

            all_stats[blender_idx].total_ads_placed += ads_placed;
            all_stats[blender_idx].total_ads_available += num_ads_available;
            all_stats[blender_idx].total_posts += posts_in_result;
            all_stats[blender_idx].total_items += result.len();
        }
    }

    eprintln!();
    eprintln!(
        "╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗"
    );
    eprintln!(
        "║                                              ADS BLENDER COMPARISON REPORT                                                         ║"
    );
    eprintln!(
        "║  {NUM_SCENARIOS} scenarios — posts: ~{:.0}% HighRisk ({:.0}% labels + {:.0}% nsfw_author), ~{:.0}% MedRisk, ~{:.0}% LowRisk, ~{:.0}% Safe — ads: ~{:.0}% BSR_LOW, ~{:.0}% BSR_IAS, ~{:.0}% BSR_HIGH, ~{:.0}% BSR_NORMAL          ║",
        expected_high_risk_post_rate() * 100.0,
        HIGH_RISK_LABEL_POST_RATE * 100.0,
        NSFW_AUTHOR_POST_RATE * 100.0,
        MEDIUM_RISK_POST_RATE * (1.0 - NSFW_AUTHOR_POST_RATE) * 100.0,
        LOW_RISK_POST_RATE * (1.0 - NSFW_AUTHOR_POST_RATE) * 100.0,
        (1.0 - HIGH_RISK_LABEL_POST_RATE - MEDIUM_RISK_POST_RATE - LOW_RISK_POST_RATE)
            * (1.0 - NSFW_AUTHOR_POST_RATE)
            * 100.0,
        BSR_LOW_AD_RATE * 100.0,
        BSR_IAS_AD_RATE * 100.0,
        BSR_HIGH_AD_RATE * 100.0,
        (1.0 - BSR_LOW_AD_RATE - BSR_IAS_AD_RATE - BSR_NO_RISK_AD_RATE - BSR_HIGH_AD_RATE) * 100.0,
    );
    eprintln!(
        "╠══════════════════════╦═════════╦═════════╦═════════╦═════════╦═════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╦══════════╣"
    );
    eprintln!(
        "║ Blender              ║ AdPlace ║ AdAvail ║  Fill%  ║  Load%  ║ AvgSpc  ║ MedAdj%  ║ LR→BSR%  ║ BSRDrop% ║ HighDrop%║ HighPlcd ║ HndlAdj  ║  KwAdj   ║"
    );
    eprintln!(
        "╠══════════════════════╬═════════╬═════════╬═════════╬═════════╬═════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╬══════════╣"
    );
    for stats in &all_stats {
        eprintln!(
            "║ {:<20} ║ {:>7} ║ {:>7} ║ {:>6.1}% ║ {:>6.1}% ║ {:>7.2} ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║ {:>7.1}% ║ {:>4}/{:<4}║ {:>8} ║ {:>8} ║",
            stats.name,
            stats.total_ads_placed,
            stats.total_ads_available,
            stats.ad_fill_rate(),
            stats.ad_load_pct(),
            stats.avg_spacing(),
            stats.medrisk_adj_pct(),
            stats.bsr_low_lowrisk_adj_pct(),
            stats.bsr_low_drop_pct(),
            stats.bsr_high_drop_pct(),
            stats.bsr_high_placed,
            stats.bsr_high_available,
            stats.handle_adj,
            stats.keyword_adj,
        );
    }
    eprintln!(
        "╚══════════════════════╩═════════╩═════════╩═════════╩═════════╩═════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╩══════════╝"
    );
    eprintln!();
    eprintln!("Placed mix (share of AdPlace; Safe = BSR_NORMAL):");
    for stats in &all_stats {
        let n = stats.total_ads_placed as f64;
        let pct = |c: usize| if n == 0.0 { 0.0 } else { c as f64 / n * 100.0 };
        eprintln!(
            "  {:<20} LOW {:>4} ({:>4.1}%)  IAS {:>4} ({:>4.1}%)  HIGH {:>4} ({:>4.1}%)  Safe {:>4} ({:>4.1}%)  other {:>4} ({:>4.1}%)",
            stats.name,
            stats.bsr_low_only_placed,
            pct(stats.bsr_low_only_placed),
            stats.bsr_ias_placed,
            pct(stats.bsr_ias_placed),
            stats.bsr_high_placed,
            pct(stats.bsr_high_placed),
            stats.bsr_normal_placed,
            pct(stats.bsr_normal_placed),
            stats.total_ads_placed
                - stats.bsr_low_only_placed
                - stats.bsr_ias_placed
                - stats.bsr_high_placed
                - stats.bsr_normal_placed,
            pct(stats.total_ads_placed
                - stats.bsr_low_only_placed
                - stats.bsr_ias_placed
                - stats.bsr_high_placed
                - stats.bsr_normal_placed),
        );
    }
    eprintln!();
    eprintln!("Legend: Fill% = ads placed / ads available");
    eprintln!("        Load% = ads placed / total items (ad load in timeline)");
    eprintln!("        AvgSpc = average number of posts between consecutive ads");
    eprintln!("        MedAdj% = % of placed ads adjacent to MediumRisk post");
    eprintln!(
        "        LR→BSR% = % of placed BSR_LOW/IAS ads adjacent to LowRisk post (should be 0 for partition_organic)"
    );
    eprintln!("        BSRDrop% = % of BSR_LOW/IAS ads dropped (not placed)");
    eprintln!("        HighDrop% = % of BSR_HIGH ads dropped (not placed)");
    eprintln!("        HighPlcd = BSR_HIGH ads placed / available");
    eprintln!(
        "        HndlAdj = placed ads adjacent to an excluded handle (should be 0 for partition_organic)"
    );
    eprintln!(
        "        KwAdj = placed ads adjacent to an excluded keyword match (should be 0 for partition_organic)"
    );
    eprintln!();

    for stats in &all_stats {
        assert_eq!(
            stats.invariant_violations, 0,
            "{} had {} invariant violations",
            stats.name, stats.invariant_violations
        );
    }

    let po_stats = all_stats
        .iter()
        .find(|s| s.name == "partition_organic")
        .expect("partition_organic blender not found in stats");
    assert_eq!(
        po_stats.bsr_low_with_lowrisk_adj, 0,
        "partition_organic had {} BSR_LOW/IAS ads adjacent to LowRisk posts!",
        po_stats.bsr_low_with_lowrisk_adj
    );
    assert_eq!(
        po_stats.handle_adj, 0,
        "partition_organic had {} ads adjacent to excluded handles!",
        po_stats.handle_adj
    );
    assert_eq!(
        po_stats.keyword_adj, 0,
        "partition_organic had {} ads adjacent to excluded keyword matches!",
        po_stats.keyword_adj
    );
    assert_eq!(
        po_stats.ads_with_medrisk_adj, 0,
        "partition_organic had {} ads adjacent to MediumRisk posts!",
        po_stats.ads_with_medrisk_adj
    );

    for name in ["multi_risk", "slot_spread", "least_movement"] {
        let npo_stats = all_stats
            .iter()
            .find(|s| s.name == name)
            .unwrap_or_else(|| panic!("{name} blender not found in stats"));
        assert_eq!(
            npo_stats.bsr_low_with_lowrisk_adj, 0,
            "{name} had {} BSR_LOW/IAS ads adjacent to LowRisk posts!",
            npo_stats.bsr_low_with_lowrisk_adj
        );
        assert_eq!(
            npo_stats.handle_adj, 0,
            "{name} had {} ads adjacent to excluded handles!",
            npo_stats.handle_adj
        );
        assert_eq!(
            npo_stats.keyword_adj, 0,
            "{name} had {} ads adjacent to excluded keyword matches!",
            npo_stats.keyword_adj
        );
        assert_eq!(
            npo_stats.non_high_med_risk_adj, 0,
            "{name} had {} non-BSR_HIGH ads adjacent to MediumRisk posts!",
            npo_stats.non_high_med_risk_adj
        );
    }
}

type BlendWithDropsFn =
    fn(Vec<ScoredPost>, Vec<AdIndexInfo>, usize, &mut Vec<DroppedAd>) -> Vec<FeedItem>;

#[test]
fn new_blenders_account_for_every_ad_and_serve_at_least_multi_risk() {
    let mut rng = StdRng::seed_from_u64(RNG_SEED);
    let scenarios = generate_scenarios(&mut rng);
    let blenders: [(&str, BlendWithDropsFn); 2] = [
        ("slot_spread", slot_spread_blender::blend_with_drops),
        ("least_movement", least_movement_blender::blend_with_drops),
    ];
    let mut baseline_total = 0;
    let mut totals = [0; 2];
    for (scenario_idx, scenario) in scenarios.iter().enumerate() {
        baseline_total += ad_count(&multi_risk_blender::blend_impl(
            scenario.posts.clone(),
            scenario.ads.clone(),
            5,
        ));
        for (b, (name, blend)) in blenders.iter().enumerate() {
            let mut drops = Vec::new();
            let result = blend(scenario.posts.clone(), scenario.ads.clone(), 5, &mut drops);
            totals[b] += ad_count(&result);
            let mut outcomes: HashMap<i64, usize> = HashMap::new();
            for item in &result {
                if let Some(feed_item::Item::Ad(ad)) = &item.item {
                    *outcomes.entry(ad.post_id).or_default() += 1;
                }
            }
            for drop in &drops {
                *outcomes.entry(drop.tweet_id as i64).or_default() += 1;
            }
            for ad in &scenario.ads {
                assert_eq!(
                    outcomes.get(&ad.post_id).copied(),
                    Some(1),
                    "[{name}] scenario {scenario_idx}: ad {} served or dropped more or less than once",
                    ad.post_id
                );
            }
        }
    }
    for ((name, _), total) in blenders.iter().zip(totals) {
        assert!(
            total >= baseline_total,
            "{name} served {total} ads vs multi_risk {baseline_total}"
        );
    }
}
