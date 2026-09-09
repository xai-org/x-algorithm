use anyhow::Result;
use dashmap::DashMap;
use log::info;
use std::collections::{HashSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::time::MissedTickBehavior;
use xai_thunder_proto::{LightPost, TweetDeleteEvent};

use crate::config::{
    DELETE_EVENT_KEY, MAX_ORIGINAL_POSTS_PER_AUTHOR, MAX_POSTING_LIST_SIZE,
    MAX_REPLY_POSTS_PER_AUTHOR, MAX_TINY_POSTS_PER_USER_SCAN, MAX_VIDEO_POSTS_PER_AUTHOR,
    TRIM_FRACTION,
};
use crate::metrics::{
    POST_STORE_DELETED_POSTS, POST_STORE_ENTITY_COUNT, POST_STORE_POSTS_RETURNED,
    POST_STORE_POSTS_RETURNED_RATIO, POST_STORE_REQUEST_TIMEOUTS, POST_STORE_REQUESTS,
    POST_STORE_TOTAL_POSTS, POST_STORE_USER_COUNT,
};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TinyPost {
    pub post_id: i64,
    pub created_at: i64,
}

impl TinyPost {
        pub fn new(post_id: i64, created_at: i64) -> Self {
        TinyPost {
            post_id,
            created_at,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CompactPost {
    pub post_id: i64,
    pub author_id: i64,
    pub created_at: i64,
    pub in_reply_to_post_id: i64,
    pub in_reply_to_user_id: i64,
    pub source_post_id: i64,
    pub source_user_id: i64,
    pub conversation_id: i64,
    pub is_retweet: bool,
    pub is_reply: bool,
    pub has_video: bool,
}

impl From<LightPost> for CompactPost {
    fn from(p: LightPost) -> Self {
        CompactPost {
            post_id: p.post_id,
            author_id: p.author_id,
            created_at: p.created_at,
            in_reply_to_post_id: p.in_reply_to_post_id.unwrap_or(0),
            in_reply_to_user_id: p.in_reply_to_user_id.unwrap_or(0),
            source_post_id: p.source_post_id.unwrap_or(0),
            source_user_id: p.source_user_id.unwrap_or(0),
            conversation_id: p.conversation_id.unwrap_or(0),
            is_retweet: p.is_retweet,
            is_reply: p.is_reply,
            has_video: p.has_video,
        }
    }
}

impl From<CompactPost> for LightPost {
    fn from(p: CompactPost) -> Self {
        let to_opt = |v: i64| if v != 0 { Some(v) } else { None };
        LightPost {
            post_id: p.post_id,
            author_id: p.author_id,
            created_at: p.created_at,
            in_reply_to_post_id: to_opt(p.in_reply_to_post_id),
            in_reply_to_user_id: to_opt(p.in_reply_to_user_id),
            source_post_id: to_opt(p.source_post_id),
            source_user_id: to_opt(p.source_user_id),
            conversation_id: to_opt(p.conversation_id),
            is_retweet: p.is_retweet,
            is_reply: p.is_reply,
            has_video: p.has_video,
        }
    }
}

#[derive(Clone)]
pub struct PostStore {
        posts: Arc<DashMap<i64, Arc<CompactPost>>>,
        original_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
        secondary_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
        video_posts_by_user: Arc<DashMap<i64, VecDeque<TinyPost>>>,
    deleted_posts: Arc<DashMap<i64, bool>>,
        retention_seconds: u64,
        request_timeout: Duration,
}

impl PostStore {
        pub fn new(retention_seconds: u64, request_timeout_ms: u64) -> Self {
        PostStore {
            posts: Arc::new(DashMap::new()),
            original_posts_by_user: Arc::new(DashMap::new()),
            secondary_posts_by_user: Arc::new(DashMap::new()),
            video_posts_by_user: Arc::new(DashMap::new()),
            deleted_posts: Arc::new(DashMap::new()),
            retention_seconds,
            request_timeout: Duration::from_millis(request_timeout_ms),
        }
    }

    pub fn mark_as_deleted(&self, posts: Vec<TweetDeleteEvent>) {
        for post in posts.into_iter() {
            self.posts.remove(&post.post_id);
            self.deleted_posts.insert(post.post_id, true);

            let mut user_posts_entry = self
                .original_posts_by_user
                .entry(DELETE_EVENT_KEY)
                .or_default();
            user_posts_entry.push_back(TinyPost {
                post_id: post.post_id,
                created_at: post.deleted_at, 
            });
        }
    }

        pub fn insert_posts(&self, mut posts: Vec<LightPost>) {
        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        posts.retain(|p| {
            p.created_at <= current_time
                && current_time - p.created_at <= (self.retention_seconds as i64)
        });

        posts.sort_unstable_by_key(|p| p.created_at);

        Self::insert_posts_internal(self, posts);
    }

    pub async fn finalize_init(&self) -> Result<()> {
        self.sort_all_user_posts().await;

        for trim_iter in 0..TRIM_FRACTION {
            self.trim_old_posts(trim_iter).await;
        }

        for entry in self.deleted_posts.iter() {
            self.posts.remove(entry.key());
        }

        Ok(())
    }

    fn insert_posts_internal(&self, posts: Vec<LightPost>) {
        for post in posts {
            let post_id = post.post_id;
            let author_id = post.author_id;
            let created_at = post.created_at;
            let is_original = !post.is_reply && !post.is_retweet;

            if self.deleted_posts.contains_key(&post_id) {
                continue;
            }

            let old = self
                .posts
                .insert(post_id, Arc::new(CompactPost::from(post)));
            if old.is_some() {
                continue;
            }

            let tiny_post = TinyPost::new(post_id, created_at);

            let mut posts_to_remove = Vec::new();

            if is_original {
                let mut user_posts_entry =
                    self.original_posts_by_user.entry(author_id).or_default();
                while user_posts_entry.len() >= MAX_POSTING_LIST_SIZE {
                    if let Some(oldest) = user_posts_entry.pop_front() {
                        posts_to_remove.push(oldest.post_id);
                    }
                }
                user_posts_entry.push_back(tiny_post.clone());
            } else {
                let mut user_posts_entry =
                    self.secondary_posts_by_user.entry(author_id).or_default();
                while user_posts_entry.len() >= MAX_POSTING_LIST_SIZE {
                    if let Some(oldest) = user_posts_entry.pop_front() {
                        posts_to_remove.push(oldest.post_id);
                    }
                }
                user_posts_entry.push_back(tiny_post.clone());
            }

            let mut video_eligible = post.has_video;

            if !video_eligible
                && post.is_retweet
                && let Some(source_post_id) = post.source_post_id
                && let Some(source_post) = self.posts.get(&source_post_id)
            {
                video_eligible = !source_post.is_reply && source_post.has_video;
            }

            if post.is_reply {
                video_eligible = false;
            }

            if video_eligible {
                let mut user_posts_entry = self.video_posts_by_user.entry(author_id).or_default();
                while user_posts_entry.len() >= MAX_POSTING_LIST_SIZE {
                    if let Some(oldest) = user_posts_entry.pop_front() {
                        posts_to_remove.push(oldest.post_id);
                    }
                }
                user_posts_entry.push_back(tiny_post);
            }

            for post_id in posts_to_remove {
                self.posts.remove(&post_id);
            }
        }

    }

        pub fn get_videos_by_users(
        &self,
        user_ids: &[i64],
        exclude_tweet_ids: &HashSet<i64>,
        start_time: Instant,
        request_user_id: i64,
    ) -> Vec<LightPost> {
        let video_posts = self.get_posts_from_map(
            &self.video_posts_by_user,
            user_ids,
            MAX_VIDEO_POSTS_PER_AUTHOR,
            exclude_tweet_ids,
            &HashSet::new(), 
            start_time,
            request_user_id,
        );

        POST_STORE_POSTS_RETURNED.observe(video_posts.len() as f64);
        video_posts
    }

        pub fn get_all_posts_by_users(
        &self,
        user_ids: &[i64],
        exclude_tweet_ids: &HashSet<i64>,
        start_time: Instant,
        request_user_id: i64,
    ) -> Vec<LightPost> {
        let following_users_set: HashSet<i64> = user_ids.iter().copied().collect();

        let mut all_posts = self.get_posts_from_map(
            &self.original_posts_by_user,
            user_ids,
            MAX_ORIGINAL_POSTS_PER_AUTHOR,
            exclude_tweet_ids,
            &HashSet::new(), 
            start_time,
            request_user_id,
        );

        let secondary_posts = self.get_posts_from_map(
            &self.secondary_posts_by_user,
            user_ids,
            MAX_REPLY_POSTS_PER_AUTHOR,
            exclude_tweet_ids,
            &following_users_set,
            start_time,
            request_user_id,
        );

        all_posts.extend(secondary_posts);
        POST_STORE_POSTS_RETURNED.observe(all_posts.len() as f64);
        all_posts
    }

    #[allow(clippy::too_many_arguments)]
    pub fn get_posts_from_map(
        &self,
        posts_map: &Arc<DashMap<i64, VecDeque<TinyPost>>>,
        user_ids: &[i64],
        max_per_user: usize,
        exclude_tweet_ids: &HashSet<i64>,
        following_users: &HashSet<i64>,
        start_time: Instant,
        request_user_id: i64,
    ) -> Vec<LightPost> {
        POST_STORE_REQUESTS.inc();
        let mut light_posts = Vec::new();

        let mut total_eligible: usize = 0;

        for (i, user_id) in user_ids.iter().enumerate() {
            if !self.request_timeout.is_zero() && start_time.elapsed() >= self.request_timeout {
                log::error!(
                    "Timed out fetching posts for user={}; Processed: {}/{}. Stage: {}",
                    request_user_id,
                    i,
                    user_ids.len(),
                    if following_users.is_empty() {
                        "original"
                    } else {
                        "secondary"
                    }
                );
                POST_STORE_REQUEST_TIMEOUTS.inc();
                break;
            }

            if let Some(user_posts_ref) = posts_map.get(user_id) {
                let user_posts = user_posts_ref.value();
                total_eligible += user_posts.len();

                let tiny_posts_iter = user_posts
                    .iter()
                    .rev()
                    .filter(|post| !exclude_tweet_ids.contains(&post.post_id))
                    .take(MAX_TINY_POSTS_PER_USER_SCAN);

                let light_post_iter_1 = tiny_posts_iter.filter_map(|tiny_post| {
                    self.posts.get(&tiny_post.post_id).map(|r| {
                        let post: &CompactPost = r.value();
                        LightPost::from(*post)
                    })
                });

                let light_post_iter = light_post_iter_1.filter(|post| {
                    !(post.is_retweet && post.source_user_id == Some(request_user_id))
                });

                let filtered_post_iter = light_post_iter.filter(|post| {
                    if following_users.is_empty() {
                        return true; 
                    }
                    post.in_reply_to_post_id.is_none_or(|reply_to_post_id| {
                        if let Some(replied_to_post) = self.posts.get(&reply_to_post_id) {
                            if !replied_to_post.is_retweet && !replied_to_post.is_reply {
                                return true;
                            }

                            return post.conversation_id.is_some_and(|convo_id| {
                                let reply_to_reply_to_original =
                                    replied_to_post.in_reply_to_post_id == convo_id;
                                let reply_to_followed_user = post
                                    .in_reply_to_user_id
                                    .map(|uid| following_users.contains(&uid))
                                    .unwrap_or(false);

                                reply_to_reply_to_original && reply_to_followed_user
                            });
                        }

                        false
                    })
                });

                light_posts.extend(filtered_post_iter.take(max_per_user));
            }
        }

        if total_eligible > 0 {
            let ratio = light_posts.len() as f64 / total_eligible as f64;
            POST_STORE_POSTS_RETURNED_RATIO.observe(ratio);
        }

        light_posts
    }

        pub fn start_stats_logger(self: Arc<Self>) {
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));

            loop {
                interval.tick().await;

                let user_count = self.original_posts_by_user.len();
                let total_posts = self.posts.len();
                let deleted_posts = self.deleted_posts.len();

                let original_posts_count: usize = self
                    .original_posts_by_user
                    .iter()
                    .map(|entry| entry.value().len())
                    .sum();
                let secondary_posts_count: usize = self
                    .secondary_posts_by_user
                    .iter()
                    .map(|entry| entry.value().len())
                    .sum();
                let video_posts_count: usize = self
                    .video_posts_by_user
                    .iter()
                    .map(|entry| entry.value().len())
                    .sum();

                POST_STORE_USER_COUNT.set(user_count as f64);
                POST_STORE_TOTAL_POSTS.set(total_posts as f64);
                POST_STORE_DELETED_POSTS.set(deleted_posts as f64);

                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["users"])
                    .set(user_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["posts"])
                    .set(total_posts as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["original_posts"])
                    .set(original_posts_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["secondary_posts"])
                    .set(secondary_posts_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["video_posts"])
                    .set(video_posts_count as f64);
                POST_STORE_ENTITY_COUNT
                    .with_label_values(&["deleted_posts"])
                    .set(deleted_posts as f64);

                let posts_capacity = self.posts.capacity();
                let original_capacity = self.original_posts_by_user.capacity();
                let secondary_capacity = self.secondary_posts_by_user.capacity();
                let video_capacity = self.video_posts_by_user.capacity();
                let deleted_capacity = self.deleted_posts.capacity();

                let original_vecdeque_capacity: usize = self
                    .original_posts_by_user
                    .iter()
                    .map(|entry| entry.value().capacity())
                    .sum();
                let secondary_vecdeque_capacity: usize = self
                    .secondary_posts_by_user
                    .iter()
                    .map(|entry| entry.value().capacity())
                    .sum();
                let video_vecdeque_capacity: usize = self
                    .video_posts_by_user
                    .iter()
                    .map(|entry| entry.value().capacity())
                    .sum();

                info!(
                    "PostStore Stats: \
                     posts(len={}, cap={}), \
                     original_posts_by_user(users={}, dashmap_cap={}, items={}, vecdeque_cap={}), \
                     secondary_posts_by_user(users={}, dashmap_cap={}, items={}, vecdeque_cap={}), \
                     video_posts_by_user(users={}, dashmap_cap={}, items={}, vecdeque_cap={}), \
                     deleted_posts(len={}, cap={})",
                    total_posts,
                    posts_capacity,
                    user_count,
                    original_capacity,
                    original_posts_count,
                    original_vecdeque_capacity,
                    self.secondary_posts_by_user.len(),
                    secondary_capacity,
                    secondary_posts_count,
                    secondary_vecdeque_capacity,
                    self.video_posts_by_user.len(),
                    video_capacity,
                    video_posts_count,
                    video_vecdeque_capacity,
                    deleted_posts,
                    deleted_capacity,
                );
            }
        });
    }

        pub fn start_auto_trim(self: Arc<Self>, interval_minutes: u64) {
        let interval_duration = Duration::from_secs(interval_minutes * 60);
        tokio::spawn(async move {
            let mut interval = tokio::time::interval_at(
                tokio::time::Instant::now() + interval_duration,
                interval_duration,
            );
            interval.set_missed_tick_behavior(MissedTickBehavior::Skip);

            let mut trim_iter: usize = 0;

            loop {
                trim_iter += 1;
                interval.tick().await;
                let trimmed = self.trim_old_posts(trim_iter).await;
                if trimmed > 0 {
                    info!("Auto-trim: removed {} old posts", trimmed);
                }
            }
        });
    }

            pub async fn trim_old_posts(&self, trim_iter: usize) -> usize {
        let posts_map = Arc::clone(&self.posts);
        let original_posts_by_user = Arc::clone(&self.original_posts_by_user);
        let secondary_posts_by_user = Arc::clone(&self.secondary_posts_by_user);
        let video_posts_by_user = Arc::clone(&self.video_posts_by_user);
        let deleted_posts = Arc::clone(&self.deleted_posts);
        let retention_seconds = self.retention_seconds;

        tokio::task::spawn_blocking(move || {
            let current_time = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs();

            let mut total_trimmed = 0;

            let trim_map = |posts_by_user: &DashMap<i64, VecDeque<TinyPost>>,
                            posts_map: &DashMap<i64, Arc<CompactPost>>,
                            deleted_posts: &DashMap<i64, bool>|
             -> usize {
                let mut trimmed = 0;
                let mut users_to_remove = Vec::new();
                let mut posts_to_remove = Vec::new();
                let mut deleted_posts_to_remove = Vec::new();

                if let Some(mut entry) = posts_by_user.get_mut(&DELETE_EVENT_KEY) {
                    let user_posts = entry.value_mut();
                    while let Some(oldest_post) = user_posts.front() {
                        if current_time - (oldest_post.created_at as u64) > retention_seconds {
                            let trimmed_post = user_posts.pop_front().unwrap();
                            posts_to_remove.push(trimmed_post.post_id);
                            deleted_posts_to_remove.push(trimmed_post.post_id);
                            trimmed += 1;
                        } else {
                            break; 
                        }
                    }
                }

                for mut entry in posts_by_user.iter_mut() {
                    let user_id = *entry.key();

                    if user_id == DELETE_EVENT_KEY {
                        continue;
                    }

                    if user_id as usize % TRIM_FRACTION != trim_iter % TRIM_FRACTION {
                        continue;
                    }

                    let user_posts = entry.value_mut();

                    while let Some(oldest_post) = user_posts.front() {
                        if user_posts.len() > MAX_POSTING_LIST_SIZE
                            || current_time - (oldest_post.created_at as u64) > retention_seconds
                        {
                            let trimmed_post = user_posts.pop_front().unwrap();
                            posts_to_remove.push(trimmed_post.post_id);
                            trimmed += 1;
                        } else {
                            break; 
                        }
                    }

                    if user_posts.capacity() > user_posts.len() * 2 {
                        let new_cap = user_posts.len() as f32 * 1.5_f32;
                        user_posts.shrink_to(new_cap as usize);
                    }

                    if user_posts.is_empty() {
                        users_to_remove.push(user_id);
                    }
                }

                for user_id in users_to_remove {
                    posts_by_user.remove_if(&user_id, |_, posts| posts.is_empty());
                }

                for post_id in posts_to_remove {
                    posts_map.remove(&post_id);
                }

                for post_id in deleted_posts_to_remove {
                    deleted_posts.remove(&post_id);
                }

                trimmed
            };

            let start = Instant::now();
            total_trimmed += trim_map(&original_posts_by_user, &posts_map, &deleted_posts);
            info!("Trimmed original posts in {:?}", start.elapsed());

            let start = Instant::now();
            total_trimmed += trim_map(&secondary_posts_by_user, &posts_map, &deleted_posts);
            info!("Trimmed secondary posts in {:?}", start.elapsed());

            let start = Instant::now();
            trim_map(&video_posts_by_user, &posts_map, &deleted_posts);
            info!("Trimmed video posts in {:?}", start.elapsed());

            total_trimmed
        })
        .await
        .expect("spawn_blocking failed")
    }

        pub async fn sort_all_user_posts(&self) {
        let original_posts_by_user = Arc::clone(&self.original_posts_by_user);
        let secondary_posts_by_user = Arc::clone(&self.secondary_posts_by_user);
        let video_posts_by_user = Arc::clone(&self.video_posts_by_user);

        tokio::task::spawn_blocking(move || {
            for mut entry in original_posts_by_user.iter_mut() {
                let user_posts = entry.value_mut();
                user_posts
                    .make_contiguous()
                    .sort_unstable_by_key(|a| a.created_at);
            }
            for mut entry in secondary_posts_by_user.iter_mut() {
                let user_posts = entry.value_mut();
                user_posts
                    .make_contiguous()
                    .sort_unstable_by_key(|a| a.created_at);
            }
            for mut entry in video_posts_by_user.iter_mut() {
                let user_posts = entry.value_mut();
                user_posts
                    .make_contiguous()
                    .sort_unstable_by_key(|a| a.created_at);
            }
        })
        .await
        .expect("spawn_blocking failed");
    }

        pub fn clear(&self) {
        self.posts.clear();
        self.original_posts_by_user.clear();
        self.secondary_posts_by_user.clear();
        self.video_posts_by_user.clear();
        info!("PostStore cleared");
    }
}

impl Default for PostStore {
    fn default() -> Self {
        Self::new(2 * 24 * 60 * 60, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_post_store_basic_operations() {
        let store = PostStore::new(2 * 24 * 60 * 60, 0); 

        assert_eq!(store.original_posts_by_user.len(), 0);
        assert_eq!(store.posts.len(), 0);

        assert_eq!(
            store.get_posts_from_map(
                &store.original_posts_by_user,
                &[123],
                100,
                &HashSet::new(),
                &HashSet::new(),
                Instant::now(),
                1, 
            ),
            Vec::new()
        );

        assert_eq!(
            store.get_posts_from_map(
                &store.original_posts_by_user,
                &[123, 456, 789],
                100,
                &HashSet::new(),
                &HashSet::new(),
                Instant::now(),
                1, 
            ),
            Vec::new()
        );
    }

    #[test]
    fn test_exclude_tweet_ids_filtering() {
        let store = PostStore::new(2 * 24 * 60 * 60, 0); 

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let mut posts = Vec::new();
        for i in 1..=10 {
            posts.push(LightPost {
                post_id: i,
                author_id: 100,
                created_at: current_time - 1,
                in_reply_to_post_id: None,
                in_reply_to_user_id: None,
                is_retweet: false,
                is_reply: false,
                source_post_id: None,
                source_user_id: None,
                has_video: false,
                conversation_id: None,
            });
        }

        store.insert_posts(posts);

        let result = store.get_posts_from_map(
            &store.original_posts_by_user,
            &[100],
            5,
            &HashSet::new(),
            &HashSet::new(),
            Instant::now(),
            1, 
        );
        assert_eq!(result.len(), 5);
        let post_ids: Vec<i64> = result.iter().map(|p| p.post_id).collect();
        assert!(post_ids.contains(&6));
        assert!(post_ids.contains(&7));
        assert!(post_ids.contains(&8));
        assert!(post_ids.contains(&9));
        assert!(post_ids.contains(&10));

        let exclude_tweet_ids: HashSet<i64> = vec![8, 9, 10].into_iter().collect();
        let result_with_exclude = store.get_posts_from_map(
            &store.original_posts_by_user,
            &[100],
            5,
            &exclude_tweet_ids,
            &HashSet::new(),
            Instant::now(),
            1, 
        );
        assert_eq!(result_with_exclude.len(), 5);
        let post_ids_excluded: Vec<i64> = result_with_exclude.iter().map(|p| p.post_id).collect();
        assert!(post_ids_excluded.contains(&3));
        assert!(post_ids_excluded.contains(&4));
        assert!(post_ids_excluded.contains(&5));
        assert!(post_ids_excluded.contains(&6));
        assert!(post_ids_excluded.contains(&7));
        assert!(!post_ids_excluded.contains(&8));
        assert!(!post_ids_excluded.contains(&9));
        assert!(!post_ids_excluded.contains(&10));
    }

    #[test]
    fn test_reply_and_retweet_filtering() {
        let store = PostStore::new(2 * 24 * 60 * 60, 0);

        let current_time = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;

        let posts = vec![
            LightPost {
                post_id: 1_000,
                author_id: 10,
                created_at: current_time - 6,
                in_reply_to_post_id: None,
                in_reply_to_user_id: None,
                is_retweet: false,
                is_reply: false,
                source_post_id: None,
                source_user_id: None,
                has_video: false,
                conversation_id: Some(1_000),
            },
            LightPost {
                post_id: 1_001,
                author_id: 2,
                created_at: current_time - 5,
                in_reply_to_post_id: Some(1_000),
                in_reply_to_user_id: Some(10),
                is_retweet: false,
                is_reply: true,
                source_post_id: None,
                source_user_id: None,
                has_video: false,
                conversation_id: Some(1_000),
            },
            LightPost {
                post_id: 1_002,
                author_id: 3,
                created_at: current_time - 4,
                in_reply_to_post_id: Some(1_001),
                in_reply_to_user_id: Some(2),
                is_retweet: false,
                is_reply: true,
                source_post_id: None,
                source_user_id: None,
                has_video: false,
                conversation_id: Some(1_000),
            },
            LightPost {
                post_id: 1_003,
                author_id: 4,
                created_at: current_time - 3,
                in_reply_to_post_id: Some(1_001),
                in_reply_to_user_id: Some(99),
                is_retweet: false,
                is_reply: true,
                source_post_id: None,
                source_user_id: None,
                has_video: false,
                conversation_id: Some(1_000),
            },
            LightPost {
                post_id: 1_004,
                author_id: 5,
                created_at: current_time - 2,
                in_reply_to_post_id: Some(1_000),
                in_reply_to_user_id: Some(10),
                is_retweet: false,
                is_reply: true,
                source_post_id: None,
                source_user_id: None,
                has_video: false,
                conversation_id: Some(1_000),
            },
            LightPost {
                post_id: 1_005,
                author_id: 2,
                created_at: current_time - 1,
                in_reply_to_post_id: None,
                in_reply_to_user_id: None,
                is_retweet: true,
                is_reply: false,
                source_post_id: Some(5000),
                source_user_id: Some(1), 
                has_video: false,
                conversation_id: Some(1_005),
            },
            LightPost {
                post_id: 1_006,
                author_id: 3,
                created_at: current_time - 2,
                in_reply_to_post_id: Some(1_002),
                in_reply_to_user_id: Some(2),
                is_retweet: false,
                is_reply: true,
                source_post_id: None,
                source_user_id: None,
                has_video: false,
                conversation_id: Some(1_000),
            },
        ];

        store.insert_posts(posts);

        let following_users: HashSet<i64> = vec![2, 3, 4, 5].into_iter().collect();
        let result = store.get_posts_from_map(
            &store.secondary_posts_by_user,
            &[2, 3, 4, 5],
            10,
            &HashSet::new(),
            &following_users,
            Instant::now(),
            1, 
        );

        let post_ids: HashSet<i64> = result.iter().map(|p| p.post_id).collect();
        assert_eq!(post_ids.len(), 3);
        assert!(post_ids.contains(&1_001)); 
        assert!(post_ids.contains(&1_002)); 
        assert!(post_ids.contains(&1_004)); 
        assert!(!post_ids.contains(&1_003)); 
        assert!(!post_ids.contains(&1_005)); 
        assert!(!post_ids.contains(&1_006)); 
    }
}
