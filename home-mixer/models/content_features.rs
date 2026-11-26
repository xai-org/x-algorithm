use crate::models::candidate::PostCandidate;
use xai_recsys_proto::ContentFeatures;

const URL_WEIGHTED_LEN: u32 = 23;
const SINGLE_WEIGHT_CODEPOINT_RANGES: [(u32, u32); 4] =
    [(0, 4351), (8192, 8205), (8208, 8223), (8242, 8247)];

pub fn build(candidate: &PostCandidate) -> ContentFeatures {
    let media_attachment_urls = u32::from(candidate.has_media.unwrap_or(false));
    let text = TextMetrics::of(&candidate.tweet_text, media_attachment_urls);
    ContentFeatures {
        has_video: candidate.has_video.unwrap_or(false),
        max_video_duration_ms: non_negative(candidate.max_video_duration_ms),
        has_photo: candidate.has_photo.unwrap_or(false),
        media_count: non_negative(candidate.media_count),
        weighted_text_len: text.weighted_len,
        newline_count: candidate.tweet_text.matches('\n').count() as u32,
        has_url: text.external_urls > 0,
    }
}

pub fn build_quoted(candidate: &PostCandidate) -> Option<ContentFeatures> {
    candidate.quoted_tweet_id?;
    let quoted_text = candidate.quoted_tweet_text.as_deref().unwrap_or("");
    let media_attachment_urls = u32::from(candidate.quoted_has_media.unwrap_or(false));
    let text = TextMetrics::of(quoted_text, media_attachment_urls);
    Some(ContentFeatures {
        has_video: candidate.quoted_has_video.unwrap_or(false),
        max_video_duration_ms: non_negative(candidate.quoted_max_video_duration_ms),
        has_photo: candidate.quoted_has_photo.unwrap_or(false),
        media_count: non_negative(candidate.quoted_media_count),
        weighted_text_len: text.weighted_len,
        newline_count: quoted_text.matches('\n').count() as u32,
        has_url: text.external_urls > 0,
    })
}

#[derive(Debug, Default, PartialEq)]
pub struct TextMetrics {
    pub weighted_len: u32,
    pub external_urls: u32,
}

impl TextMetrics {
    pub fn of(text: &str, attachment_urls: u32) -> Self {
        let bytes = text.as_bytes();
        let mut weighted_len = 0u32;
        let mut urls = 0u32;
        let mut in_token = false;
        let mut in_url = false;
        let mut i = 0;
        while i < bytes.len() {
            let (weight, is_whitespace, len) = if bytes[i].is_ascii() {
                (1, matches!(bytes[i], b'\t'..=b'\r' | b' '), 1)
            } else {
                let ch = text[i..].chars().next().unwrap_or_default();
                (char_weight(ch), ch.is_whitespace(), ch.len_utf8())
            };
            if is_whitespace {
                weighted_len += weight;
                in_token = false;
            } else {
                if !in_token {
                    in_token = true;
                    in_url = is_url(&text[i..]);
                    urls += u32::from(in_url);
                }
                if !in_url {
                    weighted_len += weight;
                }
            }
            i += len;
        }
        let external_urls = urls.saturating_sub(attachment_urls);
        Self {
            weighted_len: weighted_len + URL_WEIGHTED_LEN * external_urls,
            external_urls,
        }
    }
}

fn is_url(token: &str) -> bool {
    token.starts_with("https://") || token.starts_with("http://")
}

fn char_weight(ch: char) -> u32 {
    let cp = ch as u32;
    if cp <= SINGLE_WEIGHT_CODEPOINT_RANGES[0].1 {
        return 1;
    }
    if matches!(cp, 0xFE0E | 0xFE0F | 0x1F3FB..=0x1F3FF | 0xE0020..=0xE007F) {
        return 0;
    }
    if SINGLE_WEIGHT_CODEPOINT_RANGES
        .iter()
        .any(|&(lo, hi)| (lo..=hi).contains(&cp))
    {
        1
    } else {
        2
    }
}

fn non_negative(value: Option<i32>) -> u32 {
    value.unwrap_or(0).max(0) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weighted(text: &str) -> u32 {
        TextMetrics::of(text, 0).weighted_len
    }

    fn reference(text: &str, attachment_urls: u32) -> TextMetrics {
        let whitespace: u32 = text
            .chars()
            .filter(|c| c.is_whitespace())
            .map(char_weight)
            .sum();
        let mut urls = 0u32;
        let mut words = 0u32;
        for token in text.split_whitespace() {
            if is_url(token) {
                urls += 1;
            } else {
                words += token.chars().map(char_weight).sum::<u32>();
            }
        }
        let external_urls = urls.saturating_sub(attachment_urls);
        TextMetrics {
            weighted_len: whitespace + words + URL_WEIGHTED_LEN * external_urls,
            external_urls,
        }
    }

    #[test]
    fn matches_the_reference_implementation() {
        let samples = [
            "",
            "   ",
            "hello world",
            "  leading and trailing  ",
            "see https://t.co/abc123 now",
            "https://t.co/start then http://x.y/z and end https://t.co/end",
            "not a url: http:// alone, https:/broken, xhttps://inside, HTTPS://upper",
            "line one\nline two\thttps://t.co/x\r\n",
            "vertical\u{b}tab\u{c}feed\u{85}next-line",
            "日本\u{3000}語 and ｆｕｌｌｗｉｄｔｈ",
            "ok 👍🏽 ❤️ 🏳️‍🌈 \u{e0067}tag",
            "Ça va très bien, ünïcödé ÀÉÎÕÜ",
            "\u{2003}em\u{2009}thin\u{a0}nbsp\u{2028}sep",
            "\u{2010}\u{2015}\u{201c}quotes\u{201d}\u{2032}\u{2037}",
            "\u{10ff}\u{1100}\u{1101} boundary",
        ];
        for text in samples {
            for attachment_urls in 0..3 {
                assert_eq!(
                    TextMetrics::of(text, attachment_urls),
                    reference(text, attachment_urls),
                    "{text:?} with {attachment_urls} attachment URLs"
                );
            }
        }
    }

    #[test]
    fn weighted_text_len_follows_twitter_text_rules() {
        assert_eq!(weighted(""), 0);
        assert_eq!(weighted("hello world"), 11);
        assert_eq!(weighted("日本語"), 6);
        assert_eq!(weighted("see https://t.co/abc123 now"), 3 + 1 + 23 + 1 + 3);
        assert_eq!(weighted("ok 👍🏽"), 2 + 1 + 2);
        assert_eq!(weighted("日本\u{3000}語"), 4 + 2 + 2);
    }

    #[test]
    fn media_attachment_url_is_not_an_external_link() {
        let media_post = TextMetrics::of("So nice ❤️ https://t.co/kKPUyA0ztX", 1);
        assert_eq!(media_post.external_urls, 0);
        assert_eq!(media_post.weighted_len, 2 + 1 + 4 + 1 + 2 + 1);

        let media_post_with_link = TextMetrics::of("read https://t.co/link https://t.co/media", 1);
        assert_eq!(media_post_with_link.external_urls, 1);
        assert_eq!(media_post_with_link.weighted_len, 4 + 2 + 23);

        let plain_link = TextMetrics::of("read https://t.co/link", 0);
        assert_eq!(plain_link.external_urls, 1);
    }

    #[test]
    fn build_reads_retained_media_and_text_fields() {
        let candidate = PostCandidate {
            tweet_text: "line one\nline two https://t.co/x".to_string(),
            has_media: Some(true),
            has_photo: Some(false),
            has_video: Some(true),
            media_count: Some(1),
            max_video_duration_ms: Some(42_000),
            ..Default::default()
        };
        let features = build(&candidate);
        assert!(features.has_video);
        assert!(!features.has_photo);
        assert_eq!(features.media_count, 1);
        assert_eq!(features.max_video_duration_ms, 42_000);
        assert_eq!(features.newline_count, 1);
        assert!(!features.has_url);
        assert_eq!(features.weighted_text_len, 8 + 1 + 8 + 1);
        assert!(build_quoted(&candidate).is_none());

        let with_link = build(&PostCandidate {
            tweet_text: "see https://t.co/article".to_string(),
            ..Default::default()
        });
        assert!(with_link.has_url);
        assert_eq!(with_link.weighted_text_len, 3 + 1 + 23);
    }

    #[test]
    fn build_quoted_reads_quoted_post_fields() {
        let candidate = PostCandidate {
            tweet_text: "agree".to_string(),
            quoted_tweet_id: Some(7),
            quoted_tweet_text: Some("original take https://t.co/pic".to_string()),
            quoted_has_media: Some(true),
            quoted_has_photo: Some(true),
            quoted_has_video: Some(false),
            quoted_media_count: Some(2),
            quoted_max_video_duration_ms: None,
            ..Default::default()
        };
        let quoted = build_quoted(&candidate).expect("quoted features");
        assert!(quoted.has_photo);
        assert!(!quoted.has_video);
        assert_eq!(quoted.media_count, 2);
        assert_eq!(quoted.max_video_duration_ms, 0);
        assert!(!quoted.has_url);
        assert_eq!(quoted.weighted_text_len, 8 + 1 + 4 + 1);

        let unhydrated_quote = build_quoted(&PostCandidate {
            quoted_tweet_id: Some(7),
            ..Default::default()
        });
        assert_eq!(unhydrated_quote, Some(ContentFeatures::default()));
    }

    #[test]
    fn build_defaults_when_nothing_is_hydrated() {
        let features = build(&PostCandidate::default());
        assert_eq!(features, ContentFeatures::default());
    }
}
