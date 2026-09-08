use crate::models::candidate::PostCandidate;

/// PR-D lightweight serve-mix classification.
/// Full Studio / creator telemetry is product-side; this is a pure hook for
/// counters / tracing only (no analytics surface required in home-mixer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ServeCardType {
    Original,
    Rt,
    Quote,
}

/// Classify a serve card as original / rt / quote from candidate fields.
/// RT wins over quote if both were somehow set (RT is the outer card).
pub fn classify_serve_card(candidate: &PostCandidate) -> ServeCardType {
    if candidate.retweeted_tweet_id.is_some() {
        ServeCardType::Rt
    } else if candidate.quoted_tweet_id.is_some() {
        ServeCardType::Quote
    } else {
        ServeCardType::Original
    }
}

/// Side-effect stub: would increment serve-mix counters; logs at debug for now.
pub fn record_serve_card_type(candidate: &PostCandidate) -> ServeCardType {
    let card = classify_serve_card(candidate);
    tracing::debug!(
        tweet_id = candidate.tweet_id,
        card_type = ?card,
        "spacexai.serve_card_type"
    );
    card
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_original_rt_quote() {
        assert_eq!(
            classify_serve_card(&PostCandidate {
                tweet_id: 1,
                ..Default::default()
            }),
            ServeCardType::Original
        );
        assert_eq!(
            classify_serve_card(&PostCandidate {
                tweet_id: 2,
                retweeted_tweet_id: Some(100),
                ..Default::default()
            }),
            ServeCardType::Rt
        );
        assert_eq!(
            classify_serve_card(&PostCandidate {
                tweet_id: 3,
                quoted_tweet_id: Some(100),
                ..Default::default()
            }),
            ServeCardType::Quote
        );
    }

    #[test]
    fn rt_wins_over_quote_if_both_set() {
        assert_eq!(
            classify_serve_card(&PostCandidate {
                tweet_id: 4,
                retweeted_tweet_id: Some(100),
                quoted_tweet_id: Some(200),
                ..Default::default()
            }),
            ServeCardType::Rt
        );
    }
}
