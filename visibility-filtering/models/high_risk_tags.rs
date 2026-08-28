use std::collections::HashSet;
use std::sync::LazyLock;

/// High-risk / “killer-zone” tags.
/// Maintain this list from enforcement metrics (tags with sustained high
/// child-safety / severe-policy ban volume). Keep normalized (lowercase, no #).
pub static HIGH_RISK_TAGS: LazyLock<HashSet<&'static str>> = LazyLock::new(|| {
    HashSet::from([
        "teenageer",          // example from the issue – replace / extend with real high-volume tags
        // add more normalized tags here as metrics dictate
    ])
});

/// Returns true if the post text contains any high-risk tag.
/// Simple whitespace / punctuation split; can be replaced by the real
/// tokenizer later if available.
pub fn contains_high_risk_tag(text: &str) -> bool {
    if text.is_empty() {
        return false;
    }
    let lower = text.to_lowercase();
    // crude but effective tokenization for hashtags
    for token in lower
        .split(|c: char| !c.is_alphanumeric() && c != '_')
        .filter(|t| !t.is_empty())
    {
        if HIGH_RISK_TAGS.contains(token) {
            return true;
        }
    }
    false
}
