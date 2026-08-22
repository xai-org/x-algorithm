//! MAGA: Make Argentina Great Again.
//!
//! Detects posts of Argentine origin and hands `RankingScorer` a scalar
//! multiplier for them.

use crate::models::candidate::PostCandidate;
use crate::models::query::ScoredPostsQuery;
use crate::params::*;

/// Regional flag emoji for Argentina (U+1F1E6 U+1F1F7).
const AR_FLAG: &str = "\u{1F1E6}\u{1F1F7}";

/// Markers of Argentine origin, matched on whole words after normalization.
/// Entries may span several words. Kept deliberately small: these are the
/// signals that carry the most weight per token, not an exhaustive lexicon.
///
/// Several entries ("mate", "messi", "argentina") are ordinary words in other
/// languages, so the list is only consulted for posts tagged as Spanish.
const MAGA_MARKERS: &[&str] = &[
    "che",
    "boludo",
    "mate",
    "asado",
    "milei",
    "messi",
    "maradona",
    "quilombo",
    "argentina",
    "el mejor pais del mundo",
];

/// Lowercases, strips Spanish diacritics and collapses everything that is not
/// alphanumeric into single spaces, so that markers can be matched on word
/// boundaries regardless of accents or punctuation.
fn normalize(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 2);
    out.push(' ');
    let mut pending_space = false;

    for c in text.to_lowercase().chars() {
        let folded = match c {
            'á' => 'a',
            'é' => 'e',
            'í' => 'i',
            'ó' => 'o',
            'ú' | 'ü' => 'u',
            'ñ' => 'n',
            other => other,
        };

        if folded.is_alphanumeric() {
            if pending_space {
                out.push(' ');
                pending_space = false;
            }
            out.push(folded);
        } else {
            pending_space = true;
        }
    }

    out.push(' ');
    out
}

/// True when `marker` occurs in `normalized` on word boundaries. `normalized`
/// is space-padded and single-spaced, so it is enough to check that the
/// characters surrounding a match are spaces.
fn contains_word(normalized: &str, marker: &str) -> bool {
    normalized.match_indices(marker).any(|(start, _)| {
        let before = normalized.as_bytes().get(start.wrapping_sub(1));
        let after = normalized.as_bytes().get(start + marker.len());
        start > 0 && before == Some(&b' ') && after == Some(&b' ')
    })
}

/// True when the post carries at least one signal of Argentine origin.
///
/// The flag emoji is accepted in any language; the lexical markers require the
/// post to be tagged as Spanish.
pub(crate) fn is_argentine(text: &str, language_code: Option<&str>) -> bool {
    if text.contains(AR_FLAG) {
        return true;
    }

    if !language_code.is_some_and(|code| code.eq_ignore_ascii_case("es")) {
        return false;
    }

    let normalized = normalize(text);
    MAGA_MARKERS
        .iter()
        .any(|marker| contains_word(&normalized, marker))
}

/// Scalar multiplier applied to a candidate's score, alongside author
/// diversity and the out-of-network factor.
///
/// Returns 1.0 (a no-op) unless the boost is enabled and the post looks
/// Argentine.
pub(crate) fn maga_multiplier(query: &ScoredPostsQuery, candidate: &PostCandidate) -> f64 {
    if !query.params.get(EnableMagaBoost) {
        return 1.0;
    }

    if is_argentine(&candidate.tweet_text, candidate.language_code.as_deref()) {
        query.params.get(MagaBoostFactor)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_word_marker_is_argentine() {
        assert!(is_argentine("che qué quilombo se armó", Some("es")));
        assert!(is_argentine("mañana hay asado", Some("es")));
    }

    #[test]
    fn multi_word_marker_is_argentine() {
        assert!(is_argentine(
            "argenmemes: el mejor país del mundo, sin dudas",
            Some("es")
        ));
    }

    #[test]
    fn markers_match_regardless_of_accents_and_punctuation() {
        assert!(is_argentine("¡MILEI!", Some("es")));
        assert!(is_argentine("el mejor pais del mundo", Some("es")));
    }

    #[test]
    fn flag_emoji_is_argentine_in_any_language() {
        assert!(is_argentine("world champions \u{1F1E6}\u{1F1F7}", Some("en")));
        assert!(is_argentine("\u{1F1E6}\u{1F1F7}", None));
    }

    #[test]
    fn markers_must_be_whole_words() {
        // "che" inside "coche", "noche", "leche" must not match.
        assert!(!is_argentine("esta noche compro leche", Some("es")));
        // "mate" inside "matemática" must not match.
        assert!(!is_argentine("tarea de matemática", Some("es")));
    }

    #[test]
    fn ambiguous_markers_require_spanish() {
        // "mate" and "messi" read as ordinary words elsewhere.
        assert!(!is_argentine("thanks mate", Some("en")));
        assert!(!is_argentine("Messi ist der Beste", Some("de")));
    }

    #[test]
    fn unmarked_spanish_post_is_not_argentine() {
        assert!(!is_argentine("después de todo, ya estás aquí", Some("es")));
    }

    #[test]
    fn untagged_language_falls_back_to_flag_only() {
        assert!(!is_argentine("che boludo", None));
    }

    #[test]
    fn empty_text_is_not_argentine() {
        assert!(!is_argentine("", Some("es")));
    }
}
