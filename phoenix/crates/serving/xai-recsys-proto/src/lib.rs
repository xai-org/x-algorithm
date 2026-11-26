// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
mod pb {
    tonic::include_proto!("xai.recsys.v1");

    pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("recsys_descriptor");
}

pub use pb::*;

pub mod brand_engagement;
pub mod grok_topics;
pub mod installed_apps;
pub mod purchase_value;
pub mod starter_packs;

pub const SAFETY_BIT_AUTHOR_NSFW: u64 = 1 << 2;
pub const SAFETY_BIT_TWEET_NSFW: u64 = 1 << 3;
pub const SAFETY_BIT_TWEET_NSFW_HIGH_RECALL: u64 = 1 << 4;
pub const SAFETY_BIT_TWEET_SOFT_NSFW: u64 = 1 << 5;

pub const AD_TARGETING_MAX_INTEREST_CATEGORY_IDS: usize = 8;

pub fn timezone_string_to_enum(tz: &str) -> Timezone {
    match tz {
        "" | "unknown" => Timezone::Unknown,
        "America/New_York" | "US/Eastern" => Timezone::AmericaNewYork,
        "America/Chicago" | "US/Central" => Timezone::AmericaChicago,
        "America/Los_Angeles" | "US/Pacific" => Timezone::AmericaLosAngeles,
        "America/Denver" | "US/Mountain" => Timezone::AmericaDenver,
        "America/Sao_Paulo" | "Brazil/East" => Timezone::AmericaSaoPaulo,
        "Europe/London" | "GB" => Timezone::EuropeLondon,
        "Europe/Paris" => Timezone::EuropeParis,
        "Europe/Berlin" => Timezone::EuropeBerlin,
        "Europe/Madrid" => Timezone::EuropeMadrid,
        "Europe/Rome" => Timezone::EuropeRome,
        "Europe/Amsterdam" => Timezone::EuropeAmsterdam,
        "Europe/Istanbul" | "Turkey" => Timezone::EuropeIstanbul,
        "Europe/Moscow" => Timezone::EuropeMoscow,
        "Asia/Tokyo" | "Japan" => Timezone::AsiaTokyo,
        "Asia/Kolkata" | "Asia/Calcutta" => Timezone::AsiaKolkata,
        "Asia/Shanghai" | "PRC" => Timezone::AsiaShanghai,
        "Asia/Seoul" | "ROK" => Timezone::AsiaSeoul,
        "Asia/Jakarta" => Timezone::AsiaJakarta,
        "Asia/Bangkok" => Timezone::AsiaBangkok,
        "Asia/Dubai" => Timezone::AsiaDubai,
        "Asia/Riyadh" => Timezone::AsiaRiyadh,
        "Asia/Manila" => Timezone::AsiaManila,
        "Asia/Singapore" | "Singapore" => Timezone::AsiaSingapore,
        "Australia/Sydney" => Timezone::AustraliaSydney,
        "Pacific/Honolulu" | "US/Hawaii" => Timezone::PacificHonolulu,
        "America/Phoenix" => Timezone::AmericaPhoenix,
        "America/Toronto" | "Canada/Eastern" => Timezone::AmericaToronto,
        "America/Mexico_City" => Timezone::AmericaMexicoCity,
        "America/Argentina/Buenos_Aires" => Timezone::AmericaArgentinaBuenosAires,
        "Africa/Lagos" => Timezone::AfricaLagos,
        _ => Timezone::Other,
    }
}

pub fn language_code_string_to_enum(lang: &str) -> LanguageCode {
    match lang.to_ascii_uppercase().as_str() {
        "" => LanguageCode::LangUnknown,
        "IN" => LanguageCode::LangInOld,
        "IW" => LanguageCode::LangIwOld,
        code => {
            let enum_name = format!("LANG_{}", code);
            LanguageCode::from_str_name(&enum_name).unwrap_or(LanguageCode::LangUnknown)
        }
    }
}

pub fn country_code_string_to_enum(country: &str) -> CountryCode {
    match country.to_ascii_uppercase().as_str() {
        "" => CountryCode::CountryUnknown,
        code => {
            let enum_name = format!("COUNTRY_{}", code);
            CountryCode::from_str_name(&enum_name).unwrap_or(CountryCode::CountryUnknown)
        }
    }
}

pub fn is_video_card_type(card_type: i32) -> bool {
    match UnifiedCardType::try_from(card_type) {
        Ok(
            UnifiedCardType::VideoWebsite
            | UnifiedCardType::VideoApp
            | UnifiedCardType::VideoAndUrls
            | UnifiedCardType::VideoCarouselWebsite
            | UnifiedCardType::VideoCarouselApp
            | UnifiedCardType::VideoPlayableApp
            | UnifiedCardType::Video
            | UnifiedCardType::VideoAndButtonWebsite
            | UnifiedCardType::VideoMultiDestCarouselApp
            | UnifiedCardType::VideoMultiDestCarouselWebsite
            | UnifiedCardType::MixedMediaSingleDestCarouselApp
            | UnifiedCardType::MixedMediaSingleDestCarouselWebsite
            | UnifiedCardType::MixedMediaMultiDestCarouselApp
            | UnifiedCardType::MixedMediaMultiDestCarouselWebsite
            | UnifiedCardType::VideoPlayableWebsite,
        ) => true,
        Ok(_) => false,
        Err(_) => true,
    }
}

impl HistorySourceInfo {
    pub fn of(event_times_ms: impl IntoIterator<Item = i64>) -> Self {
        let mut length = 0u32;
        let mut span: Option<(u64, u64)> = None;
        for ms in event_times_ms {
            length += 1;
            if ms > 0 {
                let ms = ms as u64;
                span = Some(span.map_or((ms, ms), |(lo, hi)| (lo.min(ms), hi.max(ms))));
            }
        }
        Self {
            length,
            first_event_ms: span.map(|(lo, _)| lo),
            last_event_ms: span.map(|(_, hi)| hi),
            first_kafka_timestamp_ms: None,
            last_kafka_timestamp_ms: None,
        }
    }

    pub fn with_kafka_timestamps(
        self,
        kafka_timestamps_ms: impl IntoIterator<Item = Option<i64>>,
    ) -> Self {
        let mut span: Option<(u64, u64)> = None;
        let mut missing = false;
        for ms in kafka_timestamps_ms {
            match ms {
                Some(ms) if ms > 0 => {
                    let ms = ms as u64;
                    span = Some(span.map_or((ms, ms), |(lo, hi)| (lo.min(ms), hi.max(ms))));
                }
                _ => missing = true,
            }
        }
        Self {
            first_kafka_timestamp_ms: span.map(|(lo, _)| if missing { 0 } else { lo }),
            last_kafka_timestamp_ms: span.map(|(_, hi)| hi),
            ..self
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_source_counts_every_entry_and_spans_the_positive_times() {
        assert_eq!(
            HistorySourceInfo::of([5_000, 0, 2_000, -1, 9_000]),
            HistorySourceInfo {
                length: 5,
                first_event_ms: Some(2_000),
                last_event_ms: Some(9_000),
                ..Default::default()
            }
        );
        assert_eq!(
            HistorySourceInfo::of([0, 0]),
            HistorySourceInfo {
                length: 2,
                first_event_ms: None,
                last_event_ms: None,
                ..Default::default()
            }
        );
        assert_eq!(HistorySourceInfo::of([]), HistorySourceInfo::default());
    }

    #[test]
    fn the_kafka_range_spans_the_positive_timestamps() {
        let kafka = |timestamps: &[Option<i64>]| {
            let s = HistorySourceInfo::of([1_000]).with_kafka_timestamps(timestamps.to_vec());
            (s.first_kafka_timestamp_ms, s.last_kafka_timestamp_ms)
        };
        assert_eq!(
            kafka(&[Some(7_000), Some(3_000)]),
            (Some(3_000), Some(7_000))
        );
        assert_eq!(
            kafka(&[Some(7_000), None, Some(0), Some(3_000), Some(-1)]),
            (Some(0), Some(7_000))
        );
        assert_eq!(kafka(&[None, Some(0)]), (None, None));

        let source =
            HistorySourceInfo::of([1_000, 2_000]).with_kafka_timestamps([Some(1_500), None]);
        assert_eq!(source.length, 2);
        assert_eq!(source.first_event_ms, Some(1_000));
    }

    #[test]
    fn test_language_code_string_to_enum() {
        assert_eq!(language_code_string_to_enum("en"), LanguageCode::LangEn);
        assert_eq!(language_code_string_to_enum("fr"), LanguageCode::LangFr);
        assert_eq!(language_code_string_to_enum("ja"), LanguageCode::LangJa);
        assert_eq!(language_code_string_to_enum("zh"), LanguageCode::LangZh);
        assert_eq!(language_code_string_to_enum("ar"), LanguageCode::LangAr);
        assert_eq!(language_code_string_to_enum("ko"), LanguageCode::LangKo);
        assert_eq!(language_code_string_to_enum("de"), LanguageCode::LangDe);
        assert_eq!(language_code_string_to_enum("es"), LanguageCode::LangEs);
        assert_eq!(language_code_string_to_enum("pt"), LanguageCode::LangPt);
        assert_eq!(language_code_string_to_enum("hi"), LanguageCode::LangHi);
        assert_eq!(language_code_string_to_enum("ru"), LanguageCode::LangRu);

        assert_eq!(language_code_string_to_enum("EN"), LanguageCode::LangEn);
        assert_eq!(language_code_string_to_enum("Fr"), LanguageCode::LangFr);

        assert_eq!(language_code_string_to_enum("in"), LanguageCode::LangInOld);
        assert_eq!(language_code_string_to_enum("iw"), LanguageCode::LangIwOld);

        assert_eq!(language_code_string_to_enum("art"), LanguageCode::LangArt);
        assert_eq!(language_code_string_to_enum("und"), LanguageCode::LangUnd);
        assert_eq!(language_code_string_to_enum("zxx"), LanguageCode::LangZxx);
        assert_eq!(language_code_string_to_enum("ckb"), LanguageCode::LangCkb);
        assert_eq!(language_code_string_to_enum("qme"), LanguageCode::LangQme);
        assert_eq!(language_code_string_to_enum("qst"), LanguageCode::LangQst);
        assert_eq!(language_code_string_to_enum("qht"), LanguageCode::LangQht);
        assert_eq!(language_code_string_to_enum("qct"), LanguageCode::LangQct);
        assert_eq!(language_code_string_to_enum("qam"), LanguageCode::LangQam);

        assert_eq!(language_code_string_to_enum(""), LanguageCode::LangUnknown);
        assert_eq!(
            language_code_string_to_enum("xyz"),
            LanguageCode::LangUnknown
        );
        assert_eq!(
            language_code_string_to_enum("invalid"),
            LanguageCode::LangUnknown
        );
    }

    #[test]
    fn test_country_code_string_to_enum() {
        assert_eq!(country_code_string_to_enum("US"), CountryCode::CountryUs);
        assert_eq!(country_code_string_to_enum("GB"), CountryCode::CountryGb);
        assert_eq!(country_code_string_to_enum("JP"), CountryCode::CountryJp);
        assert_eq!(country_code_string_to_enum("DE"), CountryCode::CountryDe);
        assert_eq!(country_code_string_to_enum("FR"), CountryCode::CountryFr);
        assert_eq!(country_code_string_to_enum("BR"), CountryCode::CountryBr);
        assert_eq!(country_code_string_to_enum("IN"), CountryCode::CountryIn);
        assert_eq!(country_code_string_to_enum("CN"), CountryCode::CountryCn);
        assert_eq!(country_code_string_to_enum("AU"), CountryCode::CountryAu);
        assert_eq!(country_code_string_to_enum("CA"), CountryCode::CountryCa);
        assert_eq!(country_code_string_to_enum("ZW"), CountryCode::CountryZw);

        assert_eq!(country_code_string_to_enum("us"), CountryCode::CountryUs);
        assert_eq!(country_code_string_to_enum("gb"), CountryCode::CountryGb);
        assert_eq!(country_code_string_to_enum("Jp"), CountryCode::CountryJp);

        assert_eq!(country_code_string_to_enum(""), CountryCode::CountryUnknown);
        assert_eq!(
            country_code_string_to_enum("XX"),
            CountryCode::CountryUnknown
        );
        assert_eq!(
            country_code_string_to_enum("invalid"),
            CountryCode::CountryUnknown
        );
    }

    #[test]
    fn retrieval_dataset_type_ids() {
        assert_eq!(RetrievalDatasetType::Home as i32, 1);
        assert_eq!(RetrievalDatasetType::HomeCold as i32, 13);
        assert_eq!(RetrievalDatasetType::HomeHot as i32, 14);
        assert_eq!(RetrievalDatasetType::Trending as i32, 12);
    }

    #[test]
    fn ranking_score_weights_completion_field_is_wire_compatible() {
        use prost::Message;

        #[derive(Clone, PartialEq, prost::Message)]
        struct RankingScoreWeightsBeforeCompletion {
            #[prost(float, tag = "1")]
            favorite_weight: f32,
            #[prost(float, tag = "15")]
            share_via_copy_link_weight: f32,
        }

        let new = RankingScoreWeights {
            favorite_weight: 0.5,
            share_via_copy_link_weight: 2.0,
            completion_weight: 3.0,
            ..Default::default()
        };
        let old =
            RankingScoreWeightsBeforeCompletion::decode(new.encode_to_vec().as_slice()).unwrap();
        assert_eq!(old.favorite_weight, 0.5);
        assert_eq!(old.share_via_copy_link_weight, 2.0);

        let from_old = RankingScoreWeights::decode(old.encode_to_vec().as_slice()).unwrap();
        assert_eq!(from_old.favorite_weight, 0.5);
        assert_eq!(from_old.completion_weight, 0.0);
    }

    #[test]
    fn test_is_video_card_type() {
        assert!(is_video_card_type(UnifiedCardType::VideoWebsite as i32));
        assert!(is_video_card_type(UnifiedCardType::VideoApp as i32));
        assert!(is_video_card_type(
            UnifiedCardType::VideoCarouselWebsite as i32
        ));
        assert!(is_video_card_type(
            UnifiedCardType::VideoPlayableWebsite as i32
        ));
        assert!(is_video_card_type(
            UnifiedCardType::VideoMultiDestCarouselWebsite as i32
        ));
        assert!(is_video_card_type(
            UnifiedCardType::MixedMediaSingleDestCarouselApp as i32
        ));
        assert!(is_video_card_type(
            UnifiedCardType::MixedMediaMultiDestCarouselWebsite as i32
        ));
        assert!(!is_video_card_type(UnifiedCardType::Unspecified as i32));
        assert!(!is_video_card_type(UnifiedCardType::ImageWebsite as i32));
        assert!(!is_video_card_type(UnifiedCardType::ImageApp as i32));
        assert!(!is_video_card_type(
            UnifiedCardType::ImageCarouselWebsite as i32
        ));
        assert!(!is_video_card_type(
            UnifiedCardType::ImageMultiDestCarouselWebsite as i32
        ));
        assert!(!is_video_card_type(
            UnifiedCardType::ImageCollectionWebsite as i32
        ));
        assert!(is_video_card_type(9999));
    }
}
