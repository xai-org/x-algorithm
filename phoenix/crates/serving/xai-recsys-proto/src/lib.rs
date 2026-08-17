// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
mod pb {
    tonic::include_proto!("xai.recsys.v1");

    pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("recsys_descriptor");
}

pub use pb::*;

pub mod grok_topics;
pub mod installed_apps;
pub mod starter_packs;

pub const SAFETY_BIT_AUTHOR_NSFW: u64 = 1 << 2;

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

#[cfg(test)]
mod tests {
    use super::*;

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
}
