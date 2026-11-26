const REGIONS: &[(&str, &[&str])] = &[
    (
        "AFR",
        &[
            "AO", "BF", "BI", "BJ", "BW", "CD", "CF", "CG", "CI", "CM", "CV", "DJ", "DZ", "EG",
            "EH", "ER", "ET", "GA", "GH", "GM", "GN", "GQ", "GW", "IO", "KE", "KM", "LR", "LS",
            "LY", "MA", "MG", "ML", "MR", "MU", "MW", "MZ", "NA", "NE", "NG", "RE", "RW", "SC",
            "SD", "SH", "SL", "SN", "SO", "SS", "ST", "SZ", "TD", "TG", "TN", "TZ", "UG", "YT",
            "ZA", "ZM", "ZW",
        ],
    ),
    (
        "AUS",
        &[
            "AQ", "AS", "AU", "BV", "CC", "CK", "CX", "FJ", "FM", "GS", "GU", "HM", "KI", "MH",
            "MP", "NC", "NF", "NR", "NU", "NZ", "PF", "PG", "PN", "PW", "SB", "TF", "TK", "TO",
            "TV", "UM", "VU", "WF", "WS",
        ],
    ),
    ("CAS", &["AF", "KG", "KZ", "TJ", "TM", "UZ"]),
    ("EAS", &["CN", "HK", "JP", "KP", "KR", "MN", "MO", "TW"]),
    (
        "EUR",
        &[
            "AD", "AL", "AT", "AX", "BA", "BE", "BG", "BY", "CH", "CZ", "DE", "DK", "EE", "ES",
            "FI", "FO", "FR", "GB", "GG", "GI", "GL", "GR", "HR", "HU", "IE", "IM", "IS", "IT",
            "JE", "KV", "LI", "LT", "LU", "LV", "MC", "MD", "ME", "MK", "MT", "NL", "NO", "PL",
            "PT", "RO", "RS", "RU", "SE", "SI", "SJ", "SK", "SM", "UA", "VA", "XK",
        ],
    ),
    (
        "NAM",
        &[
            "AG", "AI", "AW", "BB", "BL", "BM", "BQ", "BS", "BZ", "CA", "CR", "CU", "CW", "DM",
            "DO", "GD", "GP", "GT", "HN", "HT", "JM", "KN", "KY", "LC", "MF", "MQ", "MS", "MX",
            "NI", "PA", "PM", "PR", "SV", "SX", "TC", "TT", "US", "VC", "VG", "VI",
        ],
    ),
    (
        "SAM",
        &[
            "AR", "BO", "BR", "CL", "CO", "EC", "FK", "GF", "GY", "PE", "PY", "SR", "UY", "VE",
        ],
    ),
    ("SAS", &["BD", "BT", "IN", "LK", "MV", "NP", "PK"]),
    (
        "SEA",
        &[
            "BN", "ID", "KH", "LA", "MM", "MY", "PH", "SG", "TH", "TL", "VN",
        ],
    ),
    (
        "WES",
        &[
            "AE", "AM", "AZ", "BH", "CY", "GE", "IL", "IQ", "IR", "JO", "KW", "LB", "OM", "PS",
            "QA", "SA", "SY", "TR", "YE",
        ],
    ),
];

pub(crate) fn allows_country(allowed_codes: &[String], country: &str) -> bool {
    allowed_codes.iter().any(|code| {
        match REGIONS
            .iter()
            .find(|(region, _)| region.eq_ignore_ascii_case(code))
        {
            Some((_, countries)) => countries.iter().any(|c| c.eq_ignore_ascii_case(country)),
            None => code.eq_ignore_ascii_case(country),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allowed_codes_expand_regions_and_ignore_case() {
        let codes = |codes: &[&str]| codes.iter().map(|c| (*c).to_string()).collect::<Vec<_>>();
        for (allowed, country, expected) in [
            (codes(&["NAM"]), "US", true),
            (codes(&["nam"]), "us", true),
            (codes(&["EUR"]), "US", false),
            (codes(&["AFR"]), "CI", true),
            (codes(&["EUR", "SAM"]), "BR", true),
            (codes(&["BR"]), "br", true),
            (codes(&["BR"]), "US", false),
            (codes(&["US"]), "NAM", false),
            (codes(&[]), "US", false),
        ] {
            assert_eq!(
                allows_country(&allowed, country),
                expected,
                "{allowed:?} {country}"
            );
        }
    }
}
