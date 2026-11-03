// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn find_feature_config() -> PathBuf {
    if let Ok(p) = env::var("FEATURE_CONFIG_PY") {
        return PathBuf::from(p);
    }
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
    Path::new(&manifest)
        .join("../../../xrex/data/recsys/feature_config.py")
        .canonicalize()
        .expect("cannot find feature_config.py relative to crate root")
}

fn to_upper_snake(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 8);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_uppercase());
    }
    out
}

fn to_snake_module(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 8);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

struct EnumMember {
    py_name: String,
    value: usize,
}

struct EnumClass {
    py_class_name: String,
    members: Vec<EnumMember>,
}

struct ModuleConst {
    py_name: String,
    value: i64,
}

fn is_upper_snake(name: &str) -> bool {
    !name.is_empty()
        && name.chars().next().unwrap().is_ascii_uppercase()
        && name
            .chars()
            .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
}

fn parse_feature_config(src: &str) -> (Vec<EnumClass>, Vec<ModuleConst>) {
    let mut classes = Vec::new();
    let mut consts = Vec::new();
    let mut current: Option<EnumClass> = None;

    for line in src.lines() {
        let trimmed = line.trim();

        if !line.starts_with(' ')
            && !line.starts_with('\t')
            && let Some((name, val_str)) = trimmed.split_once('=')
            && is_upper_snake(name.trim())
            && let Ok(value) = val_str.trim().replace('_', "").parse::<i64>()
        {
            consts.push(ModuleConst {
                py_name: name.trim().to_string(),
                value,
            });
        }

        if trimmed.starts_with("class ")
            && trimmed.contains("(enum.IntEnum)")
            && trimmed.ends_with(':')
        {
            if let Some(c) = current.take() {
                classes.push(c);
            }
            let class_name = trimmed
                .strip_prefix("class ")
                .unwrap()
                .split('(')
                .next()
                .unwrap()
                .to_string();
            current = Some(EnumClass {
                py_class_name: class_name,
                members: Vec::new(),
            });
            continue;
        }

        if let Some(ref mut cls) = current {
            if line.starts_with("    ") || line.starts_with('\t') {
                if trimmed.starts_with('#') || trimmed == "pass" || trimmed.is_empty() {
                    continue;
                }
                if let Some((name, val_str)) = trimmed.split_once('=') {
                    let name = name.trim();
                    let val_str = val_str.trim();
                    if let Ok(value) = val_str.parse::<usize>() {
                        cls.members.push(EnumMember {
                            py_name: name.to_string(),
                            value,
                        });
                    }
                }
            } else if !trimmed.is_empty() {
                classes.push(current.take().unwrap());
            }
        }
    }
    if let Some(c) = current {
        classes.push(c);
    }
    (classes, consts)
}

const ADS_DPA_CONST_NAMES: [&str; 6] = [
    "ADS_PRODUCT_KEY_TABLE_SIZE",
    "ADS_PRODUCT_KEY_HASH_SCALE",
    "ADS_PRODUCT_KEY_HASH_BIAS",
    "ADS_PRODUCT_KEY_HASH_SCALE_2",
    "ADS_PRODUCT_KEY_HASH_BIAS_2",
    "ADS_PRODUCT_KEY_HASH_MODULUS",
];
const ADS_DPA_INT64_MEMBER_NAMES: [&str; 2] = ["firstDpaProductKey", "firstDpaProductKeyHash2"];

fn emit_ads_dpa_cfg(classes: &[EnumClass], consts: &[ModuleConst]) {
    println!("cargo:rustc-check-cfg=cfg(recsys_ads_dpa)");

    let missing_consts: Vec<&str> = ADS_DPA_CONST_NAMES
        .into_iter()
        .filter(|n| !consts.iter().any(|c| c.py_name == *n))
        .collect();
    let int64_class = classes.iter().find(|c| c.py_class_name == "Int64Feature");
    let missing_members: Vec<&str> = ADS_DPA_INT64_MEMBER_NAMES
        .into_iter()
        .filter(|n| !int64_class.is_some_and(|c| c.members.iter().any(|m| m.py_name == *n)))
        .collect();

    let missing = missing_consts.len() + missing_members.len();
    let total = ADS_DPA_CONST_NAMES.len() + ADS_DPA_INT64_MEMBER_NAMES.len();
    if missing == 0 {
        println!("cargo:rustc-cfg=recsys_ads_dpa");
    } else if missing < total {
        panic!(
            "feature_config.py carries a PARTIAL ads-DPA block: missing constants {:?}, \
             missing Int64Feature members {:?}. Ship the full block or none of it.",
            missing_consts, missing_members
        );
    }
}

const TIME_GAP_INT64_MEMBER_NAMES: [&str; 2] =
    ["requestTimeSecSeq", "latestHistoryImpressionTimeSecSeq"];

fn emit_time_gap_cfg(classes: &[EnumClass]) {
    println!("cargo:rustc-check-cfg=cfg(recsys_time_gap)");

    let int64_class = classes.iter().find(|c| c.py_class_name == "Int64Feature");
    let missing: Vec<&str> = TIME_GAP_INT64_MEMBER_NAMES
        .into_iter()
        .filter(|n| !int64_class.is_some_and(|c| c.members.iter().any(|m| m.py_name == *n)))
        .collect();
    if missing.is_empty() {
        println!("cargo:rustc-cfg=recsys_time_gap");
    } else if missing.len() < TIME_GAP_INT64_MEMBER_NAMES.len() {
        panic!(
            "feature_config.py carries a PARTIAL time-gap block: missing Int64Feature members \
             {:?}. Ship both or neither.",
            missing
        );
    }
}

fn main() {
    let py_path = find_feature_config();

    println!("cargo:rerun-if-changed={}", py_path.display());

    let src = fs::read_to_string(&py_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", py_path.display(), e));

    let (classes, consts) = parse_feature_config(&src);

    emit_ads_dpa_cfg(&classes, &consts);
    emit_time_gap_cfg(&classes);

    let mut code = String::new();
    code.push_str("// @generated by build.rs from feature_config.py \u{2014} DO NOT EDIT\n\n");

    if !consts.is_empty() {
        code.push_str("/// Module-level integer constants from `feature_config.py`.\n");
        code.push_str("pub mod constants {\n");
        for c in &consts {
            code.push_str(&format!(
                "    /// `feature_config.{} = {}`\n",
                c.py_name, c.value
            ));
            code.push_str(&format!(
                "    pub const {}: i64 = {};\n",
                c.py_name, c.value
            ));
        }
        code.push_str("}\n\n");
    }

    for cls in &classes {
        let mod_name = to_snake_module(&cls.py_class_name);
        code.push_str(&format!(
            "/// Feature indices from Python `{}(enum.IntEnum)` in `feature_config.py`.\n",
            cls.py_class_name
        ));
        code.push_str(&format!("pub mod {} {{\n", mod_name));
        for m in &cls.members {
            let const_name = to_upper_snake(&m.py_name);
            code.push_str(&format!(
                "    /// `{}.{} = {}`\n",
                cls.py_class_name, m.py_name, m.value
            ));
            code.push_str(&format!(
                "    pub const {}: usize = {};\n",
                const_name, m.value
            ));
            code.push_str(&format!(
                "    pub const {}_NAME: &str = \"{}\";\n",
                const_name, m.py_name
            ));
            if let Some(base) = m.py_name.strip_suffix("Seq") {
                code.push_str(&format!(
                    "    pub const {}_COLUMN: &str = \"{}\";\n",
                    const_name, base
                ));
            }
        }
        code.push_str("}\n\n");
    }

    let out_dir = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&out_dir).join("feature_config_generated.rs");
    fs::write(&dest, &code).unwrap_or_else(|e| panic!("Failed to write {}: {}", dest.display(), e));
}
