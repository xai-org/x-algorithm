#!/usr/bin/env python3
"""Offline contract tests for #202. Uses production files, no Cargo dependencies.

The public checkout lacks the home-mixer build graph/internal crates. This runner
compiles full new modules plus unchanged extracted scorer, history reader/writer
function bodies with explicit protocol/parameter/service test doubles.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def extract(path, name):
    source = (ROOT / path).read_text()
    match = re.search(r"^\s*(?:pub(?:\(crate\))? )?(?:async )?fn " + re.escape(name) + r"\(", source, re.M)
    if not match:
        raise RuntimeError(f"Missing {path}:{name}")
    opening = source.index("{", match.end())
    end, depth = opening + 1, 1
    while depth:
        depth += (source[end] == "{") - (source[end] == "}")
        end += 1
    return source[match.start():end].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mutation-check", action="store_true", help="verify the cross-request regression fails if history is disconnected")
    args = parser.parse_args()
    rustc = os.environ.get("RUSTC") or shutil.which("rustc")
    if not rustc:
        raise SystemExit("Set RUSTC to rustc 1.90+; see README.md")
    # Generated sources/logs outside checkout unless requested otherwise.
    output = Path(os.environ.get("AUTHOR_DIVERSITY_TEST_OUT") or tempfile.mkdtemp(prefix="author-diversity-tests-"))
    output.mkdir(parents=True, exist_ok=True)
    manifest = []

    def functions(path, names):
        result = []
        for name in names:
            body = extract(path, name)
            manifest.append({"path": path, "function": name, "sha256": hashlib.sha256(body.encode()).hexdigest()})
            result.append(body)
        return "\n\n".join(result)

    replacements = {
        "/* SCORER */": functions("home-mixer/scorers/ranking_scorer.rs", ["offset_score", "diversity_multiplier", "author_pool_counts", "served_slate_contexts", "stored_slate_contexts", "author_diversity_multipliers", "apply_author_diversity", "score"]),
        "/* HYDRATOR */": functions("home-mixer/query_hydrators/served_history_query_hydrator.rs", ["enable", "hydrate", "update"]),
        "/* HISTORY_FUNCTIONS */": functions("home-mixer/query_hydrators/served_history_query_hydrator.rs", ["recently_served_ids", "is_module_eligible"]),
        "/* WRITER */": functions("home-mixer/side_effects/update_served_history_side_effect.rs", ["build_post_entries", "nonzero", "build_tweet_entry"]),
        "/* FILTER */": functions("home-mixer/filters/previously_served_posts_filter.rs", ["filter"]),
        "/* RELATED_IDS */": functions("home-mixer/util/candidates_util.rs", ["related_post_ids_iter"]),
        "/* STRING_CASE */": functions("home-mixer/util/string_case.rs", ["upper_snake_to_pascal"]),
        "/* UPSTREAM_TESTS */": functions("home-mixer/scorers/ranking_scorer.rs", ["candidate", "candidate_with_retweet", "recent_author_history", "recent_author_is_penalized_in_both_scoring_paths", "history_and_pool_use_the_same_original_author_key"]),
    }
    needed = "EnableCrossRequestAuthorDiversity AuthorDiversityHistoryWindowSeconds AuthorDiversityHistoryHalfLifeSeconds AuthorDiversityHistoryWeight AuthorDiversityHistoryMaxCount AuthorDiversityHistoryMaxItems AuthorDiversityDecay AuthorDiversityFloor EnableAuthorDiversity EnableOonRescoreForInNetworkRepliesRetweets MultiplierPreOffset EnableUrtMigrationComponents ExcludeServedTweetIdsDuration ExcludeServedTweetIdsNumber FeedSurveyFatigueHours WhoToFollowFatigueHours".split()
    params = (ROOT / "home-mixer/params/param.rs").read_text()
    generated = []
    for name in needed:
        m = re.search(r"param!\(\s*" + name + r',\s*(\w+),\s*"([^\"]+)",\s*([^\s,)]+)', params)
        if not m:
            raise RuntimeError("Missing param " + name)
        typ, key, default = m.groups()
        generated.append(f'struct {name}; impl Param for {name} {{ type Value={typ}; const KEY: &\'static str="{key}"; fn default_value()->{typ} {{{default}}} }}')
    replacements["/* PARAMS */"] = "\n".join(generated)
    fixture = (HERE / "fixture.rs").read_text()
    for marker, value in replacements.items():
        assert marker in fixture, marker
        fixture = fixture.replace(marker, value)
    for name in ["author_diversity", "author_history"]:
        path = ROOT / "home-mixer/util" / (name + ".rs")
        fixture = fixture.replace("@" + name.upper() + "@", str(path))
        manifest.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    # Preserve the production storage key expression exactly, without attempting
    # to emulate the unavailable RPC/serialization library.
    client_path = "home-mixer/clients/served_history_client.rs"
    client_source = (ROOT / client_path).read_text()
    key_line = "let lkey = vec![(i64::MAX - served_time_ms).to_be_bytes().to_vec()];"
    if client_source.count(key_line) != 1:
        raise SystemExit("Production storage key changed; re-review the scheduling model")
    scheduling_source = (HERE / "scheduling.rs").read_text()
    scheduling_source = scheduling_source.replace("/* STORAGE_KEY */", "fn storage_lkey(served_time_ms: i64) -> Vec<Vec<u8>> { " + key_line + " lkey }")
    scheduling = output / "scheduling.rs"
    scheduling.write_text(scheduling_source)
    fixture = fixture.replace("@SCHEDULING@", str(scheduling))
    manifest.append({"path": client_path, "expression": key_line, "sha256": hashlib.sha256(key_line.encode()).hexdigest()})
    manifest.append({"path": "tools/author-diversity-tests/scheduling.rs", "sha256": hashlib.sha256((HERE / "scheduling.rs").read_bytes()).hexdigest()})
    src = output / "contracts.rs"
    src.write_text(fixture)
    binary = output / "contracts"
    compiled = subprocess.run([rustc, "--edition=2024", "--test", str(src), "-o", str(binary)], text=True, capture_output=True, timeout=60)
    (output / "compile.log").write_text(compiled.stdout + compiled.stderr)
    if compiled.returncode:
        raise SystemExit(compiled.stderr)
    tests = subprocess.run([str(binary), "--nocapture", "--test-threads=1"], text=True, capture_output=True, timeout=30)
    (output / "tests.log").write_text(tests.stdout + tests.stderr)
    (output / "source_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    simulations = [json.loads(line.removeprefix("SIMULATION ")) for line in tests.stdout.splitlines() if line.startswith("SIMULATION ")]
    (output / "simulations.json").write_text(json.dumps(simulations, indent=2) + "\n")
    print("\n".join(line for line in tests.stdout.splitlines() if line.strip() and not line.startswith("SIMULATION ")) + tests.stderr)
    print(f"Simulation records: {len(simulations)} (simulations.json)")
    if tests.returncode:
        raise SystemExit(tests.returncode)
    for kind, expected in [("interleaving", 90), ("delay_matrix", 20), ("property_checks", 1)]:
        count = sum(row["kind"] == kind for row in simulations)
        if count != expected:
            raise SystemExit(f"Incomplete simulation coverage: {kind} {count} != {expected}")
    if args.mutation_check:
        mutated, changes = re.subn(r"let prior = history.*?\.unwrap_or\(0\.0\);", "let prior = 0.0;", fixture, flags=re.S)
        if changes != 1:
            raise SystemExit("Could not locate exactly one historical contribution")
        mutant_src = output / "history_disconnected.rs"
        mutant_src.write_text(mutated)
        mutant_bin = output / "history_disconnected"
        compiled = subprocess.run([rustc, "--edition=2024", "--test", str(mutant_src), "-o", str(mutant_bin)], text=True, capture_output=True, timeout=60)
        if compiled.returncode:
            raise SystemExit(compiled.stderr)
        negative = subprocess.run([str(mutant_bin), "--exact", "contracts::served_writer_reader_and_scorer_preserve_author_history_across_requests", "--nocapture"], text=True, capture_output=True, timeout=30)
        (output / "mutation_check.log").write_text(negative.stdout + negative.stderr)
        if negative.returncode != 101 or "1 failed" not in negative.stdout:
            raise SystemExit("Disconnected-history mutant was not rejected")
        print("Mutation check: disconnected history correctly fails the cross-request regression.")
    print("Artifacts:", output)
    raise SystemExit(tests.returncode)


if __name__ == "__main__":
    main()
