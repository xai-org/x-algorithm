# Abuse enforcement rules

The baked-in user and post rule pipelines are evaluated from top to bottom. Explicit
allowlists and missing-entity checks remain the only identity-level guardrails that
run before abuse signals. Follower count, PageRank, and user-credibility scores must
not be used as blanket skip conditions: those signals can contribute evidence to a
calibrated decision, but they do not make an account or author immune from otherwise
applicable enforcement.

The YAML files in `service-lib/rules/` are fallback copies mirrored from GrowthBook.
A valid GrowthBook override takes precedence over these files, and an invalid override
falls back to the last known good pipeline. Deploying a rule change therefore requires
updating and validating the corresponding GrowthBook configuration as well as the
checked-in fallback; changing only this repository may leave production behavior
unchanged.

`skip_author_credibility_prechecks` remains part of the score-to-CEL contract for
backward compatibility with dynamic rules that explicitly opt into that behavior. It
does not restore a follower-count or credibility-based bypass in the baked-in rules.
