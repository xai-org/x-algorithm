# verification

Config and goldens for the repo-root `./verify` tool.

## Commands

Run from the repository root:

```bash
./verify seed
./verify run <flow>
./verify screenshot <page>
./verify check
```

| Command | Purpose |
|---------|---------|
| `seed` | Write a fixed seed state under `.verify/seed/` (seed constant `20260721`). |
| `run <flow>` | Execute a named flow from `verification/flows/` → `.verify/runs/<flow>.json`. |
| `screenshot <page>` | Capture a deterministic page snapshot into `.verify/screenshots/`. |
| `check` | Validate every rule in `verification/rules.json`. Exit `0` only if all pass; otherwise exit `1`. |

Identical inputs always produce identical digests. Artifacts omit wall-clock timestamps.

## Flows

- `repo-sanity` — LICENSE, README, core directories
- `phoenix-synth-smoke` — phoenix quickstart wired to seed `20260721` (docs-only; no GPU)

## Pages

- `readme` — full `README.md` text snapshot
- `overview` — `## Overview` section snapshot

## Goldens

Committed digests live in `verification/golden/`. After an intentional content change:

```bash
./verify seed
./verify run repo-sanity
./verify run phoenix-synth-smoke
./verify screenshot readme
./verify screenshot overview
./verify check --write-golden
./verify check
```

Review the golden diff before committing.
