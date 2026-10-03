# Under the Hood account affinities

The optional `accountAffinityProfile` member in `reportJson` exposes up to ten
precomputed SimClusters **known-for** memberships for the report's account.
It is gated by `under_the_hood_report_account_affinities_enabled`, default `false`.
The existing authenticated-viewer route and account/report eligibility checks
still apply. No lookup runs for an ineligible report or when the switch is off.

Example of the new member (illustrative IDs and weights):

```json
{
  "accountAffinityProfile": {
    "modelVersion": "20M_145K_2020",
    "embeddingType": "knownFor",
    "retrievedAt": "2026-09-20T12:00:00Z",
    "primaryClusters": [
      { "clusterId": 1420, "weight": 0.82 },
      { "clusterId": 804, "weight": 0.54 },
      { "clusterId": 31, "weight": 0.28 }
    ],
    "notes": "The export includes notes explaining freshness, score semantics, and limitations."
  }
}
```

## Source and meaning

[`mh/accountKnownFor`](strato/columns/mh/accountKnownFor.strato) reads the existing
Athena `simclusters_v2_known_for_20m_145k_2020` dataset using the storage contract in
[`UserKnownForReadableStore`](../simclusters/simclusters_v2/summingbird/stores/UserKnownForReadableStore.scala).
The record's `knownForModelVersion` must match `20M_145K_2020`; changing models
requires updating both the dataset and the version check. This is an account
membership profile, distinct from the consumer **interested-in** embeddings.

- `retrievedAt` is the lookup time. The source record has no computation timestamp.
  This is the latest available offline record, **not** historical data for the
  report's monthly `period`. The report's existing `generatedAt` retains its meaning.
- `weight` is the raw `knownForScore`. Scores are not probabilities, percentages,
  normalized top-ten shares, ranking coefficients, or estimates of reach.
- Clusters sort by descending weight, then ascending ID for ties. Missing,
  non-positive, non-finite scores and negative IDs are excluded before truncation.
- IDs are meaningful only within the specified model. The source has no curated
  human-readable topic labels, so the export does not fabricate a `label` field.
- A missing record, model mismatch, or lookup failure omits the entire member and
  preserves the existing report JSON. A present record with no usable scores
  produces `primaryClusters: []`. Neither outcome establishes anything about
  enforcement or suppression.

The read-only column allows calls only from `underTheHoodReport.User`, which keeps
its existing access policy. The feature performs one key-value lookup per eligible
enabled export, followed by local filtering and sorting (`O(N log N)` for `N`
stored memberships). It adds no impression logging, per-post queries, or batch job.

## Validation and rollout

For a local check, install/provide a JDK and the Scala 2.12 compiler, library, and
reflect JARs, then run (use `;` instead of `:` for the classpath on Windows):

```sh
python3 under-the-hood/tests/run_account_affinity_tests.py \
  --scala-classpath '/path/scala-compiler.jar:/path/scala-library.jar:/path/scala-reflect.jar'
```

This harness compiles the unchanged Scala-compatible selection and JSON-append
functions extracted from the Strato library. It also exercises the report's
enrichment helper with a stubbed catalog call and serializer, including disabled,
missing, rejected, and failing lookups. It checks non-finite values, deterministic
top-ten selection, and parses enriched JSON for all four label-list shapes.
It does **not** validate Strato typing, profile serialization, or access policies.

The public snapshot does not include a Strato compiler/runtime, generated
SimClusters types, or access to the Manhattan service. The implementation needs
internal validation before enabling the switch:

1. Compile the library, storage column, and report column with the generated
   `ClustersUserIsKnownFor` type. Verify the native big-endian Long key and compact
   Thrift value encodings against an existing known-for record, and authorize the
   serving identity to read the dataset if needed.
2. Evaluate [`accountAffinityProfileTest`](strato/tests/accountAffinityProfileTest.strato).
   Every exported `checks` entry and `allPassed` should be true. Fixtures cover
   selection, ties, invalid/missing scores, raw weights, model mismatch, empty
   records, timestamp semantics, and JSON serialization.
3. With a stubbed lookup, verify zero reads when disabled or ineligible, exactly
   one read for an eligible enabled report, and unchanged label JSON on a miss,
   version mismatch, or exception. Check all four combinations of empty/nonempty
   account and post labels, including the existing conditional `total*Labels` fields.
4. Verify the authenticated route cannot request another account and direct
   untrusted reads of `mh/accountKnownFor` are rejected. Check suspended accounts,
   incomplete months, and month fallback retain their existing behavior.

The feature switch stays off until those checks pass. This change does not add
topic naming, historical embeddings, or client-side presentation of the profile.
