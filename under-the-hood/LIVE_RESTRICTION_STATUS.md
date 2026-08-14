# Live restriction status

`UnderTheHoodReport.liveRestrictionStatus` is an additive, typed GraphQL view of
recommendation restrictions that are visible to the authenticated creator. It complements the
existing monthly `reportJson`; it does not replace that historical aggregate.

## Coverage and freshness

- Current account labels come from the Gizmoduck composite user record.
- Post labels come from `visibility/baseTweetSafetyLabelMap` for at most 100 unique, authored
  posts returned by the creator timeline, restricted to the last 30 days.
- Retweets are excluded. A retweeted original's label is not attributed to the creator.
- Only labels in `underTheHoodLabels`' public account and post allowlists are returned.
- A label is current when its applied timestamp is not in the future and its optional expiry is
  still in the future. Missing expiry means the published source supplied no expiry.
- `postScanTruncated` and `postScanStatus` say when the timeline limit prevented a complete scan.
  An empty result therefore means no allowlisted active restriction was found within the stated
  coverage, not that every X enforcement system was queried.
- `accountLabelStatus` is `UNAVAILABLE` if the Gizmoduck user record or its requested label field
  is absent; an empty account-label list is not silently presented as complete in that case.

The outer `underTheHoodReport` column resolves only the authenticated viewer's user id. Its
internal keyed implementation retains the existing limited administrative access policy.

## Privacy and unavailable metadata

Raw account-label `source` / `byUser` values and post-label source variants are never returned.
Those values can contain internal actor names, LDAP identifiers, or rule ids. `provenance`
therefore remains absent until a stable, explicitly public provenance taxonomy exists.

The published label stores do not provide a reliable end-to-end model version or per-label appeal
availability or a machine-readable public surface taxonomy. `modelVersion`,
`appealAvailability`, and `affectedSurfaces` remain absent and are listed in `unavailableFields`.
Timestamps are returned only when the source record actually supplies them; no applied, expiry,
review, model, appeal, or surface value is inferred.

The human-readable `effect` remains the authoritative public description of affected product
surfaces. It does not expose internal rule or pipeline names.
