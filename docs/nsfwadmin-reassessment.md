# NsfwAdmin reassessment boundary

`NsfwAdmin` is consumed by the published visibility code as a Gizmoduck safety flag. The public
repository includes readers of that flag, but it does not include the production writer, removal
workflow, evidence record, or mutation API. The flag exposed through `User.safety` also does not
carry an application time, expiry time, or next-review time.

Consequently, this repository cannot honestly implement automatic expiry or scheduled
reassessment for `NsfwAdmin` itself. The Under the Hood report identifies that limitation and does
not infer a threshold or review date from daily observations of the flag.

The published `postToUserLabelRules` producer writes a different, expiring account label:
`NsfwHighPrecision`. Its checked-in defaults now require 10 matching original posts among the 20
most recent original posts, each no more than 60 days old. Labels it applies expire after 7 days.
Evaluation is event-driven: a later qualifying post-label event can reapply the label after it has
expired, but no periodic scheduler is present in the published path.

The timeline lookup is bounded to four times the configured evidence window. Reposts are excluded;
if that bounded scan contains too few original posts, the account does not qualify from incomplete
evidence.

An end-to-end `NsfwAdmin` reassessment would additionally require the unpublished owner to provide:

- a persisted decision record containing the evidence window and decision timestamp;
- bounded `expiresAt` and `nextReviewAt` values;
- an authorized removal or downgrade mutation;
- a periodic reevaluation trigger independent of new post-label events; and
- a creator-safe serving view for those fields.

Until those dependencies exist, treating the expiring `NsfwHighPrecision` label as though it
automatically clears or replaces `NsfwAdmin` would misrepresent production behavior.
