# Post IDs in label reports

Post ID reporting connects a post-label statistic to the logical posts counted by that statistic. Each daily and monthly label aggregate stores at most 20 distinct IDs, ordered by creation time, newest first, then by larger ID. Creation time uses the same `snowflakeCreatedMs(logicalId)` calculation as the existing post-count pipeline. Counts are independent of the ID cap and include the existing `everApply` population, including posts whose label was later removed.

## Invocation chain

1. `UthDailyPostsProd` runs `UthDailyPostsApp` on a daily date range. The job reads flat tweets, safety-label events and optional snapshots, then writes daily eligible-post and post-label datasets. `UthDailyPostsBackfillAdhoc` applies the same semantics over a requested date range. Both preserve the original count reductions and reduce IDs separately by the same user, authored day and label keys.
2. `UthUserMonthMhPublisherProd` runs `UthUserMonthMhPublisherApp`. It reads the daily datasets, selects the latest observations and the same daily rows used for carried/removed counts, then merges their bounded ID lists. It writes monthly Parquet and Manhattan aggregates. `postIdsTruncated` propagates a daily truncation or a monthly overflow.
3. The authenticated `underTheHoodReport` wrapper fetches `underTheHoodReport.User`. The serving column reads the monthly aggregate and the `under_the_hood_report_post_ids_max` feature switch. The switch defaults to `0`, which uses the original JSON serializer. Positive values include ID fields and are capped at the storage maximum of 20.
4. Jetfuel fetches the report JSON through the existing GraphQL field. The parser accepts decimal strings for IDs and rejects numeric IDs. The report page links each ID to `https://x.com/i/status/<id>` and shows availability or truncation text. The download returns the same JSON.

## Contracts and failures

The persisted thrift additions are optional: `postIds` and `postIdsTruncated` on `UthDailyPostLabel` and `UthPostLabelAggregate`. IDs use the `TweetId` personal-data annotation. Existing field IDs and count fields are unchanged.

When any selected daily row lacks IDs, the monthly aggregate leaves its ID fields absent. This avoids presenting a partial old/new month as complete. Takedown labels also leave IDs absent. With serving enabled, these rows emit `postIdsAvailable: false` and empty ID lists. Available rows emit string IDs and `postIdsTruncated`, including when a smaller serving cap omits stored IDs. With serving disabled, all new JSON fields are absent.

The ID list explains the aggregate; a linked post can subsequently be deleted or inaccessible. Failure of the existing report lookup retains its existing missing/unavailable behavior.

## Verification

The public checkout has no build configuration for these jobs or the serving column. Internal compilation, scheduling, dataset publication and an end-to-end production run are **UNVERIFIED** here. Nothing watches this document. The source declares the daily and monthly scheduled entry points; their deployed orchestration is **UNVERIFIED**.

Local static checks:

```sh
git diff --check main..uth-post-ids
tsc --noEmit --strict under-the-hood/jetfuel/routes/_lib/report.ts
```

Before enabling the switch, generate thrift bindings and run the Scala and Strato builds in their owning environment. Compare count outputs on identical inputs, including edited logical posts, duplicate events, removed labels and tweet flags. Verify newest-first IDs, the storage cap and truncation across daily/monthly boundaries. Compare the disabled JSON byte for byte with the prior serializer, then exercise positive serving caps, older rows and takedowns. Jetfuel mock modes `ids`, `ids_truncated` and `ids_unavailable` cover the three display states.

The source of truth is `scalding/UnderTheHoodCommon.scala` for the storage cap/order, the daily/backfill/monthly jobs for population and aggregation, `thrift/uth_serving.thrift` for persisted fields, `strato/lib/params.strato` and `strato/columns/underTheHoodReport.User.strato` for serving, and `jetfuel/routes/_lib/report.ts` plus `jetfuel/routes/report.tsx` for parsing and display.
