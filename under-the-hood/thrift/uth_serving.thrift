namespace java com.twitter.visibility.under_the_hood.thriftjava
namespace py gen.twitter.visibility.under_the_hood.uth_serving


struct UthDayCount {
  1: optional i32 dayOfMonth
  2: optional i64 count
}(persisted = 'true')

struct UthDayCarriedRemoved {
  1: optional i32 dayOfMonth
  2: optional i64 carried
  3: optional i64 removed
}(persisted = 'true')

struct UthDaysIncluded {
  1: optional i64 snapshotEpoch
  2: optional i32 firstCoveredDay
  3: optional i32 completeThroughDay
  4: optional i64 generatedAtMs
  5: optional i32 postObservationDays
  6: optional i32 finalThroughDay
}(persisted = 'true')

struct UthRateStats {
  1: optional double mean
  2: optional double p10
  3: optional double p25
  4: optional double p50
  5: optional double p75
  6: optional double p90
  7: optional double p99
  8: optional i64 nUsers
}(persisted = 'true')

// Brand safety is experimental-only / unused now
enum UthBrandSafetyCategory {
  SAFE = 1
  LOW_RISK = 2
  MEDIUM_RISK = 3
  NOT_OBSERVED = 4
}

enum UthFollowerClass {
  ALL = 1
  LT_1K = 2
  FROM_1K_TO_10K = 3
  FROM_10K_TO_100K = 4
  GTE_100K = 5
}


struct UthPostLabelAggregate {
  1: optional string label (personalDataType = 'TweetSafetyLabels')
  2: optional list<UthDayCarriedRemoved> days
  // Newest logical post IDs represented by days, bounded by the month publisher.
  4: optional list<i64> postIds (personalDataType = 'TweetId')
}(persisted = 'true', hasPersonalData = 'true')

struct UthAccountLabelAggregate {
  1: optional string label (personalDataType = 'UserSafetyLabels')
  2: optional list<i32> activeDays
}(persisted = 'true', hasPersonalData = 'true')

// Brand safety is experimental-only / unused now
struct UthBrandSafetyAggregate {
  1: optional UthBrandSafetyCategory category
  2: optional list<UthDayCount> days
}(persisted = 'true')


struct UthUserMonthAggregate {
  1: optional i32 monthBucket
  2: optional UthDaysIncluded daysIncluded
  3: optional list<UthDayCount> eligiblePostAgg
  4: optional list<UthPostLabelAggregate> postLabelAgg
  5: optional list<UthAccountLabelAggregate> accountLabelAgg
  // Experimental / not used: always empty; not served in reportJson.
  6: optional list<UthBrandSafetyAggregate> brandSafetyAgg
}(persisted = 'true', hasPersonalData = 'true')

struct UthUserMonthKey {
  1: required i64 userId (personalDataType = 'UserId')
  2: required i32 monthBucket
}(persisted = 'true', hasPersonalData = 'true')


struct UthLabeledRateStats {
  1: optional string label (personalDataType = 'TweetSafetyLabels, UserSafetyLabels')
  2: optional UthRateStats stats
}(persisted = 'true', hasPersonalData = 'true')

// Experimental / not used (see UthBrandSafetyCategory).
struct UthBrandSafetyRateStats {
  1: optional UthBrandSafetyCategory category
  2: optional UthRateStats stats
}(persisted = 'true')

struct UthReferenceCohortStats {
  1: optional UthFollowerClass followerClass
  2: optional string cohortDimension
  3: optional string cohortValue
  4: optional i64 nUsers
  5: optional list<UthLabeledRateStats> postLabelAgg
  6: optional list<UthLabeledRateStats> accountLabelAgg
  // Experimental / not used: always empty; not served in reportJson.
  7: optional list<UthBrandSafetyRateStats> brandSafetyAgg
}(persisted = 'true', hasPersonalData = 'true')


struct UthReferenceMonthAggregate {
  1: optional i32 monthBucket
  2: optional UthDaysIncluded daysIncluded
  3: optional list<UthReferenceCohortStats> cohorts
}(persisted = 'true', hasPersonalData = 'true')


struct UthDailyEligiblePost {
  1: optional i64 userId (personalDataType = 'UserId')
  2: optional i32 authoredYyyymmdd
  3: optional i64 count
}(persisted = 'true', hasPersonalData = 'true')

struct UthDailyPostLabel {
  1: optional i64 userId (personalDataType = 'UserId')
  2: optional i32 authoredYyyymmdd
  3: optional string label (personalDataType = 'TweetSafetyLabels')
  4: optional i64 carried
  5: optional i64 removed
  6: optional i32 asOfYyyymmdd
  7: optional i32 observationAgeDays
  8: optional bool isFinal
  9: optional i32 postObservationDays
  // Logical post IDs represented by carried; absent on pre-migration rows.
  11: optional list<i64> postIds (personalDataType = 'TweetId')
}(persisted = 'true', hasPersonalData = 'true')

struct UthDailyAccountLabel {
  1: optional i64 userId (personalDataType = 'UserId')
  2: optional i32 dayYyyymmdd
  3: optional string label (personalDataType = 'UserSafetyLabels')
}(persisted = 'true', hasPersonalData = 'true')


struct UthUserMonthAggregateRow {
  1: optional i64 userId (personalDataType = 'UserId')
  2: optional i32 monthBucket
  3: optional UthDaysIncluded daysIncluded
  4: optional list<UthDayCount> eligiblePostAgg
  5: optional list<UthPostLabelAggregate> postLabelAgg
  6: optional list<UthAccountLabelAggregate> accountLabelAgg
  7: optional list<UthBrandSafetyAggregate> brandSafetyAgg
}(persisted = 'true', hasPersonalData = 'true')
