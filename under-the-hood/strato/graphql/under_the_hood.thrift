
struct UnderTheHoodReportInfo {
  1: optional string reportPeriod
  2: optional map<string, bool> eligibilityChecks (
    strato.graphql.typename = "UnderTheHoodReportEligibilityCheckEntry"
  )
} (strato.graphql.typename = "UnderTheHoodReportInfo")

// A creator-visible, best-effort view of a currently active recommendation restriction.
// This contract intentionally does not expose enforcement actors, internal rule identifiers,
// or raw label-source payloads.
struct UnderTheHoodLiveRestriction {
  1: optional string scope
  2: optional i64 postId (
    GraphQlEncoding = "Int53",
    personalDataType = "TweetId"
  )
  3: optional string label (personalDataType = "TweetSafetyLabels, UserSafetyLabels")
  4: optional string about
  5: optional string effect
  6: optional list<string> affectedSurfaces
  7: optional i64 appliedAt (GraphQlEncoding = "Int53")
  8: optional i64 expiresAt (GraphQlEncoding = "Int53")
  9: optional string provenance
  10: optional string modelVersion
  11: optional string appealAvailability
} (strato.graphql.typename = "UnderTheHoodLiveRestriction")

struct UnderTheHoodLiveRestrictionStatus {
  1: optional i64 generatedAt (GraphQlEncoding = "Int53")
  2: optional string freshness
  3: optional list<UnderTheHoodLiveRestriction> restrictions
  4: optional i32 postScanLimit
  5: optional i32 postLookbackDays
  6: optional i32 postsScanned
  7: optional bool postScanTruncated
  8: optional string postScanStatus
  9: optional list<string> unavailableFields
  10: optional string accountLabelStatus
} (strato.graphql.typename = "UnderTheHoodLiveRestrictionStatus")

struct UnderTheHoodReport {
  1: optional i64 generatedAt (GraphQlEncoding = "Int53")
  2: optional string reportJson
  3: optional UnderTheHoodReportInfo reportInfo
  4: optional UnderTheHoodLiveRestrictionStatus liveRestrictionStatus
} (strato.graphql.typename = "UnderTheHoodReport")
