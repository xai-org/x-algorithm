
include "com/twitter/timelines/render/richtext.thrift"

struct UnderTheHoodReportInfo {
  1: optional string reportPeriod
  2: optional map<string, bool> eligibilityChecks (
    strato.graphql.typename = "UnderTheHoodReportEligibilityCheckEntry"
  )
  3: optional richtext.RichText note
  4: optional string servedPeriod
  5: optional string targetPeriod
  6: optional string periodStatus
  7: optional string nextReportNotBefore
  8: optional list<i32> availableMonths
} (strato.graphql.typename = "UnderTheHoodReportInfo")

struct UnderTheHoodReport {
  1: optional i64 generatedAt (GraphQlEncoding = "Int53")
  2: optional string reportJson
  3: optional UnderTheHoodReportInfo reportInfo
} (strato.graphql.typename = "UnderTheHoodReport")
