
include "com/twitter/timelines/render/richtext.thrift"

struct UnderTheHoodReportInfo {
  1: optional string reportPeriod
  2: optional map<string, bool> eligibilityChecks (
    strato.graphql.typename = "UnderTheHoodReportEligibilityCheckEntry"
  )
  3: optional richtext.RichText note
  4: optional map<string, richtext.RichText> postLabelAppeals (
    strato.graphql.typename = "UnderTheHoodReportPostLabelAppealEntry"
  )
  5: optional map<string, richtext.RichText> accountLabelAppeals (
    strato.graphql.typename = "UnderTheHoodReportAccountLabelAppealEntry"
  )
  6: optional i32 maxPostIdsPerLabel
} (strato.graphql.typename = "UnderTheHoodReportInfo")

struct UnderTheHoodReport {
  1: optional i64 generatedAt (GraphQlEncoding = "Int53")
  2: optional string reportJson
  3: optional UnderTheHoodReportInfo reportInfo
} (strato.graphql.typename = "UnderTheHoodReport")
