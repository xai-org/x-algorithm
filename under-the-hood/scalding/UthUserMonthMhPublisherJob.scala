package com.twitter.visibility.under_the_hood

import com.twitter.scalding.Args
import com.twitter.scalding.DateOps
import com.twitter.scalding.DateParser
import com.twitter.scalding.DateRange
import com.twitter.scalding.Days
import com.twitter.scalding.Execution
import com.twitter.scalding.RichDate
import com.twitter.scalding.TypedPipe
import com.twitter.common.util.Clock
import com.twitter.scalding_internal.dalv2.DAL
import com.twitter.scalding_internal.dalv2.DALWrite._
import com.twitter.scalding_internal.job.TwitterExecutionApp
import com.twitter.scalding_internal.job.analytics_batch._
import com.twitter.scalding_internal.multiformat.format.keyval.KeyVal
import com.twitter.visibility.under_the_hood.thriftscala._
import java.util.TimeZone

class UthUserMonthMhPublisherApp {
  import UnderTheHoodCommon._
  import UthUserMonthMhPublisherApp._

  implicit val tz: TimeZone = DateOps.UTC

  def runOnDateRange(dateRange: DateRange, config: UthUserMonthMhConfig): Execution[Unit] = {
    val asOfDay = yyyymmdd(dateRange.end.timestamp)
    val monthPublishEpoch = dateRange.end.timestamp
    val lookbackStart = dateRange.end - Days(config.dailyLookbackDays - 1)
    val dataFloor = RichDate(yyyymmddToMs(config.dailyDataFirstDay))
    val readStart = if (lookbackStart < dataFloor) dataFloor else lookbackStart
    val readRange = DateRange(readStart, dateRange.end)
    val basePath = config.outputPath

    val eligible = DAL.read(UthDailyEligiblePostsScalaDataset, readRange).toTypedPipe
    val postLabels = DAL.read(UthDailyPostLabelsScalaDataset, readRange).toTypedPipe
    val accountLabels = DAL.read(UthDailyAccountLabelsScalaDataset, readRange).toTypedPipe
    val lookbackStartYyyymmdd = yyyymmdd(readStart.timestamp)

    val userMonths =
      assembleUserMonths(
        eligible,
        postLabels,
        accountLabels,
        asOfDay,
        lookbackStartYyyymmdd,
        monthPublishEpoch,
        config.postObservationDays,
        config.testUserIds,
        config.reducers)

    val monthMeta = monthMetaSentinelRows(
      userMonths,
      asOfDay,
      lookbackStartYyyymmdd,
      monthPublishEpoch,
      config.postObservationDays,
      config.reducers)

    val sharded = (userMonths ++ monthMeta).shard(config.writeShards)

    val mhWrite =
      sharded
        .map { row => KeyVal(UthUserMonthKey(row.userId.get, row.monthBucket.get), toMhValue(row)) }
        .writeDALVersionedKeyValExecution(
          dataset = UthUserMonthAggregatesMhScalaDataset,
          pathLayout = D.Suffix(s"$basePath/mh/uth_user_month_aggregates"),
          version = UseClockTime(Clock.SYSTEM_CLOCK)
        )

    val parquetWrite =
      sharded.writeDALSnapshotExecution(
        UthUserMonthAggregatesScalaDataset,
        D.Daily,
        D.Suffix(s"$basePath/uth_user_month_aggregates"),
        D.Parquet,
        dateRange.end
      )

    Execution.zip(mhWrite, parquetWrite).unit
  }
}

object UthUserMonthMhPublisherApp {
  import UnderTheHoodCommon._

  val MetadataUserId: Long = -1L

  private def toMhValue(row: UthUserMonthAggregateRow): UthUserMonthAggregate =
    UthUserMonthAggregate(
      monthBucket = row.monthBucket,
      daysIncluded = row.daysIncluded,
      eligiblePostAgg = row.eligiblePostAgg,
      postLabelAgg = row.postLabelAgg,
      accountLabelAgg = row.accountLabelAgg,
      brandSafetyAgg = row.brandSafetyAgg
    )

  private[under_the_hood] def monthMetaSentinelRows(
    userMonths: TypedPipe[UthUserMonthAggregateRow],
    asOfDay: Int,
    lookbackStartYyyymmdd: Int,
    monthPublishEpoch: Long,
    postObservationDays: Int,
    reducers: Int
  ): TypedPipe[UthUserMonthAggregateRow] = {
    val months = applyReducers(
      userMonths.flatMap(_.monthBucket).map(_ -> 1).group,
      reducers
    ).sum.keys
    months.map { month =>
      val (firstCov, lastCov) =
        monthCoverageYyyymmdd(month, lookbackStartYyyymmdd, asOfDay)
      val coverage = UthDaysIncluded(
        snapshotEpoch = Some(monthPublishEpoch),
        firstCoveredDay = Some(firstCov),
        completeThroughDay = Some(lastCov),
        generatedAtMs = Some(monthPublishEpoch),
        postObservationDays = Some(postObservationDays),
        finalThroughDay = Some(finalThroughDay(lastCov, asOfDay, postObservationDays))
      )
      UthUserMonthAggregateRow(
        userId = Some(MetadataUserId),
        monthBucket = Some(month),
        daysIncluded = Some(coverage),
        eligiblePostAgg = Some(Nil),
        postLabelAgg = Some(Nil),
        accountLabelAgg = Some(Nil),
        brandSafetyAgg = Some(Nil)
      )
    }
  }

  private[under_the_hood] def assembleUserMonths(
    eligible: TypedPipe[UthDailyEligiblePost],
    postLabels: TypedPipe[UthDailyPostLabel],
    accountLabels: TypedPipe[UthDailyAccountLabel],
    asOfDay: Int,
    lookbackStartYyyymmdd: Int,
    monthPublishEpoch: Long,
    postObservationDays: Int,
    testUserIds: Set[Long],
    reducers: Int
  ): TypedPipe[UthUserMonthAggregateRow] = {

    val latestPostLabels = latestAsOfPostLabelRows(postLabels, reducers)
    def inWindow(day: Int): Boolean = day >= lookbackStartYyyymmdd && day <= asOfDay

    val eligibleByUserMonth = groupToList(
      eligible.flatMap { row =>
        for {
          userId <- row.userId if inScope(testUserIds, userId)
          day <- row.authoredYyyymmdd if inWindow(day)
          count <- row.count
        } yield ((userId, monthBucket(day)), (dayOfMonth(day), count))
      },
      reducers
    )

    val labelsByUserMonth = groupToList(
      latestPostLabels.flatMap { row =>
        for {
          userId <- row.userId if inScope(testUserIds, userId)
          day <- row.authoredYyyymmdd if inWindow(day)
          label <- row.label
          carried <- row.carried
          removed <- row.removed
          postIds = row.postIds.getOrElse(Seq.empty)
        } yield (
          (userId, monthBucket(day)),
          (label, dayOfMonth(day), carried, removed, postIds)
        )
      },
      reducers
    )

    val distinctAccountRows = applyReducers(
      accountLabels.flatMap { row =>
        for {
          userId <- row.userId if inScope(testUserIds, userId)
          day <- row.dayYyyymmdd if inWindow(day)
          label <- row.label
        } yield (((userId, monthBucket(day)), (label, dayOfMonth(day))), 1)
      }.group,
      reducers
    ).sum.keys
    val accountByUserMonth = groupToList(distinctAccountRows, reducers)

    val joined = eligibleByUserMonth
      .outerJoin(labelsByUserMonth)
      .outerJoin(accountByUserMonth)
    (if (reducers > 0) joined.withReducers(reducers) else joined).toTypedPipe
      .map {
        case ((userId, month), (joinedOpt, accountOpt)) =>
          val (eligibleOpt, labelsOpt) = joinedOpt.getOrElse((None, None))

          val eligibleDays = eligibleOpt
            .getOrElse(Nil)
            .groupBy { case (day, _) => day }
            .map {
              case (day, pairs) =>
                UthDayCount(Some(day), Some(pairs.map(_._2).sum))
            }
            .toList
            .sortBy(_.dayOfMonth.getOrElse(0))

          val postLabelAgg = labelsOpt
            .getOrElse(Nil)
            .groupBy { case (label, _, _, _, _) => label }
            .map {
              case (label, rows) =>
                val selectedRows = rows
                  .groupBy { case (_, day, _, _, _) => day }
                  .values
                  .map { dayRows =>
                    dayRows.maxBy {
                      case (_, _, carried, removed, postIds) =>
                        (carried, removed, postIds.size)
                    }
                  }
                  .toList
                val days = selectedRows
                  .map {
                    case (_, day, carried, removed, _) =>
                      UthDayCarriedRemoved(Some(day), Some(carried), Some(removed))
                  }
                  .sortBy(_.dayOfMonth.getOrElse(0))
                val postIds = UthPostIds.newestDistinct(
                  selectedRows.flatMap(_._5),
                  UthPostIds.MaxPostIdsPerLabel
                )
                UthPostLabelAggregate(
                  label = Some(label),
                  days = Some(days),
                  postIds = Some(postIds)
                )
            }
            .toList
            .sortBy(_.label.getOrElse(""))

          val accountLabelAgg = accountOpt
            .getOrElse(Nil)
            .groupBy { case (label, _) => label }
            .map {
              case (label, rows) =>
                UthAccountLabelAggregate(
                  Some(label),
                  Some(rows.map(_._2).distinct.sorted)
                )
            }
            .toList
            .sortBy(_.label.getOrElse(""))

          val (firstCov, lastCov) =
            monthCoverageYyyymmdd(month, lookbackStartYyyymmdd, asOfDay)
          val coverage = UthDaysIncluded(
            snapshotEpoch = Some(monthPublishEpoch),
            firstCoveredDay = Some(firstCov),
            completeThroughDay = Some(lastCov),
            generatedAtMs = Some(monthPublishEpoch),
            postObservationDays = Some(postObservationDays),
            finalThroughDay = Some(finalThroughDay(lastCov, asOfDay, postObservationDays))
          )

          UthUserMonthAggregateRow(
            userId = Some(userId),
            monthBucket = Some(month),
            daysIncluded = Some(coverage),
            eligiblePostAgg = Some(eligibleDays),
            postLabelAgg = Some(postLabelAgg),
            accountLabelAgg = Some(accountLabelAgg),
            brandSafetyAgg = Some(Nil)
          )
      }
  }

  private def groupToList[K: Ordering, V](
    pipe: TypedPipe[(K, V)],
    reducers: Int
  ) = {
    val grouped = if (reducers > 0) pipe.group.withReducers(reducers) else pipe.group
    grouped.toList
  }
}

case class UthUserMonthMhConfig(
  testUserIds: Set[Long],
  reducers: Int,
  dailyLookbackDays: Int,
  postObservationDays: Int,
  dailyDataFirstDay: Int,
  writeShards: Int,
  outputPath: String)

object UthUserMonthMhConfig {
  def fromArgs(args: Args): UthUserMonthMhConfig =
    UthUserMonthMhConfig(
      testUserIds = UnderTheHoodCommon.parseUserIds(args),
      reducers = args.int("reducers", 20000),
      dailyLookbackDays =
        UnderTheHoodCommon.preferredInt(args, "dailyLookbackDays", "lookbackDays", 90),
      postObservationDays =
        UnderTheHoodCommon.preferredInt(args, "postObservationDays", "observationDays", 7),
      dailyDataFirstDay = args.int("dailyDataFirstDay", 20260701),
      writeShards = {
        val n = args.int("writeShards", 50)
        require(n > 0, s"--writeShards must be > 0; got $n")
        n
      },
      outputPath = args.optional("outputPath").getOrElse("/user/<hadoop-role>/under_the_hood")
    )
}

object UthUserMonthMhPublisherAdhoc extends UthUserMonthMhPublisherApp with TwitterExecutionApp {
  override def job: Execution[Unit] = Execution.withArgs { args =>
    runOnDateRange(UnderTheHoodDates.resolve(args), UthUserMonthMhConfig.fromArgs(args))
  }
}

object UthUserMonthMhPublisherProd
    extends UthUserMonthMhPublisherApp
    with TwitterScheduledExecutionApp {
  implicit val dp: DateParser = DateParser.default
  override def scheduledJob: Execution[Unit] = {
    val execArgs = AnalyticsBatchExecutionArgs(
      batchDesc = BatchDescription("uth_user_month_mh_publisher_prod"),
      firstTime = BatchFirstTime(RichDate("2026-07-31")),
      batchIncrement = BatchIncrement(Days(1))
    )
    Execution.withArgs { args =>
      AnalyticsBatchExecution(execArgs) { dateRange =>
        runOnDateRange(dateRange, UthUserMonthMhConfig.fromArgs(args))
      }
    }
  }
}
