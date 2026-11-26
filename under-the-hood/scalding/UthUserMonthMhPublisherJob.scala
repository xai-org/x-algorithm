package com.twitter.visibility.under_the_hood

import com.twitter.scalding.Args
import com.twitter.scalding.DateOps
import com.twitter.scalding.DateParser
import com.twitter.scalding.DateRange
import com.twitter.scalding.Days
import com.twitter.scalding.Execution
import com.twitter.scalding.RichDate
import com.twitter.scalding.TypedPipe
import com.twitter.scalding.TypedTsv
import com.twitter.common.util.Clock
import com.twitter.scalding_internal.dalv2.DAL
import com.twitter.scalding_internal.dalv2.DALWrite._
import com.twitter.scalding_internal.job.TwitterExecutionApp
import com.twitter.scalding_internal.job.analytics_batch._
import com.twitter.scalding_internal.multiformat.format.keyval.KeyVal
import com.twitter.statebird.v2.thriftscala.Environment
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

    def readFromFloor[T](firstDay: Option[Int], read: DateRange => TypedPipe[T]): TypedPipe[T] =
      firstDay.fold(TypedPipe.empty: TypedPipe[T]) { day =>
        val floor = RichDate(yyyymmddToMs(day))
        val start = if (readStart < floor) floor else readStart
        if (start > dateRange.end) TypedPipe.empty else read(DateRange(start, dateRange.end))
      }

    val env = config.dailyDataEnvironment
    val eligible =
      DAL.read(UthDailyEligiblePostsScalaDataset, readRange).withEnvironment(env).toTypedPipe
    val postLabels =
      DAL.read(UthDailyPostLabelsScalaDataset, readRange).withEnvironment(env).toTypedPipe ++
        readFromFloor(
          config.postTakedownDataFirstDay,
          DAL.read(UthDailyPostTakedownLabelsScalaDataset, _).withEnvironment(env).toTypedPipe)
    val accountLabels =
      DAL.read(UthDailyAccountLabelsScalaDataset, readRange).withEnvironment(env).toTypedPipe ++
        readFromFloor(
          config.accountTakedownDataFirstDay,
          DAL.read(UthDailyAccountTakedownLabelsScalaDataset, _).withEnvironment(env).toTypedPipe)
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
        config.reducers,
        config.publishEpochOverrides,
        config.maxPostIdsPerLabel
      )

    val monthMeta = monthMetaSentinelRows(
      userMonths,
      asOfDay,
      lookbackStartYyyymmdd,
      monthPublishEpoch,
      config.postObservationDays,
      config.reducers,
      config.publishEpochOverrides)

    val sharded = (userMonths ++ monthMeta).shard(config.writeShards)

    if (config.tsvOnly) writeDebugTsvs(sharded, s"$basePath/month_tsv_debug/$asOfDay")
    else {

      val mhWrite =
        sharded
          .map { row =>
            KeyVal(UthUserMonthKey(row.userId.get, row.monthBucket.get), toMhValue(row))
          }
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
}

object UthUserMonthMhPublisherApp {
  import UnderTheHoodCommon._

  private[under_the_hood] def writeDebugTsvs(
    rows: TypedPipe[UthUserMonthAggregateRow],
    dir: String
  ): Execution[Unit] = {
    val userMonths = rows.map { r =>
      val cov = r.daysIncluded
      (
        r.userId.getOrElse(0L),
        r.monthBucket.getOrElse(0),
        r.eligiblePostAgg.getOrElse(Nil).flatMap(_.count).sum,
        cov.flatMap(_.firstCoveredDay).getOrElse(0),
        cov.flatMap(_.completeThroughDay).getOrElse(0),
        cov.flatMap(_.finalThroughDay).getOrElse(0))
    }
    val postLabels = rows.flatMap { r =>
      r.postLabelAgg.getOrElse(Nil).map { a =>
        val days = a.days.getOrElse(Nil)
        (
          r.userId.getOrElse(0L),
          r.monthBucket.getOrElse(0),
          a.label.getOrElse(""),
          a.countryCodes.fold("")(_.mkString(",")),
          days.flatMap(_.carried).sum,
          days.flatMap(_.removed).sum,
          days
            .map(d =>
              s"${d.dayOfMonth.getOrElse(0)}:${d.carried.getOrElse(0L)}:${d.removed.getOrElse(0L)}")
            .mkString(","),
          a.postIds.getOrElse(Nil).mkString(","))
      }
    }
    val accountLabels = rows.flatMap { r =>
      r.accountLabelAgg.getOrElse(Nil).map { a =>
        (
          r.userId.getOrElse(0L),
          r.monthBucket.getOrElse(0),
          a.label.getOrElse(""),
          a.countryCodes.fold("")(_.mkString(",")),
          a.activeDays.getOrElse(Nil).mkString(","))
      }
    }
    Execution
      .zip(
        userMonths.writeExecution(TypedTsv[(Long, Int, Long, Int, Int, Int)](s"$dir/user_months")),
        postLabels.writeExecution(
          TypedTsv[(Long, Int, String, String, Long, Long, String, String)](s"$dir/post_labels")),
        accountLabels.writeExecution(
          TypedTsv[(Long, Int, String, String, String)](s"$dir/account_labels"))
      )
      .unit
  }

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
    reducers: Int,
    publishEpochOverrides: Map[Int, Long] = Map.empty
  ): TypedPipe[UthUserMonthAggregateRow] = {
    val months = applyReducers(
      userMonths.flatMap(_.monthBucket).map(_ -> 1).group,
      reducers
    ).sum.keys
    months.map { month =>
      val (firstCov, lastCov) =
        monthCoverageYyyymmdd(month, lookbackStartYyyymmdd, asOfDay)
      val epoch = publishEpochOverrides.getOrElse(month, monthPublishEpoch)
      val coverage = UthDaysIncluded(
        snapshotEpoch = Some(epoch),
        firstCoveredDay = Some(firstCov),
        completeThroughDay = Some(lastCov),
        generatedAtMs = Some(epoch),
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
    reducers: Int,
    publishEpochOverrides: Map[Int, Long] = Map.empty,
    maxPostIdsPerLabel: Int = DefaultMaxPostIdsPerLabel
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
        } yield (
          (userId, monthBucket(day)),
          (label, dayOfMonth(day), carried, removed, row.postIds.getOrElse(Nil)))
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
                val perDay = rows
                  .groupBy { case (_, day, _, _, _) => day }
                  .map {
                    case (day, dayRows) =>
                      val best = dayRows.maxBy {
                        case (_, _, carried, removed, _) => (carried, removed)
                      }
                      (UthDayCarriedRemoved(Some(day), Some(best._3), Some(best._4)), best._5)
                  }
                  .toList
                val days = perDay.map(_._1).sortBy(_.dayOfMonth.getOrElse(0))
                val postIds = newestPostIds(perDay.flatMap(_._2), maxPostIdsPerLabel)
                val (name, countryCodes) = TakedownLabels.splitForServing(label)
                UthPostLabelAggregate(
                  Some(name),
                  Some(days),
                  countryCodes,
                  postIds = Option(postIds).filter(_.nonEmpty))
            }
            .toList
            .sortBy(a => (a.label.getOrElse(""), a.countryCodes.getOrElse(Nil).mkString(",")))

          val accountLabelAgg = accountOpt
            .getOrElse(Nil)
            .groupBy { case (label, _) => label }
            .map {
              case (label, rows) =>
                val (name, countryCodes) = TakedownLabels.splitForServing(label)
                UthAccountLabelAggregate(
                  Some(name),
                  Some(rows.map(_._2).distinct.sorted),
                  countryCodes
                )
            }
            .toList
            .sortBy(a => (a.label.getOrElse(""), a.countryCodes.getOrElse(Nil).mkString(",")))

          val (firstCov, lastCov) =
            monthCoverageYyyymmdd(month, lookbackStartYyyymmdd, asOfDay)
          val epoch = publishEpochOverrides.getOrElse(month, monthPublishEpoch)
          val coverage = UthDaysIncluded(
            snapshotEpoch = Some(epoch),
            firstCoveredDay = Some(firstCov),
            completeThroughDay = Some(lastCov),
            generatedAtMs = Some(epoch),
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
  postTakedownDataFirstDay: Option[Int],
  accountTakedownDataFirstDay: Option[Int],
  writeShards: Int,
  outputPath: String,
  tsvOnly: Boolean = false,
  publishEpochOverrides: Map[Int, Long] = Map.empty,
  maxPostIdsPerLabel: Int = UnderTheHoodCommon.DefaultMaxPostIdsPerLabel,
  dailyDataEnvironment: Environment = Environment.Prod)

object UthUserMonthMhConfig {

  private[under_the_hood] def parseEpochOverrides(args: Args): Map[Int, Long] =
    args
      .list("publishEpochOverride")
      .flatMap(_.split(","))
      .map(_.trim)
      .filter(_.nonEmpty)
      .map { kv =>
        kv.split("=", 2) match {
          case Array(m, e) => m.trim.toInt -> e.trim.toLong
          case _ =>
            throw new IllegalArgumentException(
              s"--publishEpochOverride expects yyyymm=epochMs; got '$kv'")
        }
      }
      .toMap

  def fromArgs(args: Args): UthUserMonthMhConfig =
    UthUserMonthMhConfig(
      testUserIds = UnderTheHoodCommon.parseUserIds(args),
      reducers = args.int("reducers", 20000),
      dailyLookbackDays =
        UnderTheHoodCommon.preferredInt(args, "dailyLookbackDays", "lookbackDays", 90),
      postObservationDays =
        UnderTheHoodCommon.preferredInt(args, "postObservationDays", "observationDays", 7),
      dailyDataFirstDay = args.int("dailyDataFirstDay", 20260701),
      postTakedownDataFirstDay = args.optional("postTakedownDataFirstDay").map(_.toInt),
      accountTakedownDataFirstDay = args.optional("accountTakedownDataFirstDay").map(_.toInt),
      writeShards = {
        val n = args.int("writeShards", 50)
        require(n > 0, s"--writeShards must be > 0; got $n")
        n
      },
      outputPath = args.optional("outputPath").getOrElse("/user/<hadoop-role>/under_the_hood"),
      tsvOnly = args.boolean("tsvOnly"),
      publishEpochOverrides = parseEpochOverrides(args),
      maxPostIdsPerLabel =
        args.int("maxPostIdsPerLabel", UnderTheHoodCommon.DefaultMaxPostIdsPerLabel),
      dailyDataEnvironment = args
        .optional("dailyDataEnvironment")
        .map(name =>
          Environment.valueOf(name).getOrElse(sys.error(s"unknown DAL environment: $name")))
        .getOrElse(Environment.Prod)
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
