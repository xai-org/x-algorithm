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
import com.twitter.scalding_internal.dalv2.DAL
import com.twitter.scalding_internal.dalv2.DALWrite._
import com.twitter.scalding_internal.job.TwitterExecutionApp
import com.twitter.scalding_internal.job.analytics_batch._
import com.twitter.visibility.under_the_hood.thriftscala.UthDailyPostLabel
import java.util.TimeZone
import tweetsource.common.UnhydratedFlatScalaDataset
import twadoop_config.configuration.log_categories.group.tweetypie.TweetEventsScalaDataset
import twadoop_config.configuration.log_categories.group.useng.AuditServiceDmcaTakedownsScalaDataset
import twadoop_config.configuration.log_categories.group.useng.AuditServicePctdActionsScalaDataset

class UthDailyPostTakedownLabelsApp {
  import UnderTheHoodCommon._
  import UthDailyPostTakedownLabelsApp._

  implicit val tz: TimeZone = DateOps.UTC

  def runOnDateRange(
    dateRange: DateRange,
    config: UthDailyPostTakedownLabelsConfig
  ): Execution[Unit] = {
    val rangeStartMs = dateRange.start.timestamp
    val rangeEndMs = dateRange.end.timestamp + 1L
    require(
      rangeStartMs % DayMs == 0 && rangeEndMs % DayMs == 0,
      s"range must cover whole UTC days; got [$rangeStartMs, $rangeEndMs)"
    )
    val asOfDayStarts: Seq[Long] =
      Iterator.iterate(rangeStartMs)(_ + DayMs).takeWhile(_ < rangeEndMs).toSeq
    val observationMs = config.observationDays * DayMs

    val logRange = DateRange(
      RichDate(rangeStartMs - observationMs - DayMs),
      RichDate(rangeEndMs - 1L)
    )
    val auditDeltas: TypedPipe[TakedownObservation] =
      DAL
        .read(AuditServicePctdActionsScalaDataset, logRange)
        .toTypedPipe
        .flatMap(TakedownDeltas.fromPctdAction(_, config.testUserIds)) ++
        DAL
          .read(AuditServiceDmcaTakedownsScalaDataset, logRange)
          .toTypedPipe
          .flatMap(TakedownDeltas.fromDmcaAudit(_, config.testUserIds))

    val logDays: Seq[Int] = Iterator
      .iterate(logRange.start.timestamp)(_ + DayMs)
      .takeWhile(_ <= logRange.end.timestamp)
      .map(yyyymmdd)
      .toSeq
    val basePath = config.outputPath
    def tweetEventsDir(day: Int): String = s"$basePath/$TweetEventsDir/$day"

    val extractMissing: Execution[Unit] =
      if (!config.readTweetEvents) Execution.unit
      else
        Execution.getMode.flatMap { mode =>
          val missing =
            logDays.filterNot(day => mode.fileExists(s"${tweetEventsDir(day)}/_SUCCESS"))
          Execution
            .sequence(missing.map { day =>
              val dayStartMs = yyyymmddToMs(day)
              DAL
                .read(
                  TweetEventsScalaDataset,
                  DateRange(RichDate(dayStartMs), RichDate(dayStartMs + DayMs - 1L)))
                .toTypedPipe
                .flatMap(TakedownStates.fromTweetEvent(_, testUserIds = Set.empty))
                .map(TakedownStates.toTsvRow)
                .distinct
                .shard(2)
                .writeExecution(TypedTsv[TakedownStates.TsvRow](tweetEventsDir(day)))
            })
            .unit
        }

    def tweetEventStates: TypedPipe[TakedownObservation] =
      if (!config.readTweetEvents) TypedPipe.empty
      else
        logDays
          .map { day =>
            TypedPipe
              .from(TypedTsv[TakedownStates.TsvRow](tweetEventsDir(day)))
              .map(TakedownStates.fromTsvRow)
              .filter(st => inScope(config.testUserIds, st.userId))
              .map(st => st: TakedownObservation)
          }
          .reduce(_ ++ _)

    val eligiblePosts: TypedPipe[(Long, (Long, Long))] =
      UthDailyPostsApp
        .loadPosts(
          DAL
            .read(
              UnhydratedFlatScalaDataset,
              DateRange(RichDate(rangeStartMs - observationMs), RichDate(rangeEndMs - 1L)))
            .withColumns(
              Set("userId", "tweetId", "initial_tweet_id", "shareSourceTweetId", "nullcast"))
            .toTypedPipe,
          config.testUserIds,
          rangeStartMs - observationMs,
          rangeEndMs
        )
        .map { case (tweetId, userId, _, logicalId, _) => (tweetId, (userId, logicalId)) }

    def observations: TypedPipe[TakedownObservation] =
      toLogicalPosts(
        (auditDeltas ++ tweetEventStates)
          .map(_.toTsvRow)
          .distinct
          .flatMap(TakedownObservation.fromTsvRow),
        eligiblePosts,
        config.reducers
      )

    extractMissing.flatMap(_ => observations.forceToDiskExecution).flatMap { cached =>
      Execution
        .sequence(asOfDayStarts.map { dayStartMs =>
          val dayEndMs = dayStartMs + DayMs
          val asOfDay = yyyymmdd(dayStartMs)
          implicit val partitionRange: DateRange =
            DateRange(RichDate(dayStartMs), RichDate(dayEndMs - 1L))

          val postRows = labelRowsAsOf(cached, dayStartMs, dayEndMs, observationMs, config.reducers)

          val labelRows = postRows
            .map {
              case (userId, authoredDay, tweetId, label, carried, removed, _, _) =>
                val keepId = carried > 0 &&
                  TakedownLabels.parseLabel(label).exists {
                    case (t, _) => config.postIdLabels.contains(t.name)
                  }
                (
                  (userId, authoredDay, label),
                  (carried.toLong, removed.toLong, if (keepId) Seq(tweetId) else Seq.empty[Long]))
            }
            .group
            .reduce { (a, b) =>
              (a._1 + b._1, a._2 + b._2, newestPostIds(a._3 ++ b._3, config.maxPostIdsPerLabel))
            }
            .toTypedPipe
            .map {
              case ((userId, authoredDay, label), (carried, removed, postIds)) =>
                val age = calendarDaysBetween(authoredDay, asOfDay)
                UthDailyPostLabel(
                  userId = Some(userId),
                  authoredYyyymmdd = Some(authoredDay),
                  label = Some(label),
                  carried = Some(carried),
                  removed = Some(removed),
                  asOfYyyymmdd = Some(asOfDay),
                  observationAgeDays = Some(age),
                  isFinal = Some(age >= config.observationDays),
                  postObservationDays = Some(config.observationDays),
                  postIds = Option(postIds).filter(_.nonEmpty)
                )
            }
            .shard(config.writeShards)

          labelRows
            .writeDALExecution(
              UthDailyPostTakedownLabelsScalaDataset,
              D.Daily,
              D.Suffix(s"$basePath/daily_post_takedown_labels"),
              D.Parquet
            )
            .zip(postRows
              .map {
                case (userId, authoredDay, tweetId, label, carried, removed, first, withheld) =>
                  (asOfDay, userId, tweetId, authoredDay, label, carried, removed, first, withheld)
              }
              .shard(config.writeShards)
              .writeExecution(
                TypedTsv[PostTsvRow](s"$basePath/daily_post_takedown_labels_posts_tsv/$asOfDay")))
            .unit
        }).unit
    }
  }
}

object UthDailyPostTakedownLabelsApp {
  import UnderTheHoodCommon._

  val TweetEventsDir = "tweet_takedown_events"

  type PostTsvRow = (Int, Long, Long, Int, String, Int, Int, Long, Boolean)

  private[under_the_hood] def toLogicalPosts(
    observations: TypedPipe[TakedownObservation],
    eligiblePosts: TypedPipe[(Long, (Long, Long))],
    reducers: Int
  ): TypedPipe[TakedownObservation] =
    applyReducers(observations.groupBy(_.tweetId), reducers)
      .join(eligiblePosts.group)
      .values
      .map { case (o, (userId, logicalId)) => o.toTsvRow.copy(_1 = userId, _2 = logicalId) }
      .distinct
      .flatMap(TakedownObservation.fromTsvRow)

  private[under_the_hood] def labelRowsAsOf(
    observations: TypedPipe[TakedownObservation],
    dayStartMs: Long,
    dayEndMs: Long,
    observationMs: Long,
    reducers: Int
  ): TypedPipe[(Long, Int, Long, String, Int, Int, Long, Boolean)] = {
    val inHorizon = observations.filter { o =>
      val createdMs = snowflakeCreatedMs(o.tweetId)
      createdMs >= dayStartMs - observationMs && createdMs < dayEndMs &&
      o.eventMs < math.min(createdMs + observationMs, dayEndMs)
    }
    applyReducers(inHorizon.groupBy(o => (o.userId, o.tweetId)), reducers).toList.flatMap {
      case ((userId, tweetId), postObservations) =>
        val authoredDay = yyyymmdd(snowflakeCreatedMs(tweetId))
        TakedownReplay.postLabels(TakedownReplay.effectiveStates(postObservations)).map { pl =>
          (
            userId,
            authoredDay,
            tweetId,
            pl.label,
            pl.carried,
            pl.removed,
            pl.firstWithheldMs.getOrElse(0L),
            pl.withheldAtAsOf)
        }
    }
  }
}

case class UthDailyPostTakedownLabelsConfig(
  testUserIds: Set[Long],
  observationDays: Int,
  reducers: Int,
  writeShards: Int,
  outputPath: String,
  readTweetEvents: Boolean = true,
  postIdLabels: Set[String] = Set.empty,
  maxPostIdsPerLabel: Int = UnderTheHoodCommon.DefaultMaxPostIdsPerLabel)

object UthDailyPostTakedownLabelsConfig {
  def fromArgs(args: Args): UthDailyPostTakedownLabelsConfig =
    UthDailyPostTakedownLabelsConfig(
      testUserIds = UnderTheHoodCommon.parseUserIds(args),
      observationDays = {
        val n = UnderTheHoodCommon.preferredInt(args, "postObservationDays", "observationDays", 7)
        require(n > 0, s"--postObservationDays must be > 0; got $n")
        n
      },
      reducers = args.int("reducers", 50),
      writeShards = {
        val n = args.int("writeShards", 4)
        require(n > 0, s"--writeShards must be > 0; got $n")
        n
      },
      outputPath = args.optional("outputPath").getOrElse("/user/<hadoop-role>/under_the_hood"),
      readTweetEvents = !args.boolean("noTweetEvents"),
      postIdLabels =
        UnderTheHoodCommon.parsePostIdLabels(args, UthPostIdLabels.postTakedownReasons),
      maxPostIdsPerLabel =
        args.int("maxPostIdsPerLabel", UnderTheHoodCommon.DefaultMaxPostIdsPerLabel)
    )
}

object UthDailyPostTakedownLabelsAdhoc
    extends UthDailyPostTakedownLabelsApp
    with TwitterExecutionApp {
  override def job: Execution[Unit] = Execution.withArgs { args =>
    runOnDateRange(UnderTheHoodDates.resolve(args), UthDailyPostTakedownLabelsConfig.fromArgs(args))
  }
}

object UthDailyPostTakedownLabelsProd
    extends UthDailyPostTakedownLabelsApp
    with TwitterScheduledExecutionApp {
  implicit val dp: DateParser = DateParser.default
  override def scheduledJob: Execution[Unit] = {
    val execArgs = AnalyticsBatchExecutionArgs(
      batchDesc = BatchDescription("uth_daily_post_takedown_labels_prod"),
      firstTime = BatchFirstTime(RichDate("2026-08-01")),
      batchIncrement = BatchIncrement(Days(1))
    )
    Execution.withArgs { args =>
      AnalyticsBatchExecution(execArgs) { dateRange =>
        runOnDateRange(dateRange, UthDailyPostTakedownLabelsConfig.fromArgs(args))
      }
    }
  }
}
