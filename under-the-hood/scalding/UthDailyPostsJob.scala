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
import com.twitter.spam.rtf.thriftscala.ActionType
import com.twitter.spam.rtf.thriftscala.FlattenedSafetyLabel
import com.twitter.spam.rtf.thriftscala.TweetSafetyLabelEvent
import com.twitter.tweetsource.common.thriftscala.UnhydratedFlatTweet
import com.twitter.visibility.tweet_labels.TweetSafetyLabelsScalaDataset
import com.twitter.visibility.under_the_hood.thriftscala.UthDailyEligiblePost
import com.twitter.visibility.under_the_hood.thriftscala.UthDailyPostLabel
import java.util.TimeZone
import tweetsource.common.UnhydratedFlatScalaDataset
import twadoop_config.configuration.log_categories.group.visibility.TweetSafetyLabelEventsScalaDataset

class UthDailyPostsApp {
  import UnderTheHoodCommon._
  import UthDailyPostsApp._

  implicit val tz: TimeZone = DateOps.UTC

  def runOnDateRange(dateRange: DateRange, config: UthDailyPostsConfig): Execution[Unit] = {
    val dayStartMs = dateRange.start.timestamp
    val dayEndMs = dateRange.end.timestamp + 1L
    require(dayStartMs % DayMs == 0, s"batch day must start at UTC midnight; got $dayStartMs")
    require(
      dayEndMs - dayStartMs == DayMs,
      s"UthDailyPosts expects a single UTC day DateRange; got [$dayStartMs, $dayEndMs). Loop --date for multi-day."
    )
    require(
      !(config.dateRangeFilteredSnapshotPath.nonEmpty && !config.snapshotUnion),
      "--dateRangeFilteredSnapshotPath requires snapshot gap-fill (do not pass --noSnapshotUnion)"
    )

    val asOfDay = yyyymmdd(dayStartMs)
    val observationMs = config.postObservationDays * DayMs
    val lookbackStartMs = dayStartMs - config.postObservationDays * DayMs
    val tweetAndEventRange = DateRange(RichDate(lookbackStartMs), dateRange.end)

    val rawTweets =
      DAL
        .read(UnhydratedFlatScalaDataset, tweetAndEventRange)
        .withColumns(
          Set("userId", "tweetId", "initial_tweet_id", "shareSourceTweetId", "nullcast") ++
            (if (config.tweetFlagLabels.nonEmpty) TweetFlagColumns else Set.empty))
        .toTypedPipe

    val posts = loadPosts(
      rawTweets,
      config.testUserIds,
      lookbackStartMs,
      dayEndMs
    )

    val labelEvents = loadLabelEvents(
      DAL.read(TweetSafetyLabelEventsScalaDataset, tweetAndEventRange).toTypedPipe,
      lookbackStartMs,
      dayEndMs
    )

    val tweetSafetyLabelSnapshotRows: TypedPipe[LabelRow] =
      if (!config.snapshotUnion) TypedPipe.empty
      else
        config.dateRangeFilteredSnapshotPath match {
          case Some(path) =>
            loadFilteredSnapshotRows(
              TypedPipe.from(TypedTsv[(Long, String, String, String)](path)),
              lookbackStartMs,
              dayEndMs
            )
          case None =>
            loadTweetSafetyLabelSnapshotRows(
              DAL
                .readMostRecentSnapshotNoOlderThan(
                  TweetSafetyLabelsScalaDataset,
                  Days(config.snapshotMaxAgeDays)
                )
                .withColumns(Set("tweet_id", "label_type", "created_at_msec", "expires_at_msec"))
                .toTypedPipe,
              lookbackStartMs,
              dayEndMs
            )
        }

    val eligibleRows = countEligibleOnBatchDay(posts, dayStartMs, dayEndMs, config.reducers)
      .map {
        case (userId, day, count) =>
          UthDailyEligiblePost(Some(userId), Some(day), Some(count))
      }

    val labelRows =
      countLabelsAsOf(
        posts,
        labelEvents ++ tweetSafetyLabelSnapshotRows,
        dayEndMs,
        observationMs,
        config)
        .map {
          case (userId, authoredDay, label, carried, removed, postIds) =>
            val age = calendarDaysBetween(authoredDay, asOfDay)
            UthDailyPostLabel(
              userId = Some(userId),
              authoredYyyymmdd = Some(authoredDay),
              label = Some(label),
              carried = Some(carried),
              removed = Some(removed),
              asOfYyyymmdd = Some(asOfDay),
              observationAgeDays = Some(age),
              isFinal = Some(age >= config.postObservationDays),
              postObservationDays = Some(config.postObservationDays),
              postIds = Some(postIds)
            )
        }

    val tweetFlagRows =
      countTweetFlagsOnBatchDay(
        rawTweets,
        config.testUserIds,
        config.tweetFlagLabels,
        dayStartMs,
        dayEndMs,
        config.reducers)
        .map {
          case (userId, authoredDay, label, carried, postIds) =>
            UthDailyPostLabel(
              userId = Some(userId),
              authoredYyyymmdd = Some(authoredDay),
              label = Some(label),
              carried = Some(carried),
              removed = Some(0L),
              asOfYyyymmdd = Some(asOfDay),
              observationAgeDays = Some(0),
              isFinal = Some(true),
              postObservationDays = Some(config.postObservationDays),
              postIds = Some(postIds)
            )
        }

    val basePath = config.outputPath
    implicit val jobDr: DateRange = dateRange

    val eligibleWrite =
      eligibleRows
        .shard(config.writeShards)
        .writeDALExecution(
          UthDailyEligiblePostsScalaDataset,
          D.Daily,
          D.Suffix(s"$basePath/daily_eligible_posts"),
          D.Parquet
        )

    val labelsWrite =
      (labelRows ++ tweetFlagRows)
        .shard(config.writeShards)
        .writeDALExecution(
          UthDailyPostLabelsScalaDataset,
          D.Daily,
          D.Suffix(s"$basePath/daily_post_labels"),
          D.Parquet
        )

    Execution.zip(eligibleWrite, labelsWrite).unit
  }
}

object UthDailyPostsApp {
  import UnderTheHoodCommon._

  type LabelRow = (Long, (String, Long, Boolean, Long))

  val NsfwAdminStampedLabel = "NSFW_ADMIN_STAMPED"
  val TweetFlagColumns: Set[String] = Set("nsfwAdmin")

  private def tweetFlags(t: UnhydratedFlatTweet): Map[String, Boolean] =
    Map(
      NsfwAdminStampedLabel -> t.nsfwAdmin
    )

  private[under_the_hood] def countTweetFlagsOnBatchDay(
    tweets: TypedPipe[UnhydratedFlatTweet],
    testUserIds: Set[Long],
    flagLabels: Set[String],
    dayStartMs: Long,
    dayEndMs: Long,
    reducers: Int
  ): TypedPipe[(Long, Int, String, Long, Seq[Long])] =
    if (flagLabels.isEmpty) TypedPipe.empty
    else {
      val flagged = tweets.flatMap { t =>
        if (inScope(testUserIds, t.userId) && t.shareSourceTweetId.isEmpty && !t.nullcast) {
          val logicalId = t.initialTweetId.getOrElse(t.tweetId)
          val createdMs = snowflakeCreatedMs(logicalId)
          if (createdMs >= dayStartMs && createdMs < dayEndMs) {
            tweetFlags(t).collect {
              case (label, true) if flagLabels.contains(label) =>
                ((t.userId, yyyymmdd(createdMs), logicalId, label), 1)
            }
          } else Nil
        } else Nil
      }
      val distinctPosts = applyReducers(flagged.group, reducers).sum.keys
      applyReducers(
        distinctPosts.map {
          case (userId, day, logicalId, label) =>
            ((userId, day, label), UthPostIds.single(logicalId, 0L))
        }.group,
        reducers
      ).reduce(UthPostIds.merge).toTypedPipe.map {
        case ((userId, day, label), summary) =>
          (userId, day, label, summary.carried, summary.postIds)
      }
    }

  private[under_the_hood] def loadPosts(
    tweets: TypedPipe[UnhydratedFlatTweet],
    testUserIds: Set[Long],
    startMs: Long,
    endMs: Long
  ): TypedPipe[(Long, Long, Int, Long, Long)] =
    tweets.flatMap { tweet =>
      if (inScope(
          testUserIds,
          tweet.userId) && tweet.shareSourceTweetId.isEmpty && !tweet.nullcast) {
        val logicalId = tweet.initialTweetId.getOrElse(tweet.tweetId)
        val createdMs = snowflakeCreatedMs(logicalId)
        if (createdMs >= startMs && createdMs < endMs)
          Some((tweet.tweetId, tweet.userId, yyyymmdd(createdMs), logicalId, createdMs))
        else None
      } else None
    }

  private[under_the_hood] def countEligibleOnBatchDay(
    posts: TypedPipe[(Long, Long, Int, Long, Long)],
    dayStartMs: Long,
    dayEndMs: Long,
    reducers: Int
  ): TypedPipe[(Long, Int, Long)] = {
    val firstDay = yyyymmdd(dayStartMs)
    val lastDay = yyyymmdd(dayEndMs - 1L)
    val grouped = applyReducers(
      posts
        .collect {
          case (_, userId, day, logicalId, _) if day >= firstDay && day <= lastDay =>
            (userId, day, logicalId)
        }
        .distinct
        .map { case (userId, day, _) => ((userId, day), 1L) }
        .group,
      reducers
    )
    grouped.sum.toTypedPipe.map { case ((userId, day), count) => (userId, day, count) }
  }

  private[under_the_hood] def loadLabelEvents(
    events: TypedPipe[TweetSafetyLabelEvent],
    startMs: Long,
    endMs: Long
  ): TypedPipe[LabelRow] =
    events.flatMap { event =>
      val name = event.labelType.originalName
      val isApply = event.actionType == ActionType.Applied
      val isRemove = event.actionType == ActionType.Removed
      val eventMs = event.timestampMs.orElse(event.label.flatMap(_.createdAtMsec)).getOrElse(0L)
      if ((isApply || isRemove) && eventMs >= startMs && eventMs < endMs) {
        val expiresMs =
          if (isApply) event.label.flatMap(_.expiresAtMsec).getOrElse(Long.MaxValue)
          else Long.MaxValue
        Some((event.tweetId, (name, eventMs, isApply, expiresMs)))
      } else None
    }

  private[under_the_hood] def loadTweetSafetyLabelSnapshotRows(
    raw: TypedPipe[FlattenedSafetyLabel],
    lookbackStartMs: Long,
    dayEndMs: Long
  ): TypedPipe[LabelRow] =
    raw.flatMap { f =>
      val name = f.labelType.originalName
      val expires = f.expiresAtMsec.getOrElse(Long.MaxValue)
      if (!(expires > lookbackStartMs))
        None
      else
        f.createdAtMsec match {
          case Some(created) =>
            if (created < dayEndMs) Some((f.tweetId, (name, created, true, expires))) else None
          case None => Some((f.tweetId, (name, Long.MinValue, true, expires)))
        }
    }

  private[under_the_hood] def loadFilteredSnapshotRows(
    raw: TypedPipe[(Long, String, String, String)],
    lookbackStartMs: Long,
    dayEndMs: Long
  ): TypedPipe[LabelRow] =
    raw.flatMap {
      case (tweetId, labelName, createdStr, expiresStr) =>
        val expires =
          if (expiresStr == null || expiresStr.isEmpty) Long.MaxValue else expiresStr.toLong
        if (expires <= lookbackStartMs) None
        else if (createdStr == null || createdStr.isEmpty)
          Some((tweetId, (labelName, Long.MinValue, true, expires)))
        else {
          val created = createdStr.toLong
          if (created < dayEndMs) Some((tweetId, (labelName, created, true, expires))) else None
        }
    }

  private[under_the_hood] def actionInHorizon(
    eventMs: Long,
    isApply: Boolean,
    expiresMs: Long,
    createdMs: Long,
    deadline: Long
  ): Option[(Boolean, Long, Boolean, Long)] = {
    val inObservationHorizon =
      if (eventMs == Long.MinValue) true
      else eventMs >= createdMs && eventMs < deadline
    if (!inObservationHorizon) None
    else Some((isApply, eventMs, isApply, expiresMs))
  }

  private[under_the_hood] def removedAfterLastAction(
    lastIsApply: Boolean,
    lastExpiresMs: Long,
    createdMs: Long,
    deadline: Long
  ): Boolean =
    if (!lastIsApply) true
    else
      lastExpiresMs < Long.MaxValue &&
      lastExpiresMs >= createdMs &&
      lastExpiresMs < deadline

  private[under_the_hood] def countLabelsAsOf(
    posts: TypedPipe[(Long, Long, Int, Long, Long)],
    labelRows: TypedPipe[LabelRow],
    dayEndMs: Long,
    observationMs: Long,
    config: UthDailyPostsConfig
  ): TypedPipe[(Long, Int, String, Long, Long, Seq[Long])] = {
    val postByTweetId = posts.map {
      case (tweetId, userId, day, logicalId, createdMs) =>
        (tweetId, (logicalId, userId, day, createdMs))
    }.distinct

    val scopedRows =
      if (config.testUserIds.isEmpty) labelRows
      else {
        val postIds = postByTweetId.map { case (tweetId, _) => (tweetId, ()) }.distinct.group
        labelRows.hashJoin(postIds).map { case (tweetId, (row, _)) => (tweetId, row) }
      }

    val joined =
      if (config.reducers > 0) scopedRows.join(postByTweetId).withReducers(config.reducers)
      else scopedRows.join(postByTweetId)

    val inHorizon = joined.flatMap {
      case (_, ((label, eventMs, isApply, expiresMs), (logicalId, userId, day, createdMs))) =>
        val deadline = math.min(createdMs + observationMs, dayEndMs)
        actionInHorizon(eventMs, isApply, expiresMs, createdMs, deadline).map {
          case (everApply, ts, lastApply, exp) =>
            ((logicalId, userId, day, label, createdMs), (everApply, ts, lastApply, exp))
        }
    }

    val reduced = applyReducers(inHorizon.group, config.reducers).reduce { (a, b) =>
      val everApply = a._1 || b._1
      val last =
        if (a._2 != b._2) {
          if (a._2 > b._2) (a._2, a._3, a._4) else (b._2, b._3, b._4)
        } else if (a._3 || b._3) {
          val exp = if (a._3) a._4 else b._4
          (a._2, true, exp)
        } else (a._2, false, a._4)
      (everApply, last._1, last._2, last._3)
    }.toTypedPipe

    applyReducers(
      reduced.collect {
        case (
              (logicalId, userId, day, label, createdMs),
              (everApply, _, lastApply, lastExpires)
            ) if everApply =>
          val deadline = math.min(createdMs + observationMs, dayEndMs)
          val removed =
            if (removedAfterLastAction(lastApply, lastExpires, createdMs, deadline)) 1L else 0L
          ((userId, day, label), UthPostIds.single(logicalId, removed))
      }.group,
      config.reducers
    ).reduce(UthPostIds.merge).toTypedPipe.map {
      case ((userId, day, label), summary) =>
        (userId, day, label, summary.carried, summary.removed, summary.postIds)
    }
  }
}

case class UthDailyPostsConfig(
  testUserIds: Set[Long],
  reducers: Int,
  postObservationDays: Int,
  snapshotUnion: Boolean,
  snapshotMaxAgeDays: Int,
  dateRangeFilteredSnapshotPath: Option[String],
  writeShards: Int,
  outputPath: String,
  tweetFlagLabels: Set[String] = Set.empty)

object UthDailyPostsConfig {
  import UthDailyPostsApp._

  private val KnownTweetFlags =
    Map(
      "nsfwAdmin" -> NsfwAdminStampedLabel
    )

  def fromArgs(args: Args): UthDailyPostsConfig =
    UthDailyPostsConfig(
      testUserIds = UnderTheHoodCommon.parseUserIds(args),
      reducers = args.int("reducers", 50),
      postObservationDays =
        UnderTheHoodCommon.preferredInt(args, "postObservationDays", "observationDays", 7),
      snapshotUnion = !args.boolean("noSnapshotUnion"),
      snapshotMaxAgeDays = args.int("snapshotMaxAgeDays", 10),
      dateRangeFilteredSnapshotPath = args.optional("dateRangeFilteredSnapshotPath"),
      writeShards = {
        val n = args.int("writeShards", 50)
        require(n > 0, s"--writeShards must be > 0; got $n")
        n
      },
      outputPath = args.optional("outputPath").getOrElse("/user/<hadoop-role>/under_the_hood"),
      tweetFlagLabels = parseTweetFlags(args)
    )

  private[under_the_hood] def parseTweetFlags(args: Args): Set[String] = {
    val requested = args.list("tweetFlags").flatMap(_.split(",")).map(_.trim).filter(_.nonEmpty)
    if (requested == Seq("none")) Set.empty
    else if (requested.isEmpty) KnownTweetFlags.values.toSet
    else requested.flatMap(KnownTweetFlags.get).toSet
  }
}

object UthDailyPostsAdhoc extends UthDailyPostsApp with TwitterExecutionApp {
  override def job: Execution[Unit] = Execution.withArgs { args =>
    runOnDateRange(UnderTheHoodDates.resolve(args), UthDailyPostsConfig.fromArgs(args))
  }
}

object UthDailyPostsProd extends UthDailyPostsApp with TwitterScheduledExecutionApp {
  implicit val dp: DateParser = DateParser.default
  override def scheduledJob: Execution[Unit] = {
    val execArgs = AnalyticsBatchExecutionArgs(
      batchDesc = BatchDescription("uth_daily_posts_prod"),
      firstTime = BatchFirstTime(RichDate("2026-06-30")),
      batchIncrement = BatchIncrement(Days(1))
    )
    Execution.withArgs { args =>
      AnalyticsBatchExecution(execArgs) { dateRange =>
        runOnDateRange(dateRange, UthDailyPostsConfig.fromArgs(args))
      }
    }
  }
}
