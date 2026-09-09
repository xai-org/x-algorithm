package com.twitter.visibility.under_the_hood

import com.twitter.scalding.DateRange
import com.twitter.scalding.Days
import com.twitter.scalding.Execution
import com.twitter.scalding.RichDate
import com.twitter.scalding.TypedPipe
import com.twitter.scalding.TypedTsv
import com.twitter.scalding_internal.dalv2.DAL
import com.twitter.scalding_internal.dalv2.DALWrite._
import com.twitter.scalding_internal.job.TwitterExecutionApp
import com.twitter.visibility.tweet_labels.TweetSafetyLabelsScalaDataset
import com.twitter.visibility.under_the_hood.thriftscala.UthDailyEligiblePost
import com.twitter.visibility.under_the_hood.thriftscala.UthDailyPostLabel
import java.util.TimeZone
import com.twitter.scalding.DateOps
import tweetsource.common.UnhydratedFlatScalaDataset
import twadoop_config.configuration.log_categories.group.visibility.TweetSafetyLabelEventsScalaDataset

class UthDailyPostsBackfillApp {
  import UnderTheHoodCommon._
  import UthDailyPostsBackfillApp._

  implicit val tz: TimeZone = DateOps.UTC

  def runOnDateRange(dateRange: DateRange, config: UthDailyPostsConfig): Execution[Unit] = {
    val rangeStartMs = dateRange.start.timestamp
    val rangeEndMs = dateRange.end.timestamp + 1L
    require(
      rangeStartMs % DayMs == 0 && rangeEndMs % DayMs == 0,
      s"range must cover whole UTC days; got [$rangeStartMs, $rangeEndMs)"
    )
    require(
      !(config.dateRangeFilteredSnapshotPath.nonEmpty && !config.snapshotUnion),
      "--dateRangeFilteredSnapshotPath requires snapshot gap-fill (do not pass --noSnapshotUnion)"
    )

    val observationMs = config.postObservationDays * DayMs
    val lookbackStartMs = rangeStartMs - observationMs
    val readRange = DateRange(RichDate(lookbackStartMs), dateRange.end)

    val rawTweets =
      DAL
        .read(UnhydratedFlatScalaDataset, readRange)
        .withColumns(
          Set("userId", "tweetId", "initial_tweet_id", "shareSourceTweetId", "nullcast") ++
            (if (config.tweetFlagLabels.nonEmpty) UthDailyPostsApp.TweetFlagColumns
             else Set.empty))
        .toTypedPipe

    val posts = UthDailyPostsApp.loadPosts(
      rawTweets,
      config.testUserIds,
      lookbackStartMs,
      rangeEndMs
    )

    val labelEvents = UthDailyPostsApp.loadLabelEvents(
      DAL.read(TweetSafetyLabelEventsScalaDataset, readRange).toTypedPipe,
      lookbackStartMs,
      rangeEndMs
    )

    val snapshotRows: TypedPipe[UthDailyPostsApp.LabelRow] =
      if (!config.snapshotUnion) TypedPipe.empty
      else
        config.dateRangeFilteredSnapshotPath match {
          case Some(path) =>
            loadFilteredSnapshotRowsWithExpiry(
              TypedPipe.from(TypedTsv[(Long, String, String, String)](path)),
              lookbackStartMs,
              rangeEndMs
            )
          case None =>
            UthDailyPostsApp.loadTweetSafetyLabelSnapshotRows(
              DAL
                .readMostRecentSnapshotNoOlderThan(
                  TweetSafetyLabelsScalaDataset,
                  Days(config.snapshotMaxAgeDays)
                )
                .withColumns(Set("tweet_id", "label_type", "created_at_msec", "expires_at_msec"))
                .toTypedPipe,
              lookbackStartMs,
              rangeEndMs
            )
        }

    val eligibleRows = countEligiblePerDay(posts, rangeStartMs, rangeEndMs, config.reducers)
      .map {
        case (userId, day, count) =>
          UthDailyEligiblePost(Some(userId), Some(day), Some(count))
      }

    val labelRows =
      countLabelsPerAsOfDay(
        posts,
        labelEvents ++ snapshotRows,
        rangeStartMs,
        rangeEndMs,
        observationMs,
        config)
        .map {
          case (userId, authoredDay, label, asOfDay, carried, removed, postIds) =>
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
      UthDailyPostsApp
        .countTweetFlagsOnBatchDay(
          rawTweets,
          config.testUserIds,
          config.tweetFlagLabels,
          rangeStartMs,
          rangeEndMs,
          config.reducers)
        .map {
          case (userId, authoredDay, label, carried, postIds) =>
            UthDailyPostLabel(
              userId = Some(userId),
              authoredYyyymmdd = Some(authoredDay),
              label = Some(label),
              carried = Some(carried),
              removed = Some(0L),
              asOfYyyymmdd = Some(authoredDay),
              observationAgeDays = Some(0),
              isFinal = Some(true),
              postObservationDays = Some(config.postObservationDays),
              postIds = Some(postIds)
            )
        }

    val basePath = config.outputPath
    val asOfDayStarts: Seq[Long] =
      Iterator.iterate(rangeStartMs)(_ + DayMs).takeWhile(_ < rangeEndMs).toSeq

    Execution
      .zip(eligibleRows.forceToDiskExecution, (labelRows ++ tweetFlagRows).forceToDiskExecution)
      .flatMap {
        case (eligiblePipe, labelPipe) =>
          Execution
            .sequence(asOfDayStarts.map { dStartMs =>
              val day = yyyymmdd(dStartMs)
              implicit val partitionRange: DateRange =
                DateRange(RichDate(dStartMs), RichDate(dStartMs + DayMs - 1L))
              val eligibleWrite = eligiblePipe
                .filter(_.authoredYyyymmdd.contains(day))
                .shard(config.writeShards)
                .writeDALExecution(
                  UthDailyEligiblePostsScalaDataset,
                  D.Daily,
                  D.Suffix(s"$basePath/daily_eligible_posts"),
                  D.Parquet
                )
              val labelsWrite = labelPipe
                .filter(_.asOfYyyymmdd.contains(day))
                .shard(config.writeShards)
                .writeDALExecution(
                  UthDailyPostLabelsScalaDataset,
                  D.Daily,
                  D.Suffix(s"$basePath/daily_post_labels"),
                  D.Parquet
                )
              Execution.zip(eligibleWrite, labelsWrite).unit
            }).unit
      }
  }
}

object UthDailyPostsBackfillApp {
  import UnderTheHoodCommon._

  private[under_the_hood] def dayStartMs(ms: Long): Long = (ms / DayMs) * DayMs

  private[under_the_hood] def asOfDayStartsFor(
    createdMs: Long,
    rangeStartMs: Long,
    rangeEndMs: Long,
    observationMs: Long
  ): Seq[Long] = {
    val aStart = dayStartMs(createdMs)
    val first = math.max(aStart, rangeStartMs)
    val last = math.min(aStart + observationMs, rangeEndMs - DayMs)
    Iterator.iterate(first)(_ + DayMs).takeWhile(_ <= last).toSeq
  }

  private[under_the_hood] def loadFilteredSnapshotRowsWithExpiry(
    raw: TypedPipe[(Long, String, String, String)],
    rangeLookbackStartMs: Long,
    rangeEndMs: Long
  ): TypedPipe[UthDailyPostsApp.LabelRow] =
    raw.flatMap {
      case (tweetId, labelName, createdStr, expiresStr) =>
        val expires =
          if (expiresStr == null || expiresStr.isEmpty) Long.MaxValue else expiresStr.toLong
        if (expires <= rangeLookbackStartMs) None
        else if (createdStr == null || createdStr.isEmpty)
          Some((tweetId, (labelName, Long.MinValue, true, expires)))
        else {
          val created = createdStr.toLong
          if (created < rangeEndMs) Some((tweetId, (labelName, created, true, expires)))
          else None
        }
    }

  private[under_the_hood] def countEligiblePerDay(
    posts: TypedPipe[(Long, Long, Int, Long, Long)],
    rangeStartMs: Long,
    rangeEndMs: Long,
    reducers: Int
  ): TypedPipe[(Long, Int, Long)] = {
    val firstDay = yyyymmdd(rangeStartMs)
    val lastDay = yyyymmdd(rangeEndMs - 1L)
    val distinctPosts = applyReducers(
      posts.collect {
        case (_, userId, day, logicalId, _) if day >= firstDay && day <= lastDay =>
          ((userId, day, logicalId), 1)
      }.group,
      reducers
    ).sum.keys
    applyReducers(
      distinctPosts.map { case (userId, day, _) => ((userId, day), 1L) }.group,
      reducers
    ).sum.toTypedPipe.map { case ((userId, day), count) => (userId, day, count) }
  }

  private[under_the_hood] def countLabelsPerAsOfDay(
    posts: TypedPipe[(Long, Long, Int, Long, Long)],
    labelRows: TypedPipe[UthDailyPostsApp.LabelRow],
    rangeStartMs: Long,
    rangeEndMs: Long,
    observationMs: Long,
    config: UthDailyPostsConfig
  ): TypedPipe[(Long, Int, String, Int, Long, Long, Seq[Long])] = {
    val postByTweetId = applyReducers(
      posts.map {
        case (tweetId, userId, day, logicalId, createdMs) =>
          ((tweetId, (logicalId, userId, day, createdMs)), 1)
      }.group,
      config.reducers
    ).sum.keys

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
        asOfDayStartsFor(createdMs, rangeStartMs, rangeEndMs, observationMs).flatMap { dStartMs =>
          val dayEndMs = dStartMs + DayMs
          val deadline = math.min(createdMs + observationMs, dayEndMs)
          val lookbackStartMs = dStartMs - observationMs
          if (expiresMs <= lookbackStartMs) Nil
          else
            UthDailyPostsApp
              .actionInHorizon(eventMs, isApply, expiresMs, createdMs, deadline)
              .map {
                case (everApply, ts, lastApply, exp) =>
                  (
                    (logicalId, userId, day, label, yyyymmdd(dStartMs), createdMs),
                    (everApply, ts, lastApply, exp)
                  )
              }
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
              (logicalId, userId, day, label, asOfDay, createdMs),
              (everApply, _, lastApply, lastExpires)
            ) if everApply =>
          val deadline = math.min(createdMs + observationMs, yyyymmddToMs(asOfDay) + DayMs)
          val removed =
            if (UthDailyPostsApp
                .removedAfterLastAction(lastApply, lastExpires, createdMs, deadline)) 1L
            else 0L
          ((userId, day, label, asOfDay), UthPostIds.single(logicalId, removed))
      }.group,
      config.reducers
    ).reduce(UthPostIds.merge).toTypedPipe.map {
      case ((userId, day, label, asOfDay), summary) =>
        (
          userId,
          day,
          label,
          asOfDay,
          summary.carried,
          summary.removed,
          summary.postIds
        )
    }
  }
}

object UthDailyPostsBackfillAdhoc extends UthDailyPostsBackfillApp with TwitterExecutionApp {
  override def job: Execution[Unit] = Execution.withArgs { args =>
    runOnDateRange(UnderTheHoodDates.resolve(args), UthDailyPostsConfig.fromArgs(args))
  }
}
