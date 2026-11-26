package com.twitter.simclusters_v2.summingbird.storm

import com.twitter.simclusters_v2.summingbird.common.Configs
import com.twitter.simclusters_v2.summingbird.common.Implicits
import com.twitter.simclusters_v2.summingbird.common.StatsUtil
import com.twitter.simclusters_v2.thriftscala._
import com.twitter.snowflake.id.SnowflakeId
import com.twitter.summingbird._
import com.twitter.summingbird.option.JobId
import com.twitter.simclusters_v2.common.TweetId
import com.twitter.simclusters_v2.common.UserId
import com.twitter.unified_user_actions.thriftscala.ActionType
import com.twitter.unified_user_actions.thriftscala.Item.TweetInfo
import com.twitter.unified_user_actions.thriftscala.UnifiedUserAction
import com.twitter.unified_user_actions.thriftscala.UserIdentifier

object TwhinTweetJob {

  import Implicits._
  import StatsUtil._

  object NodeName {
    final val UUASourceNodeName: String = "UUASource"

    final val TweetUserEmbeddingFlatMapNodeName: String = "TweetUserEmbeddingFlatMap"
    final val TweetEmbeddingSummerNodeName: String = "TweetEmbeddingSummer"
    final val KeyedTweetEmbeddingMapNodeName: String = "KeyedTweetEmbeddingMap"
    final val KeyedPersistentTwhinTweetEmbeddingWriteNodeName: String =
      "KeyedPersistentTwhinTweetEmbeddingWrite"

    final val VideoUserEmbeddingFlatMapNodeName: String = "VideoUserEmbeddingFlatMap"
    final val VideoEmbeddingSummerNodeName: String = "VideoEmbeddingSummer"
    final val KeyedVideoEmbeddingMapNodeName: String = "KeyedVideoEmbeddingMap"
    final val KeyedPersistentTwhinVideoEmbeddingWriteNodeName: String =
      "KeyedPersistentTwhinVideoEmbeddingWrite"
  }

  def generate[P <: Platform[P]](
    uuaEventSource: Producer[P, UnifiedUserAction],
    userEmbeddingService: P#Service[UserId, TwhinTweetEmbedding],
    tweetEmbeddingStore: P#Store[TweetId, PersistentTwhinTweetEmbedding],
    videoEmbeddingStore: P#Store[TweetId, PersistentTwhinTweetEmbedding],
    tweetEmbeddingKafkaPipe: P#Sink[KeyedPersistentTwhinTweetEmbedding],
    videoEmbeddingKafkaPipe: P#Sink[KeyedPersistentTwhinTweetEmbedding]
  )(
    implicit jobId: JobId
  ): TailProducer[P, Any] = {

    val uuaSource = uuaEventSource.name(NodeName.UUASourceNodeName)

    val favOrVideoQualityViewEvents = uuaSource.filter { uua =>
      uua.actionType == ActionType.ServerTweetFav ||
      uua.actionType == ActionType.ClientTweetVideoQualityView
    }

    val qualifiedFavEvents = favOrVideoQualityViewEvents
      .collect {
        case UnifiedUserAction(
              UserIdentifier(Some(userId), _, _, _, _, _, _, _, _, _, _, _, _, _, _),
              TweetInfo(tweetInfo),
              ActionType.ServerTweetFav,
              eventMetadata,
              _,
              _)
            if tweetInfo.actionTweetAuthorInfo.exists(!_.authorId.contains(userId))
              && !isTweetTooOld(tweetInfo.actionTweetId) =>
          (
            userId,
            (
              tweetInfo.actionTweetId,
              tweetInfo.actionTweetAuthorInfo.flatMap(_.authorId),
              eventMetadata.receivedTimestampMs))
      }
      .observe("num_qualified_favorite_events")

    val tweetEmbeddingPipe = qualifiedFavEvents
      .leftJoin(userEmbeddingService)
      .collect {
        case (_, ((tweetId, authorId, updatedAt), Some(embedding))) =>
          tweetId ->
            PersistentTwhinTweetEmbedding(embedding, 1, updatedAt, authorId)
      }
      .observe("user_embedding_fetch")
      .name(NodeName.TweetUserEmbeddingFlatMapNodeName)
      .sumByKey(tweetEmbeddingStore)
      .name(NodeName.TweetEmbeddingSummerNodeName)
      .map {
        case (tweetId, (existingEmbedding, delta)) =>
          val newEmbedding = persistentTwhinTweetEmbeddingMonoid.plus(
            existingEmbedding.getOrElse(persistentTwhinTweetEmbeddingMonoid.zero),
            delta)
          KeyedPersistentTwhinTweetEmbedding(tweetId, newEmbedding)
      }
      .name(NodeName.KeyedTweetEmbeddingMapNodeName)
      .write(tweetEmbeddingKafkaPipe)
      .name(NodeName.KeyedPersistentTwhinTweetEmbeddingWriteNodeName)

    val qualifiedVideoWatchEvents = favOrVideoQualityViewEvents
      .collect {
        case UnifiedUserAction(
              UserIdentifier(Some(userId), _, _, _, _, _, _, _, _, _, _, _, _, _, _),
              TweetInfo(tweetInfo),
              ActionType.ClientTweetVideoQualityView,
              eventMetadata,
              _,
              _)
            if tweetInfo.actionTweetAuthorInfo.exists(!_.authorId.contains(userId))
              && !isTweetTooOld(tweetInfo.actionTweetId) =>
          (
            userId,
            (
              tweetInfo.actionTweetId,
              tweetInfo.actionTweetAuthorInfo.flatMap(_.authorId),
              eventMetadata.receivedTimestampMs))
      }
      .observe("num_qualified_video_watch_events")

    val videoEmbeddingPipe = qualifiedVideoWatchEvents
      .leftJoin(userEmbeddingService)
      .collect {
        case (_, ((tweetId, authorId, updatedAt), Some(embedding))) =>
          tweetId ->
            PersistentTwhinTweetEmbedding(embedding, 1, updatedAt, authorId)
      }
      .observe("user_embedding_fetch")
      .name(NodeName.VideoUserEmbeddingFlatMapNodeName)
      .sumByKey(videoEmbeddingStore)
      .name(NodeName.VideoEmbeddingSummerNodeName)
      .map {
        case (tweetId, (existingEmbedding, delta)) =>
          val newEmbedding = persistentTwhinTweetEmbeddingMonoid.plus(
            existingEmbedding.getOrElse(persistentTwhinTweetEmbeddingMonoid.zero),
            delta)
          KeyedPersistentTwhinTweetEmbedding(tweetId, newEmbedding)
      }
      .name(NodeName.KeyedVideoEmbeddingMapNodeName)
      .write(videoEmbeddingKafkaPipe)
      .name(NodeName.KeyedPersistentTwhinVideoEmbeddingWriteNodeName)

    tweetEmbeddingPipe.also(videoEmbeddingPipe)
  }

  private def isTweetTooOld(tweetId: TweetId): Boolean = {
    SnowflakeId.unixTimeMillisOptFromId(tweetId).exists { millis =>
      System.currentTimeMillis() - millis >= Configs.OldestTweetFavEventTimeInMillis
    }
  }

}
