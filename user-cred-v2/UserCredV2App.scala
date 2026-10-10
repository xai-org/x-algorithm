package com.twitter.health.platform_manipulation.user_cred_v2

import com.twitter.scalding._
import com.twitter.scalding_internal.dalv2.DAL
import com.twitter.scalding_internal.dalv2.DALWrite._
import com.twitter.scalding_internal.dalv2.remote_access.ExplicitLocation
import com.twitter.scalding_internal.dalv2.remote_access.ProcAtla
import com.twitter.scalding_internal.job.TwitterExecutionApp
import com.twitter.scalding_internal.job.analytics_batch._
import com.twitter.scalding_internal.multiformat.format.keyval.KeyVal
import com.twitter.usersource.snapshot.flat.UsersourceFlatScalaDataset
import com.twitter.iesource.processing.events.batch.ServerEngagementsScalaDataset
import com.twitter.investigations.account_expansion.AccountExpansionInvestigationsScalaDataset
import com.twitter.util.logging.Logging
import graphstore.common.FlockFollowsJavaDataset
import java.util.TimeZone

class UserCredV2App {
  import UserCredV2App._

  def runOnDateRange(dateRange: DateRange, config: UserCredV2Config): Execution[Unit] = {
    val validUserInfoPipe = readValidUserInfoPipe(dateRange)

    val followPipe = DAL
      .readMostRecentSnapshot(FlockFollowsJavaDataset, dateRange)
      .withRemoteReadPolicy(ExplicitLocation(ProcAtla))
      .toTypedPipe
      .flatMap(Edge.fromFlockEdge)

    val allEdges = filterLinkedUserEdges(followPipe)

    val engagementPipe = filterLinkedUserEdges(readEngagementEdges(dateRange))

    val outputUserMassExec = pageRankMain(
      validUserInfoPipe,
      allEdges,
      engagementPipe,
      config,
    )

    outputUserMassExec.flatMap { outputUserMassPipe =>
      val userCredPipe = outputUserMassPipe.map { userMass =>
        UserCredV2.fromMass(userMass.id, userMass.mass)
      }
      gateAndPublish(userCredPipe, config, dateRange)
    }
  }
}

object UserCredV2App extends Logging {
  implicit val tz: TimeZone = DateOps.UTC
  implicit val dp: DateParser = DateParser.default

  private def readValidUserInfoPipe(dateRange: DateRange): TypedPipe[ValidUserInfo] = {
    val readColumns = Set(
      "deactivated",
      "erased",
      "id",
      "restricted",
      "suspended",
      "user_state",
    )

    DAL
      .readMostRecentSnapshot(UsersourceFlatScalaDataset, dateRange)
      .withRemoteReadPolicy(ExplicitLocation(ProcAtla))
      .withColumns(readColumns)
      .toTypedPipe
      .flatMap(ValidUserInfo.fromFlatUser)
  }

  import com.twitter.iesource.thriftscala.InteractionType

  private val EngagementTypes: Set[InteractionType] = Set(
    InteractionType.Favorite,
    InteractionType.Retweet,
  )

  private def readEngagementEdges(dateRange: DateRange): TypedPipe[Edge] = {
    val engagementDateRange =
      DateRange(dateRange.end - Days(UserCredV2Config.EngagementWindowDays), dateRange.end)
    DAL
      .read(ServerEngagementsScalaDataset, engagementDateRange)
      .withRemoteReadPolicy(ExplicitLocation(ProcAtla))
      .toTypedPipe
      .flatMap { event =>
        for {
          interactionType <- event.interactionType
          if EngagementTypes.contains(interactionType)
          authorUserId <- event.tweetAuthorUserId
          if event.engagingUserId > 0L
          if event.engagingUserId != authorUserId
        } yield Edge(sourceId = event.engagingUserId, destinationId = authorUserId)
      }
  }

  private[user_cred_v2] def readLinkedUserPairs(): TypedPipe[(Long, Long)] = {
    DAL
      .readMostRecentSnapshotNoOlderThan(AccountExpansionInvestigationsScalaDataset, Days(7))
      .withRemoteReadPolicy(ExplicitLocation(ProcAtla))
      .toTypedPipe
      .map(row => (row.userId, row.linkedUserId))
  }

  private[user_cred_v2] def filterLinkedUserEdges(
    edges: TypedPipe[Edge],
  ): TypedPipe[Edge] = {
    val linkedPairs = readLinkedUserPairs().map(p => (p, ())).group
    edges
      .groupBy(e => (e.sourceId, e.destinationId))
      .leftJoin(linkedPairs)
      .collect { case (_, (edge, None)) => edge }
  }

  private[user_cred_v2] def readPreviousMass(): TypedPipe[UserMass] = {
    DAL
      .readMostRecentSnapshotNoOlderThan(UserCredV2ScalaDataset, Days(7))
      .withRemoteReadPolicy(ExplicitLocation(ProcAtla))
      .toTypedPipe
      .map(uc => UserMass(uc.userId, uc.mass))
  }

  private[user_cred_v2] def joinPreviousMass(
    freshMassPipe: TypedPipe[UserMass],
    previousMassPipe: TypedPipe[UserMass],
  ): TypedPipe[UserMass] = {
    freshMassPipe
      .groupBy(_.id)
      .leftJoin(previousMassPipe.groupBy(_.id))
      .map {
        case (_, (_, Some(prevMass))) => prevMass
        case (_, (freshMass, None)) => freshMass
      }
  }

  private[user_cred_v2] def computeTeleportWeights(
    validUserInfoPipe: TypedPipe[ValidUserInfo],
    engagementPipe: TypedPipe[Edge],
    priorMassPipe: TypedPipe[UserMass],
    beta: Double,
  ): TypedPipe[UserMass] = {
    val engagementCounts = engagementPipe
      .map(e => ((e.sourceId, e.destinationId), 1L))
      .group
      .sum

    val totalCountsPerEngager = engagementPipe
      .map(e => (e.sourceId, 1L))
      .group
      .sum

    val rawEngagementWeights = engagementCounts
      .map { case ((engagerId, destinationId), count) => (engagerId, (destinationId, count)) }
      .group
      .join(totalCountsPerEngager)
      .join(priorMassPipe.groupBy(_.id))
      .map {
        case (_, (((destinationId, count), totalCount), engagerMass)) =>
          (destinationId, engagerMass.mass * count.toDouble / totalCount.toDouble)
      }
      .group
      .sum

    val normalizedEngagement = getNormalizedUserMassPipe(
      rawEngagementWeights.toTypedPipe.map { case (id, weight) => UserMass(id, weight) }
    )

    val neutralPriorEligibleUsers = Stat("neutral_teleport_prior_eligible_users")
    val neutralPriorNearZeroExcludedUsers =
      Stat("neutral_teleport_prior_near_zero_excluded_users")
    val normalizedNeutralPrior = getNormalizedUserMassPipe(
      validUserInfoPipe.flatMap { user =>
        val mass = neutralTeleportMass(user)
        if (mass.isDefined) neutralPriorEligibleUsers.inc()
        else neutralPriorNearZeroExcludedUsers.inc()
        mass
      }
    )

    val blended = normalizedNeutralPrior
      .map(u => (u.id, u.mass))
      .group
      .outerJoin(normalizedEngagement.groupBy(_.id).mapValues(_.mass))
      .map {
        case (id, (Some(neutralPrior), Some(engagement))) =>
          UserMass(id, (1.0 - beta) * neutralPrior + beta * engagement)
        case (id, (Some(neutralPrior), None)) =>
          UserMass(id, (1.0 - beta) * neutralPrior)
        case (id, (None, Some(engagement))) =>
          UserMass(id, beta * engagement)
        case (id, (None, None)) =>
          UserMass(id, 0.0)
      }

    getNormalizedUserMassPipe(blended)
  }

  /**
   * The uniform component of the teleport prior covers every graph-eligible user equally.
   * Verification and subscription state are deliberately not inputs to this decision.
   */
  private[user_cred_v2] def neutralTeleportMass(user: ValidUserInfo): Option[UserMass] = {
    if (user.isNearZero) None else Some(UserMass(user.id, 1.0))
  }

  private[user_cred_v2] def pageRankMain(
    validUserInfoPipe: TypedPipe[ValidUserInfo],
    edgePipe: TypedPipe[Edge],
    engagementPipe: TypedPipe[Edge],
    config: UserCredV2Config,
  ): Execution[TypedPipe[UserMass]] = {
    val freshMassPipe = getNormalizedUserMassPipe(
      validUserInfoPipe.map(u => UserMass.calcInitMass(u))
    )
    val priorMassPipe =
      if (config.firstSnapshot) freshMassPipe
      else readPreviousMass()

    val initialUserMassPipe =
      if (config.firstSnapshot) {
        info("First snapshot run: initializing mass uniformly")
        freshMassPipe
      } else {
        info("Warm-start run: reading previous mass from prior snapshot")
        getNormalizedUserMassPipe(joinPreviousMass(freshMassPipe, priorMassPipe))
      }

    info(
      s"Config: beta=${config.engagementTeleportBeta}, alpha=${config.jumpProbability}, " +
        s"maxIter=${config.maxIterations}, threshold=${config.diffThreshold}, " +
        s"engagementWindow=${UserCredV2Config.EngagementWindowDays}d")

    val teleportWeightsPipe = computeTeleportWeights(
      validUserInfoPipe,
      engagementPipe,
      priorMassPipe,
      config.engagementTeleportBeta
    )
    val userNodePipe = getPageRankGraph(validUserInfoPipe, edgePipe)

    Execution
      .zip(
        userNodePipe.forceToDiskExecution,
        initialUserMassPipe.forceToDiskExecution,
        teleportWeightsPipe.forceToDiskExecution,
      ).flatMap {
        case (userNodePipe, initialUserMassPipe, teleportWeightsPipe) =>
          runPageRank(
            userNodePipe = userNodePipe,
            initialUserMassPipe = initialUserMassPipe,
            teleportWeightsPipe = teleportWeightsPipe,
            config = config,
          )
      }
  }

  private[user_cred_v2] def runPageRank(
    userNodePipe: TypedPipe[UserNode],
    initialUserMassPipe: TypedPipe[UserMass],
    teleportWeightsPipe: TypedPipe[UserMass],
    config: UserCredV2Config,
  ): Execution[TypedPipe[UserMass]] = {
    info(s"Entering runPageRank, iteration ${config.curIteration}")
    doPageRank(userNodePipe, initialUserMassPipe, teleportWeightsPipe, config.jumpProbability)
      .flatMap {
        case (outputUserMassPipe, diff) =>
          if (diff > config.diffThreshold
            && config.curIteration + 1 < config.maxIterations) {
            val newConfig = config.nextIteration
            info(s"Diff after iteration ${config.curIteration}: $diff")
            runPageRank(
              userNodePipe = userNodePipe,
              initialUserMassPipe = outputUserMassPipe,
              teleportWeightsPipe = teleportWeightsPipe,
              config = newConfig,
            )
          } else {
            outputUserMassPipe.forceToDiskExecution
          }
      }
  }

  private[user_cred_v2] def getNormalizedUserMassPipe(
    userMassPipe: TypedPipe[UserMass]
  ): TypedPipe[UserMass] = {
    val safeMassPipe = userMassPipe.map { userMass =>
      userMass.copy(mass = sanitizeMass(userMass.mass))
    }
    val massSum = safeMassPipe.map(_.mass).sum

    safeMassPipe
      .cross(massSum)
      .map {
        case (UserMass(id, mass), totalMass) if totalMass > 0 =>
          UserMass(id, mass / totalMass)
        case (UserMass(id, _), _) =>
          UserMass(id, 0.0)
      }
  }

  private[user_cred_v2] def sanitizeMass(mass: Double): Double = {
    if (java.lang.Double.isFinite(mass) && mass > 0.0) mass else 0.0
  }

  private[user_cred_v2] def getPageRankGraph(
    validUserInfoPipe: TypedPipe[ValidUserInfo],
    edgePipe: TypedPipe[Edge],
  ): TypedPipe[UserNode] = {
    val allUserPipe = validUserInfoPipe.map(u => (u.id, u.isNearZero)).group

    val activeUserPipe = validUserInfoPipe.filter(!_.isNearZero).map(u => (u.id, true)).group

    val flippedEdgePipe = edgePipe.map(_.asTuple.swap)

    val validEdgePipe = flippedEdgePipe
      .join(allUserPipe)
      .map { case (destinationId, (sourceId, _)) => (sourceId, destinationId) }
      .join(activeUserPipe)
      .map { case (sourceId, (destinationId, _)) => Edge(sourceId, destinationId) }

    val validNeighbourListPipe = validEdgePipe
      .groupBy(_.sourceId)
      .mapValues(_.destinationId)
      .toList

    allUserPipe
      .leftJoin(validNeighbourListPipe)
      .map {
        case (sourceId, (isNearZero, destinationIds)) =>
          UserNode(
            id = sourceId,
            neighbourIds = destinationIds.getOrElse(List.empty),
            isNearZero = isNearZero,
          )
      }
  }

  private[user_cred_v2] def doPageRank(
    userNodePipe: TypedPipe[UserNode],
    inputUserMassPipe: TypedPipe[UserMass],
    teleportWeightsPipe: TypedPipe[UserMass],
    alpha: Double,
  ): Execution[(TypedPipe[UserMass], Double)] = {
    val groupedUserMassPipe = inputUserMassPipe.groupBy(_.id)

    val joinedUserPipe = userNodePipe
      .groupBy(_.id)
      .join(groupedUserMassPipe)

    val normalMassPipe = joinedUserPipe
      .flatMap {
        case (_, (userNode, userMass)) =>
          if (userNode.neighbourIds.nonEmpty) {
            val length = userNode.neighbourIds.length
            val distributeMass = userMass.mass / length
            userNode.neighbourIds.map(id => UserMass(id, distributeMass))
          } else {
            Seq.empty
          }
      }
      .groupBy(_.id)
      .mapValues(_.mass)
      .sum
      .map { case (id, mass) => UserMass(id, mass) }

    val totalMissingMassPipe = normalMassPipe
      .map(_.mass)
      .sum
      .map(distributedMass => 1.0 - distributedMass)

    val jumpMassPipe = userNodePipe
      .map(u => (u.id, ()))
      .group
      .leftJoin(teleportWeightsPipe.groupBy(_.id))
      .toTypedPipe
      .cross(totalMissingMassPipe)
      .map {
        case (((id, (_, teleportOpt)), missingMass)) =>
          val p = teleportOpt.map(_.mass).getOrElse(0.0)
          UserMass(id, p * alpha + p * missingMass * (1.0 - alpha))
      }

    val completeMassPipe = jumpMassPipe
      .groupBy(_.id)
      .leftJoin(normalMassPipe.groupBy(_.id))
      .map {
        case (id, (jumpMass, Some(normalMass))) =>
          UserMass(id, normalMass.mass * (1 - alpha) + jumpMass.mass)
        case (id, (jumpMass, _)) =>
          UserMass(id, jumpMass.mass)
      }
      .forceToDiskExecution

    val diff = completeMassPipe.flatMap { outputUserMassPipe =>
      inputUserMassPipe
        .groupBy(_.id)
        .join(outputUserMassPipe.groupBy(_.id))
        .map {
          case (_, (inputMass, outputMass)) =>
            scala.math.abs(outputMass.mass - inputMass.mass)
        }
        .sum
        .toOptionExecution
        .map(_.getOrElse(0D))
    }

    Execution.zip(completeMassPipe, diff)
  }

  private[user_cred_v2] def gateAndPublish(
    userCredPipe: TypedPipe[UserCredV2],
    config: UserCredV2Config,
    dateRange: DateRange,
  ): Execution[Unit] = {
    val currentMetricsExec = SnapshotSafeguard.computeMetrics(userCredPipe, config.vitThreshold)
    val previousMetricsExec: Execution[Option[GateMetrics]] =
      if (config.firstSnapshot) {
        Execution.from(None)
      } else {
        val previousCredPipe = readPreviousMass().map(um => UserCredV2.fromMass(um.id, um.mass))
        SnapshotSafeguard.computeMetrics(previousCredPipe, config.vitThreshold).map(Some(_))
      }

    Execution.zip(currentMetricsExec, previousMetricsExec).flatMap {
      case (current, previousOpt) =>
        info(s"Snapshot gate metrics: current=$current baseline=$previousOpt")
        val results = SnapshotSafeguard.evaluate(current, previousOpt, config)
        results.foreach { r =>
          info(s"Snapshot gate check ${r.name}: ${if (r.ok) "PASS" else "FAIL"} (${r.detail})")
        }
        val failures = results.filterNot(_.ok)
        if (failures.isEmpty) {
          writeResults(userCredPipe, config, dateRange)
        } else if (config.skipGate) {
          warn(
            s"skip_gate is set: publishing despite ${failures.size} failed gate check(s): " +
              failures.map(_.name).mkString(", "))
          writeResults(userCredPipe, config, dateRange)
        } else {
          error("Snapshot gate failed: writing snapshot to quarantine instead of publishing")
          writeQuarantine(userCredPipe, config.outputPath, dateRange).flatMap { _ =>
            Execution.failed(
              new RuntimeException(s"Snapshot gate failed: " +
                failures.map(r => s"${r.name} (${r.detail})").mkString("; ")))
          }
        }
    }
  }

  private[user_cred_v2] def writeQuarantine(
    userCredPipe: TypedPipe[UserCredV2],
    outputPath: String,
    dateRange: DateRange,
  ): Execution[Unit] = {
    userCredPipe
      .map(uc =>
        thriftscala.UserCredV2(
          userId = uc.id,
          mass = uc.mass,
          score = uc.score,
        ))
      .writeDALSnapshotExecution(
        UserCredV2BadSnapshotScalaDataset,
        D.Daily,
        D.Suffix(s"${outputPath}_bad_snapshot"),
        D.Parquet,
        dateRange.end
      )
  }

  private[user_cred_v2] def writeResults(
    userCredPipe: TypedPipe[UserCredV2],
    config: UserCredV2Config,
    dateRange: DateRange,
  ): Execution[Unit] = {
    val hdfsWriteExecution = writeHdfsSnapshot(userCredPipe, config.outputPath, dateRange)

    config.mhOutputPath match {
      case Some(mhOutputPath) =>
        val mhWriteExecution = userCredPipe
          .map { uc =>
            KeyVal(
              key = uc.id,
              value = thriftscala.UserCredV2Scores(
                score = uc.score,
                mass = uc.mass,
                snapshotTimestampMsec = dateRange.end.timestamp,
              ),
            )
          }
          .writeDALVersionedKeyValExecution(
            UserCredV2ManhattanScalaDataset,
            D.Suffix(mhOutputPath),
          )
        Execution
          .zip(
            hdfsWriteExecution,
            mhWriteExecution,
          ).unit
      case None =>
        hdfsWriteExecution
    }
  }

  private[user_cred_v2] def writeHdfsSnapshot(
    userCredPipe: TypedPipe[UserCredV2],
    outputPath: String,
    dateRange: DateRange,
  ): Execution[Unit] = {
    userCredPipe
      .map(uc =>
        thriftscala.UserCredV2(
          userId = uc.id,
          mass = uc.mass,
          score = uc.score,
        ))
      .writeDALSnapshotExecution(
        UserCredV2ScalaDataset,
        D.Daily,
        D.Suffix(outputPath),
        D.Parquet,
        dateRange.end
      )
  }
}

object UserCredV2AppAdhoc extends UserCredV2App with TwitterExecutionApp {
  import UserCredV2App._
  override def job: Execution[Unit] = Execution.withArgs { args =>
    val dateRange = DateRange.parse(args.list("date"))
    val config = UserCredV2Config.fromArgs(args)
    runOnDateRange(dateRange, config)
  }
}

object UserCredV2AppProd extends UserCredV2App with TwitterScheduledExecutionApp {
  import UserCredV2App._
  override def scheduledJob: Execution[Unit] = {
    val analyticsArgs = AnalyticsBatchExecutionArgs(
      batchDesc = BatchDescription("user_cred_v2_app_prod"),
      firstTime = BatchFirstTime(RichDate("2026-04-27")),
      batchIncrement = BatchIncrement(Days(1)),
    )

    Execution.withArgs { args =>
      AnalyticsBatchExecution(analyticsArgs) { dateRange =>
        val config = UserCredV2Config.fromArgs(args)
        runOnDateRange(dateRange, config)
      }
    }
  }
}
