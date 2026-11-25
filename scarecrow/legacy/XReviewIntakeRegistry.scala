package com.twitter.botmaker.app.scarecrow.legacy

import com.google.inject.Exposed
import com.google.inject.Provides
import com.google.inject.Singleton
import com.twitter.botmaker.ASTNode
import com.twitter.botmaker.BotMakerFinatraModule.Migration
import com.twitter.botmaker.Context
import com.twitter.botmaker.DownstreamService
import com.twitter.botmaker.FunctionUnit4
import com.twitter.botmaker.FunctionUnit5O2
import com.twitter.botmaker.app.scarecrow.ScarecrowRuntime
import com.twitter.botmaker.compiler.ActionLevel
import com.twitter.botmaker.runtime.localMode
import com.twitter.botmaker.runtime.personalAccountMode
import com.twitter.finagle.mtls.authentication.ServiceIdentifier
import com.twitter.finagle.stats.StatsReceiver
import com.twitter.inject.Injector
import com.twitter.inject.TwitterPrivateModule
import com.twitter.inject.annotations.Flag
import com.twitter.useng.common.xreview.{
  XReviewIntakeClientModule => SharedXReviewIntakeClientModule
}
import com.twitter.util.Future
import java.lang.{Long => JLong}

object XReviewIntakeRegistry extends TwitterPrivateModule {

  flag[String](
    "xreview.tls.ca-cert",
    SharedXReviewIntakeClientModule.DefaultCaCertPath,
    "CA cert path for XReview intake mTLS"
  )

  flag[String](
    "xreview.tls.client-fullchain",
    "",
    "Client cert chain for XReview intake mTLS"
  )

  flag[String](
    "xreview.tls.client-key",
    "",
    "Client key for XReview intake mTLS"
  )

  @Provides
  @Singleton
  @Exposed
  def providesXReviewReportSubmitter(
    serviceIdentifier: ServiceIdentifier,
    @Migration statsReceiver: StatsReceiver,
    @Flag("xreview.tls.ca-cert") caCertPath: String,
    @Flag("xreview.tls.client-fullchain") clientFullchain: String,
    @Flag("xreview.tls.client-key") clientKeyPath: String
  ): XReviewReportSubmitter = {
    if (localMode() || personalAccountMode()) {
      XReviewReportSubmitter.noop(statsReceiver)
    } else {
      val prodClient = SharedXReviewIntakeClientModule.provideXReviewIntakeClient(
        serviceIdentifier = serviceIdentifier,
        statsReceiver = statsReceiver.scope("xreview"),
        caCertPath = caCertPath,
        clientFullchain = clientFullchain,
        clientKeyPath = clientKeyPath
      )
      val stagingClient = SharedXReviewIntakeClientModule.provideXReviewStagingIntakeClient(
        serviceIdentifier = serviceIdentifier,
        statsReceiver = statsReceiver.scope("xreview_staging"),
        caCertPath = caCertPath,
        clientFullchain = clientFullchain,
        clientKeyPath = clientKeyPath
      )
      XReviewReportSubmitter(prodClient, stagingClient, statsReceiver)
    }
  }

  override def singletonShutdown(injector: Injector): Unit =
    injector.instance[XReviewReportSubmitter].close()
}

object CreateXReviewReportProd
    extends FunctionUnit5O2[
      ScarecrowRuntime,
      String,
      Long,
      Long,
      String,
      Long,
      String,
      JLong,
      Future[Unit]
    ] {

  override def cacheLevel: ASTNode.CacheLevel = ASTNode.CacheLevel.Event
  override def actionLevel: ActionLevel = ActionLevel.PROD_AGENT_WORKFLOW_ACTION
  override def downstreams: Set[DownstreamService] = Set(DownstreamServices.XReviewIntake)
  override def description: String =
    "Submits a report to production XReview Intake over gRPC. " +
      "entityType must be post or profile. reportType must be an XReview-allowlisted value. " +
      "Impersonation bots should use bystander_impersonation (lane tags key on that type, " +
      "not generic impersonation). Each evaluation includes detection_timestamp_ms so " +
      "intake's content-hash report id is unique; evidence rollup then applies."
  override def arguments: Seq[String] = Seq(
    "reported entity type (post or profile)",
    "reported entity id (tweet id or user id)",
    "reported user id",
    "XReview report_type (e.g. bystander_impersonation)",
    "bot id (report-bag detection_bot_id; reporter_id is 0)",
    "optional note",
    "optional victim user id (victim_user_id)"
  )
  override def examples: Seq[String] = Seq(
    "CreateXReviewReportProd(\"profile\", :userId, :userId, \"bystander_impersonation\", :botId)",
    "CreateXReviewReportProd(\"profile\", :userId, :userId, \"bystander_impersonation\", :botId, :note, :victimId)"
  )

  override def evaluate(
    context: Context[ScarecrowRuntime],
    entityType: String,
    entityId: Long,
    userId: Long,
    reportType: String,
    botId: Long,
    note: Option[String],
    victimId: Option[JLong]
  ): Future[Unit] = {
    context.getRuntime.fetcher20.xreviewReportSubmitter.submit(
      entityType,
      entityId,
      userId,
      reportType,
      botId,
      note,
      victimId,
      staging = false,
      detectionTimestampMs = context.startMillis()
    )
  }
}

object CreateXReviewReportStaging
    extends FunctionUnit5O2[
      ScarecrowRuntime,
      String,
      Long,
      Long,
      String,
      Long,
      String,
      JLong,
      Future[Unit]
    ] {

  override def cacheLevel: ASTNode.CacheLevel = ASTNode.CacheLevel.Event
  override def actionLevel: ActionLevel = ActionLevel.NO_ACTION
  override def downstreams: Set[DownstreamService] = Set(DownstreamServices.XReviewIntake)
  override def description: String =
    "Submits a report to staging XReview Intake over gRPC. " +
      "entityType must be post or profile. reportType must be an XReview-allowlisted value. " +
      "Impersonation bots should use bystander_impersonation (lane tags key on that type, " +
      "not generic impersonation)."
  override def arguments: Seq[String] = Seq(
    "reported entity type (post or profile)",
    "reported entity id (tweet id or user id)",
    "reported user id",
    "XReview report_type (e.g. bystander_impersonation)",
    "bot id (report-bag detection_bot_id; reporter_id is 0)",
    "optional note",
    "optional victim user id (victim_user_id)"
  )
  override def examples: Seq[String] = Seq(
    "CreateXReviewReportStaging(\"profile\", :userId, :userId, \"bystander_impersonation\", :botId)"
  )

  override def evaluate(
    context: Context[ScarecrowRuntime],
    entityType: String,
    entityId: Long,
    userId: Long,
    reportType: String,
    botId: Long,
    note: Option[String],
    victimId: Option[JLong]
  ): Future[Unit] = {
    context.getRuntime.fetcher20.xreviewReportSubmitter.submit(
      entityType,
      entityId,
      userId,
      reportType,
      botId,
      note,
      victimId,
      staging = true,
      detectionTimestampMs = context.startMillis()
    )
  }
}

object CreateXReviewSpaceReportProd
    extends FunctionUnit4[ScarecrowRuntime, String, Long, String, Long, Future[Unit]] {

  override def cacheLevel: ASTNode.CacheLevel = ASTNode.CacheLevel.Event
  override def actionLevel: ActionLevel = ActionLevel.PROD_AGENT_WORKFLOW_ACTION
  override def downstreams: Set[DownstreamService] = Set(DownstreamServices.XReviewIntake)
  override def description: String =
    "Submits a live Space report to production XReview Intake over gRPC, where it lands in " +
      "the Reported Spaces queue of the report type's project. reportType must be an " +
      "XReview-allowlisted value. Each evaluation includes detection_timestamp_ms so intake's " +
      "content-hash report id is unique."
  override def arguments: Seq[String] = Seq(
    "Space id (broadcastId)",
    "reported user id (usually the Space creator)",
    "XReview report_type (e.g. hate_abuse_or_harassment)",
    "bot id (report-bag detection_bot_id; reporter_id is 0)"
  )
  override def examples: Seq[String] = Seq(
    "CreateXReviewSpaceReportProd(broadcastId, :space.creator_twitter_user_id, \"hate_abuse_or_harassment\", RULE_ID)"
  )

  override def evaluate(
    context: Context[ScarecrowRuntime],
    spaceId: String,
    userId: Long,
    reportType: String,
    botId: Long
  ): Future[Unit] =
    context.getRuntime.fetcher20.xreviewReportSubmitter
      .submitSpace(spaceId, userId, reportType, botId, context.startMillis())
}
