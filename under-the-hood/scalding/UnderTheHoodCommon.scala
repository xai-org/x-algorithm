package com.twitter.visibility.under_the_hood

import com.twitter.scalding.Args
import com.twitter.scalding.TypedPipe
import com.twitter.scalding.typed.Grouped
import com.twitter.visibility.under_the_hood.thriftscala.UthDailyPostLabel
import java.util.Calendar
import java.util.TimeZone
import scala.util.Try

object UnderTheHoodCommon {

  val DayMs: Long = 86400000L
  val TwitterEpochMs: Long = 1288834974657L

  def snowflakeCreatedMs(id: Long): Long = (id >> 22) + TwitterEpochMs

  private def utcCal(atMsec: Long): Calendar = {
    val cal = Calendar.getInstance(TimeZone.getTimeZone("UTC"))
    cal.setTimeInMillis(atMsec)
    cal
  }

  def yyyymm(atMsec: Long): Int = {
    val cal = utcCal(atMsec)
    (cal.get(Calendar.YEAR) * 100) + (cal.get(Calendar.MONTH) + 1)
  }

  def yyyymmdd(atMsec: Long): Int = {
    val cal = utcCal(atMsec)
    (cal.get(Calendar.YEAR) * 10000) +
      ((cal.get(Calendar.MONTH) + 1) * 100) +
      cal.get(Calendar.DAY_OF_MONTH)
  }

  def dayOfMonth(yyyymmdd: Int): Int = yyyymmdd % 100

  def monthBucket(yyyymmdd: Int): Int = yyyymmdd / 100

  def yyyymmddToMs(d: Int): Long = {
    val y = d / 10000
    val m = (d / 100) % 100
    val day = d % 100
    java.time.LocalDate
      .of(y, m, day)
      .atStartOfDay(java.time.ZoneOffset.UTC)
      .toInstant
      .toEpochMilli
  }

  def followerClass(followers: Long): String =
    if (followers < 1000L) "LT_1K"
    else if (followers < 10000L) "FROM_1K_TO_10K"
    else if (followers < 100000L) "FROM_10K_TO_100K"
    else "GTE_100K"

  def monthCoverageYyyymmdd(
    monthBucket: Int,
    lookbackStartYyyymmdd: Int,
    asOfDay: Int
  ): (Int, Int) = {
    val y = monthBucket / 100
    val m = monthBucket % 100
    val monthStart = y * 10000 + m * 100 + 1
    val monthEnd = y * 10000 + m * 100 + java.time.YearMonth.of(y, m).lengthOfMonth()
    val first = math.max(monthStart, lookbackStartYyyymmdd)
    val last = math.min(monthEnd, asOfDay)
    require(
      first <= last,
      s"empty coverage for month $monthBucket in [$lookbackStartYyyymmdd, $asOfDay]"
    )
    (first, last)
  }

  def inclusiveDayCount(firstYyyymmdd: Int, lastYyyymmdd: Int): Int = {
    require(firstYyyymmdd <= lastYyyymmdd, s"first $firstYyyymmdd > last $lastYyyymmdd")
    val first = java.time.LocalDate
      .of(firstYyyymmdd / 10000, (firstYyyymmdd / 100) % 100, firstYyyymmdd % 100)
    val last =
      java.time.LocalDate.of(lastYyyymmdd / 10000, (lastYyyymmdd / 100) % 100, lastYyyymmdd % 100)
    java.time.temporal.ChronoUnit.DAYS.between(first, last).toInt + 1
  }

  def calendarDaysBetween(startYyyymmdd: Int, endYyyymmdd: Int): Int = {
    require(startYyyymmdd <= endYyyymmdd, s"start $startYyyymmdd > end $endYyyymmdd")
    inclusiveDayCount(startYyyymmdd, endYyyymmdd) - 1
  }

  def addCalendarDays(yyyymmddDay: Int, days: Int): Int = {
    val d = java.time.LocalDate
      .of(yyyymmddDay / 10000, (yyyymmddDay / 100) % 100, yyyymmddDay % 100)
      .plusDays(days.toLong)
    d.getYear * 10000 + d.getMonthValue * 100 + d.getDayOfMonth
  }

  def finalThroughDay(completeThroughDay: Int, asOfDay: Int, postObservationDays: Int): Int =
    math.min(completeThroughDay, addCalendarDays(asOfDay, -postObservationDays))

  def latestAsOfPostLabelRows(
    rows: TypedPipe[UthDailyPostLabel],
    reducers: Int
  ): TypedPipe[UthDailyPostLabel] = {
    def asOf(r: UthDailyPostLabel): Int = r.asOfYyyymmdd.getOrElse(Int.MinValue)
    val isTakedown = (r: UthDailyPostLabel) =>
      r.label.exists(TakedownLabels.parseLabel(_).isDefined)

    val safety = applyReducers(
      rows.filterNot(isTakedown).groupBy(r => (r.userId, r.authoredYyyymmdd, r.label)),
      reducers
    ).reduce { (a, b) => if (asOf(a) >= asOf(b)) a else b }.values

    val takedown = applyReducers(
      rows.filter(isTakedown).groupBy(r => (r.userId, r.authoredYyyymmdd)),
      reducers
    ).toList.values
      .flatMap { group =>
        val freshest = group.map(asOf).max
        group.filter(asOf(_) == freshest).groupBy(_.label).values.map(_.head)
      }.filter(_.carried.exists(_ > 0))

    safety ++ takedown
  }

  def parseUserIds(args: Args): Set[Long] = {
    val raw = args.list("userIds").flatMap(_.split(",")).map(_.trim).filter(_.nonEmpty)
    val invalid = raw.filter(s => Try(s.toLong).isFailure)
    require(
      invalid.isEmpty,
      s"--userIds supplied but contains non-numeric ids: ${invalid.mkString(", ")}. " +
        "Omit --userIds entirely for an all-user run."
    )
    raw.flatMap(s => Try(s.toLong).toOption).toSet
  }

  def parsePostIdLabels(args: Args, default: Set[String]): Set[String] =
    args.optional("postIdLabels") match {
      case None => default
      case Some("none") => Set.empty
      case Some(_) =>
        args.list("postIdLabels").flatMap(_.split(",")).map(_.trim).filter(_.nonEmpty).toSet
    }

  val DefaultMaxPostIdsPerLabel = 500

  def newestPostIds(ids: Seq[Long], max: Int): Seq[Long] =
    ids.distinct.sorted(Ordering[Long].reverse).take(max)

  def preferredInt(args: Args, preferred: String, deprecated: String, default: Int): Int = {
    val raw = args.optional(preferred).orElse(args.optional(deprecated))
    raw match {
      case None => default
      case Some(s) =>
        Try(s.toInt).getOrElse {
          throw new IllegalArgumentException(
            s"--$preferred (or deprecated --$deprecated) must be an integer; got '$s'")
        }
    }
  }

  def inScope(testUserIds: Set[Long], userId: Long): Boolean =
    testUserIds.isEmpty || testUserIds.contains(userId)

  def applyReducers[K, V](g: Grouped[K, V], reducers: Int): Grouped[K, V] =
    if (reducers > 0) g.withReducers(reducers) else g
}
