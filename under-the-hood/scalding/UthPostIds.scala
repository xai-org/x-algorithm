package com.twitter.visibility.under_the_hood

object UthPostIds {
  val MaxPostIdsPerLabel: Int = 1000

  final case class Summary(carried: Long, removed: Long, postIds: Seq[Long])

  val empty: Summary = Summary(0L, 0L, Vector.empty)

  def single(logicalId: Long, removed: Long): Summary =
    Summary(1L, if (removed > 0L) 1L else 0L, Vector(logicalId))

  def merge(a: Summary, b: Summary): Summary =
    Summary(
      carried = a.carried + b.carried,
      removed = a.removed + b.removed,
      postIds = newestDistinct(a.postIds ++ b.postIds, MaxPostIdsPerLabel)
    )

  def summarize(rows: Iterable[(Long, Long)]): Summary =
    rows.foldLeft(empty) {
      case (acc, (logicalId, removed)) => merge(acc, single(logicalId, removed))
    }

  def newestDistinct(postIds: Iterable[Long], limit: Int): Seq[Long] = {
    require(limit > 0, s"limit must be > 0; got $limit")
    postIds.toSet.toSeq.sorted.takeRight(limit)
  }
}
