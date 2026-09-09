package com.twitter.visibility.under_the_hood

import org.scalatest.matchers.should.Matchers
import org.scalatest.wordspec.AnyWordSpec

class UthPostIdsSpec extends AnyWordSpec with Matchers {

  private def summaryOf(rows: (Long, Long)*): UthPostIds.Summary =
    UthPostIds.summarize(rows)

  "UthPostIds.single" should {
    "count one carried post and normalize any positive removal to one" in {
      UthPostIds.single(10L, 0L) shouldBe UthPostIds.Summary(1L, 0L, Vector(10L))
      UthPostIds.single(10L, 1L) shouldBe UthPostIds.Summary(1L, 1L, Vector(10L))
      UthPostIds.single(10L, 5L).removed shouldBe 1L
    }
  }

  "UthPostIds.merge" should {
    "sum carried and removed the way the upstream summed rows" in {
      val merged = UthPostIds.merge(UthPostIds.single(1L, 1L), UthPostIds.single(2L, 0L))
      merged.carried shouldBe 2L
      merged.removed shouldBe 1L
      merged.postIds shouldBe Seq(1L, 2L)
    }

    "treat empty as an identity" in {
      val one = UthPostIds.single(7L, 1L)
      UthPostIds.merge(UthPostIds.empty, one) shouldBe one
      UthPostIds.merge(one, UthPostIds.empty) shouldBe one
    }

    "be associative and order independent so map-side combining is safe" in {
      val a = UthPostIds.single(3L, 0L)
      val b = UthPostIds.single(1L, 1L)
      val c = UthPostIds.single(2L, 0L)
      val left = UthPostIds.merge(UthPostIds.merge(a, b), c)
      val right = UthPostIds.merge(a, UthPostIds.merge(b, c))
      left shouldBe right
      UthPostIds.merge(a, b) shouldBe UthPostIds.merge(b, a)
    }

    "never let a merged id list exceed the bound" in {
      val big = UthPostIds.Summary(
        carried = UthPostIds.MaxPostIdsPerLabel.toLong,
        removed = 0L,
        postIds = (1L to UthPostIds.MaxPostIdsPerLabel.toLong).toVector
      )
      val other = UthPostIds.Summary(0L, 0L, (5000L to 5100L).toVector)
      UthPostIds.merge(big, other).postIds.size shouldBe UthPostIds.MaxPostIdsPerLabel
    }
  }

  "UthPostIds.summarize" should {
    "produce ids in deterministic ascending order" in {
      summaryOf((30L, 0L), (10L, 0L), (20L, 0L)).postIds shouldBe Seq(10L, 20L, 30L)
    }

    "reconcile removed counts against carried counts" in {
      val s = summaryOf((1L, 1L), (2L, 0L), (3L, 1L))
      s.carried shouldBe 3L
      s.removed shouldBe 2L
    }

    "return the empty summary for no rows" in {
      UthPostIds.summarize(Nil) shouldBe UthPostIds.Summary(0L, 0L, Vector.empty)
    }

    "bound the id list while leaving carried counts exact" in {
      val rows = (1L to 2500L).map(id => (id, 0L))
      val s = UthPostIds.summarize(rows)
      s.carried shouldBe 2500L
      s.postIds.size shouldBe UthPostIds.MaxPostIdsPerLabel
      s.postIds.head shouldBe 1501L
      s.postIds.last shouldBe 2500L
    }
  }

  "UthPostIds.newestDistinct" should {
    "keep the newest ids by snowflake ordering" in {
      UthPostIds.newestDistinct(Seq(5L, 1L, 9L, 3L), 2) shouldBe Seq(5L, 9L)
    }

    "deduplicate before applying the bound" in {
      UthPostIds.newestDistinct(Seq(4L, 4L, 4L, 1L), 10) shouldBe Seq(1L, 4L)
    }

    "return everything when the bound is not reached" in {
      UthPostIds.newestDistinct(Seq(2L, 1L), 10) shouldBe Seq(1L, 2L)
    }

    "reject a non positive bound" in {
      an[IllegalArgumentException] should be thrownBy UthPostIds.newestDistinct(Seq(1L), 0)
    }
  }

  "a per day cap" should {
    "not change which ids survive the monthly cap" in {
      val dayA = (1L to 2500L).map(id => (id, 0L))
      val dayB = (9000L to 9010L).map(id => (id, 0L))
      val cappedPerDay =
        UthPostIds.summarize(dayA).postIds ++ UthPostIds.summarize(dayB).postIds
      val uncapped = (dayA ++ dayB).map(_._1)
      UthPostIds.newestDistinct(cappedPerDay, UthPostIds.MaxPostIdsPerLabel) shouldBe
        UthPostIds.newestDistinct(uncapped, UthPostIds.MaxPostIdsPerLabel)
    }
  }
}
