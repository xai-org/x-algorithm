"""Exercise unchanged Scala-compatible Strato helpers with stubbed catalog access.

This does not compile Strato or test its serializers/access policies. Use the
native fixture library and internal integration checks for those boundaries.
Provide Scala 2.12 compiler, library, and reflect JARs via --scala-classpath.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scala-classpath', required=True, help='OS-separated paths to Scala 2.12 compiler, library, and reflect JARs')
parser.add_argument('--java', default='java', help='Java executable (JDK 17 recommended)')
args = parser.parse_args()
ROOT = Path(__file__).resolve().parents[2]
source = (ROOT / 'under-the-hood/strato/lib/accountAffinityProfile.strato').read_text(encoding='utf-8')
report = (ROOT / 'under-the-hood/strato/columns/underTheHoodReport.User.strato').read_text(encoding='utf-8')

def between(text, start, end):
    return text[text.index(start):text.index(end, text.index(start))]

selection = between(source, 'def primaryClusterScores(', '\ndef profileJson(')
append = between(source, 'def appendToReport(', '\nLibrary(')
fetch = between(report, 'def withAccountAffinityProfile(', '\ndef userWithQueryFields(')
fetch = fetch.replace('#<visibility/under-the-hood/mh/accountKnownFor>.fetch(userId, ()).v', 'fetchKnownFor(userId)')
max_clusters = between(source, 'val maxPrimaryClusters = ', '\n\n// Scores')

harness = r'''
object AffinityHarness extends App {
  MAX_CLUSTERS
  SELECTION
  APPEND
  case class KnownFor(json: Option[String])
  object params {
    var enabled = false
    def underTheHoodReportAccountAffinitiesEnabled(): Boolean = enabled
  }
  object Time { object now { val inMillis = 1234L } }
  def formatGeneratedAtIso(ms: Long): String = "2026-09-20T12:00:00Z"
  var reads = 0
  var seenUserId = 0L
  var result: Option[KnownFor] = None
  var failFetch = false
  def fetchKnownFor(id: Long): Option[KnownFor] = {
    reads += 1
    seenUserId = id
    if (failFetch) throw new RuntimeException("store unavailable")
    result
  }
  object accountAffinityProfile {
    // This harness substitutes the Strato serializer; native fixtures test it.
    def profileJson(record: KnownFor, retrievedAt: String): Option[String] = {
      assert(retrievedAt == "2026-09-20T12:00:00Z")
      record.json
    }
    def appendToReport(base: String, profile: Option[String]): String = AffinityHarness.appendToReport(base, profile)
  }
  FETCH

  var tests = 0
  def check(name: String)(body: => Unit): Unit = {
    body
    tests += 1
    println("PASS " + name)
  }
  check("empty source") { assert(primaryClusterScores(Map.empty).isEmpty) }
  check("top ten with raw scores") {
    val scores = (1 to 40).map(id => id -> Some(id.toDouble)).toMap
    assert(primaryClusterScores(scores) == (40 to 31 by -1).map(id => (id, id.toDouble)))
  }
  check("stable tie cutoff regardless of insertion order") {
    val pairs = (0 until 100).map(id => id -> Some(0.5))
    for (_ <- 1 to 100) {
      val shuffled = scala.util.Random.shuffle(pairs).toMap
      assert(primaryClusterScores(shuffled) == (0 until 10).map(id => (id, 0.5)))
    }
  }
  check("missing and invalid scores filtered before truncation") {
    val bad: Map[Int, Option[Double]] = Map(
      -1 -> Some(2.0), 1 -> None, 2 -> Some(0.0), 3 -> Some(-1.0),
      4 -> Some(Double.NaN), 5 -> Some(Double.PositiveInfinity),
      6 -> Some(Double.NegativeInfinity), 7 -> Some(-0.0),
      8 -> Some(0.4), 9 -> Some(0.1))
    assert(primaryClusterScores(bad) == Seq((8, 0.4), (9, 0.1)))
  }
  check("finite double boundaries and cluster zero") {
    assert(primaryClusterScores(Map(0 -> Some(Double.MaxValue), 1 -> Some(java.lang.Double.MIN_VALUE))) ==
      Seq((0, Double.MaxValue), (1, java.lang.Double.MIN_VALUE)))
  }
  check("output invariants across random maps") {
    val random = new scala.util.Random(210)
    for (_ <- 1 to 500) {
      val input = (0 until 200).map(id => id -> Some(random.nextInt(50).toDouble - 10.0)).toMap
      val output = primaryClusterScores(input)
      assert(output.size <= 10)
      assert(output.map(_._1).distinct.size == output.size)
      assert(output.forall { case (id, score) => score > 0.0 && input(id).contains(score) })
      assert(output.sliding(2).forall(pair => pair.size < 2 || pair.head._2 > pair.last._2 ||
        (pair.head._2 == pair.last._2 && pair.head._1 < pair.last._1)))
      assert(input.toSeq.filter { case (id, score) => score.exists(_ > 0.0) && !output.exists(_._1 == id) }
        .forall { case (_, score) => output.size == 10 && score.get <= output.last._2 })
    }
  }
  val bases = Seq(
    """{"notes":"quoted \"text\" and brace }","postLabels":[],"accountLabels":[],"totalPostLabels":0,"totalAccountLabels":0}""",
    """{"postLabels":[],"accountLabels":[{"label":"test"}],"totalPostLabels":0}""",
    """{"postLabels":[{"label":"test"}],"accountLabels":[],"totalAccountLabels":0}""",
    """{"postLabels":[{"label":"test"}],"accountLabels":[{"label":"test"}]}""")
  val profile = """{"modelVersion":"20M_145K_2020","primaryClusters":[{"clusterId":42,"weight":0.8}]}"""
  check("absent profile preserves every report shape byte for byte") {
    bases.foreach(base => assert(appendToReport(base, None) == base))
  }
  check("disabled switch performs no read and preserves JSON") {
    params.enabled = false
    reads = 0
    assert(withAccountAffinityProfile(210L, bases.head) == bases.head)
    assert(reads == 0)
  }
  check("enabled profile performs one read for the requested user") {
    params.enabled = true
    result = Some(KnownFor(Some(profile)))
    reads = 0
    assert(withAccountAffinityProfile(210L, bases.head) == appendToReport(bases.head, Some(profile)))
    assert(reads == 1 && seenUserId == 210L)
  }
  check("lookup miss preserves all report shapes") {
    result = None
    bases.foreach(base => assert(withAccountAffinityProfile(210L, base) == base))
  }
  check("rejected profile preserves all report shapes") {
    result = Some(KnownFor(None))
    bases.foreach(base => assert(withAccountAffinityProfile(210L, base) == base))
  }
  check("lookup exceptions preserve all report shapes") {
    failFetch = true
    bases.foreach(base => assert(withAccountAffinityProfile(210L, base) == base))
  }
  bases.foreach(base => println("JSON " + appendToReport(base, Some(profile))))
  println("PASS total=" + tests)
}
'''
harness = harness.replace('MAX_CLUSTERS', max_clusters).replace('SELECTION', selection).replace('APPEND', append).replace('FETCH', fetch)
with tempfile.TemporaryDirectory(prefix='uth-affinity-') as scratch:
    scala = Path(scratch) / 'AffinityHarness.scala'
    scala.write_text(harness, encoding='utf-8')
    classpath = args.scala_classpath
    subprocess.run([args.java, '-cp', classpath, 'scala.tools.nsc.Main', '-usejavacp', '-d', scratch, str(scala)], check=True)
    run = subprocess.run([args.java, '-cp', scratch + os.pathsep + classpath, 'AffinityHarness'], text=True, capture_output=True, check=True)
json_count = 0
for line in run.stdout.splitlines():
    if line.startswith('JSON '):
        parsed = json.loads(line[5:])
        assert parsed.pop('accountAffinityProfile')['primaryClusters'] == [{'clusterId': 42, 'weight': 0.8}]
        assert ('totalPostLabels' in parsed) == (not parsed['postLabels'])
        assert ('totalAccountLabels' in parsed) == (not parsed['accountLabels'])
        json_count += 1
    else:
        print(line)
assert json_count == 4
print('PASS Python JSON parsing and conditional totals for four report shapes')
print('LIMIT: Scala compatibility harness; not a native Strato compilation or live store test')
