import type { Metadata } from "next";
import Image from "next/image";
import Link from "next/link";
import { Github } from "lucide-react";
import Background from "@/components/Background";
import SiteHeader from "@/components/SiteHeader";
import SocialLinks from "@/components/SocialLinks";

const DESCRIPTION =
  "What every number, flag and button in the BESC Engagement Checker means, and exactly which part of X's open-sourced For You algorithm each one comes from.";

export const metadata: Metadata = {
  title: "Docs · BESC Engagement Checker",
  description: DESCRIPTION,
  alternates: { canonical: "/docs" },
  openGraph: {
    title: "Docs · BESC Engagement Checker",
    description: DESCRIPTION,
    type: "article",
    url: "/docs",
    images: ["/besc-banner.png"],
  },
};

const SECTIONS = [
  { id: "what-this-is", label: "What this is" },
  { id: "the-score", label: "The BESC Score" },
  { id: "inputs", label: "Every input explained" },
  { id: "signals", label: "Signal breakdown" },
  { id: "risks", label: "Visibility-filtering risk" },
  { id: "tips", label: "Tips" },
  { id: "optimize", label: "Optimize" },
  { id: "ai", label: "AI rewrite & generate" },
  { id: "import", label: "Import a live post" },
  { id: "tracking", label: "Track record" },
  { id: "ai-estimate", label: "The AI second opinion" },
  { id: "sources", label: "Where the numbers come from" },
];

export default function Docs() {
  return (
    <main className="relative min-h-screen">
      <Background />

      <SiteHeader />

      <section className="mx-auto max-w-7xl px-5 pb-6 pt-4 sm:px-8">
        <span className="rounded-full border border-besc-500/25 bg-besc-500/10 px-3 py-1 text-[11px] font-medium uppercase tracking-widest text-besc-300">
          Documentation
        </span>
        <h1 className="mt-4 font-display text-4xl font-bold leading-[1.05] tracking-tight sm:text-5xl">
          What every number here <span className="text-gold">actually means</span>
        </h1>
        <p className="mt-4 max-w-2xl text-balance text-[15px] leading-relaxed text-white/50">
          This tool makes a lot of specific claims. This page explains each one, what it&apos;s
          based on, and — just as importantly — where the limits are. If something here
          can&apos;t be traced to a real rule or weight, it says so.
        </p>
      </section>

      <div className="mx-auto grid max-w-7xl gap-8 px-5 pb-24 sm:px-8 lg:grid-cols-[220px_minmax(0,1fr)] lg:items-start">
        <nav className="lg:sticky lg:top-6">
          <div className="glass-panel p-4">
            <p className="px-2 pb-2 text-[10.5px] uppercase tracking-widest text-white/30">
              On this page
            </p>
            <ul className="space-y-0.5">
              {SECTIONS.map((s) => (
                <li key={s.id}>
                  <a
                    href={`#${s.id}`}
                    className="block rounded-lg px-2 py-1.5 text-[13px] text-white/55 transition-colors hover:bg-white/[0.04] hover:text-white/90"
                  >
                    {s.label}
                  </a>
                </li>
              ))}
            </ul>
          </div>
        </nav>

        <div className="min-w-0 space-y-6">
          <Section id="what-this-is" title="What this is (and what it isn't)">
            <P>
              In August 2026, X open-sourced the code that decides what appears in the For You
              timeline. This tool is built on that code: the ranking weights, the filters, and
              the visibility rules are read directly out of this repository rather than
              guessed at from social-media folklore.
            </P>
            <Callout tone="warn" title="The honest limitation, up front">
              X&apos;s real ranking runs <Mono>Phoenix</Mono>, a learned transformer that reads a
              specific viewer&apos;s engagement history to predict how likely{" "}
              <em>that person</em> is to reply to, share, or mute your post. We don&apos;t have
              that model, your account&apos;s history, or a viewer. So this tool estimates those
              probabilities from surface features of your draft — length, hooks, links,
              formatting — and then blends them using the <strong>real production weights</strong>.
              The weights and the rules are accurate. The probabilities are a transparent
              stand-in. Treat the score as directional coaching, not a promise of reach.
            </Callout>
            <P>
              That&apos;s also exactly why the{" "}
              <a href="#tracking" className="text-besc-300 hover:text-besc-200">
                Track record
              </a>{" "}
              feature exists — it checks the tool&apos;s predictions against your real posts so you
              don&apos;t have to take any of this on faith.
            </P>
          </Section>

          <Section id="the-score" title="The BESC Score">
            <P>
              One number, 0–100. It&apos;s built the same way X builds its own ranking score, then
              squashed into a readable range.
            </P>
            <Steps
              items={[
                <>
                  <strong>Estimate a probability for each action</strong> a viewer might take —
                  reply, like, repost, share, follow you, or mute/block/report you.
                </>,
                <>
                  <strong>Multiply each by its real production weight and sum them.</strong>{" "}
                  This is X&apos;s actual formula: <Mono>Final Score = Σ (weight × P(action))</Mono>.
                </>,
                <>
                  <strong>Apply the author-diversity multiplier</strong> if you&apos;ve posted
                  recently (see below).
                </>,
                <>
                  <strong>Squash to 0–100</strong> so it&apos;s readable. A weighted sum is
                  unbounded; the score is not.
                </>,
              ]}
            />
            <SubHead>What the grades mean</SubHead>
            <Table
              head={["Score", "Grade", "Roughly"]}
              rows={[
                ["82–100", "Excellent", "Strong hook, a real reason to reply, nothing risky"],
                ["58–81", "Strong", "Solid post, usually one lever left unpulled"],
                ["38–57", "Decent", "Publishable, but leaving reach on the table"],
                ["20–37", "Weak", "Little reason for anyone to act on it"],
                ["0–19", "High Risk", "Likely tripping a spam/negative signal — check the risk panel"],
              ]}
            />
            <SubHead>The two multipliers shown under the gauge</SubHead>
            <DefList
              items={[
                {
                  term: "Author diversity ×",
                  def: (
                    <>
                      Each additional post from you in the same window is multiplied by{" "}
                      <Mono>(1 − 0.25) × 0.5^k + 0.25</Mono>. Your 2nd post scores at ~62%, and it
                      floors at 25% by the 4th. Spacing posts out is worth more than most people
                      think.
                    </>
                  ),
                },
                {
                  term: "Out-of-network ×",
                  def: (
                    <>
                      Your own followers see the post scored at full strength. For everyone else
                      it&apos;s multiplied by <Mono>0.75</Mono> — and on a topic-matched
                      recommendation surface it&apos;s <Mono>0.5</Mono>. That surface also runs a
                      much longer list of drop rules your followers never hit.
                    </>
                  ),
                },
              ]}
            />
          </Section>

          <Section id="inputs" title="Every input explained">
            <P>
              These change the score because they change how the real algorithm treats the post.
              None of them are cosmetic.
            </P>
            <DefList
              items={[
                {
                  term: "Media type (Text / Photo / Video / GIF)",
                  def: (
                    <>
                      Media unlocks scored actions text can never earn — photo-expand, video-open,
                      and video-quality-view. One catch worth knowing:{" "}
                      <strong>a video under 10 seconds has its video-quality-view weight forced
                      to exactly 0</strong>, no matter how good it is.
                    </>
                  ),
                },
                {
                  term: "This is a reply",
                  def: (
                    <>
                      Structurally the most expensive toggle here. Replies and reposts are{" "}
                      <strong>removed entirely</strong> from recommendations to anyone who
                      doesn&apos;t already follow you — not downranked, excluded. Even shown to your
                      own followers they&apos;re rescored at the same 0.75× as out-of-network
                      content. Replies are also excluded from the cold-start boost and the
                      mutual-follow bonus below.
                    </>
                  ),
                },
                {
                  term: "Mostly mutual-follow audience",
                  def: (
                    <>
                      If people who follow you also get followed back, an{" "}
                      <strong>original post</strong> gets <Mono>+15.0</Mono> added to its reply
                      weight — the single largest situational boost in the model, taking reply
                      from 5.0 to 20.0. It never applies to a reply, which is why this is greyed
                      out when &quot;This is a reply&quot; is checked.
                    </>
                  ),
                },
                {
                  term: "Contains sensitive media",
                  def: (
                    <>
                      Costs more than the blur suggests. Followers see it behind a click-through
                      interstitial — but for everyone else it&apos;s{" "}
                      <strong>dropped from recommendations entirely</strong>. It can also
                      accumulate into an account-level label if roughly 3 of your last 5 posts are
                      flagged.
                    </>
                  ),
                },
                {
                  term: "Verified / Premium checkmark",
                  def: (
                    <>
                      Raises the character limit from 280 to 4,000 and changes which
                      length-related tips apply. It is <em>not</em> treated as a ranking boost,
                      because the open-sourced ranking code doesn&apos;t contain one.
                    </>
                  ),
                },
                {
                  term: "Posts already sent in this window",
                  def: <>Drives the author-diversity multiplier described above.</>,
                },
                {
                  term: "@handle + follower count",
                  def: (
                    <>
                      Used for the cold-start check, and to remember your track record. Under
                      1,000 followers, a post under 24 hours old and under 1,000 views can be
                      lifted in ranking — but only if it&apos;s <em>already</em> ranking in the top
                      85% of a viewer&apos;s candidates on its own merits, and it&apos;s lifted toward
                      roughly rank 15, not to the top of the feed.
                    </>
                  ),
                },
              ]}
            />
          </Section>

          <Section id="signals" title="Signal breakdown">
            <P>
              The full ledger behind the score: every action, its real weight, the estimated
              probability, and what it contributed. These weights are read straight from{" "}
              <Mono>home-mixer/params/param.rs</Mono>.
            </P>
            <Table
              head={["Action", "Weight", "Worth knowing"]}
              rows={[
                ["Share via copy link", "+20.0", "The highest single weight in the entire model"],
                ["Reply", "+5.0 → +20.0", "With the mutual-follow boost on an original post"],
                ["Quote post", "+5.0", ""],
                ["Share via DM", "+5.0", ""],
                ["Follow you", "+4.0", ""],
                ["Share (generic)", "+2.0", ""],
                ["Repost", "+1.0", ""],
                ["Like", "+0.5", "The lowest positive weight — chasing likes is chasing the least valuable action"],
                ["Not interested", "−43.2", ""],
                ["Block", "−31.2", ""],
                ["Mute", "−58.8", ""],
                ["Report", "−234.0", "The largest weight in the model, by far, and it's negative"],
              ]}
            />
            <Callout tone="info" title="The asymmetry is the whole point">
              A report is worth <strong>468 likes in the negative direction</strong>. Anything that
              nudges people toward mute, block, or &quot;not interested&quot; — shouting, spammy
              formatting, engagement bait — costs far more than the engagement it might buy.
            </Callout>
          </Section>

          <Section id="risks" title="Visibility-filtering risk">
            <P>
              Ranking decides <em>order</em>. Visibility filtering decides whether your post can be
              shown at all — a separate system with three possible answers: allow, show behind an
              interstitial, or drop. Each flag here cites the specific rule file it comes from.
            </P>
            <DefList
              items={[
                {
                  term: "Link / URL verdict",
                  def: (
                    <>
                      Shorteners, raw IP links and certain cheap TLDs draw scrutiny. A
                      &quot;low quality&quot; verdict gets downranked; an &quot;unsafe&quot; one is a
                      hard drop for every non-follower. It isn&apos;t contained to one post either —
                      when a domain&apos;s reputation flips, past posts sharing it get relabelled
                      too.
                    </>
                  ),
                },
                {
                  term: "Templated / copy-paste phrasing",
                  def: (
                    <>
                      &quot;Follow for follow&quot;, &quot;link in bio&quot;, &quot;RT if you
                      agree&quot; — the exact fingerprint duplicate-text spam detection looks for
                      across accounts.
                    </>
                  ),
                },
                {
                  term: "ALL-CAPS / !!! bursts / hashtag stuffing",
                  def: (
                    <>
                      Pushes up the three most negative weights in the model. Note that a word
                      you also use as a hashtag (a ticker or brand name) is treated as a name,
                      not as shouting.
                    </>
                  ),
                },
                {
                  term: "Post age",
                  def: (
                    <>
                      Anything older than <strong>48 hours</strong> stops being eligible for For You
                      ranking altogether — a hard exclusion before scoring even runs.
                    </>
                  ),
                },
                {
                  term: "Sensitive media & repeat NSFW",
                  def: <>Interstitial for followers, dropped from recommendations for everyone else.</>,
                },
              ]}
            />
            <P className="text-white/40">
              Checks that pass are collapsed under &quot;other checks passed clean&quot; so the panel
              shows you what actually needs attention.
            </P>
          </Section>

          <Section id="tips" title="Tips">
            <P>
              Ranked by impact, and each one is tied to a weight rather than to taste. The
              highest-impact tips are almost always the same two: give people a concrete reason
              to <strong>reply</strong> (worth 10–40× a like), and give them something specific
              enough to be worth <strong>copy-link sharing</strong> (the single heaviest action).
            </P>
            <Callout tone="info" title="Not all reply hooks are equal">
              A question tied to the specifics of your post scores substantially higher here than
              a bolt-on closer like &quot;What do you think?&quot;. That&apos;s deliberate, and it
              was a bug worth fixing: the scorer used to reward the generic closer more, which
              quietly pushed every optimized post toward the same lazy ending. Two reasons it
              runs the other way now — a question that asks nothing gives a reader no reason to
              answer, and thousands of posts ending in an identical tail is precisely the
              templated-text pattern duplicate-text spam detection looks for across accounts.
            </Callout>
            <P>
              Others cover writing craft — filler words, passive voice, weak openers, stock AI
              phrasing. Those are flagged as general craft signals, not as repo-cited weights,
              and the tool labels them that way rather than dressing them up as algorithm rules.
            </P>
          </Section>

          <Section id="optimize" title="Optimize for the algorithm">
            <P>
              A deterministic, meaning-preserving pass. It applies mechanical fixes and{" "}
              <strong>keeps each one only if the score measurably improves</strong>, verified with
              the same scorer used everywhere else. &quot;Optimized&quot; here provably means
              higher-scoring, never just &quot;reworded&quot;.
            </P>
            <Table
              head={["Fix", "Why"]}
              rows={[
                ["Toned down !!! / ??? bursts", "Feeds the same spam penalty as ALL-CAPS"],
                ["Fixed ALL-CAPS shouting", "Drives mute / report / not-interested propensity"],
                ["Trimmed hashtags to 2", "Beyond a couple reads as stuffing and dilutes the post"],
                ["Removed templated CTA phrasing", "The fingerprint duplicate-text spam detection catches"],
                ["Cut filler/hedge words", "Dilutes a claim without adding information"],
                ["Added a reply hook", "Reply is worth 10–40× a like. A deterministic pass can't write a question about your specific post, so this is a varied fallback — a specific question scores materially higher"],
                ["Trimmed to the character limit", "A hard constraint — applied whether or not it raises the score"],
              ]}
            />
            <P>
              It won&apos;t rewrite a brand name or ticker that you also use as a hashtag, and it
              won&apos;t invent facts, because it only ever deletes or reshapes text you already
              wrote.
            </P>
          </Section>

          <Section id="ai" title="AI rewrite & generate from an idea">
            <P>
              Two optional layers on top of the deterministic optimizer. Both are available only
              if the deployment has an AI provider configured, and{" "}
              <strong>every AI output is still scored and gated by the same deterministic
              scorer</strong> — the AI only ever suggests, it never decides.
            </P>
            <P>
              The model isn&apos;t just told &quot;make this engaging&quot;. It&apos;s given a
              working brief on the ranking system, assembled from the same constants the scorer
              uses: the full weight table and what the ratios imply, the structural limits that
              cap reach before wording matters (reply/repost exclusion, the out-of-network
              discount, author-diversity decay, the 48-hour cutoff), the label risks that can
              drop a post outright, and the constraints of your specific post — what media is
              attached, whether it&apos;s a reply, how many times you&apos;ve posted in this
              window. That&apos;s what lets it write something worth copy-link sharing rather
              than tacking a question onto the end.
            </P>
            <DefList
              items={[
                {
                  term: "AI rewrite",
                  def: (
                    <>
                      Rewrites your existing draft, preserving your facts and voice. Candidates are
                      only shown if they score <em>higher</em> than the mechanical result. If one
                      beats it by 5+ points it&apos;s applied automatically, with a one-click Undo
                      and the fact-check warning kept in view.
                    </>
                  ),
                },
                {
                  term: "Generate from an idea",
                  def: (
                    <>
                      For when you have something to say but no draft. Give it rough notes and it
                      writes complete posts, each already run through the deterministic optimizer
                      and scored, best first.
                    </>
                  ),
                },
              ]}
            />
            <Callout tone="warn" title="Read AI output before you post it">
              An AI can rephrase a claim in a way that changes its meaning — turning a pending
              status into a finished one, for example — even when explicitly told not to. Generated
              posts are the higher risk of the two: there&apos;s no original text to check them
              against. The prompts forbid inventing facts, names, numbers and links, but{" "}
              <strong>a higher score only means it fits the algorithm&apos;s signals better, not
              that it&apos;s true</strong>. Verify every name and number yourself.
            </Callout>
          </Section>

          <Section id="import" title="Import a live post">
            <P>
              Paste any <Mono>x.com/…/status/…</Mono> link to score a post that already exists. It
              pulls the real text, media type, author follower count, verified status, post age
              and real engagement numbers, and fills in the toggles it can determine for itself —
              including whether the post was a reply and whether it was marked sensitive.
            </P>
            <P className="text-white/40">
              Useful for scoring a competitor&apos;s post, or for going back over your own to see
              what the tool would have said before you published it.
            </P>
          </Section>

          <Section id="tracking" title="Track record">
            <P>
              The part that keeps the rest of this honest. Everything above is a{" "}
              <em>prediction</em>. This checks the predictions against reality.
            </P>
            <Steps
              items={[
                <>
                  Hit <strong>Track</strong> on a draft. It&apos;s saved with the score you&apos;re
                  looking at.
                </>,
                <>Publish the post on X, in your own composer, as normal.</>,
                <>
                  Hit <strong>Check for results</strong>. Your recent timeline is matched against
                  saved drafts — tolerant of last-minute wording tweaks and X&apos;s link rewriting —
                  and the real views and engagement are pulled once the post has had a couple of
                  hours to accumulate them.
                </>,
              ]}
            />
            <SubHead>What it will and won&apos;t tell you</SubHead>
            <P>
              Once you have <strong>6 measured posts</strong>, it splits them at the median
              predicted score and compares the real numbers of the higher-scoring half against the
              lower-scoring half. If those two columns look the same, the score isn&apos;t
              predicting anything for your account — and the panel will say exactly that rather
              than spinning it.
            </P>
            <Callout tone="info" title="Why it refuses to answer early">
              Engagement data is noisy enough that a &quot;pattern&quot; drawn from three posts
              would be invented rather than observed. So nothing is claimed below 6 measured
              posts, per-fix comparisons need at least 3 posts on each side, and everything uses
              medians rather than averages so a single post that happens to take off can&apos;t
              manufacture a trend. A tool like this is only worth having if you can trust it when
              it says &quot;not yet&quot;.
            </Callout>
            <P>
              A fix is only credited to a post when the tracked text is exactly the
              optimizer&apos;s output — edit after optimizing and nothing is recorded for that
              post, because a wrong attribution would quietly corrupt the very data this exists to
              build.
            </P>

            <SubHead>Learning from your history</SubHead>
            <P>
              <strong>Learn from my history</strong> reads your published posts and their real
              numbers in one go, rather than waiting for tracked drafts to accumulate. Views and
              per-action counts are both public on a published post, so{" "}
              <Mono>replies ÷ views</Mono> is a directly measured action rate — real ground truth,
              not an estimate.
            </P>
            <P>
              With enough of those, the guessed probabilities behind your score get replaced by
              ones fitted to your own results, using ridge regression over the same features the
              scorer already extracts. It&apos;s the same shape as what Phoenix does — predict a
              probability per action, then blend with the real production weights — just learned
              at the level of &quot;posts like this one&quot; rather than per individual viewer.
            </P>
            <Callout tone="info" title="Why it can't be Phoenix itself">
              Phoenix takes a viewer, that viewer&apos;s engagement history, and a candidate post,
              and predicts what <em>that specific person</em> will do. An unpublished draft has no
              viewer, and no API exposes the private engagement history of everyone who might see
              your post — so the model is out of reach by construction, not for lack of access. No
              trained weights are published either; what ships is training code and synthetic data
              generators. The aggregate question this tool asks — &quot;how will this post do?&quot;
              — is a different and more tractable one.
            </Callout>
            <P>
              Three guards keep it honest. Nothing is fitted below 40 posts. Every fitted action
              must clear <strong>cross-validated</strong> R² before it&apos;s used at all, so a
              model that merely memorised your history is rejected rather than shipped. And the
              fit is blended with the original heuristics in proportion to how much data backs it
              — 60 posts nudges the score, 500 largely drives it.
            </P>
            <P className="text-white/40">
              What&apos;s learned is <em>relative</em>: &quot;this post should do about 1.8× your
              typical reply rate&quot;, applied to the existing scale. Real reply rates are ~1% of
              views while the priors sit near 0.4 — they were tuned to spread the 0–100 range
              usefully, not to be literal probabilities. Substituting real magnitudes would drop
              everyone&apos;s score by a dozen points and tell them nothing, so calibration
              reorders drafts rather than rebasing the scale.
            </P>
            <P className="text-white/40">
              Tracking needs a database to be attached to the deployment. Without one it reports
              itself as unavailable and everything else works exactly as before.
            </P>
          </Section>

          <Section id="ai-estimate" title="The AI second opinion">
            <P>
              <strong>Second opinion from the AI</strong> hands your draft to a model that has been
              given the real weight table, the visibility-filtering rules and the structural limits,
              and asks it one narrow question per action: how does this post compare to a typical
              post from this account?
            </P>
            <P>
              It answers in multipliers — <Mono>1.0</Mono> is &quot;typical for you&quot;,{" "}
              <Mono>2.0</Mono> is &quot;about twice as likely&quot; — never in absolute
              probabilities. Language models are genuinely good at relative judgements about text
              and genuinely bad at absolute rates, which depend on follower count, timing and
              audience, none of which are in the words. Those multipliers then feed the same real
              production weights as every other number here; nothing about the scoring maths
              changes.
            </P>
            <P>
              It runs only when you press the button, never on every keystroke, and the read applies
              to that exact draft — edit the text and the panel clears rather than describing a post
              that no longer exists.
            </P>

            <SubHead>Why it&apos;s put on trial</SubHead>
            <P>
              A language model will produce confident numbers whether or not those numbers predict
              anything, and there is no way to tell the two apart by reading them. So{" "}
              <strong>Does the AI read beat the score?</strong> in your track record scores your
              published posts twice — once with the built-in heuristics, once with the AI&apos;s
              probabilities — and reports which one ordered your <em>real</em> results better.
              Same posts, same weights, only the probabilities swapped.
            </P>
            <Callout tone="warn" title="A tie goes to the incumbent">
              The AI head is only reported as a win when it beats the heuristic outright on your own
              published history. If it loses, the panel says so and keeps saying so — an estimator
              that sounds sophisticated and predicts nothing is precisely what this tool exists to
              catch, including when it&apos;s our own. Until you run the test, the AI read is
              labelled as an untested opinion rather than a measurement.
            </Callout>
            <P className="text-white/40">
              Its influence is also damped by how much of your score already comes from a model
              fitted to your real outcomes. Both are estimating the same thing from overlapping
              evidence, so stacking them at full strength would double-count — and where measured
              outcomes exist, they are the better evidence.
            </P>
          </Section>

          <Section id="sources" title="Where the numbers come from">
            <P>
              Every weight, threshold and rule cited in this tool is read out of the open-sourced
              algorithm in this repository. The main ones:
            </P>
            <Table
              head={["What", "Source"]}
              rows={[
                ["Action weights, OON discount, cold-start params", "home-mixer/params/param.rs"],
                ["Score arithmetic, author diversity, reply/repost rescoring", "home-mixer/scorers/ranking_scorer.rs"],
                ["Cold-start boost eligibility and targeting", "home-mixer/scorers/author_cold_start.rs"],
                ["48-hour age cutoff, reply/repost OON exclusion", "home-mixer/filters/"],
                ["Drop / interstitial rules and their evaluation order", "visibility-filtering/rules/registry.rs"],
                ["Label definitions and their real effects", "under-the-hood/strato/lib/underTheHoodLabels.strato"],
                ["Spam / URL-verdict labelling rules", "botmaker-rules/scarecrow/bot/"],
              ]}
            />
            <P>
              The scoring implementation is fully commented with a citation for every number in{" "}
              <Mono>besc-engagement-checker/lib/scoring.ts</Mono>. If you think one of them is
              wrong, it&apos;s all readable — and worth telling us about.
            </P>
            <div className="flex flex-wrap gap-2 pt-1">
              <a
                href="https://github.com/BESCLLC/x-algo-BESC-twitter-engagement-"
                target="_blank"
                rel="noreferrer"
                className="flex items-center gap-1.5 rounded-full border border-white/10 bg-white/[0.03] px-3.5 py-1.5 text-xs font-medium text-white/60 transition-colors hover:text-white/90"
              >
                <Github className="h-3.5 w-3.5" />
                Read the algorithm
              </a>
              <Link
                href="/"
                className="flex items-center gap-1.5 rounded-full border border-besc-400/40 bg-besc-500/10 px-3.5 py-1.5 text-xs font-medium text-besc-200 transition-colors hover:bg-besc-500/20"
              >
                Score a post
              </Link>
            </div>
          </Section>
        </div>
      </div>

      <footer className="mx-auto flex max-w-7xl flex-col items-center gap-5 px-5 pb-14 pt-6 sm:px-8">
        <div className="relative h-10 w-10 opacity-90">
          <Image src="/besc-logo.png" alt="BESC" fill sizes="40px" className="object-contain" />
        </div>
        <SocialLinks variant="labeled" />
        <p className="max-w-xl text-balance text-center text-xs text-white/25">
          Built by <span className="text-besc-400/80">BESC</span> · weights sourced from{" "}
          <code className="font-mono text-white/35">home-mixer/params/param.rs</code> · not
          affiliated with X Corp.
        </p>
      </footer>
    </main>
  );
}

function Section({
  id,
  title,
  children,
}: {
  id: string;
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section id={id} className="glass-panel scroll-mt-6 p-6 sm:p-7">
      <h2 className="font-display text-xl font-semibold tracking-tight sm:text-2xl">{title}</h2>
      <div className="mt-4 space-y-4">{children}</div>
    </section>
  );
}

function SubHead({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="pt-1 font-display text-[15px] font-semibold text-white/80">{children}</h3>
  );
}

function P({ children, className = "" }: { children: React.ReactNode; className?: string }) {
  return (
    <p className={`text-[14.5px] leading-relaxed text-white/60 ${className}`}>{children}</p>
  );
}

function Mono({ children }: { children: React.ReactNode }) {
  return (
    <code className="rounded bg-white/[0.06] px-1.5 py-0.5 font-mono text-[12.5px] text-white/75">
      {children}
    </code>
  );
}

function Steps({ items }: { items: React.ReactNode[] }) {
  return (
    <ol className="space-y-2.5">
      {items.map((item, i) => (
        <li key={i} className="flex gap-3 text-[14.5px] leading-relaxed text-white/60">
          <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full border border-besc-400/40 bg-besc-500/10 font-mono text-[11px] text-besc-300">
            {i + 1}
          </span>
          <span className="min-w-0">{item}</span>
        </li>
      ))}
    </ol>
  );
}

function DefList({ items }: { items: { term: string; def: React.ReactNode }[] }) {
  return (
    <dl className="space-y-3">
      {items.map((item) => (
        <div key={item.term} className="rounded-xl border border-white/10 bg-black/25 p-3.5">
          <dt className="text-[13.5px] font-semibold text-white/85">{item.term}</dt>
          <dd className="mt-1 text-[13.5px] leading-relaxed text-white/55">{item.def}</dd>
        </div>
      ))}
    </dl>
  );
}

function Table({ head, rows }: { head: string[]; rows: string[][] }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-white/10">
      <table className="w-full min-w-[520px] border-collapse text-left">
        <thead>
          <tr className="bg-white/[0.04]">
            {head.map((h) => (
              <th
                key={h}
                className="px-3.5 py-2.5 text-[10.5px] font-semibold uppercase tracking-wide text-white/40"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-t border-white/[0.07]">
              {row.map((cell, j) => (
                <td
                  key={j}
                  className={`px-3.5 py-2.5 align-top text-[13px] leading-relaxed ${
                    j === 0 ? "font-medium text-white/75" : "text-white/50"
                  } ${j === 1 && /^[+−-]?[\d.]/.test(cell) ? "font-mono tabular-nums" : ""}`}
                >
                  {cell}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Callout({
  tone,
  title,
  children,
}: {
  tone: "info" | "warn";
  title: string;
  children: React.ReactNode;
}) {
  const styles =
    tone === "warn"
      ? "border-warn/35 bg-warn/[0.07]"
      : "border-besc-400/30 bg-besc-500/[0.07]";
  const titleColor = tone === "warn" ? "text-warn/90" : "text-besc-200";
  return (
    <div className={`rounded-xl border p-4 ${styles}`}>
      <p className={`text-[13px] font-semibold ${titleColor}`}>{title}</p>
      <p className="mt-1.5 text-[13.5px] leading-relaxed text-white/60">{children}</p>
    </div>
  );
}
