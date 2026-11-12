use super::*;
use crate::clients::socialgraph_client::Graph;
use crate::filter_tweets::FilterTweetsEndpoint;
use crate::hydration::sources::Source;
use crate::hydration::sources::{suspended, InMemorySources};
use crate::result_label::state_label;
use crate::rules::WireLevel;
use crate::scenario::{tweet_data, Wire};
use vf_pb::tweet_evaluation::Outcome;
use xai_core_entities::entities::GizmoduckUser;
use xai_visibility_filtering_proto as vf_pb;
use xai_x_thrift::safety_level::SafetyLevel as ThriftLevel;

const VIEWER: u64 = 50;

async fn judge(sources: Arc<InMemorySources>, viewer: u64, levels: &[&str]) -> Vec<LevelWhy> {
    let ask = Ask {
        viewer: Some(viewer),
        post: 4,
        country: "us".to_owned(),
        client: Client::Web,
        levels: levels
            .iter()
            .map(|name| Route::parse(name).unwrap())
            .collect(),
    };
    let why = Why::new(
        sources,
        FallbackCaches::default(),
        Arc::new(CountryLists::starting_at_default()),
        ClientSwitches::for_tests(),
        Metrics::none(),
    );
    Box::pin(why.judge(&ask)).await
}

#[tokio::test]
async fn a_retweet_of_a_suspended_author_is_answered_at_every_post_level_as_its_handler_serves_it()
{
    let sources = Arc::new(
        InMemorySources::default()
            .retweet(4, 40, 6, Some(60))
            .tweet(6, 60)
            .authors(&[40])
            .user(60, suspended())
            .viewer(VIEWER, GizmoduckUser::default())
            .edge(Graph::Follows, VIEWER, 40),
    );
    let why = judge(Arc::clone(&sources), VIEWER, &[]).await;

    let hydration = why
        .iter()
        .find(|level| level.route.level == SafetyLevel::TimelineHomeHydration)
        .unwrap();
    assert_eq!(
        hydration.to_string(),
        "\
timeline_home_hydration (evaluate_tweets): drop/author_is_suspended
  rule: suspended_author/drop
  viewer: found
  failed lookups: none
  fail-open defaults: none
  retweet of 6: source_withheld, source rule: suspended_author/drop
"
    );

    let wire = Wire::new(sources);
    let filter_tweets = FilterTweetsEndpoint::new(
        Arc::clone(wire.post_evaluator()),
        ClientSwitches::for_tests(),
    );
    for answer in &why {
        let Route { level, rpc } = answer.route;
        let served = &answer.served;
        match rpc {
            Rpc::FilterTweets => {
                let proto = (0..=255)
                    .filter_map(|number| vf_pb::SafetyLevel::try_from(number).ok())
                    .find(|&proto| SafetyLevel::parse(WireLevel::FilterTweets(proto)) == Ok(level))
                    .unwrap();
                let response = filter_tweets
                    .handle(tonic::Request::new(vf_pb::VisibilityFilterRequest {
                        safety_level: proto.into(),
                        tweets: vec![vf_pb::TweetInput {
                            tweet_id: 4,
                            author_id: None,
                        }],
                        viewer_id: Some(VIEWER),
                        country_code: Some("us".to_owned()),
                    }))
                    .await
                    .unwrap()
                    .into_inner();
                let [result] = &response.results[..] else {
                    panic!("{response:?}");
                };
                assert_eq!(&filter_label(result), served, "{level:?}");
            }
            Rpc::EvaluateTweets => {
                let thrift = ThriftLevel::ENUM_VALUES
                    .iter()
                    .copied()
                    .find(|&thrift| {
                        SafetyLevel::parse(WireLevel::EvaluateTweets(thrift.0)) == Ok(level)
                    })
                    .unwrap();
                let sent = wire
                    .send_with_context(
                        &Client::Web.context(Some(VIEWER), Some("us")),
                        vf_pb::EvaluateTweetsRequest {
                            safety_level: thrift.0,
                            viewer_id: Some(VIEWER),
                            tweets: vec![tweet_data(4, None)],
                        },
                    )
                    .await
                    .unwrap();
                let [Outcome::ResultStateThriftCompact(bytes)] = &sent[..] else {
                    panic!("{sent:?}");
                };
                let state = xai_x_thrift::deserialize_compact(bytes).unwrap();
                assert_eq!(&state_label(&state), served, "{level:?}");
            }
        }
    }
}

#[tokio::test]
async fn a_failed_lookup_is_printed_with_the_default_it_put_in() {
    let sources = Arc::new(
        InMemorySources::default()
            .tweet(4, 40)
            .authors(&[40])
            .viewer(VIEWER, GizmoduckUser::default())
            .fail_graph(Graph::Blocks),
    );
    let why = judge(sources, VIEWER, &["timeline_home"]).await;

    let report: String = why.iter().map(ToString::to_string).collect();
    assert_eq!(
        report,
        "\
timeline_home (filter_tweets): allow/none
  rule: none
  viewer: found
  failed lookups: follows, blocks, mutes
  fail-open defaults: blocks, mutes
"
    );
}

#[tokio::test]
async fn a_failed_author_lookup_is_printed_for_an_id_judged_logged_out() {
    let sources = Arc::new(
        InMemorySources::default()
            .tweet(4, 40)
            .fail_key(Source::GizmoduckAuthor, 40),
    );
    let why = judge(sources, u64::MAX, &["timeline_home"]).await;

    let report: String = why.iter().map(ToString::to_string).collect();
    assert_eq!(
        report,
        "\
timeline_home (filter_tweets): drop/unspecified_reason
  rule: lookup_failed
  viewer: logged_out
  failed lookups: author
  fail-open defaults: none
"
    );
}
