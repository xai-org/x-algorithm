use super::{Feature, Leaf, Meaning, Reads};
use crate::hydration::Hydrators;
use crate::models::{ClientSwitch, Placement, VerifyBlurSupport};
use crate::params::CountryList;
use crate::rules::context::At;
use crate::rules::rule_spec::engine::{Candidate, Engine};
use crate::rules::rule_spec::rule::RuleCond;
use crate::rules::rule_spec::Predicate;

pub(in crate::rules) enum Request {}

impl Feature for Request {
    const REQUIRED: Hydrators = Hydrators::empty();
    type Value<'a> = ();
}

impl<C: Candidate> Reads<C> for Request {
    #[inline]
    fn read<'a>(_: At<'_, 'a, C>) {}
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, strum::IntoStaticStr)]
#[strum(serialize_all = "snake_case")]
pub enum RequestLeaf {
    ViewerLoggedOut,
    IsQuotedInnerTweet,
    IsSourceTweet,
    RequestCountryIn(CountryList),
    RequestCountryIs(&'static str),
    ClientVerifyBlurSupportIs(VerifyBlurSupport),
    ClientHas(ClientSwitch),
    RequestHasSoftNsfwNewUserDrop,
}

impl Leaf for RequestLeaf {
    type Feature = Request;

    #[inline]
    fn meaning<'a, M: Meaning<'a, Request>>(self, m: M) -> M::Out {
        match self {
            Self::ViewerLoggedOut => m.run(|(), facts| facts.viewer_id().is_none()),
            Self::IsQuotedInnerTweet => {
                m.run(|(), facts| matches!(facts.placement(), Placement::InnerQuoted { .. }))
            }
            Self::IsSourceTweet => m.run(|(), facts| facts.placement() == Placement::RetweetSource),
            Self::RequestCountryIn(list) => m.run_with(list, |(), facts| {
                facts
                    .request_country()
                    .is_some_and(|country| facts.in_country_list(list, country))
            }),
            Self::RequestCountryIs(country) => m.run_with(country, |(), facts| {
                facts.request_country() == Some(country)
            }),
            Self::ClientVerifyBlurSupportIs(support) => m.run_with(support, |(), facts| {
                facts.client_capability().verify_blur_support == Some(support)
            }),
            Self::ClientHas(switch) => {
                m.run_with(switch, |(), facts| facts.client_capability().has(switch))
            }
            Self::RequestHasSoftNsfwNewUserDrop => {
                m.run(|(), facts| facts.soft_nsfw_new_user_drop())
            }
        }
    }
}

impl RequestLeaf {
    pub(in crate::rules) const fn leaf<E: Engine>(self) -> RuleCond<E> {
        RuleCond::Leaf(Predicate::Request(self))
    }
}
