// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 X.AI Corp.
use crate::ActionName;

pub fn is_positive_brand_engagement(action_name: ActionName) -> bool {
    matches!(
        action_name,
        ActionName::ServerTweetFav
            | ActionName::ServerTweetReply
            | ActionName::ServerTweetRetweet
            | ActionName::ClientTweetFollowAuthor
            | ActionName::ClientTweetClick
            | ActionName::ClientTweetOpenLink
            | ActionName::ClientTweetClickProfile
            | ActionName::ClientTweetPhotoExpand
            | ActionName::ClientTweetCarouselSwipeNext
            | ActionName::ClientTweetCarouselSwipePrevious
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positive_brand_engagement_accepts_expected_actions() {
        let positive = [
            ActionName::ServerTweetFav,
            ActionName::ServerTweetReply,
            ActionName::ServerTweetRetweet,
            ActionName::ClientTweetFollowAuthor,
            ActionName::ClientTweetClick,
            ActionName::ClientTweetOpenLink,
            ActionName::ClientTweetClickProfile,
            ActionName::ClientTweetPhotoExpand,
            ActionName::ClientTweetCarouselSwipeNext,
            ActionName::ClientTweetCarouselSwipePrevious,
        ];
        for &action in &positive {
            assert!(
                is_positive_brand_engagement(action),
                "{action:?} should be a positive brand engagement"
            );
        }
    }

    #[test]
    fn positive_brand_engagement_rejects_non_positive_actions() {
        let non_positive = [
            ActionName::ClientTweetRenderImpression,
            ActionName::ClientTweetVideoQualityView,
            ActionName::ClientTweetVideoMrcView,
            ActionName::ClientTweetVideoView,
            ActionName::ClientTweetVideoViewThreshold,
            ActionName::ClientTweetVideoPlayFromTap,
            ActionName::ClientTweetBlockAuthor,
            ActionName::ClientTweetMuteAuthor,
            ActionName::ClientTweetReport,
            ActionName::ClientTweetNotInterestedIn,
            ActionName::ClientTweetRecapDwelled,
            ActionName::ClientTweetRecapNotDwelled,
            ActionName::ClientExternalLinkSession,
            ActionName::ClientTweetTakeScreenshot,
            ActionName::InvalidName,
            ActionName::ClientTweetVideo6secView,
            ActionName::ClientTweetVideoPlaybackStart,
            ActionName::ClientTweetVideoPlayback25,
            ActionName::ClientTweetVideoPlayback50,
            ActionName::ClientTweetVideoPlayback75,
            ActionName::ClientTweetVideoPlaybackComplete,
            ActionName::ClientTweetVideoPlaybackStartWith50pctView,
            ActionName::ClientTweetVideoPlayback50With50pctView,
            ActionName::ClientTweetUnfollowAuthor,
        ];
        for &action in &non_positive {
            assert!(
                !is_positive_brand_engagement(action),
                "{action:?} should NOT be a positive brand engagement"
            );
        }
    }
}
