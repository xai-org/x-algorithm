import logging
from datetime import datetime
from strato_http.queries.data_types import (
    PostWithQuoteMetadata,
    ContentUnderstandingMetadataV2,
    PostMetadata as StratoPostMetadata,
    MediaEntity as StratoMediaEntity,
    CardMetadata as StratoCardMetadata,
    UserMetadata as StratoUserMetadata,
    BroadcastMetadata as StratoBroadcastMetadata,
    CardMetadataV2 as StratoCardMetadataV2,
    PollCardMetadata as StratoPollCardMetadata,
    GrokShareCardMetadata as StratoGrokShareCardMetadata,
    GrokShareMetadata as StratoGrokShareMetadata,
    ArticleMetadata as StratoArticleMetadata,
    ListMetadata as StratoListMetadata,
    ChatGroupMetadata as StratoChatGroupMetadata,
)
from grox.core.data_loaders.data_types import (
    Post,
    User,
    Counts,
    Card,
    CardV2,
    LegacyCard,
    UnifiedCard,
    Video,
    Image,
    VideoInfo,
    VideoVariant,
    BroadcastMetadata,
    PollCard,
    GrokShareCard,
    GrokShare,
    ArticleMetadata,
    ListMetadata,
    ChatGroupMetadata,
    AffiliatedBusiness,
)

logger = logging.getLogger(__name__)


class PostMapper:
    @classmethod
    def from_strato_post_with_quote_metadata(
        cls, post_with_quote_metadata: PostWithQuoteMetadata
    ) -> Post:
        if not post_with_quote_metadata.post:
            raise ValueError("Post metadata is required")
        post = cls.from_post_metadata_strato(post_with_quote_metadata.post)
        if post_with_quote_metadata.quotedPost:
            post.quoted_post = cls.from_post_metadata_strato(
                post_with_quote_metadata.quotedPost
            )
        return post

    @classmethod
    def from_strato_content_understanding_metadata(
        cls, content_understanding_metadata: ContentUnderstandingMetadataV2
    ) -> Post:
        if not content_understanding_metadata.postMetadata:
            raise ValueError("Post metadata is required")
        post = cls.from_strato_post_with_quote_metadata(
            content_understanding_metadata.postMetadata
        )
        if content_understanding_metadata.replyThreadPostsMetadata:
            post.ancestors = [
                cls.from_strato_post_with_quote_metadata(pm)
                for pm in content_understanding_metadata.replyThreadPostsMetadata
            ]
        if content_understanding_metadata.replyRootPostMetadata:
            root_post = cls.from_strato_post_with_quote_metadata(
                content_understanding_metadata.replyRootPostMetadata
            )
            if post.ancestors and post.ancestors[-1].id != root_post.id:
                post.ancestors.append(root_post)
            elif not post.ancestors:
                post.ancestors = [root_post]
        post.ancestors = list(reversed(post.ancestors)) if post.ancestors else []
        return post

    @classmethod
    def from_post_metadata_strato(cls, post_metadata: StratoPostMetadata) -> Post:
        user = (
            cls._from_strato_user_metadata_to_user(post_metadata.authorMetadata)
            if post_metadata.authorMetadata
            else None
        )
        counts = (
            Counts.from_strato_model(post_metadata.counts)
            if post_metadata.counts
            else None
        )
        media = (
            cls._from_strato_media_entities_to_media_list(post_metadata.media)
            if post_metadata.media
            else None
        )
        created_at = (
            datetime.fromtimestamp(post_metadata.createdAt)
            if post_metadata.createdAt
            else None
        )
        card = None
        if post_metadata.cardMetadatas:
            card = cls._from_strato_cardmetadata_to_card(post_metadata.cardMetadatas[0])
        if post_metadata.broadcastMetadata:
            broadcast_metadata = (
                cls._from_strato_broadcast_metadata_to_broadcast_metadata(
                    post_metadata.broadcastMetadata
                )
            )
        else:
            broadcast_metadata = None
        cardsV2 = None
        if post_metadata.cardMetadatasV2:
            cardsV2 = [
                cls._from_strato_cardmetadataV2_to_cardV2(cardV2)
                for cardV2 in post_metadata.cardMetadatasV2
            ]
        grok_share_metadatas = None
        if post_metadata.grokShareMetadatas:
            grok_share_metadatas = [
                cls._from_strato_grok_share_metadata(m)
                for m in post_metadata.grokShareMetadatas
            ]
        article_metadata = None
        if post_metadata.articleMetadata:
            article_metadata = cls._from_strato_article_metadata_to_article_metadata(
                post_metadata.articleMetadata
            )
        list_metadata = None
        if post_metadata.listMetadata:
            list_metadata = cls._from_strato_list_metadata_to_list_metadata(
                post_metadata.listMetadata
            )
        chat_group_metadata = None
        if post_metadata.chatGroupMetadata:
            chat_group_metadata = (
                cls._from_strato_chat_group_metadata_to_chat_group_metadata(
                    post_metadata.chatGroupMetadata
                )
            )
        return Post(
            id=str(post_metadata.postId),
            user=user,
            full_text=post_metadata.text,
            language=post_metadata.language.language
            if post_metadata.language
            else None,
            counts=counts,
            urls=post_metadata.urls if post_metadata.urls else None,
            media=media,
            quoted_post=None,
            card=card,
            broadcast_metadata=broadcast_metadata,
            created_at=created_at,
            ancestors=[],
            screenshot=None,
            cardsV2=cardsV2,
            grok_share_metadatas=grok_share_metadatas,
            article_metadata=article_metadata,
            list_metadata=list_metadata,
            chat_group_metadata=chat_group_metadata,
        )

    @classmethod
    def _from_strato_media_entities_to_media_list(
        cls, media_entities: list[StratoMediaEntity]
    ) -> list[Image | Video]:
        return [
            cls._from_strato_media_entity_post_media(media_entity)
            for media_entity in media_entities
        ]

    @classmethod
    def _from_strato_media_entity_post_media(
        cls, media_entity: StratoMediaEntity
    ) -> Image | Video:
        if media_entity.videoInfo or media_entity.animatedGifInfo:
            video_info = None
            animated_gif_info = None
            if media_entity.videoInfo:
                video_info = VideoInfo(
                    durationMillis=media_entity.videoInfo.durationMillis,
                    variants=[
                        VideoVariant(
                            url=variant.url,
                            contentType=variant.contentType,
                            bitRate=variant.bitRate,
                        )
                        for variant in media_entity.videoInfo.variants
                    ],
                )
            if media_entity.animatedGifInfo:
                animated_gif_info = VideoInfo(
                    durationMillis=None,
                    variants=[
                        VideoVariant(
                            url=variant.url,
                            contentType=variant.contentType,
                            bitRate=variant.bitRate,
                        )
                        for variant in media_entity.animatedGifInfo.variants
                    ],
                )
            return Video(
                id=str(media_entity.mediaId),
                url=media_entity.url,
                videoInfo=video_info,
                animatedGifInfo=animated_gif_info,
                convo_video=None,
            )
        else:
            return cls._from_strato_media_entity_to_image(media_entity)

    @classmethod
    def _from_strato_media_entity_to_image(
        cls, media_entity: StratoMediaEntity
    ) -> Image:
        return Image(
            id=str(media_entity.mediaId),
            url=media_entity.url,
            convo_image=None,
        )

    @classmethod
    def _from_strato_cardmetadata_to_card(
        cls, card_metadata: StratoCardMetadata
    ) -> Card:
        cover_image = None
        if card_metadata.thumbnailImage:
            cover_image = Image(url=card_metadata.thumbnailImage)
        elif card_metadata.media:
            cover_image = cls._from_strato_media_entity_to_image(card_metadata.media[0])
        return Card(
            url=None,
            domain=card_metadata.domain,
            description=card_metadata.description
            if card_metadata.description
            else None,
            title=card_metadata.title if card_metadata.title else None,
            cover_image=cover_image,
        )

    @classmethod
    def _from_strato_broadcast_metadata_to_broadcast_metadata(
        cls, broadcast_metadata: StratoBroadcastMetadata
    ) -> BroadcastMetadata:
        return BroadcastMetadata(
            broadcast_id=broadcast_metadata.broadcastId,
            media_key=broadcast_metadata.mediaKey,
            thumbnail_image=Image(url=broadcast_metadata.thumbnailUrl)
            if broadcast_metadata.thumbnailUrl
            else None,
        )

    @classmethod
    def _from_strato_article_metadata_to_article_metadata(
        cls, article_metadata: StratoArticleMetadata
    ) -> ArticleMetadata:
        return ArticleMetadata(
            title=article_metadata.title
            if hasattr(article_metadata, "title")
            else None,
            media=cls._from_strato_media_entities_to_media_list(article_metadata.media)
            if article_metadata.media
            else None,
            cover_media=cls._from_strato_media_entity_post_media(
                article_metadata.coverMedia
            )
            if article_metadata.coverMedia
            else None,
        )

    @classmethod
    def _from_strato_list_metadata_to_list_metadata(
        cls, list_metadata: StratoListMetadata
    ) -> ListMetadata:
        return ListMetadata(
            banner_image=Image(url=list_metadata.bannerImageUrl)
            if list_metadata.bannerImageUrl
            else None,
        )

    @classmethod
    def _from_strato_chat_group_metadata_to_chat_group_metadata(
        cls, metadata: StratoChatGroupMetadata
    ) -> ChatGroupMetadata:
        return ChatGroupMetadata(
            group_name=metadata.groupName,
            group_avatar=Image(url=metadata.groupAvatarUrl)
            if metadata.groupAvatarUrl
            else None,
        )

    @classmethod
    def _from_strato_poll_card_metadata_to_poll_card(
        cls, poll_card_metadata: StratoPollCardMetadata
    ) -> PollCard:
        return PollCard(
            choice_label=poll_card_metadata.choiceLabel,
            choice_image=Image(url=poll_card_metadata.choiceImageUrl)
            if poll_card_metadata.choiceImageUrl
            else None,
        )

    @classmethod
    def _from_strato_grok_share_card_metadata_to_grok_share_card(
        cls, metadata: StratoGrokShareCardMetadata
    ) -> GrokShareCard:
        return GrokShareCard(sender=metadata.sender, message=metadata.message)

    @classmethod
    def _from_strato_grok_share_metadata(
        cls, metadata: StratoGrokShareMetadata
    ) -> GrokShare:
        sender = metadata.sender.name if metadata.sender is not None else None
        return GrokShare(sender=sender, message=metadata.message)

    @classmethod
    def _from_strato_user_metadata_to_user(
        cls, user_metadata: StratoUserMetadata
    ) -> User:
        return User(
            id=user_metadata.userId,
            handle=user_metadata.userHandle,
            name=user_metadata.userName,
            bio=user_metadata.userBio,
            follower_count=user_metadata.followerCount,
            following_count=user_metadata.followingCount,
            subscription_level=user_metadata.subscriptionLevel,
            signup_country_code=user_metadata.signupCountryCode,
            is_protected=user_metadata.isProtected,
            created_at=datetime.fromtimestamp(user_metadata.createdAtMsec / 1000),
            has_missing_client_events=user_metadata.hasMissingClientEvents,
            tfe_top_country=user_metadata.tfeTopCountry,
            account_lang=user_metadata.accountLang,
            num_replies_last_24hrs=user_metadata.numRepliesLast24Hrs,
            has_risky_user_safety_label=user_metadata.hasRiskyUserSafetyLabel,
            num_legit_blocks_received_last_24hrs=user_metadata.numLegitBlocksReceivedLast24Hrs,
            urls=user_metadata.urls,
            affiliated_business=AffiliatedBusiness(
                name=user_metadata.affiliatedBusinessMetadata.name,
                url=user_metadata.affiliatedBusinessMetadata.url,
            )
            if user_metadata.affiliatedBusinessMetadata
            else None,
        )

    @classmethod
    def _from_strato_cardmetadataV2_to_cardV2(
        cls, card_metadata_v2: StratoCardMetadataV2
    ) -> CardV2:
        legacy_card = None
        if card_metadata_v2.legacyCardMetadata:
            if card_metadata_v2.legacyCardMetadata.thumbnailImage:
                thumbnail_image = Image(
                    url=card_metadata_v2.legacyCardMetadata.thumbnailImage
                )
            else:
                thumbnail_image = None
            if card_metadata_v2.legacyCardMetadata.pollCardMetadatas:
                poll_cards = [
                    cls._from_strato_poll_card_metadata_to_poll_card(poll_card_metadata)
                    for poll_card_metadata in card_metadata_v2.legacyCardMetadata.pollCardMetadatas
                ]
            else:
                poll_cards = None
            if card_metadata_v2.legacyCardMetadata.grokShareCardMetadatas:
                grok_share_cards = [
                    cls._from_strato_grok_share_card_metadata_to_grok_share_card(m)
                    for m in card_metadata_v2.legacyCardMetadata.grokShareCardMetadatas
                ]
            else:
                grok_share_cards = None
            legacy_card = LegacyCard(
                title=card_metadata_v2.legacyCardMetadata.title,
                description=card_metadata_v2.legacyCardMetadata.description,
                domain=card_metadata_v2.legacyCardMetadata.domain,
                thumbnail_image=thumbnail_image,
                poll_cards=poll_cards,
                grok_share_cards=grok_share_cards,
            )

        unified_cards = None
        if card_metadata_v2.unifiedCardMetadatas:
            unified_cards = [
                UnifiedCard(
                    title=unified_card_metadata.title,
                    description=unified_card_metadata.description,
                    url=unified_card_metadata.url,
                    media=cls._from_strato_media_entity_post_media(
                        unified_card_metadata.media
                    )
                    if unified_card_metadata.media
                    else None,
                )
                for unified_card_metadata in card_metadata_v2.unifiedCardMetadatas
            ]

        return CardV2(legacy_card=legacy_card, unified_cards=unified_cards)
