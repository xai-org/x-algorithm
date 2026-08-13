import re
from datetime import datetime
from typing import Any
from urllib.parse import urlparse

import thrifts.gen.twitter.strato.columns.content_understanding.content_understanding.ttypes as t
from pydantic import BaseModel
from strato_http.queries.data_types import (
    Counts as StratoCounts,
)
from thrifts.serdes import Deserializer

from grox.core.lm.convo import Image as ConvoImage
from grox.core.lm.convo import Video as ConvoVideo


class AffiliatedBusiness(BaseModel):
    name: str | None = None
    url: str | None = None


class User(BaseModel):
    id: int | None = None
    handle: str | None = None
    name: str | None = None
    bio: str | None = None
    follower_count: int | None = None
    following_count: int | None = None
    subscription_level: str | None = None
    signup_country_code: str | None = None
    is_protected: bool | None = None
    created_at: datetime | None = None
    reply_history: list["Post"] | None = None
    profile_location: str | None = None
    has_missing_client_events: bool | None = None
    tfe_top_country: str | None = None
    account_lang: str | None = None
    num_replies_last_24hrs: float | None = None
    has_risky_user_safety_label: bool | None = None
    num_legit_blocks_received_last_24hrs: float | None = None
    urls: list[str] | None = None
    affiliated_business: AffiliatedBusiness | None = None
    recent_posts: list["Post"] | None = None

    @classmethod
    def from_thrift_model(cls, author_metadata: t.AuthorMetadata) -> "User":
        return cls(
            id=int(author_metadata.userId)
            if author_metadata.userId is not None
            else None,
            handle=author_metadata.userHandle,
            name=author_metadata.userName,
            bio=author_metadata.userBio,
            follower_count=int(author_metadata.followerCount)
            if author_metadata.followerCount is not None
            else None,
            following_count=int(author_metadata.followingCount)
            if author_metadata.followingCount is not None
            else None,
            subscription_level=author_metadata.subscriptionLevel,
            signup_country_code=author_metadata.signupCountryCode,
            is_protected=author_metadata.isProtected,
            created_at=datetime.fromtimestamp(author_metadata.createdAtMsec // 1000)
            if author_metadata.createdAtMsec
            else None,
            has_missing_client_events=author_metadata.hasMissingClientEvents
            if hasattr(author_metadata, "hasMissingClientEvents")
            else None,
            tfe_top_country=author_metadata.tfeTopCountry
            if hasattr(author_metadata, "tfeTopCountry")
            else None,
            account_lang=author_metadata.accountLang
            if hasattr(author_metadata, "accountLang")
            else None,
            num_replies_last_24hrs=author_metadata.numRepliesLast24Hrs
            if hasattr(author_metadata, "numRepliesLast24Hrs")
            else None,
            has_risky_user_safety_label=author_metadata.hasRiskyUserSafetyLabel
            if hasattr(author_metadata, "hasRiskyUserSafetyLabel")
            else None,
            num_legit_blocks_received_last_24hrs=author_metadata.numLegitBlocksReceivedLast24Hrs
            if hasattr(author_metadata, "numLegitBlocksReceivedLast24Hrs")
            else None,
            urls=author_metadata.urls,
            affiliated_business=AffiliatedBusiness(
                name=author_metadata.affiliatedBusinessMetadata.name,
                url=author_metadata.affiliatedBusinessMetadata.url,
            )
            if author_metadata.affiliatedBusinessMetadata
            else None,
        )


class Image(BaseModel):
    id: str | None = None
    url: str | None = None
    convo_image: ConvoImage | None = None
    dark_mode_image: ConvoImage | None = None
    light_mode_image: ConvoImage | None = None
    clahe_image: ConvoImage | None = None

    @classmethod
    def from_thrift_model(cls, media_entity: t.MediaEntity) -> "Image":
        return cls(
            id=str(media_entity.mediaId) if media_entity.mediaId is not None else None,
            url=media_entity.url,
            convo_image=None,
        )

    def to_convo(self, idx: int = 0) -> list[str | ConvoImage]:
        result: list[str | ConvoImage] = []
        if self.convo_image:
            result.extend([f" [{idx + 1}: Image] ", self.convo_image, " "])
        if self.dark_mode_image:
            result.extend(
                [
                    f" [{idx + 1}: Dark Mode Image is also attached for reference, it might reveal some hidden data] ",
                    self.dark_mode_image,
                    " ",
                ]
            )
        if self.light_mode_image:
            result.extend(
                [
                    f" [{idx + 1}: Light Mode Image is also attached for reference, it might reveal some hidden data] ",
                    self.light_mode_image,
                    " ",
                ]
            )
        if self.clahe_image:
            result.extend(
                [
                    f" [{idx + 1}: Contrast-Enhanced Image is also attached for reference, it might reveal some hidden data] ",
                    self.clahe_image,
                    " ",
                ]
            )
        return result


class VideoVariant(BaseModel):
    url: str | None = None
    contentType: str | None = None
    bitRate: int | None = None

    @classmethod
    def from_thrift_model(cls, video_variant: t.VideoVariant) -> "VideoVariant":
        return cls(
            url=video_variant.url,
            contentType=video_variant.contentType,
            bitRate=video_variant.bitRate,
        )


class VideoInfo(BaseModel):
    durationMillis: int | None = None
    variants: list[VideoVariant] | None = None

    @classmethod
    def from_thrift_model(
        cls, video_info: t.VideoInfo | t.AnimatedGifInfo
    ) -> "VideoInfo":
        durationMillis = (
            video_info.durationMillis if isinstance(video_info, t.VideoInfo) else None
        )
        variants = (
            [VideoVariant.from_thrift_model(variant) for variant in video_info.variants]
            if video_info.variants
            else None
        )
        return cls(
            durationMillis=durationMillis,
            variants=variants,
        )

    def get_best_variant(self) -> VideoVariant | None:
        if not self.variants:
            return None
        variants = sorted([v for v in self.variants], key=lambda v: v.bitRate or 0)
        for var in variants:
            if var.bitRate:
                return var
        return variants[0]

    def get_highest_variant(self) -> VideoVariant | None:
        if not self.variants:
            return None
        variants = sorted(
            [v for v in self.variants], key=lambda v: v.bitRate or 0, reverse=True
        )
        for var in variants:
            if var.bitRate:
                return var
        return variants[0]

    def get_highest_variant_within(
        self, duration_ms: int, max_bytes: int
    ) -> VideoVariant | None:
        if not self.variants:
            return None
        variants = sorted(
            [v for v in self.variants], key=lambda v: v.bitRate or 0, reverse=True
        )
        for var in variants:
            if var.bitRate and (var.bitRate / 8) * (duration_ms / 1000) <= max_bytes:
                return var
        return self.get_best_variant()

    def get_highest_bitrate(self) -> int | None:
        if not self.variants:
            return None
        bitrates = [v.bitRate for v in self.variants if v.bitRate is not None]
        return max(bitrates) if bitrates else None


class Video(BaseModel):
    id: str | None = None
    url: str | None = None
    videoInfo: VideoInfo | None = None
    animatedGifInfo: VideoInfo | None = None
    convo_video: ConvoVideo | None = None

    @classmethod
    def from_thrift_model(cls, media_entity: t.MediaEntity) -> "Video":
        videoInfo = (
            VideoInfo.from_thrift_model(media_entity.videoInfo)
            if media_entity.videoInfo
            else None
        )
        animatedGifInfo = (
            VideoInfo.from_thrift_model(media_entity.animatedGifInfo)
            if media_entity.animatedGifInfo
            else None
        )
        return cls(
            id=str(media_entity.mediaId) if media_entity.mediaId is not None else None,
            url=media_entity.url,
            videoInfo=videoInfo,
            animatedGifInfo=animatedGifInfo,
            convo_video=None,
        )

    def to_convo(self, idx: int = 0) -> list[str | ConvoVideo]:
        return [f" [{idx + 1}: Video] ", self.convo_video] if self.convo_video else []


class Card(BaseModel):
    url: str | None = None
    domain: str | None = None
    description: str | None = None
    title: str | None = None
    cover_image: Image | None = None

    @classmethod
    def from_thrift_model_card_metadata(
        cls, card_metadata: t.CardMetadata
    ) -> "Card | None":
        media = card_metadata.media[0] if card_metadata.media else None
        cover_image = None
        if card_metadata.thumbnailImage:
            cover_image = Image(url=card_metadata.thumbnailImage)
        elif media:
            cover_image = Image.from_thrift_model(media)
        return cls(
            url=None,
            domain=card_metadata.domain,
            description=card_metadata.description
            if card_metadata.description
            else None,
            title=card_metadata.title if card_metadata.title else None,
            cover_image=cover_image,
        )

    def to_convo(self) -> list[str | ConvoImage]:
        res: list[str | ConvoImage] = ["\n\nThis post has the following card attached:"]
        if self.title:
            res.append(f"\n\nTitle: {self.title}")
        if self.description:
            res.append(f"\n\nDescription: {self.description}")
        if self.url:
            res.append(f"\n\nUrl: {self.url}")
        if self.domain:
            res.append(f"\n\nDomain: {self.domain}")
        if self.cover_image and self.cover_image.convo_image:
            res.append(self.cover_image.convo_image)
        return res


class PollCard(BaseModel):
    choice_label: str | None = None
    choice_image: Image | None = None

    @classmethod
    def from_thrift_model(cls, poll_card_metadata: t.PollCardMetadata) -> "PollCard":
        if poll_card_metadata.choiceImageUrl:
            choice_image = Image(url=poll_card_metadata.choiceImageUrl)
        else:
            choice_image = None
        return cls(
            choice_label=poll_card_metadata.choiceLabel,
            choice_image=choice_image,
        )

    def to_convo(self) -> list[str | ConvoImage]:
        res: list[str | ConvoImage] = ["\n\n[Poll Choice] ", " "]
        if self.choice_label:
            res.append(f"\n\nLabel: {self.choice_label}")
        if self.choice_image and self.choice_image.convo_image:
            res.extend(["\n\nImage: ", self.choice_image.convo_image, " "])
        return res


class GrokShareCard(BaseModel):
    sender: str | None = None
    message: str | None = None

    @classmethod
    def from_thrift_model(cls, metadata: t.GrokShareCardMetadata) -> "GrokShareCard":
        return cls(sender=metadata.sender, message=metadata.message)

    def to_convo(self) -> list[str | ConvoImage]:
        if not self.sender and not self.message:
            return []
        res: list[str | ConvoImage] = ["\n\n[Grok Share Message] ", " "]
        if self.sender:
            res.append(f"\n\nSender: {self.sender}")
        if self.message:
            res.append(f"\n\nMessage: {self.message}")
        return res


class LegacyCard(BaseModel):
    title: str | None = None
    description: str | None = None
    domain: str | None = None
    thumbnail_image: Image | None = None
    poll_cards: list[PollCard] | None = None
    grok_share_cards: list[GrokShareCard] | None = None

    @classmethod
    def from_thrift_model(
        cls, legacy_card_metadata: t.LegacyCardMetadata
    ) -> "LegacyCard":
        if legacy_card_metadata.thumbnailImage:
            thumbnail_image = Image(url=legacy_card_metadata.thumbnailImage)
        else:
            thumbnail_image = None
        if legacy_card_metadata.pollCardMetadatas:
            poll_cards = [
                PollCard.from_thrift_model(poll_card_metadata)
                for poll_card_metadata in legacy_card_metadata.pollCardMetadatas
            ]
        else:
            poll_cards = None
        if legacy_card_metadata.grokShareCardMetadatas:
            grok_share_cards = [
                GrokShareCard.from_thrift_model(m)
                for m in legacy_card_metadata.grokShareCardMetadatas
            ]
        else:
            grok_share_cards = None
        return cls(
            title=legacy_card_metadata.title,
            description=legacy_card_metadata.description,
            domain=legacy_card_metadata.domain,
            thumbnail_image=thumbnail_image,
            poll_cards=poll_cards,
            grok_share_cards=grok_share_cards,
        )

    def to_convo(self) -> list[str | ConvoImage]:
        res: list[str | ConvoImage] = ["\n\n[Card] ", " "]
        if self.title:
            res.append(f"\n\nTitle: {self.title}")
        if self.description:
            res.append(f"\n\nDescription: {self.description}")
        if self.domain:
            res.append(f"\n\nDomain: {self.domain}")
        if self.thumbnail_image and self.thumbnail_image.convo_image:
            res.extend(["\n\n[Card Image] ", self.thumbnail_image.convo_image, " "])
        if self.poll_cards:
            res.append(
                "\n\nThis post includes a poll where user can vote their choices. The choices are as rendered below:"
            )
            for poll_card in self.poll_cards:
                res.extend(poll_card.to_convo())
        if self.grok_share_cards:
            share_parts: list[str | ConvoImage] = []
            for grok_share_card in self.grok_share_cards:
                share_parts.extend(grok_share_card.to_convo())
            if share_parts:
                res.append(
                    "\n\nThis post includes a shared Grok conversation. The conversation messages are as rendered below:"
                )
                res.extend(share_parts)
        return res


class UnifiedCard(BaseModel):
    title: str | None = None
    description: str | None = None
    url: str | None = None
    media: Image | Video | None = None

    @classmethod
    def from_thrift_model(
        cls, unified_card_metadata: t.UnifiedCardMetadata
    ) -> "UnifiedCard":
        if unified_card_metadata.media:
            if (
                unified_card_metadata.media.videoInfo
                or unified_card_metadata.media.animatedGifInfo
            ):
                media = Video.from_thrift_model(unified_card_metadata.media)
            else:
                media = Image.from_thrift_model(unified_card_metadata.media)
        else:
            media = None
        return cls(
            title=unified_card_metadata.title,
            description=unified_card_metadata.description,
            url=unified_card_metadata.url,
            media=media,
        )

    def to_convo(self) -> list[str | ConvoImage | ConvoVideo]:
        res: list[str | ConvoImage | ConvoVideo] = ["\n\n[Card] ", " "]
        if self.title:
            res.append(f"\n\nTitle: {self.title}")
        if self.description:
            res.append(f"\n\nDescription: {self.description}")
        if self.url:
            res.append(f"\n\nUrl: {self.url}")
        if self.media:
            if isinstance(self.media, Image) and self.media.convo_image:
                res.extend(["\n\n[Card Image] ", self.media.convo_image, " "])
            elif isinstance(self.media, Video) and self.media.convo_video:
                res.extend(["\n\n[Card Video] ", self.media.convo_video, " "])
        return res


class CardV2(BaseModel):
    legacy_card: LegacyCard | None = None
    unified_cards: list[UnifiedCard] | None = None

    @classmethod
    def from_thrift_model(cls, card_metadata_v2: t.CardMetadataV2) -> "CardV2":
        if card_metadata_v2.legacyCardMetadata:
            legacy_card = LegacyCard.from_thrift_model(
                card_metadata_v2.legacyCardMetadata
            )
        else:
            legacy_card = None
        if card_metadata_v2.unifiedCardMetadatas:
            unified_cards = [
                UnifiedCard.from_thrift_model(unified_card_metadata)
                for unified_card_metadata in card_metadata_v2.unifiedCardMetadatas
            ]
        else:
            unified_cards = None
        return cls(legacy_card=legacy_card, unified_cards=unified_cards)

    def to_convo(self) -> list[str | ConvoImage | ConvoVideo]:
        res: list[str | ConvoImage | ConvoVideo] = []
        if self.legacy_card:
            res.extend(self.legacy_card.to_convo())
        if self.unified_cards:
            for unified_card in self.unified_cards:
                res.extend(unified_card.to_convo())
        return res


class Counts(BaseModel):
    retweets: int | None = None
    replies: int | None = None
    likes: int | None = None
    quotes: int | None = None
    bookmarks: int | None = None

    @classmethod
    def from_strato_model(cls, counts: StratoCounts) -> "Counts":
        return cls(
            retweets=counts.retweetCount,
            replies=counts.replyCount,
            likes=counts.favoriteCount,
            quotes=counts.quoteCount,
            bookmarks=counts.bookmarkCount,
        )

    @classmethod
    def from_thrift_model(cls, counts: t.StatusCounts) -> "Counts":
        return cls(
            retweets=int(counts.retweetCount)
            if counts.retweetCount is not None
            else None,
            replies=int(counts.replyCount) if counts.replyCount is not None else None,
            likes=int(counts.favoriteCount)
            if counts.favoriteCount is not None
            else None,
            quotes=int(counts.quoteCount) if counts.quoteCount is not None else None,
            bookmarks=int(counts.bookmarkCount)
            if counts.bookmarkCount is not None
            else None,
        )


class BroadcastMetadata(BaseModel):
    broadcast_id: str
    media_key: str | None = None
    thumbnail_image: Image | None = None
    video: Video | None = None

    @classmethod
    def from_thrift_model(
        cls, broadcastMetadata: t.BroadcastMetadata
    ) -> "BroadcastMetadata":
        return cls(
            broadcast_id=broadcastMetadata.broadcastId,
            media_key=broadcastMetadata.mediaKey,
            thumbnail_image=Image(url=broadcastMetadata.thumbnailUrl)
            if broadcastMetadata.thumbnailUrl
            else None,
        )

    def to_convo(self) -> list[str | ConvoImage]:
        res: list[str | ConvoImage] = [
            "\n\nThis post has the following broadcast metadata attached:"
        ]
        if self.thumbnail_image and self.thumbnail_image.convo_image:
            res.append(self.thumbnail_image.convo_image)
        if self.video and self.video.convo_video:
            res.append(self.video.convo_video)
        return res


class ListMetadata(BaseModel):
    banner_image: Image | None = None

    @classmethod
    def from_thrift_model(cls, listMetadata: t.ListMetadata) -> "ListMetadata":
        return cls(
            banner_image=Image(url=listMetadata.bannerImageUrl)
            if listMetadata.bannerImageUrl
            else None,
        )

    def to_convo(self) -> list[str | ConvoImage]:
        res: list[str | ConvoImage] = []
        if self.banner_image and self.banner_image.convo_image:
            res.append(self.banner_image.convo_image)
        if res:
            return ["\n\nThis post has the following list metadata attached:"] + res
        else:
            return res


class ChatGroupMetadata(BaseModel):
    group_name: str | None = None
    group_avatar: Image | None = None

    @classmethod
    def from_thrift_model(cls, metadata: t.ChatGroupMetadata) -> "ChatGroupMetadata":
        return cls(
            group_name=metadata.groupName,
            group_avatar=Image(url=metadata.groupAvatarUrl)
            if metadata.groupAvatarUrl
            else None,
        )

    def to_convo(self) -> list[str | ConvoImage]:
        res: list[str | ConvoImage] = []
        if self.group_name:
            res.append(f"\nGroup name: {self.group_name}")
        if self.group_avatar and self.group_avatar.convo_image:
            res.append(self.group_avatar.convo_image)
        if res:
            return [
                "\n\nThis post has the following chat group invite metadata attached:"
            ] + res
        return res


class ArticleMetadata(BaseModel):
    title: str | None = None
    media: list[Image | Video] | None = None
    cover_media: Image | None = None

    @classmethod
    def _transform_media_entities(
        cls, media_entities: list[t.MediaEntity]
    ) -> list[Image | Video]:
        media_list: list[Image | Video] = []
        for entity in media_entities:
            if entity.videoInfo or entity.animatedGifInfo:
                video = Video.from_thrift_model(entity)
                media_list.append(video)
            else:
                image = Image.from_thrift_model(entity)
                media_list.append(image)
        return media_list

    @classmethod
    def from_thrift_model(cls, articleMetadata: t.ArticleMetadata) -> "ArticleMetadata":
        return cls(
            title=articleMetadata.title if hasattr(articleMetadata, "title") else None,
            media=cls._transform_media_entities(articleMetadata.media)
            if articleMetadata.media
            else None,
            cover_media=cls._transform_media_entities([articleMetadata.coverMedia])[0]
            if articleMetadata.coverMedia
            else None,
        )

    def to_convo(self) -> list[str | ConvoImage | ConvoVideo]:
        res: list[str | ConvoImage | ConvoVideo] = []
        if self.title:
            res.append(f"\n[Article Title] {self.title}")
        if self.cover_media:
            res.extend(["\n[Article Cover] "] + self.cover_media.to_convo())
        if self.media:
            for oneMedia in self.media:
                res.extend(oneMedia.to_convo())
        if res:
            return ["\n\nThis post has the following article metadata attached:"] + res
        else:
            return res


class Reply(BaseModel):
    inReplyToStatus: str | None = None
    inReplyToUser: str | None = None
    inReplyToScreenName: str | None = None


class Post(BaseModel):
    id: str
    user: User | None = None
    full_text: str | None = None
    language: str | None = None
    counts: Counts | None = None
    urls: list[str] | None = None
    media: list[Image | Video] | None = None
    url_videos: list[Video] | None = None
    media_descriptions: list[str] | None = None
    quoted_post: "Post | None" = None
    card: Card | None = None
    broadcast_metadata: BroadcastMetadata | None = None
    created_at: datetime | None = None
    ancestors: list["Post"] | None = None
    screenshot: Image | None = None
    reply: Reply | None = None
    cardsV2: list[CardV2] | None = None
    article_metadata: ArticleMetadata | None = None
    descendants: list["Post"] | None = None
    user_agent: str | None = None
    app_attestation_status: str | None = None
    is_pasted: bool | None = None
    composition_source: str | None = None
    safety_labels: list[str] | None = None
    list_metadata: ListMetadata | None = None
    chat_group_metadata: ChatGroupMetadata | None = None

    def get_images(self) -> list[Image]:
        images: list[Image] = []
        if self.media:
            for m in self.media:
                if isinstance(m, Image):
                    images.append(m)
        if self.quoted_post and self.quoted_post.media:
            for m in self.quoted_post.media:
                if isinstance(m, Image):
                    images.append(m)
        return images

    def get_fav_count(self) -> int:
        if self.counts:
            like_count = self.counts.likes or 0
            return like_count
        else:
            return 0

    def _collect_urls(self) -> list[str]:
        urls: list[str] = list(self.urls or [])
        if self.user and self.user.urls:
            urls.extend(self.user.urls)
        if self.card and hasattr(self.card, "url") and self.card.url:
            urls.append(self.card.url)
        if self.cardsV2:
            for cv in self.cardsV2:
                if hasattr(cv, "unified_cards") and cv.unified_cards:
                    for uc in cv.unified_cards:
                        if hasattr(uc, "url") and uc.url:
                            urls.append(uc.url)
        return urls

    def extract_domains(self) -> list[str]:
        def hostname(url: str) -> str | None:
            try:
                return urlparse(url).hostname
            except Exception:
                return None

        domains = set()
        for url in self._collect_urls():
            h = (hostname(url) or "").strip().lower()
            if h and re.fullmatch(r"[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?", h):
                domains.add(h)
        return list(domains)

    @classmethod
    def from_post_metadata(cls, post_metadata: t.PostMetadataV2) -> "Post":
        user = (
            User.from_thrift_model(post_metadata.authorMetadata)
            if post_metadata.authorMetadata
            else None
        )
        counts = (
            Counts.from_thrift_model(post_metadata.counts)
            if post_metadata.counts
            else None
        )
        media = (
            cls._transform_media_entities(post_metadata.media)
            if post_metadata.media
            else None
        )
        created_at = (
            datetime.fromtimestamp(post_metadata.createdAt)
            if post_metadata.createdAt
            else None
        )
        if post_metadata.cardMetadatas:
            card = Card.from_thrift_model_card_metadata(post_metadata.cardMetadatas[0])
        else:
            card = None
        if post_metadata.broadcastMetadata:
            broadcast_metadata = BroadcastMetadata.from_thrift_model(
                post_metadata.broadcastMetadata
            )
        else:
            broadcast_metadata = None
        if post_metadata.cardMetadatasV2:
            cardsV2 = [
                CardV2.from_thrift_model(cardV2)
                for cardV2 in post_metadata.cardMetadatasV2
            ]
        else:
            cardsV2 = None
        if post_metadata.articleMetadata:
            article_metadata = ArticleMetadata.from_thrift_model(
                post_metadata.articleMetadata
            )
        else:
            article_metadata = None
        if post_metadata.listMetadata:
            list_metadata = ListMetadata.from_thrift_model(post_metadata.listMetadata)
        else:
            list_metadata = None
        if post_metadata.chatGroupMetadata:
            chat_group_metadata = ChatGroupMetadata.from_thrift_model(
                post_metadata.chatGroupMetadata
            )
        else:
            chat_group_metadata = None

        return cls(
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
            article_metadata=article_metadata,
            user_agent=post_metadata.userAgent
            if hasattr(post_metadata, "userAgent")
            else None,
            app_attestation_status=post_metadata.appAttestationStatus
            if hasattr(post_metadata, "appAttestationStatus")
            else None,
            is_pasted=post_metadata.isPasted
            if hasattr(post_metadata, "isPasted")
            else None,
            composition_source=post_metadata.compositionSource
            if hasattr(post_metadata, "compositionSource")
            else None,
            list_metadata=list_metadata,
            chat_group_metadata=chat_group_metadata,
        )

    @classmethod
    def _transform_media_entities(
        cls, media_entities: list[t.MediaEntity]
    ) -> list[Image | Video]:
        media_list: list[Image | Video] = []
        for entity in media_entities:
            if entity.videoInfo or entity.animatedGifInfo:
                video = Video.from_thrift_model(entity)
                media_list.append(video)
            else:
                image = Image.from_thrift_model(entity)
                media_list.append(image)
        return media_list

    @classmethod
    def from_post_quote_metadata(
        cls, post_quote_metadata: t.PostQuoteMetadata
    ) -> "Post":
        if not post_quote_metadata.post:
            raise ValueError("Post metadata is required")
        post = cls.from_post_metadata(post_quote_metadata.post)
        if post_quote_metadata.quotedPost:
            post.quoted_post = cls.from_post_metadata(post_quote_metadata.quotedPost)
        return post

    @classmethod
    def from_content_understanding_metadata(
        cls, metadata: t.ContentUnderstandingMetadataV2
    ) -> "Post":
        if not metadata.postMetadata:
            raise ValueError("Post metadata is required")
        post = cls.from_post_quote_metadata(metadata.postMetadata)
        if metadata.replyThreadPostsMetadata:
            post.ancestors = [
                cls.from_post_quote_metadata(pm)
                for pm in metadata.replyThreadPostsMetadata
            ]
        if metadata.replyRootPostMetadata:
            root_post = cls.from_post_quote_metadata(metadata.replyRootPostMetadata)
            if post.ancestors and post.ancestors[-1].id != root_post.id:
                post.ancestors.append(root_post)
            elif not post.ancestors:
                post.ancestors = [root_post]
        post.ancestors = list(reversed(post.ancestors)) if post.ancestors else []
        return post

    @classmethod
    def from_thrift_content_understanding_metadata(cls, message_value: Any) -> "Post":
        metadata = Deserializer.deserialize(
            message_value, t.ContentUnderstandingMetadataV2
        )
        return cls.from_content_understanding_metadata(metadata)
