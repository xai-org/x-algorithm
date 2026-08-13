import asyncio
import logging
import os
import traceback
from urllib.parse import urlparse

from blobstore_http.cdn_downloader import CDNDownloader
from grox_fetcher_client import GroxFetcherClient
from html_render.tweet_render_for_grox import TweetRenderForGrox
from monitor.metrics import Metrics
from strato_http.queries.video_subtitle import StratoVideoSubtitle
from tenacity import retry, stop_after_attempt, wait_fixed
from video_tools.image import enhance_image_with_clahe, process_image_bytes, resize_tile
from video_tools.subtitles import SubtitleAligner
from video_tools.video_frames import VideoFramesExtractor

from grox.config.config import grox_config
from grox.core.clients.nightowl_client import NightOwlClient
from grox.core.data_loaders.data_types import BroadcastMetadata, Image, Post, Video
from grox.core.data_loaders.descendant_hydrator import DescendantHydrator
from grox.core.data_loaders.descendant_id_provider import DescendantIdProvider
from grox.core.lm.convo import Image as ConvoImage
from grox.core.lm.convo import Video as ConvoVideo

logger = logging.getLogger(__name__)

_VIDEO_EXTENSIONS = (".mp4", ".mov", ".webm")
_VIDEO_DURATION_LIMIT_MINUTES = 360
_VIDEO_MAX_ESTIMATED_BYTES = 2 * 1024**3
_MAX_URL_VIDEOS_PER_POST = 7


def _create_descendant_id_provider() -> DescendantIdProvider | None:
    nightowl_config = grox_config.nightowl
    if nightowl_config is not None:
        logger.info("Creating NightOwl descendant ID provider")
        client = NightOwlClient(
            endpoint=nightowl_config.endpoint,
            timeout=nightowl_config.timeout,
        )
        return DescendantIdProvider(client)

    logger.warning("NightOwl config not set; descendant ID provider unavailable")
    return None


class MediaLoader:
    strato_video_subtitle = StratoVideoSubtitle()
    descendant_id_provider = _create_descendant_id_provider()
    descendant_hydrator = DescendantHydrator()
    cdn_downloader = CDNDownloader()
    tweet_render = TweetRenderForGrox(pool_size=2, max_contexts_per_browser=5)
    grox_fetcher_client = GroxFetcherClient(use_sharding=False, timeout=120)

    @classmethod
    def _metrics_attributes(cls) -> dict[str, str]:
        return {"pid": str(os.getpid())}

    @classmethod
    @retry(stop=stop_after_attempt(2), wait=wait_fixed(1), reraise=True)
    async def _download(cls, url: str) -> bytes | None:
        return await cls.cdn_downloader.getNonGraceful(url)

    @classmethod
    def _ton_vid_url(cls, url: str) -> str | None:
        parsed = urlparse(url)
        path_parts = parsed.path.strip("/").split("/")
        media_type, blob_id = path_parts[0], path_parts[1]
        if not str.isdigit(blob_id):
            return None
        remaining_path = [part for part in path_parts[2:] if part != "pu"]
        ext = remaining_path[-1].split(".")[-1] if remaining_path else "mp4"
        remaining_path[-1] = f"{blob_id}.{ext}"
        new_path = "/".join([media_type, blob_id] + remaining_path)
        return f"https://bs.atla.twitter.com/{new_path}"

    @classmethod
    def get_fav_count(cls, post: Post) -> int:
        if post and post.counts:
            like_count = post.counts.likes or 0
            return like_count
        else:
            return 0

    @classmethod
    def post_media_is_screenshot_target(cls, post: Post) -> bool:
        if not post:
            return False
        image_count = 0
        video_count = 0
        if post.media:
            for medium in post.media:
                if isinstance(medium, Image):
                    image_count += 1
                elif isinstance(medium, Video):
                    video_count += 1
        return image_count >= 2 and video_count == 0

    @classmethod
    async def hydrate_screenshot(cls, post: Post) -> None:
        if not post.screenshot or not post.screenshot.convo_image:
            Metrics.counter("media_loader.hydrate_screenshot.count").add(
                1, attributes=cls._metrics_attributes()
            )
            try:
                screenshot_bytes = await cls.tweet_render.take_screenshot(post.id)
                if screenshot_bytes:
                    resized = resize_tile(
                        screenshot_bytes,
                        grox_config.media_hydration.deluxe_image_tile_size,
                    )
                    post.screenshot = Image(
                        id=post.id, url=None, convo_image=ConvoImage(content=resized)
                    )
                else:
                    logger.error(f"screenshot bytes for post {post.id}, returned empty")
            except Exception:
                logger.error(
                    f"failed to hydrate screenshot for post {post.id}, error: {traceback.format_exc()}"
                )

    @classmethod
    async def hydrate_post(cls, post: Post) -> None:
        Metrics.counter("media_loader.hydrate_post.count").add(
            1, attributes=cls._metrics_attributes()
        )
        is_high_fav = (
            cls.get_fav_count(post)
            >= grox_config.media_hydration.deluxe_fav_count_threshold
        )

        if is_high_fav and not post.ancestors:
            if cls.descendant_id_provider is None:
                logger.warning("No descendant id provider available")
            else:
                try:
                    descendant_ids = await cls.descendant_id_provider.fetch(
                        int(post.id)
                    )
                    hydrated = await cls.descendant_hydrator.hydrate(descendant_ids)
                    post.descendants = hydrated if len(hydrated) >= 3 else []
                except Exception:
                    Metrics.counter("media_loader.hydrate_post.error").add(
                        1, attributes=cls._metrics_attributes()
                    )
                    logger.error(
                        f"Failed to fetch/hydrate descendants, error: {traceback.format_exc()}"
                    )
                    post.descendants = []

        ranked_posts = cls.ranked_posts_associated(post)
        max_post_count = grox_config.media_hydration.max_post_count
        posts_to_hydrate = ranked_posts[:max_post_count]
        media_tasks = [
            cls.hydrate_media(
                post_to_hydrate, post.id == post_to_hydrate.id, is_high_fav
            )
            for post_to_hydrate in posts_to_hydrate
        ]
        await asyncio.gather(*media_tasks)

        if (
            (
                cls.post_media_is_screenshot_target(post)
                or cls.post_media_is_screenshot_target(post.quoted_post)
            )
            and not post.ancestors
            and is_high_fav
        ):
            await cls.hydrate_screenshot(post)

    @classmethod
    def ranked_posts_associated(cls, post: Post) -> list[Post]:
        posts = sorted(
            post.ancestors or [],
            key=lambda p: p.counts.likes if p.counts and p.counts.likes else 0,
        )
        posts.append(post)
        return posts[::-1]

    @classmethod
    async def _hydrate_url_video(
        cls, video: Video, is_main_post: bool, is_high_fav: bool
    ) -> None:
        try:
            await cls.hydrate_video(video, is_main_post, is_high_fav)
        except Exception:
            Metrics.counter("media_loader.hydrate_url_video_failed.count").add(
                1, attributes=cls._metrics_attributes()
            )
            logger.warning(
                f"failed to hydrate URL video id={video.id} url={video.url} "
                f"is_main_post={is_main_post} is_high_fav={is_high_fav}, error: {traceback.format_exc()}"
            )

    @classmethod
    def _extract_video_urls(cls, post: Post) -> list[str]:
        if not post.urls:
            return []
        existing_video_urls: set[str] = set()
        if post.media:
            for medium in post.media:
                if isinstance(medium, Video) and medium.url:
                    existing_video_urls.add(medium.url)
        video_urls = []
        for url in post.urls:
            parsed = urlparse(url)
            if (
                parsed.path.lower().endswith(_VIDEO_EXTENSIONS)
                and url not in existing_video_urls
            ):
                video_urls.append(url)
                if len(video_urls) >= _MAX_URL_VIDEOS_PER_POST:
                    break
        return video_urls

    @classmethod
    async def hydrate_media(
        cls, post: Post, is_main_post: bool = False, is_high_fav: bool = False
    ) -> None:
        logger.info(f"hydrating medium {post.id}", extra=cls._metrics_attributes())
        tasks = []
        if post.media:
            should_enable_clahe_enhancement = (
                grox_config.media_hydration.enable_clahe_enhancement
                and is_main_post
                and is_high_fav
            )
            should_enable_light_dark_enhancement = (
                grox_config.media_hydration.enable_light_dark_enhancement
                and is_main_post
            )
            for medium in post.media:
                if isinstance(medium, Image):
                    tasks.append(
                        cls.hydrate_image(
                            medium,
                            should_enable_light_dark_enhancement,
                            is_high_fav and is_main_post,
                            should_enable_clahe_enhancement,
                        )
                    )
                elif isinstance(medium, Video):
                    tasks.append(
                        cls.hydrate_video(
                            medium,
                            is_main_post,
                            is_high_fav,
                            should_enable_clahe_enhancement,
                        )
                    )
        if post.broadcast_metadata and post.broadcast_metadata.thumbnail_image:
            tasks.append(cls.hydrate_image(post.broadcast_metadata.thumbnail_image))
            tasks.append(cls.hydrate_broadcast(post.broadcast_metadata))
        if post.cardsV2:
            for cardV2 in post.cardsV2:
                if cardV2.legacy_card:
                    if cardV2.legacy_card.thumbnail_image:
                        tasks.append(
                            cls.hydrate_image(
                                cardV2.legacy_card.thumbnail_image, False, False
                            )
                        )
                    if cardV2.legacy_card.poll_cards:
                        for poll_card in cardV2.legacy_card.poll_cards:
                            if poll_card.choice_image:
                                tasks.append(
                                    cls.hydrate_image(
                                        poll_card.choice_image, False, False
                                    )
                                )
                elif cardV2.unified_cards:
                    for unified_card in cardV2.unified_cards:
                        if unified_card.media:
                            if isinstance(unified_card.media, Image):
                                tasks.append(
                                    cls.hydrate_image(unified_card.media, False, False)
                                )
                            elif isinstance(unified_card.media, Video):
                                tasks.append(
                                    cls.hydrate_video(
                                        unified_card.media, is_main_post, False
                                    )
                                )
        if post.article_metadata and is_high_fav:
            if post.article_metadata.media:
                for medium in post.article_metadata.media:
                    if isinstance(medium, Image):
                        tasks.append(cls.hydrate_image(medium, False, False))
                    elif isinstance(medium, Video):
                        tasks.append(cls.hydrate_video(medium, is_main_post, False))
            if post.article_metadata.cover_media:
                tasks.append(
                    cls.hydrate_image(post.article_metadata.cover_media, False, False)
                )
        if post.list_metadata and is_high_fav:
            if post.list_metadata.banner_image:
                tasks.append(
                    cls.hydrate_image(post.list_metadata.banner_image, False, False)
                )
        if post.chat_group_metadata and is_high_fav:
            if post.chat_group_metadata.group_avatar:
                tasks.append(
                    cls.hydrate_image(
                        post.chat_group_metadata.group_avatar, False, False
                    )
                )
        if is_main_post and is_high_fav:
            video_urls = cls._extract_video_urls(post)
            if video_urls:
                post.url_videos = [Video(url=url) for url in video_urls]
                for video in post.url_videos:
                    tasks.append(
                        cls._hydrate_url_video(video, is_main_post, is_high_fav)
                    )
        await asyncio.gather(*tasks)
        if post.quoted_post:
            await cls.hydrate_media(post.quoted_post, is_main_post, is_high_fav)

    @classmethod
    async def hydrate_image(
        cls,
        image: Image,
        enable_light_dark_enhancement: bool = False,
        is_deluxe: bool = False,
        enable_clahe_enhancement: bool = False,
    ) -> None:
        if not image.url:
            return
        Metrics.counter("media_loader.hydrate_image.count").add(
            1, attributes=cls._metrics_attributes()
        )
        url = image.url
        try:
            bs = await cls._download(url)
        except Exception:
            logger.error(
                f"failed to download image {url}, error: {traceback.format_exc()}"
            )
            Metrics.counter("media_loader.hydrate_image_failed.count").add(
                1, attributes=cls._metrics_attributes()
            )
            raise
        if not bs:
            return
        image_tile_size = (
            grox_config.media_hydration.deluxe_image_tile_size
            if is_deluxe
            else grox_config.media_hydration.image_tile_size
        )
        logger.info(f"image config image_tile_size {image_tile_size}")

        if enable_clahe_enhancement:
            try:
                image.clahe_image = ConvoImage(
                    content=enhance_image_with_clahe(bs, image_tile_size)
                )
            except Exception:
                logger.info(
                    f"failed to apply CLAHE enhancement for image {image.url}, error: {traceback.format_exc()}"
                )
                Metrics.counter(
                    "media_loader.clahe_enhancement_image_failed.count"
                ).add(1, attributes=cls._metrics_attributes())

        if enable_light_dark_enhancement:
            processed = process_image_bytes(bs, image_tile_size)
            image.convo_image = ConvoImage(content=processed.convo_image)
            if processed.dark_mode_image:
                image.dark_mode_image = ConvoImage(content=processed.dark_mode_image)
            if processed.light_mode_image:
                image.light_mode_image = ConvoImage(content=processed.light_mode_image)
        else:
            image.convo_image = ConvoImage(content=resize_tile(bs, image_tile_size))

    @classmethod
    async def hydrate_video(
        cls,
        video: Video,
        is_main_post: bool = False,
        is_high_fav: bool = False,
        enable_clahe_enhancement: bool = False,
    ) -> None:
        url = None
        if video.videoInfo and video.videoInfo.durationMillis:
            duration_ms = video.videoInfo.durationMillis
            if duration_ms < 1000 * 60 * 30:
                Metrics.counter("media_loader.hydrate_video_under_30m.count").add(
                    1, attributes=cls._metrics_attributes()
                )
            if duration_ms >= 1000 * 60 * 30:
                Metrics.counter("media_loader.hydrate_video_over_30m.count").add(
                    1, attributes=cls._metrics_attributes()
                )
            if duration_ms >= 1000 * 60 * 60 * 5:
                Metrics.counter("media_loader.hydrate_video_over_5hr.count").add(
                    1, attributes=cls._metrics_attributes()
                )

            duration_limit_ms = _VIDEO_DURATION_LIMIT_MINUTES * 60 * 1000
            if duration_ms <= duration_limit_ms:
                best_variant = (
                    video.videoInfo.get_highest_variant_within(
                        duration_ms, _VIDEO_MAX_ESTIMATED_BYTES
                    )
                    if is_main_post and is_high_fav
                    else video.videoInfo.get_best_variant()
                )
                if best_variant and best_variant.url:
                    url = best_variant.url
            else:
                logger.info(
                    f"Skipping video download since duration is {duration_ms} and limit is {duration_limit_ms}"
                )

        elif video.animatedGifInfo:
            Metrics.counter("media_loader.hydrate_gif.count").add(
                1, attributes=cls._metrics_attributes()
            )
            logger.info("Downloading a gif")
            best_variant = (
                video.animatedGifInfo.get_highest_variant()
                if is_main_post and is_high_fav
                else video.animatedGifInfo.get_best_variant()
            )
            if best_variant and best_variant.url:
                url = best_variant.url
        else:
            url = video.url
        if not url:
            return
        Metrics.counter("media_loader.hydrate_video.count").add(
            1, attributes=cls._metrics_attributes()
        )
        try:
            video_bytes = await cls._download(url)
            if not video_bytes:
                url = cls._ton_vid_url(url)
                if url:
                    video_bytes = await cls._download(url)
            if not video_bytes:
                Metrics.counter("media_loader.hydrate_video_failed.count").add(
                    1, attributes=cls._metrics_attributes()
                )
                return
            subtitles = None
            if video.id:
                try:
                    subtitles = await cls.fetch_video_subtitles(video.id)
                except Exception:
                    logger.warning(
                        f"failed to fetch subtitles for video {video.id}, error: {traceback.format_exc()}"
                    )
                    Metrics.counter(
                        "media_loader.fetch_video_subtitles_failed.count"
                    ).add(1, attributes=cls._metrics_attributes())

            video.convo_video = await cls.construct_convo_video(
                video_bytes,
                subtitles,
                is_main_post,
                is_high_fav,
                enable_clahe_enhancement,
            )
            Metrics.counter("media_loader.hydrate_video_success.count").add(
                1, attributes=cls._metrics_attributes()
            )
        except Exception:
            logger.error(
                f"failed to download video {url}, error: {traceback.format_exc()}"
            )
            Metrics.counter("media_loader.hydrate_video_failed.count").add(
                1, attributes=cls._metrics_attributes()
            )
            raise

    @classmethod
    async def hydrate_broadcast(cls, broadcast_metadata: BroadcastMetadata) -> None:
        Metrics.counter("media_loader.hydrate_broadcast.count").add(
            1, attributes=cls._metrics_attributes()
        )
        logger.info("Hydrating broadcast frames from grox-fetcher")
        try:
            broadcast_data = await cls.grox_fetcher_client.fetch_broadcast(
                broadcast_metadata.broadcast_id,
                broadcast_metadata.media_key,
                dedup_video_frames=False,
            )
            if not broadcast_data or not broadcast_data.broadcast_content:
                Metrics.counter("media_loader.hydrate_broadcast_failed.count").add(
                    1, attributes=cls._metrics_attributes()
                )
                return
            broadcast_content = broadcast_data.broadcast_content
            frames = [frame.frame for frame in broadcast_content.frames]
            broadcast_metadata.video = Video(
                convo_video=ConvoVideo(
                    frames=frames,
                    subtitles=None,
                    duration=broadcast_content.duration,
                    total_duration=broadcast_content.total_duration,
                )
            )
            Metrics.counter("media_loader.hydrate_broadcast_success.count").add(
                1, attributes=cls._metrics_attributes()
            )
        except Exception:
            logger.error(
                f"failed to download broadcast {broadcast_metadata.broadcast_id}, error: {traceback.format_exc()}"
            )
            Metrics.counter("media_loader.hydrate_broadcast_failed.count").add(
                1, attributes=cls._metrics_attributes()
            )

    @classmethod
    async def fetch_video_subtitles(cls, media_id: str) -> list[str] | None:
        Metrics.counter("media_loader.fetch_video_subtitles.count").add(
            1, attributes=cls._metrics_attributes()
        )
        res = await cls.strato_video_subtitle.fetch(media_id)
        if not res:
            return None
        return [item.subtitle for item in res]

    @classmethod
    async def construct_convo_video(
        cls,
        video_bytes: bytes,
        subtitles: list[str] | None,
        is_main_post: bool = False,
        is_high_fav: bool = False,
        enable_clahe_enhancement: bool = False,
    ) -> ConvoVideo:
        video_max_frames = grox_config.media_hydration.video_max_frames_light
        video_tile_size = grox_config.media_hydration.video_tile_size
        if is_main_post:
            if is_high_fav:
                video_max_frames = grox_config.media_hydration.deluxe_video_max_frames
                video_tile_size = grox_config.media_hydration.deluxe_video_tile_size
            else:
                video_max_frames = grox_config.media_hydration.video_max_frames
        logger.info(
            f"video config video_tile_size {video_tile_size}, video_max_frames {video_max_frames}"
        )
        video_data = await VideoFramesExtractor.extract_frames(
            video_bytes,
            video_max_frames,
            video_tile_size,
            enable_clahe=enable_clahe_enhancement,
            include_combined_video_bytes=False,
        )
        times = [frame.time_sec for frame in video_data.frames]
        frames = [frame.frame for frame in video_data.frames]
        duration = (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0
        if video_data.total_duration is not None:
            total_duration = video_data.total_duration
        else:
            total_duration = times[-1] if times else duration
        if subtitles:
            try:
                subtitles = SubtitleAligner(subtitles).align(times)
            except Exception:
                logger.warning(
                    f"failed to align video subtitles, error: {traceback.format_exc()}"
                )
                Metrics.counter("media_loader.align_video_subtitles_failed.count").add(
                    1, attributes=cls._metrics_attributes()
                )
                subtitles = None

        if subtitles:
            Metrics.counter("media_loader.subtitles_found.count").add(
                1, attributes=cls._metrics_attributes()
            )
        else:
            Metrics.counter("media_loader.subtitles_not_found.count").add(
                1, attributes=cls._metrics_attributes()
            )

        return ConvoVideo(
            frames=frames,
            subtitles=subtitles,
            duration=duration,
            total_duration=total_duration,
            is_deluxe_target=is_main_post and is_high_fav,
        )
