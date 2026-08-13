import re

from grox.core.data_loaders.data_types import Post, User
from grox.core.lm.convo import Content
from grox.core.lm.convo import Image as ConvoImage
from grox.core.lm.convo import Video as ConvoVideo


def remove_tco_links(text, remove_twitter_links: bool = False):
    if not text:
        return ""
    if remove_twitter_links:
        pattern = r"https?://(?:t\.co/[A-Za-z0-9]+|twitter\.com/[^\s]+)"
    else:
        pattern = r"https?://t\.co/[A-Za-z0-9]+"

    result = re.sub(pattern, "", text).strip()

    return result


def get_replies_handle_string(post: Post) -> str:
    if not post.ancestors:
        return ""
    return " ".join(
        [f"@{a.user.handle}" for a in post.ancestors if a.user and a.user.handle]
    )


class PostRenderer:
    @classmethod
    def has_media(cls, post: Post) -> bool:
        return any(
            isinstance(part, (ConvoImage, ConvoVideo)) for part in cls.render(post)
        )

    @classmethod
    def render(
        cls,
        post: Post,
        indent: int = 0,
        max_media: int | None = None,
        include_reply_to: bool = False,
        cards_note_override: str | None = None,
    ) -> list[Content]:
        indent_str = ">" + (" " * indent) if indent > 0 else ""
        res = []
        if not post.user:
            post.user = User(name="unknown", handle="unknown")
        res.append(f"\n{indent_str}Metadata: {post.user.name} @{post.user.handle}")
        all_media = list(post.media or []) + list(post.url_videos or [])
        if all_media:
            res.append(f"\n{indent_str}Media:")
            if max_media is not None:
                all_media = all_media[:max_media]
            for idx, m in enumerate(all_media):
                res.extend(m.to_convo(idx))
        formatted_text = (
            post.full_text.replace("\n", f"\n{indent_str}") if post.full_text else ""
        )
        if post.ancestors:
            for ancestor in post.ancestors:
                if ancestor.user and ancestor.user.handle:
                    formatted_text = formatted_text.replace(
                        f"@{ancestor.user.handle}", "", 1
                    )
        res.append(f"\n{indent_str}Post: {post.id}")
        if include_reply_to and post.ancestors:
            reply_handles = get_replies_handle_string(post)
            res.append(f"\n{indent_str}Replying to: {reply_handles}")
        res.append(f"\n{indent_str}Text: {formatted_text}")
        if post.urls:
            res.append(
                f"\n{indent_str}The Post contains these URLs: {', '.join(post.urls)}"
            )
        if post.broadcast_metadata:
            res.extend(post.broadcast_metadata.to_convo())
        if cards_note_override is not None:
            res.append(f"\n{indent_str}{cards_note_override}")
        elif post.cardsV2:
            res.append(f"\n{indent_str}This post has the following cards attached: ")
            for cardV2 in post.cardsV2:
                res.extend(cardV2.to_convo())
        if post.article_metadata:
            res.append(f"\n{indent_str}[Article Post] This post is an Article.")
            res.extend(post.article_metadata.to_convo())
        if post.list_metadata:
            res.extend(post.list_metadata.to_convo())
        if post.chat_group_metadata:
            res.extend(post.chat_group_metadata.to_convo())
        if post.quoted_post:
            res.append(
                f"\n\n{indent_str}This Post quotes Post {post.quoted_post.id}\n\n"
            )
            res.extend(
                cls.render(post.quoted_post, indent=indent + 4, max_media=max_media)
            )
        if post.descendants:
            res.append(
                f"\n\n{indent_str}This Post also received these replies. Please use it as a reference to understand the post more holistically\n\n"
            )
            for descendant_post in post.descendants:
                res.extend(
                    cls.render(descendant_post, indent=indent + 4, max_media=max_media)
                )
        if post.screenshot:
            res.append(
                f"\n\n{indent_str}Sometimes user present their content by combining multiple images to one view. So screenshot is attached for reference\n\n"
            )
            res.extend(post.screenshot.to_convo())
        return [m for m in res if m]


class LitePostRenderer:
    @classmethod
    def render(
        cls,
        post: Post,
        indent: int = 0,
        max_media: int | None = None,
        include_reply_to: bool = False,
        include_author_mode: int = 0,
        include_safety_labels: bool = False,
    ) -> list[Content]:
        indent_str = ">" + (" " * indent) if indent > 0 else ""
        res = []
        post_text = remove_tco_links(post.full_text)
        formatted_text = post_text.replace("\n", f"\n{indent_str}")

        assert post.user is not None
        if include_author_mode > 0:
            res.append(f"\n{indent_str}Author: {post.user.name} (@{post.user.handle})")
            if include_author_mode > 1:
                res.append(f"\n{indent_str}Author Bio: {post.user.bio}")
            if include_author_mode > 2:
                res.append(
                    f"\n{indent_str}Author Info: Signup Country: {post.user.signup_country_code}, Follower Count: {post.user.follower_count}"
                )
        if include_reply_to:
            reply_handles = get_replies_handle_string(post)
            res.append(f"\n{indent_str}Replying to: {reply_handles}")
        if post.created_at:
            res.append(
                f"\n{indent_str}Post created At: {post.created_at.strftime('%Y-%m-%d %H:%M:%S')}"
            )
        res.append(f"\n{indent_str}Text: {formatted_text}")
        if post.media:
            res.append(f"\n{indent_str}Media:")
            media = post.media
            if max_media is not None:
                media = media[:max_media]
            for idx, m in enumerate(media):
                res.extend(m.to_convo(idx))
        if post.media_descriptions:
            res.append(f"\n{indent_str}Media Description: ")
            media_desc = post.media_descriptions
            for idx, media_description in enumerate(media_desc):
                res.append(f"\n[{idx + 1}: Media Description] {media_description}\n")
        if post.cardsV2:
            res.append(f"\n{indent_str}This post has the following cards attached: ")
            for cardV2 in post.cardsV2:
                res.extend(cardV2.to_convo())
        if post.quoted_post:
            res.append(f"\n\n{indent_str}This Post quotes Post::\n\n")
            res.extend(
                cls.render(
                    post.quoted_post,
                    indent=indent + 4,
                    max_media=max_media,
                    include_author_mode=include_author_mode,
                    include_reply_to=include_reply_to,
                )
            )
        if include_safety_labels and post.safety_labels:
            res.append(
                f"\n{indent_str}This post has the following safety labels attached: {', '.join(post.safety_labels)}"
            )
        return [m for m in res if m]
