from grox.core.data_loaders.data_types import User
from grox.core.lm.convo import Content
from grox.core.lm.post import LitePostRenderer


class UserRenderer:
    @classmethod
    def render(
        cls,
        user: User | None,
        include_stats: bool = False,
        include_profile_urls: bool = False,
        include_affiliate_business: bool = False,
        include_recent_posts: bool = False,
    ) -> list[Content]:
        if not user:
            return []
        profile = f"""# User Profile
Here are the user details.
  - Name: {user.name}
  - Username: @{user.handle}
  - User Bio: '{user.bio or ""}'
  - Profile Location: {user.profile_location or ""}
"""
        if include_profile_urls and user.urls:
            profile += f"""  - Profile URLs: {", ".join(user.urls)}
"""
        if include_affiliate_business and user.affiliated_business:
            profile += f"""  - Affiliated Business: Name is {user.affiliated_business.name} and URL is {user.affiliated_business.url}"""
        if include_stats:
            profile += f"""
  - Account Created: {user.created_at.strftime("%Y-%m-%d") if user.created_at else "N/A"}
  - Signup Country: {user.signup_country_code or "N/A"}
  - Followers: {user.follower_count if user.follower_count is not None else "N/A"}
  - Following: {user.following_count if user.following_count is not None else "N/A"}
  - Subscription: {user.subscription_level or "N/A"}
"""
        res: list[Content] = [profile]
        if include_recent_posts and user.recent_posts:
            res.append("\n## Recent Posts by This User\n")
            for post in user.recent_posts:
                if user.id != post.user.id:
                    res.append("\n# Reposted Post\n")
                res.extend(
                    LitePostRenderer.render(
                        post, include_author_mode=1, include_safety_labels=True
                    )
                )
                res.append("\n---\n")
        return res
