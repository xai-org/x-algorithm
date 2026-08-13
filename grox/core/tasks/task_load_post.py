from grox.core.data_loaders.data_types import Post
from grox.core.data_loaders.strato_loader import TweetStratoLoader
from grox.core.schedules.types import TaskContext
from grox.core.tasks.task import TaskStopExecution, TaskWithPost


class TaskLoadPost(TaskWithPost):
    @classmethod
    async def _exec_with_post(cls, ctx: TaskContext, post: Post) -> None:
        loaded_post = await TweetStratoLoader.load_post(post.id)
        if loaded_post is None:
            raise TaskStopExecution(f"Post not found: {post.id}")
        ctx.payload.post = loaded_post
