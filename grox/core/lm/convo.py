import io
import logging
import math
from enum import Enum
from base64 import b64decode, b64encode
from PIL import Image as PILImage
from pydantic import Field, BaseModel, field_validator, field_serializer

from xai_sdk.chat import image as xai_image, text as xai_text, system, user, assistant
from xai_sdk.proto import chat_pb2

from grox.config.config import grox_config

logger = logging.getLogger(__name__)
SEPARATOR_COMMAND = grox_config.prompt_tokens.separator
SEPARATOR = f"{SEPARATOR_COMMAND}\n\n"
THINKING_CONTROL_END = grox_config.prompt_tokens.thinking_control_end
THINKING_CONTROL_START = grox_config.prompt_tokens.thinking_control_start
STORYBOARD_COLUMNS = 3
STORYBOARD_TILE_SIZE = 448
NO_THINKING_PROMPT = grox_config.prompt_tokens.no_thinking_prompt

PREVIEW_IMAGE_LABEL = "[Video Preview Thumbnail]"

MOTION_REVEAL_DESCRIPTION = (
    "Motion-reveal stills for this video follow. They subtract what stays still and amplify what changes, to expose "
    "content hidden under the visible video as a faint semi-transparent layer, so colors and textures are distorted "
    "and the hidden layer often looks blurry, ghostly or low-contrast. A hidden layer is a different scene from the "
    "visible video: different people or bodies, "
    "in different positions or a different setting, often skin-colored bodies, close-ups of body parts, or sexual "
    "motion behind an innocent-looking cover such as a selfie, a photo, a cartoon, food or text. Judge the hidden "
    "layer as part of the video's actual content, as if it were shown openly, even when it is low-resolution, "
    "blurred, or partially obscured. The visible video is likely a decoy: when the hidden layer is more severe than "
    "the visible video, classify the post by the hidden layer. Ordinary motion of the visible people appears as "
    "their own outlines in the same places as in the video frames; that alone is not hidden content."
)

SHOT_KEY_FRAMES_DESCRIPTION = (
    "Shot key frames of this video's opening follow: one still per distinct shot, including shots only a frame or "
    "two long (a flash or a quick cut). Judge what they show as part of the video's actual content."
)

RANKED_KEY_FRAMES_DESCRIPTION = (
    "Ranked key frames of this video follow: stills picked by an automated search for the moments that most need a "
    "close look. A moment can be on screen for only a fraction of a second; judge what they show as part of the "
    "video's actual content."
)


class Role(str, Enum):
    USER = "User"
    ASSISTANT = "Assistant"
    SYSTEM = "System"
    HUMAN = "Human"


class Image(BaseModel):
    content: bytes = Field()

    @field_serializer("content", when_used="json")
    def serialize_content(self, value: bytes) -> str:
        return b64encode(value).decode("utf-8")

    @field_validator("content", mode="before")
    @classmethod
    def deserialize_content(cls, value: str | bytes) -> bytes:
        if isinstance(value, str):
            return b64decode(value)
        return value


class Video(BaseModel):
    frames: list[bytes] = Field(default_factory=lambda: [])
    subtitles: list[str] | None = None
    asr_transcript: str | None = None
    duration: float
    total_duration: float
    is_deluxe_target: bool = False
    motion_reveal_frames: list[bytes] = Field(default_factory=lambda: [])
    preview_image: bytes | None = None
    shot_key_frames: list[bytes] = Field(default_factory=lambda: [])
    ranked_key_frames: list[bytes] = Field(default_factory=lambda: [])

    @field_serializer(
        "frames",
        "motion_reveal_frames",
        "shot_key_frames",
        "ranked_key_frames",
        when_used="json",
    )
    def serialize_frames(self, value: list[bytes]) -> list[str]:
        return [b64encode(frame).decode("utf-8") for frame in value]

    @field_serializer("preview_image", when_used="json")
    def serialize_preview_image(self, value: bytes | None) -> str | None:
        return b64encode(value).decode("utf-8") if value is not None else None

    @field_validator("preview_image", mode="before")
    @classmethod
    def decode_preview_image(cls, value: str | bytes | None) -> bytes | None:
        return b64decode(value) if isinstance(value, str) else value

    @field_validator(
        "frames",
        "motion_reveal_frames",
        "shot_key_frames",
        "ranked_key_frames",
        mode="before",
    )
    @classmethod
    def decode_frames(cls, value: list[str] | list[bytes]) -> list[bytes]:
        is_list_str = len(value) > 0 and isinstance(value[0], str)
        if is_list_str:
            return [b64decode(frame) for frame in value]
        return value

    def stills(self) -> list[bytes]:
        return [*self.frames, *self.shot_key_frames, *self.ranked_key_frames]

    def key_frame_parts(self) -> list[str | bytes]:
        parts: list[str | bytes] = []
        for description, stills in (
            (SHOT_KEY_FRAMES_DESCRIPTION, self.shot_key_frames),
            (RANKED_KEY_FRAMES_DESCRIPTION, self.ranked_key_frames),
        ):
            if stills:
                parts.append(f"{description}\n")
                parts.extend(stills)
        return parts

    def _create_storyboard(
        self, frame_chunk: list[bytes], aspect_ratio: float | None = None
    ) -> bytes:
        num_tiles = len(frame_chunk)
        if num_tiles == 0:
            img = PILImage.new("RGB", (1, 1), (255, 255, 255))
            with io.BytesIO() as output:
                img.save(output, format="JPEG")
                return output.getvalue()

        tile_height = (
            grox_config.media_hydration.deluxe_video_tile_size
            if self.is_deluxe_target
            else grox_config.media_hydration.video_tile_size
        )
        tile_width = int(tile_height * aspect_ratio) if aspect_ratio else tile_height

        cols = STORYBOARD_COLUMNS
        rows = math.ceil(num_tiles / cols)
        canvas_width = cols * tile_width
        canvas_height = rows * tile_height

        canvas = PILImage.new("RGB", (canvas_width, canvas_height), (255, 255, 255))

        for i, frame_bytes in enumerate(frame_chunk):
            with PILImage.open(io.BytesIO(frame_bytes)) as img:
                if img.mode == "RGBA":
                    img = img.convert("RGB")
                img = img.resize((tile_width, tile_height), PILImage.Resampling.LANCZOS)

                col = i % cols
                row = i // cols
                x = col * tile_width
                y = row * tile_height

                canvas.paste(img, (x, y))

        with io.BytesIO() as output:
            canvas.save(output, format="JPEG")
            return output.getvalue()

    def interleave_storyboard(
        self, aspect_ratio: float | None = None
    ) -> list[str | bytes]:
        res: list[str | bytes] = []

        if self.preview_image:
            res.extend([f"{PREVIEW_IMAGE_LABEL} ", self.preview_image, "\n"])

        if self.frames:
            res.append(
                f"The video has a duration of {self.total_duration:.2f} seconds. "
                "Below is a storyboard with composite panels of frames (which are sorted left to right, top to bottom) "
                "and subtitles interleaved.\n"
            )

            frames_per_storyboard = STORYBOARD_COLUMNS**2
            storyboards: list[bytes] = []
            storyboard_indices: list[int] = []

            for i in range(0, len(self.frames), frames_per_storyboard):
                chunk_frames = self.frames[i : i + frames_per_storyboard]
                storyboard_bytes = self._create_storyboard(chunk_frames, aspect_ratio)
                storyboards.append(storyboard_bytes)
                storyboard_indices.append(i)

            for idx, (storyboard_bytes, frame_idx) in enumerate(
                zip(storyboards, storyboard_indices, strict=True)
            ):
                start_frame = frame_idx
                end_frame = min(frame_idx + frames_per_storyboard, len(self.frames))
                start_time = start_frame * self.duration
                end_time = (end_frame - 1) * self.duration

                res.extend(
                    [
                        " ",
                        storyboard_bytes,
                        f" Storyboard panel {idx + 1} (frames {start_frame + 1}-{end_frame} at {start_time:.2f}s-{end_time:.2f}s)",
                    ]
                )

                if self.subtitles:
                    subtitle_texts = []
                    for i in range(start_frame, end_frame):
                        if i < len(self.subtitles) and self.subtitles[i]:
                            frame_time = i * self.duration
                            subtitle_texts.append(
                                f"[{frame_time:.2f}s] {self.subtitles[i]}"
                            )

                    if subtitle_texts:
                        res.append(f" Subtitles: {'; '.join(subtitle_texts)}")

                res.append("\n")

        if self.motion_reveal_frames:
            res.append(f"{MOTION_REVEAL_DESCRIPTION}\n")
            for idx, reveal_bytes in enumerate(self.motion_reveal_frames):
                res.extend([" ", reveal_bytes, f" Motion-reveal still {idx + 1}\n"])

        res.extend(self.key_frame_parts())

        return res


type Content = str | Image | Video


class Message(BaseModel):
    role: Role
    content: list[Content] = Field(default_factory=lambda: [])
    separator: str = SEPARATOR

    def is_empty(self) -> bool:
        return not self.content

    def sanitize_text(self, text: str):
        for token in (
            THINKING_CONTROL_START,
            THINKING_CONTROL_END,
            SEPARATOR_COMMAND,
            NO_THINKING_PROMPT,
        ):
            if token:
                text = text.replace(token, "")
        return text

    def interleave(self) -> list[str | bytes]:
        role_str = f"{self.role.value}:"
        interleaved: list[str | bytes] = [role_str]
        for piece in self.content:
            if isinstance(piece, str):
                if isinstance(interleaved[-1], str):
                    interleaved[-1] = " ".join([interleaved[-1], piece])
                else:
                    interleaved.append(piece)
            elif isinstance(piece, Image):
                interleaved.append(piece.content)
            elif isinstance(piece, Video):
                interleaved.extend(piece.interleave_storyboard())
        if not self.is_empty() and self.separator:
            if isinstance(interleaved[-1], str):
                interleaved[-1] = "".join([interleaved[-1], self.separator])
            else:
                interleaved.append(self.separator)
        return interleaved

    def convert_grox_content_to_xai_content_list(
        self, content_list: list[Content]
    ) -> list[chat_pb2.Content]:
        result: list[chat_pb2.Content] = []
        text_buffer: list[str] = []

        def flush_text_buffer():
            if text_buffer:
                result.append(xai_text("".join(text_buffer)))
                text_buffer.clear()

        for item in content_list:
            if isinstance(item, str):
                text_buffer.append(self.sanitize_text(item))
            elif isinstance(item, Image):
                flush_text_buffer()
                image_b64 = b64encode(item.content).decode("utf-8")
                result.append(
                    xai_image(
                        image_url=f"data:image/jpeg;base64,{image_b64}", detail="high"
                    )
                )
            elif isinstance(item, Video):
                flush_text_buffer()
                storyboard_content = item.interleave_storyboard()

                for storyboard_item in storyboard_content:
                    if isinstance(storyboard_item, str):
                        text_buffer.append(storyboard_item)
                    elif isinstance(storyboard_item, bytes):
                        flush_text_buffer()
                        storyboard_b64 = b64encode(storyboard_item).decode("utf-8")
                        result.append(
                            xai_image(
                                image_url=f"data:image/jpeg;base64,{storyboard_b64}",
                                detail="high",
                            )
                        )

        flush_text_buffer()
        return result

    def interleaveToEapi(self) -> chat_pb2.Message:
        interleaved = self.convert_grox_content_to_xai_content_list(self.content)
        match self.role:
            case Role.USER | Role.HUMAN:
                return user(*interleaved)
            case Role.ASSISTANT:
                return assistant(*interleaved)
            case Role.SYSTEM:
                return system(*interleaved)
            case _:
                raise ValueError("Invalid Role")

    def to_prompt(self) -> str:
        for c in self.content:
            if (
                not isinstance(c, str)
                and not isinstance(c, Image)
                and not isinstance(c, Video)
            ):
                raise ValueError(
                    f"Message content must be a str, Image, or Video, got {type(c)}"
                )
        msg = "\n".join([c for c in self.content if isinstance(c, str)])
        return (
            f"{self.role.value}: {msg}{self.separator}"
            if not self.is_empty()
            else f"{self.role.value}: "
        )

    def to_prompt_eapi(self) -> chat_pb2.Message:
        for c in self.content:
            if (
                not isinstance(c, str)
                and not isinstance(c, Image)
                and not isinstance(c, Video)
            ):
                raise ValueError(
                    f"Message content must be a str, Image, or Video, got {type(c)}"
                )
        msg = "\n".join(
            [self.sanitize_text(c) for c in self.content if isinstance(c, str)]
        )
        match self.role:
            case Role.USER | Role.HUMAN:
                return user(msg)
            case Role.ASSISTANT:
                return assistant(msg)
            case Role.SYSTEM:
                return system(msg)
            case _:
                raise ValueError("Invalid Role")


class Conversation(BaseModel):
    conversation_id: str | None = None
    messages: list[Message] = Field(default_factory=lambda: [])

    def interleave(self) -> list[str | bytes]:
        res: list[str | bytes] = []
        for message in self.messages:
            for part in message.interleave():
                if isinstance(part, str) and res and isinstance(res[-1], str):
                    res[-1] = "".join([res[-1], part])
                else:
                    res.append(part)
        return res

    def interleaveToEapi(self) -> list[chat_pb2.Message]:
        return [
            message.interleaveToEapi()
            for message in self.messages
            if message.role != Role.ASSISTANT
        ]

    def to_openai_messages(self) -> list[dict]:
        _CONTROL_TOKENS = {
            tok
            for tok in (THINKING_CONTROL_START, THINKING_CONTROL_END, SEPARATOR_COMMAND)
            if tok
        }
        role_map = {
            Role.SYSTEM: "system",
            Role.USER: "user",
            Role.ASSISTANT: "assistant",
            Role.HUMAN: "user",
        }

        def _clean_text(text: str) -> str:
            for tok in _CONTROL_TOKENS:
                text = text.replace(tok, "")
            return text.strip()

        def _image_part(image_bytes: bytes) -> dict:
            b64 = b64encode(image_bytes).decode("utf-8")
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }

        result = []
        for msg in self.messages:
            parts = []
            for c in msg.content:
                if isinstance(c, str):
                    cleaned = _clean_text(c)
                    if cleaned:
                        parts.append({"type": "text", "text": cleaned})
                elif isinstance(c, Image):
                    parts.append(_image_part(c.content))
                elif isinstance(c, Video):
                    if c.preview_image:
                        parts.append({"type": "text", "text": PREVIEW_IMAGE_LABEL})
                        parts.append(_image_part(c.preview_image))
                    if c.frames:
                        desc = f"The video lasts for {c.total_duration:.2f} seconds. The following {len(c.frames)} frames are sampled at equal intervals."
                        parts.append({"type": "text", "text": desc})
                        bucket_times = [i * c.duration for i in range(len(c.frames))]
                        for i, (frame, time_sec) in enumerate(
                            zip(c.frames, bucket_times, strict=True)
                        ):
                            subtitle = (
                                c.subtitles[i]
                                if c.subtitles and i < len(c.subtitles)
                                else None
                            )
                            subtitle_str = (
                                f" with subtitle: {subtitle}" if subtitle else ""
                            )
                            parts.append(
                                {
                                    "type": "text",
                                    "text": f"Frame at {time_sec:.2f}s{subtitle_str}",
                                }
                            )
                            parts.append(_image_part(frame))
                    if c.motion_reveal_frames:
                        parts.append(
                            {"type": "text", "text": MOTION_REVEAL_DESCRIPTION}
                        )
                        for idx, reveal in enumerate(c.motion_reveal_frames):
                            parts.append(
                                {
                                    "type": "text",
                                    "text": f"Motion-reveal still {idx + 1}",
                                }
                            )
                            parts.append(_image_part(reveal))
                    for part in c.key_frame_parts():
                        parts.append(
                            {"type": "text", "text": part}
                            if isinstance(part, str)
                            else _image_part(part)
                        )

            if not parts:
                continue
            if msg.role == Role.SYSTEM:
                text = " ".join(p["text"] for p in parts if p.get("type") == "text")
                if text:
                    result.append({"role": "system", "content": text})
            else:
                result.append({"role": role_map[msg.role], "content": parts})
        return result

    def to_prompt(self) -> str:
        return "".join(msg.to_prompt() for msg in self.messages)

    def to_prompt_eapi(self) -> list[chat_pb2.Message]:
        return [
            message.to_prompt_eapi()
            for message in self.messages
            if message.role != Role.ASSISTANT
        ]
