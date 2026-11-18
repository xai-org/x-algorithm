import io
import asyncio
import logging
import random
import math

import av
import cv2
import numpy as np
from PIL import Image
from pydantic import Field, BaseModel
from av.stream import Stream
from av.container import InputContainer

from video_tools.image import (
    resize_tile,
    enhance_image_with_clahe,
    build_motion_reveal_images,
)

logger = logging.getLogger(__name__)

_MOTION_REVEAL_DENSE_SAMPLES = 48
_SHARP_WINDOW_SEC = 0.5
_SHARP_MAX_FRAMES_PER_WINDOW = 60
_SHARP_GRAY_WIDTH = 320
_SHARP_GRAY_MAX_HEIGHT = 960


class VideoFrame(BaseModel):
    time_sec: float
    frame: bytes


class VideoData(BaseModel):
    frames: list[VideoFrame]
    combined_bytes: bytes | None = None
    total_duration: float | None = None
    motion_reveal_frames: list[bytes] = Field(default_factory=list)
    sharp_frames: list[VideoFrame] = Field(default_factory=list)


class VideoFramesExtractor:
    @classmethod
    async def extract_frames(
        cls,
        video_bytes: bytes,
        max_frames: int,
        tile_size: int | None = None,
        enable_clahe: bool = False,
        include_combined_video_bytes: bool = True,
        enable_motion_reveal: bool = False,
        max_duration: float | None = None,
        sharp_frames: bool = False,
    ) -> VideoData:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            cls._extract_frames,
            video_bytes,
            max_frames,
            tile_size,
            enable_clahe,
            include_combined_video_bytes,
            enable_motion_reveal,
            max_duration,
            sharp_frames,
        )

    @classmethod
    def _extract_frames(
        cls,
        video_bytes: bytes,
        max_frames: int,
        tile_size: int | None,
        enable_clahe: bool = False,
        include_combined_video_bytes: bool = True,
        enable_motion_reveal: bool = False,
        max_duration: float | None = None,
        sharp_frames: bool = False,
    ) -> VideoData:
        logger.info(f"Extracting maximum {max_frames} frames from video")

        with av.open(io.BytesIO(video_bytes)) as container:
            if not isinstance(container, InputContainer):
                raise TypeError(
                    f"Container is not an InputContainer: {type(container).__name__}"
                )
            c_duration = container.duration
            if not c_duration:
                logger.warning("No duration found for video")
                c_duration = 0
            total_duration = float(c_duration / av.time_base)
            if max_duration is not None:
                total_duration = min(total_duration, max_duration)
            sample_times = cls._sample_frames(total_duration, max_frames)
            frames = cls._extract_frames_at_times(container, sample_times)
            sharp: list[VideoFrame] = []
            if sharp_frames:
                try:
                    sharp = cls._extract_sharpest_frames(
                        container, total_duration, len(sample_times)
                    )
                except Exception:
                    logger.warning(
                        "Failed to extract sharp frames; continuing with sampled frames only",
                        exc_info=True,
                    )

            reveal_input = frames
            if enable_motion_reveal and frames and total_duration > 0:
                dense_times = [
                    i * total_duration / _MOTION_REVEAL_DENSE_SAMPLES
                    for i in range(_MOTION_REVEAL_DENSE_SAMPLES)
                ]
                try:
                    dense_frames = cls._extract_frames_at_times(container, dense_times)
                    if len(dense_frames) > len(frames):
                        reveal_input = dense_frames
                except Exception:
                    logger.warning(
                        "Failed to extract dense motion-reveal frames; using sampled frames",
                        exc_info=True,
                    )

        motion_reveal_frames: list[bytes] = []
        if enable_motion_reveal:
            try:
                motion_reveal_frames = build_motion_reveal_images(
                    [frame.frame for frame in reveal_input]
                )
            except Exception:
                logger.warning(
                    "Failed to build motion-reveal stills; continuing with RGB frames only",
                    exc_info=True,
                )

        for frame in frames:
            frame.frame = cls._process_frame(frame.frame, tile_size, enable_clahe)
        for frame in sharp:
            frame.frame = cls._process_frame(frame.frame, tile_size)
        logger.info(f"Extracted {len(frames)} frames")
        combined_bytes = (
            cls.get_video_bytes([frame.frame for frame in frames])
            if include_combined_video_bytes
            else None
        )
        return VideoData(
            frames=frames,
            combined_bytes=combined_bytes,
            total_duration=total_duration,
            motion_reveal_frames=motion_reveal_frames,
            sharp_frames=sharp,
        )

    @classmethod
    def _sample_frames(cls, total_duration: float, max_frames: int) -> list[float]:
        if max_frames <= 0 or total_duration <= 0:
            return []
        if max_frames == 1:
            return [0.0]
        if max_frames == 2:
            return [0.0, total_duration / 2]
        if max_frames == 3:
            return [0.0, total_duration / 2, total_duration]

        num_samples = max(min(max_frames, math.ceil(total_duration) + 1), 3)
        bucket_duration = total_duration / (num_samples - 2)
        jitter = bucket_duration * 0.5

        times = [0.0]
        times.append(round(random.uniform(0, jitter), 3))
        for i in range(1, num_samples - 2):
            jittered = round(i * bucket_duration + random.uniform(-jitter, jitter), 3)
            times.append(jittered)
        times.append(round(total_duration + random.uniform(-jitter, 0), 3))
        return times

    @classmethod
    def _extract_sharpest_frames(
        cls, container: InputContainer, total_duration: float, count: int
    ) -> list[VideoFrame]:
        span = max(total_duration - _SHARP_WINDOW_SEC, 0.0)
        starts = [0.0] if count == 1 else [i * span / (count - 1) for i in range(count)]
        video_stream = next(s for s in container.streams if s.type == "video")
        time_base = cls._get_video_time_base(container, video_stream)
        out = []
        for start in starts:
            container.seek(int(start / time_base), stream=video_stream)
            best, best_score, best_time, scored = None, -1.0, start, 0
            for frame in container.decode(video=0):
                if frame.pts is None or frame.pts * time_base < start:
                    continue
                if best is not None and (
                    frame.pts * time_base >= start + _SHARP_WINDOW_SEC
                    or scored == _SHARP_MAX_FRAMES_PER_WINDOW
                ):
                    break
                width = max(
                    1,
                    min(
                        _SHARP_GRAY_WIDTH,
                        frame.width * _SHARP_GRAY_MAX_HEIGHT // frame.height,
                    ),
                )
                small = frame.to_ndarray(
                    width=width,
                    height=max(1, frame.height * width // frame.width),
                    format="gray",
                )
                score = cv2.Laplacian(small, cv2.CV_64F).var()
                scored += 1
                if score > best_score:
                    best, best_score, best_time = frame, score, frame.pts * time_base
            ret, jpeg = (
                cv2.imencode(".jpeg", best.to_ndarray(format="bgr24"))
                if best is not None
                else (False, None)
            )
            if ret:
                out.append(VideoFrame(time_sec=best_time, frame=jpeg.tobytes()))
        return out

    @classmethod
    def _extract_frames_at_times(
        cls, container: InputContainer, times: list[float]
    ) -> list[VideoFrame]:
        video_stream = next(s for s in container.streams if s.type == "video")
        time_base = cls._get_video_time_base(container, video_stream)
        frames = []
        successful_times = []
        for time_sec in sorted(times):
            if time_sec < 0:
                continue
            target_pts = int(time_sec / time_base)
            container.seek(target_pts, stream=video_stream)
            for frame in container.decode(video=0):
                if frame.pts is not None and frame.pts >= target_pts:
                    img = frame.to_ndarray(format="bgr24")
                    del frame
                    ret, jpeg = cv2.imencode(".jpeg", img)
                    del img
                    if ret:
                        frames.append(jpeg.tobytes())
                        del jpeg
                        successful_times.append(time_sec)
                    break
        return [
            VideoFrame(time_sec=time_sec, frame=frame)
            for time_sec, frame in zip(successful_times, frames, strict=True)
        ]

    @classmethod
    def _get_video_time_base(
        cls, container: InputContainer, video_stream: Stream
    ) -> float:
        if video_stream.time_base is not None:
            time_base = float(video_stream.time_base)
        elif video_stream.average_rate is not None:
            time_base = float(1 / video_stream.average_rate)
        else:
            container_format = container.format.name.lower()
            if "mpegts" in container_format:
                time_base = 1 / 90000
            elif "mp4" in container_format or "mov" in container_format:
                time_base = 1 / 30000
            elif "avi" in container_format:
                time_base = 1 / 1000000
            else:
                time_base = 1 / 1000
        return float(time_base)

    @classmethod
    def _process_frame(
        cls, frame: bytes, tile_size: int | None, enable_clahe: bool = False
    ) -> bytes:
        if enable_clahe:
            try:
                return enhance_image_with_clahe(frame, tile_size)
            except Exception:
                logger.warning(
                    "Failed to apply CLAHE to video frame, falling back to resize"
                )
                if tile_size is None:
                    return frame
                return resize_tile(frame, tile_size)
        elif tile_size is None:
            return frame
        return resize_tile(frame, tile_size)

    @classmethod
    def get_video_bytes(cls, frames: list[bytes]) -> bytes | None:
        num_frames = len(frames)
        if num_frames == 0:
            logger.warning("No video frames to combine")
            return None
        combined_array = None
        for i, frame in enumerate(frames):
            with io.BytesIO(frame) as buffer:
                with Image.open(buffer) as image:
                    with image.convert("RGB") as converted_image:
                        np_arr = np.array(converted_image)
                        np_frame = np_arr.transpose(2, 0, 1)
                        del np_arr
                        if i == 0:
                            combined_array = np.empty(
                                (num_frames,) + np_frame.shape, dtype=np.uint8
                            )
                        combined_array[i] = np_frame
        with io.BytesIO() as buffer:
            np.save(buffer, combined_array)
            del combined_array
            return buffer.getvalue()
