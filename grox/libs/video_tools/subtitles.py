import logging
import re

from pydantic import BaseModel

logger = logging.getLogger(__name__)
X_WORD_MS_PATTERN = (
    r"<X-word-ms ms=([^>]+) index=\d+ character_ranges=([^>]+)>([^<]+)</X-word-ms>"
)


class Subtitle(BaseModel):
    index: int
    start_time: float
    end_time: float
    text: str


class SubtitleAligner:
    def __init__(self, subtitles: list[str]):
        self.subtitles = self._parse_video_subtitle(subtitles)

    def align(self, times: list[float]) -> list[str]:
        if not times or not self.subtitles:
            logger.warning("No times or subtitles to align")
            return []
        logger.info(f"Aligning subtitles to {len(times)} buckets")
        bucket_duration = (
            (times[-1] - times[0]) / (len(times) - 1) if len(times) > 1 else 1.0
        )
        buckets: list[list[str]] = [[] for _ in range(len(times))]
        for sub in self.subtitles:
            bucket_idx = min(int(sub.start_time / bucket_duration), len(times) - 1)
            buckets[bucket_idx].append(sub.text)
        logger.info(f"Aligned subtitles to {len(times)} buckets")
        return [" ".join(bucket).strip() for bucket in buckets]

    def _parse_video_subtitle(self, subtitle_data: list[str]) -> list[Subtitle]:
        subtitles: list[Subtitle] = []

        for subtitle_text in subtitle_data:
            blocks = (
                subtitle_text.encode("utf-8")
                .decode("utf-8-sig")
                .replace("\r", "")
                .strip()
                .split("\n\n")
            )
            for block in blocks:
                lines = block.strip().split("\n")
                if len(lines) < 2:
                    logger.warning(
                        f"Getting subtitle block with less than 2 lines: {lines}"
                    )
                    continue
                index = int(lines[0])
                timing = lines[1]
                start_time, end_time = timing.split(" --> ")
                text_lines = lines[2:]
                raw_text = "\n".join(text_lines)
                text = raw_text
                if "<X-word-ms" in raw_text:
                    text_matches = re.findall(X_WORD_MS_PATTERN, raw_text)
                    text = "\n".join(match[2] for match in text_matches).strip()

                subtitles.append(
                    Subtitle(
                        index=index,
                        start_time=self._time_to_seconds(start_time),
                        end_time=self._time_to_seconds(end_time),
                        text=text,
                    )
                )
        return subtitles

    def _time_to_seconds(self, time_str: str) -> float:
        h, m, s = time_str.replace(",", ".").split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
