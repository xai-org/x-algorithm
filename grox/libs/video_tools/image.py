import io
import math
from dataclasses import dataclass
import logging
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


MIN_WIDTH = 8
MIN_HEIGHT = 8

logger = logging.getLogger(__name__)


@dataclass
class ProcessedImage:
    convo_image: bytes | None
    dark_mode_image: bytes | None
    light_mode_image: bytes | None


def _resize_and_pad(
    image: Image.Image, tile_size: int, high_quality_jpeg: bool = False
) -> bytes:
    width, height = image.size
    aspect_ratio = width / height
    if width > height:
        new_width = tile_size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = tile_size
        new_width = int(new_height * aspect_ratio)
    resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    return _padded_image(resized_image, high_quality_jpeg)


def resize_tile(image_bytes: bytes, tile_size: int) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        with img.convert("RGB") as image:
            return _resize_and_pad(image, tile_size)


def _has_transparency(img: Image.Image) -> bool:
    if "A" in img.mode:
        return True
    if img.mode == "P" and "transparency" in img.info:
        return True
    return False


def _apply_clahe_pil(
    image: Image.Image,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> Image.Image:
    arr = np.array(image)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)
    merged = cv2.merge((cl, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def _boost_saturation(image: Image.Image, factor: float = 1.8) -> Image.Image:
    return ImageEnhance.Color(image).enhance(factor)


def _sharpen(image: Image.Image) -> Image.Image:
    return image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))


def _adaptive_brightness_correction(
    image: Image.Image, strength: float = 0.25
) -> Image.Image:
    arr = np.array(image)
    lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    x = np.arange(256, dtype=np.float32) / 255.0
    adjusted = x + strength * (0.5 - x)
    lut = np.clip(adjusted * 255, 0, 255).astype(np.uint8)

    l_adjusted = cv2.LUT(l_channel, lut)
    merged = cv2.merge((l_adjusted, a_channel, b_channel))
    result = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return Image.fromarray(result)


def enhance_image_with_clahe(image_bytes: bytes, tile_size: int | None) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        with img.convert("RGB") as rgb_img:
            enhanced = _apply_clahe_pil(rgb_img)
            enhanced = _adaptive_brightness_correction(enhanced)
            enhanced = _boost_saturation(enhanced, factor=1.5)
            enhanced = _sharpen(enhanced)
            if tile_size is None:
                return _padded_image(enhanced, high_quality_jpeg=True)
            return _resize_and_pad(enhanced, tile_size, high_quality_jpeg=True)


def process_image_bytes(image_bytes: bytes, tile_size: int) -> ProcessedImage:
    with Image.open(io.BytesIO(image_bytes)) as img:
        if _has_transparency(img):
            logger.info(f"Image has transparency (mode={img.mode})")
            img_rgba = img.convert("RGBA")
            black_bg = Image.new("RGBA", img_rgba.size, (0, 0, 0, 255))
            black_comp = Image.alpha_composite(black_bg, img_rgba)
            with black_comp.convert("RGB") as black_rgb_img:
                dark_mode = _resize_and_pad(black_rgb_img, tile_size)
            white_bg = Image.new("RGBA", img_rgba.size, (255, 255, 255, 255))
            white_comp = Image.alpha_composite(white_bg, img_rgba)
            with white_comp.convert("RGB") as white_rgb_img:
                light_mode = _resize_and_pad(white_rgb_img, tile_size)
        else:
            logger.info("Image does not have transparency")
            dark_mode = None
            light_mode = None
        with img.convert("RGB") as rgb_img:
            resized = _resize_and_pad(rgb_img, tile_size)
            return ProcessedImage(
                convo_image=resized,
                dark_mode_image=dark_mode,
                light_mode_image=light_mode,
            )


def _img_to_bytes(image: Image.Image, high_quality_jpeg: bool = False) -> bytes:
    with io.BytesIO() as output:
        if high_quality_jpeg:
            image.save(output, format="JPEG", quality=95)
        else:
            image.save(output, format="JPEG")
        return output.getvalue()


def _padded_image(image: Image.Image, high_quality_jpeg: bool = False) -> bytes:
    width, height = image.size
    if width < MIN_WIDTH or height < MIN_HEIGHT:
        new_width = max(width, MIN_WIDTH)
        new_height = max(height, MIN_HEIGHT)
        with Image.new("RGB", (new_width, new_height), (0, 0, 0)) as padded_image:
            padded_image.paste(
                image, ((new_width - width) // 2, (new_height - height) // 2)
            )
            return _img_to_bytes(padded_image, high_quality_jpeg)
    return _img_to_bytes(image, high_quality_jpeg)


def pad_image(image_bytes: bytes) -> bytes:
    with Image.open(io.BytesIO(image_bytes)) as img:
        with img.convert("RGB") as image:
            return _padded_image(image)


_MOTION_REVEAL_MIN_FRAMES = 4
_MOTION_REVEAL_MAX_INPUT_FRAMES = 48
_MOTION_REVEAL_WORK_MAX_DIM = 640
_MOTION_REVEAL_OVERVIEW_FRAMES = 9
_MOTION_REVEAL_DETAIL_STILLS = 6
_MOTION_REVEAL_PLAIN_STILLS = 2
_MOTION_REVEAL_MAX_OPAQUE_MOTION_SHARE = 0.35
_MOTION_REVEAL_JPEG_QUALITY = 90
_MOTION_REVEAL_MAX_RESIDUAL_P99 = 100.0
_MOTION_REVEAL_DENOISE_ENERGY = (1.0, 3.0, 8.0)
_MOTION_REVEAL_DENOISE_SIGMA = (2.0, 1.2, 0.0)
_MOTION_REVEAL_WINDOW_FRAMES = 6
_MOTION_REVEAL_ECC_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
    20,
    1e-3,
)
_MOTION_REVEAL_ECC_MAX_DIM = 320
_MOTION_REVEAL_STATIC_SHIFT_PX = 0.3
_MOTION_REVEAL_STATIC_MIN_RESPONSE = 0.05
_MOTION_REVEAL_MIN_COVER_GRADIENT = 5.0
_MOTION_REVEAL_MAX_WARP_TRANSLATION = 0.15
_MOTION_REVEAL_MAX_WARP_LINEAR = 0.15
_OVERLAY_WORK_MAX_DIM = 320
_OVERLAY_PATCH = 16
_OVERLAY_LOCAL_RADIUS = 3
_OVERLAY_MIN_CHANGE = 2.0
_OVERLAY_MIN_COVER_TEXTURE = 1.5
_OVERLAY_SEE_THROUGH_NCC = 0.5
_OVERLAY_DEPENDENT_NCC = 0.35
_OVERLAY_OPAQUE_NCC = 0.25
_OVERLAY_GLOBAL_WEIGHTS = (2.164, 0.581, -8.755, 22.971, -5.714, -9.539, 0.928)
_OVERLAY_LOCAL_WEIGHTS = (4.049, -0.254, 9.51, 21.771, 0.616, -1.785, 0.232)
_OVERLAY_BIAS = -4.374
_OVERLAY_MIN_SCORE = -1.0
_OVERLAY_LOCAL_OPAQUE_MARGIN = 0.05
_MOTION_REVEAL_SKIN_LAB_AB = (143.0, 148.0)
_MOTION_REVEAL_SKIN_AXIS_MAX_DISTANCE = 12.0
_MOTION_REVEAL_FLOW_MAX_DIM = 320


def _decode_reveal_frame(frame_bytes: bytes) -> np.ndarray | None:
    img = cv2.imdecode(np.frombuffer(frame_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    scale = _MOTION_REVEAL_WORK_MAX_DIM / max(h, w)
    if scale < 1.0:
        img = cv2.resize(
            img,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return img


def _temporal_median(stack_u8: np.ndarray) -> np.ndarray:
    n = stack_u8.shape[0]
    lower, upper = (n - 1) // 2, n // 2
    part = np.partition(stack_u8, [lower, upper], axis=0)
    return (part[lower].astype(np.float32) + part[upper].astype(np.float32)) * 0.5


def _cover_gradient(gray_u8: np.ndarray) -> float:
    g = gray_u8.astype(np.float32)
    return float(
        np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1)).mean()
    )


def _residual_magnitude(stack: np.ndarray, median: np.ndarray) -> np.ndarray:
    out = np.empty(stack.shape[:3], dtype=np.float32)
    buf = np.empty(stack.shape[1:], dtype=np.float32)
    for i in range(stack.shape[0]):
        np.subtract(stack[i], median, out=buf)
        np.abs(buf, out=buf)
        np.mean(buf, axis=-1, out=out[i])
    return out


def _stretch_to_u8(arr: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(arr, 2))
    hi = float(np.percentile(arr, 98))
    if hi <= lo + 1e-3:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _plain_reveal(residual: np.ndarray) -> np.ndarray:
    smoothed = cv2.GaussianBlur(residual, (0, 0), 2.0)
    lo, hi = np.percentile(smoothed, (1, 99))
    stretched = np.clip(
        (smoothed - lo) * (255.0 / max(float(hi - lo), 1e-3)), 0, 255
    ).astype(np.uint8)
    lab = cv2.cvtColor(stretched, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _opaque_motion_share(frames: np.ndarray, radius: int | None) -> float:
    def grey(image: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(
            np.clip(image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)

    def local(x: np.ndarray) -> np.ndarray:
        return cv2.GaussianBlur(x, (0, 0), 3.0)

    shares = []
    for t, frame in enumerate(frames):
        if radius is not None or t == 0:
            cover = _temporal_median(
                frames
                if radius is None
                else frames[max(0, t - radius) : t + radius + 1]
            )
            cover_grey = grey(cover)
            c = _band_pass(cover_grey)
            textured = np.sqrt(local(c * c)) > _OVERLAY_MIN_COVER_TEXTURE
        frame_grey = grey(frame)
        f = _band_pass(frame_grey)
        see_through = local(f * c) / np.sqrt(local(f * f) * local(c * c) + 1e-3)
        opaque = (
            (see_through < _OVERLAY_OPAQUE_NCC)
            & (local(np.abs(frame_grey - cover_grey)) > 6.0)
            & textured
        )
        opaque = np.clip(
            cv2.GaussianBlur(opaque.astype(np.float32), (0, 0), 4.0) * 1.5, 0.0, 1.0
        )
        change = np.abs(frame.astype(np.float32) - cover).mean(axis=-1)
        shares.append(float((change * opaque).sum() / (change.sum() + 1e-6)))
    return float(np.median(shares))


def _encode_reveal(bgr: np.ndarray) -> bytes | None:
    ok, jpeg = cv2.imencode(
        ".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), _MOTION_REVEAL_JPEG_QUALITY]
    )
    return jpeg.tobytes() if ok else None


def _skin_fraction(bgr_u8: np.ndarray) -> float:
    ycrcb = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2YCrCb)
    mask = (
        (ycrcb[:, :, 0] > 60)
        & (ycrcb[:, :, 1] > 135)
        & (ycrcb[:, :, 1] < 175)
        & (ycrcb[:, :, 2] > 80)
        & (ycrcb[:, :, 2] < 130)
    )
    return float(mask.mean())


def _to_gray_u8(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)


def _ecc_working_scale(h: int, w: int) -> float:
    return min(1.0, _MOTION_REVEAL_ECC_MAX_DIM / max(h, w))


def _ecc_downscale(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale >= 1.0:
        return gray
    h, w = gray.shape[:2]
    return cv2.resize(
        gray,
        (max(8, round(w * scale)), max(8, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _ecc_align(
    reference_small: np.ndarray, small: np.ndarray, mode: int, scale: float = 1.0
) -> np.ndarray | None:
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        cc, warp = cv2.findTransformECC(
            reference_small, small, warp, mode, _MOTION_REVEAL_ECC_CRITERIA, None, 5
        )
    except cv2.error:
        return None
    if not np.isfinite(cc) or not np.isfinite(warp).all():
        return None
    if scale < 1.0:
        warp = warp.copy()
        warp[:, 2] /= scale
    max_dim = max(reference_small.shape[:2]) / scale
    if np.hypot(warp[0, 2], warp[1, 2]) > _MOTION_REVEAL_MAX_WARP_TRANSLATION * max_dim:
        return None
    if (
        np.linalg.norm(warp[:, :2] - np.eye(2, dtype=np.float32))
        > _MOTION_REVEAL_MAX_WARP_LINEAR
    ):
        return None
    return warp


def _is_static(reference_gray: np.ndarray, gray: np.ndarray) -> bool:
    try:
        (dx, dy), response = cv2.phaseCorrelate(reference_gray, gray)
    except cv2.error:
        return False
    return bool(
        np.isfinite(dx)
        and np.isfinite(dy)
        and np.isfinite(response)
        and response >= _MOTION_REVEAL_STATIC_MIN_RESPONSE
        and abs(dx) < _MOTION_REVEAL_STATIC_SHIFT_PX
        and abs(dy) < _MOTION_REVEAL_STATIC_SHIFT_PX
    )


def _warp(frame: np.ndarray, warp: np.ndarray) -> np.ndarray:
    return cv2.warpAffine(
        frame,
        warp,
        (frame.shape[1], frame.shape[0]),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _dense_align_residual(
    reference: np.ndarray, residual: np.ndarray, gain: float
) -> tuple[np.ndarray, np.ndarray]:
    h, w = reference.shape[:2]
    scale = min(1.0, _MOTION_REVEAL_FLOW_MAX_DIM / max(h, w))
    size = (max(8, round(w * scale)), max(8, round(h * scale)))

    def flow_gray(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.resize(
            np.clip(128 + gray * gain, 0, 255).astype(np.uint8),
            size,
            interpolation=cv2.INTER_AREA,
        )

    try:
        flow = cv2.calcOpticalFlowFarneback(
            flow_gray(reference), flow_gray(residual), None, 0.5, 4, 21, 4, 7, 1.5, 0
        )
        if (
            flow is None
            or flow.shape != (size[1], size[0], 2)
            or not np.isfinite(flow).all()
        ):
            raise ValueError("Invalid motion-reveal optical flow")
        flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_LINEAR) / scale
        if w * scale < 8:
            flow[:, :, 0] *= w * scale / size[0]
        if h * scale < 8:
            flow[:, :, 1] *= h * scale / size[1]
        y, x = np.mgrid[:h, :w].astype(np.float32)
        map_x, map_y = x + flow[:, :, 0], y + flow[:, :, 1]
        if not np.isfinite(map_x).all() or not np.isfinite(map_y).all():
            raise ValueError("Invalid motion-reveal sampling coordinates")
        valid = (
            (map_x >= 0) & (map_x < w - 1) & (map_y >= 0) & (map_y < h - 1)
        ).astype(np.float32)
        return cv2.remap(
            residual, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        ), valid
    except (cv2.error, ValueError):
        logger.warning(
            "Failed to refine motion-reveal residual alignment; using the unaligned residual",
            exc_info=True,
        )
        return residual, np.ones((h, w), dtype=np.float32)


def _weighted_median(
    samples: list[np.ndarray], weights: list[np.ndarray]
) -> np.ndarray:
    stack = np.stack(samples)
    weight = np.broadcast_to(np.stack(weights)[..., None], stack.shape)
    order = np.argsort(stack, axis=0)
    sorted_values = np.take_along_axis(stack, order, axis=0)
    cumulative = np.cumsum(np.take_along_axis(weight, order, axis=0), axis=0)
    index = (cumulative >= 0.5 * cumulative[-1][None]).argmax(axis=0)
    return np.take_along_axis(sorted_values, index[None], axis=0)[0]


def _native_chroma(bgr_u8: np.ndarray) -> tuple[float, float] | None:
    lab = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)
    lum, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    bright = lum > np.percentile(lum, 70)
    if bright.sum() <= 100:
        return None
    return float(np.median(a[bright])), float(np.median(b[bright]))


def _skin_plausible(overlays: list[np.ndarray]) -> bool:
    chroma = [c for c in (_native_chroma(o) for o in overlays) if c is not None]
    if not chroma:
        return False
    point = np.array(
        [np.median([c[0] for c in chroma]), np.median([c[1] for c in chroma])],
        dtype=np.float64,
    )
    neutral = np.array([128.0, 128.0])
    axis = np.array(_MOTION_REVEAL_SKIN_LAB_AB) - neutral
    t = float(np.clip(np.dot(point - neutral, axis) / np.dot(axis, axis), 0.0, 1.0))
    return (
        float(np.linalg.norm(point - (neutral + t * axis)))
        <= _MOTION_REVEAL_SKIN_AXIS_MAX_DISTANCE
    )


def _band_pass(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (0, 0), 1.0) - cv2.GaussianBlur(gray, (0, 0), 4.0)


def _patches(x: np.ndarray) -> np.ndarray:
    p = _OVERLAY_PATCH
    h, w = x.shape[-2:]
    lead = x.shape[:-2]
    x = x[..., : h // p * p, : w // p * p]
    return (
        x.reshape(*lead, h // p, p, w // p, p)
        .swapaxes(-3, -2)
        .reshape(*lead, h // p, w // p, p * p)
    )


def _patch_ncc(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a * a).sum(-1) * (b * b).sum(-1) + 1e-6)


def _overlay_measures(gray: np.ndarray, radius: int | None) -> np.ndarray:
    n = len(gray)
    if radius is None:
        covers = np.median(gray, axis=0)[None]
    else:
        covers = np.stack(
            [
                np.median(gray[max(0, t - radius) : t + radius + 1], axis=0)
                for t in range(n)
            ]
        )
    residual = np.empty_like(gray)
    for t in range(n):
        cover = covers[0 if radius is None else t]
        cover = cover - cover.mean()
        frame = gray[t] - gray[t].mean()
        residual[t] = (
            frame
            - float((frame * cover).sum() / ((cover * cover).sum() + 1e-6)) * cover
        )
    cover_bp = _patches(np.stack([_band_pass(c) for c in covers]))
    frame_bp = _patches(np.stack([_band_pass(f) for f in gray]))
    residual_bp = _patches(np.stack([_band_pass(r) for r in residual]))
    change = _patches(
        np.stack([cv2.GaussianBlur(np.abs(r), (0, 0), 2.0) for r in residual])
    ).mean(-1)
    noise = _patches(
        np.stack([r - cv2.GaussianBlur(r, (0, 0), 1.0) for r in residual])
    ).std(-1)
    changed = change > max(_OVERLAY_MIN_CHANGE, 3.0 * float(np.median(noise)))
    candidates = changed & (cover_bp.std(-1) > _OVERLAY_MIN_COVER_TEXTURE)
    measures = np.zeros(len(_OVERLAY_GLOBAL_WEIGHTS))
    measures[0] = changed.mean()
    if not candidates.any():
        return measures
    see_through = _patch_ncc(frame_bp, cover_bp)
    dependent = np.abs(_patch_ncc(residual_bp, cover_bp))
    structured = residual_bp.std(-1) > 2.0 * noise
    transparent = (
        candidates
        & structured
        & (see_through > _OVERLAY_SEE_THROUGH_NCC)
        & (dependent < _OVERLAY_DEPENDENT_NCC)
    )
    opaque = candidates & (see_through < _OVERLAY_OPAQUE_NCC)
    n_changed = float(changed.sum())
    candidate_see_through = see_through[candidates]
    measures[1:] = (
        np.median(candidate_see_through),
        opaque.sum() / n_changed,
        transparent.sum() / n_changed,
        structured[candidates].mean(),
        (candidate_see_through < _OVERLAY_OPAQUE_NCC).mean(),
        np.log1p(np.median(change[changed])),
    )
    return measures


def _overlay_cover_radius(stack_u8: np.ndarray) -> tuple[bool, int | None]:
    h, w = stack_u8.shape[1:3]
    scale = min(1.0, _OVERLAY_WORK_MAX_DIM / max(h, w))
    size = (max(_OVERLAY_PATCH, int(w * scale)), max(_OVERLAY_PATCH, int(h * scale)))
    gray = np.stack(
        [
            cv2.resize(
                cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), size, interpolation=cv2.INTER_AREA
            )
            for f in stack_u8
        ]
    )
    gray = gray.astype(np.float32)
    whole, local = (
        _overlay_measures(gray, None),
        _overlay_measures(gray, _OVERLAY_LOCAL_RADIUS),
    )
    score = (
        float(
            whole @ np.array(_OVERLAY_GLOBAL_WEIGHTS)
            + local @ np.array(_OVERLAY_LOCAL_WEIGHTS)
        )
        + _OVERLAY_BIAS
    )
    if score < _OVERLAY_MIN_SCORE:
        return False, None
    return True, (
        _OVERLAY_LOCAL_RADIUS
        if whole[2] > local[2] + _OVERLAY_LOCAL_OPAQUE_MARGIN
        else None
    )


def _to_grey(bgr_u8: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def build_motion_reveal_images(frame_jpegs: list[bytes]) -> list[bytes]:
    if len(frame_jpegs) < _MOTION_REVEAL_MIN_FRAMES:
        return []
    if len(frame_jpegs) > _MOTION_REVEAL_MAX_INPUT_FRAMES:
        indices = np.linspace(
            0, len(frame_jpegs) - 1, _MOTION_REVEAL_MAX_INPUT_FRAMES
        ).astype(int)
        frame_jpegs = [frame_jpegs[i] for i in indices]

    decoded: list[np.ndarray] = []
    target_hw: tuple[int, int] | None = None
    for frame_bytes in frame_jpegs:
        img = _decode_reveal_frame(frame_bytes)
        if img is None:
            continue
        if target_hw is None:
            target_hw = (img.shape[0], img.shape[1])
        elif (img.shape[0], img.shape[1]) != target_hw:
            img = cv2.resize(
                img, (target_hw[1], target_hw[0]), interpolation=cv2.INTER_AREA
            )
        decoded.append(img)
    if len(decoded) < _MOTION_REVEAL_MIN_FRAMES:
        return []

    originals_u8 = np.stack(decoded, axis=0)

    raw_median = _temporal_median(originals_u8)
    raw_p99 = np.percentile(
        _residual_magnitude(originals_u8, raw_median).reshape(len(originals_u8), -1),
        99,
        axis=1,
    )
    if float(np.median(raw_p99)) > _MOTION_REVEAL_MAX_RESIDUAL_P99:
        return []
    is_overlay, cover_radius = _overlay_cover_radius(originals_u8)
    if (
        not is_overlay
        or _opaque_motion_share(originals_u8, cover_radius)
        > _MOTION_REVEAL_MAX_OPAQUE_MOTION_SHARE
    ):
        return []

    reference_gray = _to_gray_u8(raw_median)
    warps: dict[int, np.ndarray] = {}
    if (
        cover_radius is None
        and _cover_gradient(reference_gray) >= _MOTION_REVEAL_MIN_COVER_GRADIENT
    ):
        scale = _ecc_working_scale(*originals_u8.shape[1:3])
        reference_gray_f32 = reference_gray.astype(np.float32)
        reference_small = _ecc_downscale(reference_gray, scale)
        for index, frame_u8 in enumerate(originals_u8):
            gray = cv2.cvtColor(frame_u8, cv2.COLOR_BGR2GRAY)
            if _is_static(reference_gray_f32, gray.astype(np.float32)):
                continue
            warp = _ecc_align(
                reference_small, _ecc_downscale(gray, scale), cv2.MOTION_AFFINE, scale
            )
            if warp is not None:
                warps[index] = warp
    del reference_gray

    if not warps:
        stack: np.ndarray = originals_u8
        median = raw_median
    else:
        stack = np.empty(originals_u8.shape, dtype=np.float32)
        for index, frame_u8 in enumerate(originals_u8):
            warp = warps.get(index)
            stack[index] = (
                frame_u8 if warp is None else _warp(frame_u8.astype(np.float32), warp)
            )
        median = np.median(stack, axis=0)

    windows = np.array_split(
        np.arange(len(stack)), max(1, len(stack) // _MOTION_REVEAL_WINDOW_FRAMES)
    )
    covers = [
        median
        if cover_radius is None
        else _temporal_median(
            stack[max(0, window[0] - cover_radius) : window[-1] + cover_radius + 1]
        )
        for window in windows
    ]
    motion_energy = np.concatenate(
        [
            _residual_magnitude(stack[window], cover).mean(axis=(1, 2))
            for window, cover in zip(windows, covers)
        ]
    )
    sigma = float(
        np.interp(
            float(np.median(motion_energy)),
            _MOTION_REVEAL_DENOISE_ENERGY,
            _MOTION_REVEAL_DENOISE_SIGMA,
        )
    )

    overlays: list[np.ndarray] = []
    overlay_energy: list[float] = []
    skin: list[float] = []
    reference_frames: list[int] = []
    for window, cover in zip(windows, covers):
        ref = int(window[np.argmax(motion_energy[window])])
        reference = stack[ref] - cover
        samples = [reference]
        sample_weights = [np.ones(reference.shape[:2], dtype=np.float32)]
        magnitude = max(1.0, float(np.percentile(np.abs(reference), 85)))
        for i in window:
            if i == ref:
                continue
            aligned_residual, valid = _dense_align_residual(
                reference, stack[i] - cover, min(16.0, 96 / magnitude)
            )
            difference = cv2.GaussianBlur(
                np.abs(aligned_residual - reference).mean(axis=-1), (0, 0), 1.2
            )
            samples.append(aligned_residual)
            sample_weights.append(
                np.exp(-((difference / (0.7 * magnitude)) ** 2)) * valid
            )
        overlay = _weighted_median(samples, sample_weights)
        overlay_energy.append(float(np.abs(overlay).mean()))
        if sigma > 0:
            overlay = cv2.bilateralFilter(overlay, 5, max(1.0, magnitude * 0.18), sigma)
        stretched = _stretch_to_u8(overlay)
        skin.append(_skin_fraction(stretched))
        overlays.append(stretched)
        reference_frames.append(ref)

    skin_plausible = _skin_plausible(overlays)
    if not skin_plausible:
        overlays = [_to_grey(o) for o in overlays]

    score = np.array(overlay_energy) * (0.5 + np.array(skin))

    out: list[bytes] = []

    best = int(np.argmax(score))
    encoded = _encode_reveal(
        np.concatenate([originals_u8[reference_frames[best]], overlays[best]], axis=1)
    )
    if encoded:
        out.append(encoded)

    peaks = [int(window[np.argmax(motion_energy[window])]) for window in windows]
    strongest = sorted(range(len(windows)), key=lambda i: -motion_energy[peaks[i]])[
        :_MOTION_REVEAL_PLAIN_STILLS
    ]
    for i in strongest:
        plain = _plain_reveal(stack[peaks[i]].astype(np.float32) - covers[i])
        encoded = _encode_reveal(plain if skin_plausible else _to_grey(plain))
        if encoded:
            out.append(encoded)

    picks = np.linspace(
        0, len(overlays) - 1, min(_MOTION_REVEAL_OVERVIEW_FRAMES, len(overlays))
    ).astype(int)
    tiles = [overlays[i] for i in picks]
    h, w = tiles[0].shape[:2]
    cols = math.ceil(math.sqrt(len(tiles)))
    rows = math.ceil(len(tiles) / cols)
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, cols)
        canvas[row * h : (row + 1) * h, col * w : (col + 1) * w] = tile
    encoded = _encode_reveal(canvas)
    if encoded:
        out.append(encoded)

    details = sorted(
        int(i)
        for i in np.argsort(-score)[: _MOTION_REVEAL_DETAIL_STILLS + 1]
        if int(i) != best
    )[:_MOTION_REVEAL_DETAIL_STILLS]
    for i in details:
        encoded = _encode_reveal(overlays[i])
        if encoded:
            out.append(encoded)

    return out
