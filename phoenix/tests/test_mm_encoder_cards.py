# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 X.AI Corp.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "reference"))
from mm_encoder import (
    IMAGE_PAD,
    CardV2,
    Image,
    LegacyCard,
    PollCard,
    Post,
    UnifiedCard,
    render_post,
)


@pytest.mark.parametrize("max_images", [0, 1, 2, None])
def test_image_budget_keeps_all_card_text(max_images):
    post = Post(
        text="Body",
        media=[Image(content=b"body image")],
        cards=[
            CardV2(
                legacy_card=LegacyCard(
                    title="Legacy title",
                    description="Legacy description",
                    thumbnail_image=Image(content=b"thumbnail"),
                    poll_cards=[PollCard(choice_label="Poll answer", choice_image=Image(b"poll"))],
                )
            ),
            CardV2(
                unified_cards=[
                    UnifiedCard(title="Unified title", description="Unified description")
                ]
            ),
        ],
    )

    text, images = render_post(post, max_images=max_images)

    for expected in (
        "Body",
        "Card: Legacy title",
        "Legacy description",
        "Poll choice: Poll answer",
        "Card: Unified title",
        "Unified description",
    ):
        assert expected in text
    expected_images = [b"body image", b"thumbnail", b"poll"]
    if max_images is not None:
        expected_images = expected_images[:max_images]
    assert images == expected_images
    assert text.count(IMAGE_PAD) == len(images)


def test_quoted_card_text_survives_zero_per_post_image_budget():
    quoted = Post(cards=[CardV2(legacy_card=LegacyCard(title="Quoted card"))])

    text, images = render_post(Post(quoted_post=quoted), max_images=1)

    assert "Card: Quoted card" in text
    assert images == []
