from functools import partial

from com.twitter.media_understanding.common.dbv2_utils.keras_export import (
    DEFAULT_IMAGE_FEATURE_NAME,
)
from twitter.clip.model import (
    DEFAULT_CLIP_MODEL_TYPE,
    DEFAULT_FINAL_EMBEDDING_DIM,
    TwitterCLIP,
)
from twitter.clip.utils import data_utils


def export_model(
    output_dir,
    clip_model_type,
    top_feedforward,
    final_embedding_dim,
):
    model = TwitterCLIP(
        clip_model_type=clip_model_type,
        top_feedforward=top_feedforward,
        final_embedding_dim=final_embedding_dim,
    )

    image_feature_name = DEFAULT_IMAGE_FEATURE_NAME

    image_size = model.clip_model.visual.input_resolution

    preprocessing_fn = partial(
        data_utils.clip_preproc_val,
        output_image_shape=(3, image_size, image_size),
        means=data_utils.PRETRAINED_MEANS,
        stdevs=data_utils.PRETRAINED_STDEVS,
    )

    model.export_image_encoder_to_dbv2_model_file(
        output_dir, preprocessing_fn, image_feature_name
    )

    model.export_image_encoder_to_dbv2_model_file(
        output_dir, preprocessing_fn, image_feature_name=DEFAULT_IMAGE_FEATURE_NAME
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--clip_model_type", type=str, default=DEFAULT_CLIP_MODEL_TYPE)
    parser.add_argument(
        "--no_top_feedforward",
        action="store_false",
        default=True,
        dest="top_feedforward",
    )
    parser.add_argument(
        "--final_embedding_dim", type=int, default=DEFAULT_FINAL_EMBEDDING_DIM
    )

    export_model(**vars(parser.parse_args()))
