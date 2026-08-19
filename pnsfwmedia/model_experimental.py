import os

import gin
import numpy as np
import pandas
import tensorflow.compat.v2 as tf
from twitter.deepbird.data import (
    get_batch_prediction_request_decoder,
    get_batch_prediction_response_writer,
    optimized_block_format_dataset,
)
from twitter.deepbird.io.legacy.contrib.feature_config import FeatureConfigBuilder
from twitter.deepbird.stats_server.app_with_stats import run_with_stats_server


class Model(tf.keras.Model):
    def __init__(self, clip_model, feature_config=None):
        super().__init__()

        self._clip_model = clip_model
        self._clip_model.trainable = False

        self._norm_layer = tf.keras.layers.BatchNormalization()
        self._first_layer = tf.keras.layers.Dense(50, activation="relu")
        self._classifier_layer = tf.keras.layers.Dense(1, activation="sigmoid")

        if feature_config:
            self._batch_prediction_request_decoder = (
                get_batch_prediction_request_decoder(feature_config)
            )
            self._batch_prediction_response_writer = (
                get_batch_prediction_response_writer("binary")
            )
        else:
            self._batch_prediction_request_decoder = None
            self._batch_prediction_response_writer = None

    def get_config(self):
        config = super().get_config()
        config.update({"_clip_model": self._clip_model})
        return config

    def call(self, features):
        X = self._get_final_feature_tensor(features)
        X = self._norm_layer(X)
        X = self._first_layer(X)
        return self._classifier_layer(X)

    def _get_final_feature_tensor(self, features):
        image_score = self._clip_model(
            features["media.mediaunderstanding.media_embeddings.twitter_clip_as_tensor"]
        )
        user_agatha_score = features[
            "source_user.hss.user_health_scores.agatha_calibrated_nsfw_score"
        ]
        user_text_score = features[
            "source_user.hss.user_health_scores.nsfw_text_user_score"
        ]
        nsfw_consumer_follower_score = features[
            "source_user.hss.user_health_scores.nsfw_consumer_follower_score"
        ]
        log_followers = tf.math.log(
            tf.cast(
                features["source_user.core.socialgraph_counts.followed_by_count"],
                tf.float32,
            )
            + 10.0
        )

        is_agatha_score_missing = tf.cast(
            tf.math.equal(user_agatha_score, -1.0), tf.float32
        )
        is_user_text_score_missing = tf.cast(
            tf.math.equal(user_text_score, -1.0), tf.float32
        )
        is_nsfw_consumer_follower_score_missing = tf.cast(
            tf.math.equal(nsfw_consumer_follower_score, -1.0), tf.float32
        )

        feature_list = [
            image_score,
            user_agatha_score,
            user_text_score,
            log_followers,
            nsfw_consumer_follower_score,
            is_agatha_score_missing,
            is_user_text_score_missing,
            is_nsfw_consumer_follower_score_missing,
        ]

        return tf.keras.layers.Concatenate(axis=1)(feature_list)

    @tf.function(
        input_signature=[tf.TensorSpec(shape=[None], dtype=tf.uint8, name="request")]
    )
    def serve(self, request):
        assert self._batch_prediction_request_decoder, (
            "Must supply feature config if you use serve()"
        )

        features = self._batch_prediction_request_decoder(request)
        output = self.call(features)
        response = self._batch_prediction_response_writer(output)
        return response


def get_feature_config(data_spec):
    return (
        FeatureConfigBuilder(data_spec)
        .extract_feature(
            "source_user.hss.user_health_scores.agatha_calibrated_nsfw_score",
            default_value=-1.0,
        )
        .extract_feature(
            "source_user.hss.user_health_scores.nsfw_text_user_score",
            default_value=-1.0,
        )
        .extract_feature("source_user.core.socialgraph_counts.followed_by_count")
        .extract_feature(
            "source_user.hss.user_health_scores.nsfw_consumer_follower_score",
            default_value=-1.0,
        )
        .extract_tensors(
            {
                "media.mediaunderstanding.media_embeddings.twitter_clip_as_tensor": np.zeros(
                    256, dtype="float64"
                )
            }
        )
        .add_labels("label")
        .build()
    )


@gin.configurable("main")
def main(
    data_path,
    clip_model_path,
    batch_size=32,
    epochs=20,
    steps_per_epoch=100,
    save_dir=None,
):
    data_spec = os.path.join(data_path, "_data_spec.json")
    training_part_file = os.path.join(data_path, "training.lzo")
    validation_part_file = os.path.join(data_path, "validation.lzo")
    feature_config = get_feature_config(data_spec)
    parse_fn = feature_config.get_parse_fn()

    training_data = optimized_block_format_dataset(
        [training_part_file], parse_fn, batch_size=batch_size, repeat=True
    )

    validation_data = optimized_block_format_dataset(
        [validation_part_file], parse_fn, batch_size=batch_size, repeat=False
    )

    clip_model = tf.keras.models.load_model(clip_model_path)
    model = Model(clip_model, feature_config=feature_config)

    loss = tf.keras.losses.BinaryCrossentropy()
    model.compile(optimizer="adam", loss=loss, metrics=[tf.keras.metrics.AUC()])

    model.fit(
        training_data,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_data,
    )

    if save_dir:
        model.save(save_dir, signatures={"serve": model.serve})

        model = tf.keras.models.load_model(save_dir, custom_objects={"Model": Model})

        data = validation_data.unbatch()
        n_elements = sum(1 for _, _ in data)
        data = data.batch(n_elements)

        for features, labels in data:
            pandas.DataFrame(
                {
                    "score": model.predict(features).reshape(-1),
                    "label": labels.numpy().reshape(-1),
                }
            ).to_csv(save_dir + "predictions.csv", index=False)
            break


if __name__ == "__main__":
    configs = """
  # Trained model destination
  SAVE_DIR='/tmp/pnsfwmedia_experimental/'

  # Path to trained CLIP model
  CLIP_MODEL_PATH='health-sensitive-illegal-content/models/pnsfwmedia/training/clip_model'

  # Path to directory containing training data records
  DATA_PATH='health-sensitive-illegal-content/models/pnsfwmedia/training/data_images_only'

  main.save_dir=%SAVE_DIR
  main.data_path=%DATA_PATH
  main.clip_model_path=%CLIP_MODEL_PATH

  # For opening stats server
  run_with_stats_server.stats_port=6616
  run_with_stats_server.tf_log_dir=%SAVE_DIR
  """

    gin.parse_config(configs)
    run_with_stats_server(main)
