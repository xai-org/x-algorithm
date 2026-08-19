import os
import pathlib
import subprocess
import tempfile
import zipfile

import tensorflow as tf
from twitter.clip.utils import data_utils

OPEN_SOURCE_MODELS = ["ViT-B/32"]
TWITTER_MODELS = ["Twitter-ViT-B/32-256", "Twitter-ViT-B/32-128"]
PROD_MODEL = TWITTER_MODELS[0]

MODELS = OPEN_SOURCE_MODELS + TWITTER_MODELS


def get_clip_model(model_name=PROD_MODEL):
    model_path = _maybe_download_model(model_name)
    return tf.keras.models.load_model(model_path)


def get_packer_package_name(model_name):
    model_name = model_name.replace("/", "-")
    return f"CLIP-keras-{model_name}"


def _maybe_download_model(
    model_name, download_dir="~/.cache/clip/keras", cluster="atla"
):
    download_dir = os.path.expanduser(download_dir)
    package_name = get_packer_package_name(model_name)

    local_path = os.path.join(download_dir, package_name)

    expected_files = [
        "saved_model.pb",
        "keras_metadata.pb",
        "variables/variables.data-00000-of-00001",
        "variables/variables.index",
    ]

    all_files_exist_locally = True
    for file in expected_files:
        if not os.path.isfile(os.path.join(local_path, file)):
            all_files_exist_locally = False
            break

    if not all_files_exist_locally:
        with tempfile.TemporaryDirectory() as tmp_dir:
            fetch_cmd = [
                "packer",
                "fetch",
                f"--cluster={cluster}",
                "embeddings-category",
                package_name,
                "latest",
                "-v",
            ]
            print(f"Downloading model from packer: {package_name} -> {tmp_dir}")
            subprocess.check_output(fetch_cmd, cwd=tmp_dir)

            zip_file_path = os.path.join(tmp_dir, package_name + ".zip")

            pathlib.Path(download_dir).mkdir(parents=True, exist_ok=True)

            print(f"Extracting model to: {local_path}")
            with zipfile.ZipFile(zip_file_path, "r") as zip_ref:
                zip_ref.extractall(local_path)

    return local_path


class TwitterCLIPKeras(tf.keras.Model):
    IMAGE_EMBEDDINGS_OUTPUT_KEY = "image_embeddings"
    TEXT_EMBEDDINGS_OUTPUT_KEY = "text_embeddings"

    def __init__(self, image_module, text_module, image_size, embedding_dim):
        super().__init__()
        self.image_module = image_module
        self.text_module = text_module
        self.image_size = image_size
        self.embedding_dim = embedding_dim

    @tf.function(
        input_signature=[
            {
                "images": tf.TensorSpec([None, 3, None, None], tf.float32),
                "texts": tf.TensorSpec([None, None], tf.int64),
            }
        ]
    )
    def call(self, inputs):
        return {
            **self.predict_from_preprocessed_images(inputs["images"]),
            **self.predict_from_preprocessed_texts(inputs["texts"]),
        }

    @tf.function(input_signature=[])
    def get_embedding_dim(self):
        return self.embedding_dim

    @tf.function(input_signature=[tf.TensorSpec([], tf.string)])
    def decode_and_preprocess_image(self, image):
        image = tf.io.decode_image(image, channels=3)
        image = self.preprocess_image(image)
        return image

    @tf.function(input_signature=[tf.TensorSpec(None, tf.uint8)])
    def preprocess_image(self, image):
        return data_utils.clip_preproc_val(
            image, output_image_shape=(3, self.image_size, self.image_size)
        )

    @tf.function(input_signature=[tf.TensorSpec([None], tf.string)])
    def predict_from_encoded_images(self, images):
        preprocessed_images = tf.map_fn(
            self.decode_and_preprocess_image, images, fn_output_signature=tf.float32
        )
        return self.predict_from_preprocessed_images(preprocessed_images)

    @tf.function(input_signature=[tf.TensorSpec([None, 3, None, None], tf.float32)])
    def predict_from_preprocessed_images(self, images):
        image_embedding = self.image_module(image=images)
        image_keys = list(image_embedding.keys())
        image_embedding = image_embedding[image_keys[-1]]
        image_embedding /= tf.norm(
            image_embedding, ord="euclidean", axis=-1, keepdims=True
        )
        return {self.IMAGE_EMBEDDINGS_OUTPUT_KEY: image_embedding}

    @tf.function(input_signature=[tf.TensorSpec([None, None], tf.int64)])
    def predict_from_preprocessed_texts(self, texts):
        text_embedding = self.text_module(text=texts)
        text_keys = list(text_embedding.keys())
        text_embedding = text_embedding[text_keys[-1]]
        text_embedding /= tf.norm(
            text_embedding, ord="euclidean", axis=-1, keepdims=True
        )
        return {self.TEXT_EMBEDDINGS_OUTPUT_KEY: text_embedding}

    def get_export_signatures(self):
        signatures = {"serving_default": self.call.get_concrete_function()}

        tf_fn_names = [
            "predict_from_encoded_images",
            "predict_from_preprocessed_images",
            "predict_from_preprocessed_texts",
        ]

        for fn_name in tf_fn_names:
            signatures[fn_name] = getattr(self, fn_name).get_concrete_function()

        return signatures
