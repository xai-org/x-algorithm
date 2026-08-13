import os
import tempfile

import numpy as np
from twitter.clip.config import ClipConfig
from twitter.clip.keras_model import TwitterCLIPKeras
from twitter.clip.modules import ImageCLIP, TextCLIP
from twitter.clip.utils import io_utils, model_utils, train_utils

train_utils.set_memory_growth()

import onnx
import tensorflow as tf
import torch
from com.twitter.media_understanding.common.dbv2_utils.keras_export import (
    DEFAULT_IMAGE_FEATURE_NAME,
    DBV2ExportWrapper,
)
from onnx_tf.backend import prepare
from PIL import Image

import clip

DEFAULT_CLIP_MODEL_TYPE = "Twitter-ViT-B/32-256"
DEFAULT_FINAL_EMBEDDING_DIM = ClipConfig.DEFAULT_FINAL_EMBEDDING_DIM


class ImageResolutionError(Exception):
    pass


class TwitterCLIP:
    PRETRAINED_MODELS = {
        "RN50": "https://example.invalid/models/open_source/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt",
        "RN101": "https://example.invalid/models/open_source/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt",
        "RN50x4": "https://example.invalid/models/open_source/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt",
        "RN50x16": "https://example.invalid/models/open_source/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt",
        "ViT-B/32": "https://example.invalid/models/open_source/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt",
        "ViT-B/16": "https://example.invalid/models/open_source/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt",
    }

    TWITTER_MODELS = {
        "Twitter-ViT-B/32-256": "https://example.invalid/models/twitter/clip_source_d256_bs2784/clip_source_d256_bs2784_twitter-clip-d256.pt",
        "Twitter-ViT-B/32-128": "https://example.invalid/models/twitter/clip_source_d128_bs2784/clip_source_d128_bs2784_twitter-clip-d128.pt",
    }

    TWITTER_MODELS_PRETRAINED_BASE = {
        "Twitter-ViT-B/32-256": "ViT-B/32",
        "Twitter-ViT-B/32-128": "ViT-B/32",
    }

    MODELS = {**PRETRAINED_MODELS, **TWITTER_MODELS}

    MODEL_NAMES = tuple(MODELS.keys())

    def __init__(
        self,
        clip_model_type=DEFAULT_CLIP_MODEL_TYPE,
        top_feedforward=True,
        final_embedding_dim=DEFAULT_FINAL_EMBEDDING_DIM,
        truncate_text=True,
        logit_scale=None,
        use_gpu=None,
    ):
        self.clip_model_type = clip_model_type

        if clip_model_type not in self.MODELS:
            raise ValueError(
                f"Unknown model type: {clip_model_type}. Choices: {self.MODELS.keys()}"
            )

        if clip_model_type in self.PRETRAINED_MODELS:
            self.base_clip_model_type = clip_model_type
        else:
            self.base_clip_model_type = self.TWITTER_MODELS_PRETRAINED_BASE[
                clip_model_type
            ]

        io_utils.maybe_download_file(self.PRETRAINED_MODELS[self.base_clip_model_type])
        self.clip_model, self.torch_preprocess_pretrained = clip.load(
            self.base_clip_model_type, jit=False
        )

        self.image_encoder = ImageCLIP(
            self.clip_model,
            top_feedforward=top_feedforward,
            d_out=final_embedding_dim,
        )
        self.text_encoder = TextCLIP(
            self.clip_model,
            top_feedforward=top_feedforward,
            d_out=final_embedding_dim,
        )
        self.top_feedforward = top_feedforward
        self.truncate_text = truncate_text

        logit_scale = (
            logit_scale if logit_scale is not None else self.clip_model.logit_scale
        )
        self.clip_model.logit_scale = torch.nn.Parameter(torch.ones([]) * logit_scale)
        self.use_gpu = use_gpu if use_gpu is not None else torch.cuda.is_available()
        self._multi_gpu = self.use_gpu and torch.cuda.device_count() > 1
        self._device = "cuda:0" if self.use_gpu else "cpu"
        if self.use_gpu:
            self._prepare_model_for_gpu()

        self.clip_model.logit_scale.to(self._device)

        if clip_model_type in self.TWITTER_MODELS:
            map_location = None if self.use_gpu else torch.device("cpu")
            local_model_path = io_utils.maybe_download_file(
                self.TWITTER_MODELS[clip_model_type]
            )
            print(f"loading model from: {local_model_path}")
            checkpoint_contents = torch.load(
                local_model_path, map_location=map_location
            )

            config = checkpoint_contents["config"]
            if config["top_feedforward"] != top_feedforward:
                raise ValueError(
                    f"The value of top_feedforward specified ({top_feedforward}) does not match that in the "
                    f"checkpoint for {clip_model_type} ({config['top_feedforward']})."
                )

            if self._multi_gpu:
                image_state_dict = checkpoint_contents[
                    "image_encoder_state_dict_multi_gpu"
                ]
                text_state_dict = checkpoint_contents[
                    "text_encoder_state_dict_multi_gpu"
                ]
            else:
                image_state_dict = checkpoint_contents["image_encoder_state_dict"]
                text_state_dict = checkpoint_contents["text_encoder_state_dict"]

            self.image_encoder.load_state_dict(image_state_dict, strict=True)
            self.text_encoder.load_state_dict(text_state_dict, strict=True)

        if self.top_feedforward:
            self.embedding_dim = final_embedding_dim
        else:
            self.embedding_dim = self.clip_model.visual.output_dim

    def _prepare_model_for_gpu(self):
        if self._multi_gpu:
            self.image_encoder = torch.nn.DataParallel(self.image_encoder)
            self.text_encoder = torch.nn.DataParallel(self.text_encoder)
        model_utils.convert_models_to_fp32(self.image_encoder)
        model_utils.convert_models_to_fp32(self.text_encoder)
        if self._multi_gpu:
            self.image_encoder.to(f"cuda:{self.image_encoder.device_ids[0]}")
            self.text_encoder.to(f"cuda:{self.text_encoder.device_ids[0]}")
        else:
            self.image_encoder.to(self._device)
            self.text_encoder.to(self._device)

    def encode_images(self, images, return_numpy=False, normalize_outputs=True):
        images = images.numpy()

        height = images.shape[2]
        width = images.shape[3]
        input_resolution = self.clip_model.visual.input_resolution

        if height != input_resolution:
            raise ImageResolutionError(
                f"Input image height ({height}) does not match model input resolution ({input_resolution})."
            )

        if width != input_resolution:
            raise ImageResolutionError(
                f"Input image width ({width}) does not match model input resolution ({input_resolution})."
            )

        images = torch.Tensor(images).to(self._device)

        image_embeddings = self.image_encoder(images)

        if normalize_outputs:
            image_embeddings = self.normalize_outputs(image_embeddings)

        if return_numpy:
            return image_embeddings.detach().cpu().numpy()
        else:
            return image_embeddings

    def encode_texts(self, texts, return_numpy=False, normalize_outputs=True):
        texts = clip.tokenize(
            [t.decode("utf-8") for t in texts.numpy()], truncate=self.truncate_text
        )

        text_embeddings = self.text_encoder(texts.to(self._device))

        if normalize_outputs:
            text_embeddings = self.normalize_outputs(text_embeddings)

        if return_numpy:
            return text_embeddings.detach().cpu().numpy()
        else:
            return text_embeddings

    def encode(
        self,
        features,
        image_field="image",
        text_field="tweet_text",
        return_numpy=False,
        normalize_outputs=True,
    ):
        return {
            "image_embeddings": self.encode_images(
                features[image_field],
                return_numpy=return_numpy,
                normalize_outputs=normalize_outputs,
            ),
            "text_embeddings": self.encode_texts(
                features[text_field],
                return_numpy=return_numpy,
                normalize_outputs=normalize_outputs,
            ),
        }

    def get_logits(self, features, image_field="image", text_field="tweet_text"):
        out = self.encode(features, image_field=image_field, text_field=text_field)
        logit_scale = torch.clamp(self.clip_model.logit_scale.exp(), max=100)
        image_logits, text_logits = model_utils.create_logits(
            out["image_embeddings"], out["text_embeddings"], logit_scale
        )
        return {
            "image_logits": image_logits,
            "text_logits": text_logits,
        }

    @staticmethod
    def normalize_outputs(x):
        if isinstance(x, np.ndarray):
            return x / np.linalg.norm(x, axis=-1, keepdims=True)
        elif isinstance(x, tf.Tensor):
            return x / tf.norm(x, ord="euclidean", axis=-1, keepdims=True)
        else:
            return x / x.norm(dim=-1, keepdim=True)

    def _to_onnx(self, output_dir, image=True, text=True):
        is_training = self.clip_model.training

        if not is_training:
            self.clip_model.eval()

        output_path_image = None
        output_path_text = None

        if image:
            output_path_image = os.path.join(output_dir, "image_encoder.onnx")
            input_image = Image.new("RGB", (200, 200))
            input_image = (
                self.torch_preprocess_pretrained(input_image)
                .unsqueeze(0)
                .to(self._device)
            )
            torch.onnx.export(
                self.image_encoder,
                input_image,
                output_path_image,
                input_names=["image"],
            )

        if text:
            output_path_text = os.path.join(output_dir, "text_encoder.onnx")
            input_text = ["foo", "bar"]
            input_text = clip.tokenize(input_text, truncate=self.truncate_text).to(
                self._device
            )
            torch.onnx.export(
                self.text_encoder, input_text, output_path_text, input_names=["text"]
            )

        if is_training:
            self.clip_model.train()

        return output_path_image, output_path_text

    def _to_onnx_tf_rep(self, image=True, text=True):
        with tempfile.TemporaryDirectory() as tmp_dir:
            onnx_path_image, onnx_path_text = self._to_onnx(
                tmp_dir, image=image, text=text
            )

            if image:
                onnx_model_image = onnx.load(onnx_path_image)
                tf_rep_image = prepare(onnx_model_image)
                image_module = tf_rep_image.tf_module
            else:
                image_module = None

            if text:
                onnx_model_text = onnx.load(onnx_path_text)
                tf_rep_text = prepare(onnx_model_text)
                text_module = tf_rep_text.tf_module
            else:
                text_module = None

        return image_module, text_module

    def export_image_encoder_to_dbv2_model_file(
        self,
        output_dir,
        preprocessing_fn,
        image_feature_name=DEFAULT_IMAGE_FEATURE_NAME,
        normalize_outputs=True,
        classification_heads={},
    ):
        tf_rep, _ = self._to_onnx_tf_rep(image=True, text=False)

        def _model(image):
            output = {}
            output_tf_rep = tf_rep(image=image)
            keys = list(output_tf_rep.keys())
            output["embedding"] = output_tf_rep[keys[0]]
            if normalize_outputs:
                output["embedding"] = self.normalize_outputs(output["embedding"])
            return output

        image_size = self.clip_model.visual.input_resolution
        preprocessed_image_shape = (None, 3, image_size, image_size)

        wrapped_model = DBV2ExportWrapper(
            _model,
            preprocessing_fn,
            image_feature_name,
            preprocessed_image_shape=preprocessed_image_shape,
            classification_heads=classification_heads,
        )
        wrapped_model.export_for_dbv2_serving(output_dir)

    def export_to_keras_saved_model(self, output_dir):
        print("Exporting torch model to onnx...")
        tf_rep_image, tf_rep_text = self._to_onnx_tf_rep(image=True, text=True)
        print("Exporting torch model to onnx... done!")

        texts = ["hello", "world"]
        texts = clip.tokenize(texts).numpy()

        image_size = self.clip_model.visual.input_resolution
        inputs = {
            "images": np.ones((2, 3, image_size, image_size), dtype=np.float32),
            "texts": tf.constant(np.asarray(texts)),
        }

        print("Exporting onnx model to keras...")
        keras_model = TwitterCLIPKeras(
            tf_rep_image, tf_rep_text, image_size, self.embedding_dim
        )
        keras_model.predict(inputs)
        signatures = keras_model.get_export_signatures()
        keras_model.save(output_dir, signatures=signatures)

        print("Exporting onnx model to keras... done!")
