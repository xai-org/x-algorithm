from functools import partial

import tensorflow as tf
from twitter.clip.utils.data_utils import (
    clean_text as clean_text_fn,
)
from twitter.clip.utils.data_utils import (
    get_clip_decode_and_proc_fn,
    read_tfrecords_from_gcs,
)


class LanguageImageGcsDataset:
    def __init__(
        self,
        features=None,
        text_field="tweet_text",
        encoded_image_field="image_byte_string",
        image_field="image",
        input_resolution=224,
        channels=3,
        resize_side_buffer=32,
        use_data_augmentation=False,
        clean_text=True,
        batch_size=128,
        eval_batch_size=None,
        buffer_size=None,
    ):
        if features is None:
            features = {
                "tweet_text": tf.io.FixedLenFeature([], tf.string),
                "image_byte_string": tf.io.FixedLenFeature([], tf.string),
            }
        eval_batch_size = batch_size if not eval_batch_size else eval_batch_size
        buffer_size = batch_size * 100 if not buffer_size else buffer_size
        for argname, value in locals().items():
            if argname != "self":
                setattr(self, argname, value)

    def create_dataset(
        self,
        dataset_path,
        tfrec_ext="*.tfrecord",
        is_training=True,
        disable_repeat=False,
        disable_shuffle=False,
        decode_and_proc=True,
        decode_and_proc_fn=None,
        seed=None,
    ):
        batch_size = self.batch_size if is_training else self.eval_batch_size
        dataset = read_tfrecords_from_gcs(
            dataset_path=dataset_path,
            tfrec_ext=tfrec_ext,
            features=self.features,
            num_parallel_read_calls=tf.data.experimental.AUTOTUNE,
            prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
        )
        if is_training and not disable_shuffle:
            dataset = dataset.shuffle(self.buffer_size, seed=seed)
        if decode_and_proc:
            if not decode_and_proc_fn:
                data_augmentation = self.use_data_augmentation and is_training
                decode_and_proc_fn = get_clip_decode_and_proc_fn(
                    encoded_image_field=self.encoded_image_field,
                    image_field=self.image_field,
                    data_augmentation=data_augmentation,
                    encoder_input_resolution=self.input_resolution,
                    channels=self.channels,
                )
            dataset = dataset.map(
                decode_and_proc_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        dataset = dataset.apply(tf.data.experimental.ignore_errors())
        if self.clean_text:
            dataset = dataset.map(
                partial(clean_text_fn, text_field=self.text_field),
                tf.data.experimental.AUTOTUNE,
            )
        if is_training and not disable_repeat:
            dataset = dataset.repeat()
        dataset = dataset.batch(batch_size)
        return dataset.prefetch(1)
