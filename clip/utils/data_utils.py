import os
import re
import time

import numpy as np
import tensorflow as tf
import tqdm
from tensorflow_models_official_resnet import vgg_preprocessing
from twitter.image_classification.utils import parallel_file_access
from twitter.magicpony.common import file_access

PRETRAINED_MEANS = (0.48145466, 0.4578275, 0.40821073)
PRETRAINED_STDEVS = (0.26862954, 0.26130258, 0.27577711)

TWITTER_MEANS = (0.5233059176252566, 0.4864276130294983, 0.4649590681783273)
TWITTER_STDEVS = (0.3215334269073608, 0.3114275303605159, 0.3149949567754017)


def read_tfrecords_from_gcs(
    features,
    dataset_path,
    tfrec_ext="*.tfrecord",
    num_parallel_readers=None,
    num_parallel_read_calls=tf.data.experimental.AUTOTUNE,
    prefetch_buffer_size=tf.data.experimental.AUTOTUNE,
    deterministic=False,
):
    def _read_tfrecord(data):
        return tf.io.parse_single_example(data, features)

    files = tf.data.Dataset.list_files(os.path.join(dataset_path, tfrec_ext))
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        cycle_length=num_parallel_readers,
        num_parallel_calls=num_parallel_read_calls,
        deterministic=deterministic,
    )
    dataset = dataset.map(_read_tfrecord)
    return dataset.prefetch(prefetch_buffer_size)


def clean_text(features, text_field="text"):
    text = features.pop(text_field)
    text = tf.strings.strip(
        tf.strings.regex_replace(text, r"\n|\t|\r|\r\n|http\S+", "")
    )
    text = tf.strings.regex_replace(text, r"\s+", " ")
    features[text_field] = tf.cond(
        tf.strings.length(text) > 0, lambda: text, lambda: tf.constant("")
    )
    return features


def clip_preproc_train(
    image,
    output_image_shape,
    means=TWITTER_MEANS,
    stdevs=TWITTER_STDEVS,
    drop_frames=True,
):
    _, output_height, output_width = output_image_shape

    if drop_frames:
        image = _drop_frames(image)

    image = tf.image.convert_image_dtype(image, tf.float32)

    resize_side = tf.random.uniform(
        [],
        minval=vgg_preprocessing._RESIZE_SIDE_MIN,
        maxval=vgg_preprocessing._RESIZE_SIDE_MAX + 1,
        dtype=tf.int32,
    )
    image = aspect_preserving_resize(image, resize_side)
    image = vgg_preprocessing._random_crop([image], output_height, output_width)[0]
    image = tf.squeeze((image - means) / stdevs)
    tf.transpose(image, [2, 0, 1])
    image.set_shape(output_image_shape)
    return image


def clip_preproc_val(
    image,
    output_image_shape,
    means=TWITTER_MEANS,
    stdevs=TWITTER_STDEVS,
    drop_frames=True,
):
    if drop_frames:
        image = _drop_frames(image)

    image = tf.image.convert_image_dtype(image, tf.float32)

    _, output_height, output_width = output_image_shape
    smallest_side = tf.math.minimum(output_height, output_width)
    image = aspect_preserving_resize(image, smallest_side)
    image = vgg_preprocessing._central_crop([image], output_height, output_width)[0]
    image = (image - means) / stdevs

    image = tf.squeeze(image)
    image = tf.transpose(image, [2, 0, 1])
    image.set_shape(output_image_shape)
    return image


def get_clip_decode_and_proc_fn(
    encoded_image_field="image_byte_string",
    image_field="image",
    data_augmentation=True,
    encoder_input_resolution=224,
    channels=3,
):
    preprocessed_image_shape = (
        channels,
        encoder_input_resolution,
        encoder_input_resolution,
    )

    clip_preproc_fn = clip_preproc_train if data_augmentation else clip_preproc_val

    def _decode_and_proc_fn(image_byte_string):
        image = tf.image.decode_image(image_byte_string, channels)
        return clip_preproc_fn(image, preprocessed_image_shape)

    def _map_fn(features):
        image_byte_string = features.pop(encoded_image_field)
        image_features = tf.cond(
            tf.strings.length(image_byte_string) > 0,
            lambda: _decode_and_proc_fn(image_byte_string),
            lambda: tf.constant([], tf.float32),
        )
        features[image_field] = image_features
        return features

    return _map_fn


def blobstore_fetch(urls, cache_dir):
    n_urls = len(urls)
    local_paths = urls.map(
        lambda x: os.path.join(
            cache_dir,
            re.sub(r"https?://example.invalid/", "", x).replace(
                os.path.basename(x), ""
            ),
            os.path.basename(x),
        )
    )
    cache = dict(zip(urls, local_paths))
    start = time.time()
    _ = parallel_file_access.copy_multiple_files(urls, local_paths)
    elapsed = time.time() - start
    calls_per_second = float(n_urls) / elapsed
    print(
        f"Processed {n_urls} files in {elapsed:.02f} seconds. "
        f"({calls_per_second:.02f} imgs/sec)"
    )
    return cache


def init_blobstore_client(pool_size=1000, retries=0):
    thread_pool_size = pool_size
    connection_pool_max_size = pool_size
    pool_connections = pool_size
    parallel_file_access.init_default_thread_pool(thread_pool_size)

    file_access.blobstore.init_client(
        retries=retries,
        pool_maxsize=connection_pool_max_size,
        pool_connections=pool_connections,
    )


def _bytestring_feature(list_of_bytestrings):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=list_of_bytestrings))


def _float_feature(list_of_floats):
    return tf.train.Feature(float_list=tf.train.FloatList(value=list_of_floats))


def _int64_feature(list_of_ints):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=list_of_ints))


def write_dataset_to_gcs_tfrecords(
    dataset,
    dataset_path,
    gs_bucket,
    prefix="data",
    samples_per_record=1000,
    nsamples=None,
):
    if not nsamples:
        nsamples = len(list(dataset))

    dtype_map = {
        tf.string: _bytestring_feature,
        tf.int64: _int64_feature,
        tf.float32: _float_feature,
    }

    tfrecords = []
    for i, features in tqdm.tqdm(enumerate(dataset), unit="samples"):
        feature = {}
        for feature_name, feature_tensor in features.items():
            value = feature_tensor.numpy()
            if isinstance(value, np.ndarray):
                value = list(value)
            elif not isinstance(value, list):
                value = [value]
            feature[feature_name] = dtype_map[feature_tensor.dtype](value)
        tfrecords.append(tf.train.Example(features=tf.train.Features(feature=feature)))

        if (i + 1) % samples_per_record == 0 or (i + 1) == nsamples:
            shard = i // samples_per_record
            filename = os.path.join(
                f"gs://{gs_bucket}",
                dataset_path,
                f"{prefix}_shard{(shard + 1):05d}.tfrecord",
            )
            with tf.io.TFRecordWriter(filename) as outfile:
                print(
                    f"writing records {(shard) * samples_per_record + 1} "
                    f"to {(shard) * samples_per_record + len(tfrecords)} "
                    f"to {filename}"
                )
                for record in tfrecords:
                    outfile.write(record.SerializeToString())
            tfrecords = []


def aspect_preserving_resize(image, new_smallest_size):
    new_smallest_size = tf.cast(new_smallest_size, tf.float64)

    shape = tf.shape(image)

    height = tf.cast(shape[0], tf.float64)
    width = tf.cast(shape[1], tf.float64)

    scale_factor = tf.cond(
        tf.greater(height, width),
        lambda: new_smallest_size / width,
        lambda: new_smallest_size / height,
    )

    new_height = tf.math.round(height * scale_factor)
    new_width = tf.math.round(width * scale_factor)

    new_height = tf.math.maximum(new_height, new_smallest_size)
    new_width = tf.math.maximum(new_width, new_smallest_size)

    new_height = tf.cast(new_height, tf.int32)
    new_width = tf.cast(new_width, tf.int32)

    image = tf.expand_dims(image, 0)
    resized_image = tf.compat.v1.image.resize_area(
        image, [new_height, new_width], align_corners=False
    )
    resized_image = tf.squeeze(resized_image)
    resized_image.set_shape([None, None, 3])

    return resized_image


def _drop_frames(x):
    def f1():
        return x[0, :, :, :]

    def f2():
        return x

    if tf.executing_eagerly():
        if tf.rank(x).numpy() == 4:
            y = x[0, :, :, :]
        else:
            y = x

    else:
        y = tf.cond(tf.equal(tf.rank(x), tf.constant(4)), f1, f2)

    return y
