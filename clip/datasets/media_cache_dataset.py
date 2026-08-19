import copy
import importlib
import os
from functools import partial

import pkg_resources
import tensorflow as tf
from google.cloud import bigquery
from twitter.clip.utils.data_utils import blobstore_fetch
from twitter.magicpony.common import file_access


class MediaCacheBqDataset:
    def __init__(
        self,
        project_id,
        dataset_id,
        table_id,
        element_spec,
        media_field,
        download_image_size="small",
        http_urls=False,
        bq_compute_project="example-media-project",
    ):
        self.project_id = project_id
        self.dataset_id = dataset_id
        self.table_id = table_id
        self.element_spec = copy.deepcopy(element_spec)
        self.media_field = media_field
        self.download_image_size = download_image_size
        self.http_urls = http_urls
        self.bq_compute_project = bq_compute_project

    def _read_from_bq(self, row_restriction=None):
        importlib.reload(pkg_resources)
        pkg_resources.get_distribution("google-api-core")

        if row_restriction is None:
            row_restriction = ""
        print(f"Reading BQ examples {row_restriction}...")
        client = bigquery.Client(project=self.bq_compute_project)
        job = client.query(
            f"""
          SELECT
            {",".join(list(self.element_spec.keys()))}
          FROM
            `{self.project_id}.{self.dataset_id}.{self.table_id}`
          {row_restriction}
        """.replace("\n", " ")
        )
        return job.result().to_dataframe()

    def create_dataset(
        self,
        cache_dir=".cached_images",
        row_restriction=None,
        read_images=True,
        with_media_only=False,
        postprocess_fn=None,
    ):
        df = self._read_from_bq(row_restriction=row_restriction)
        urls = df.pop(self.media_field)
        urls = urls[urls.notna()].map(
            lambda url: file_access.blobstore.BlobstoreResourceLocator.from_url(url)
            .to_prod_internal()
            .url
        )
        if self.http_urls:
            urls = urls.map(lambda url: url.replace("https:", "http:"))
        _ = self.element_spec.pop(self.media_field)
        if self.download_image_size:
            urls = urls.map(lambda url: f"{url}:{self.download_image_size}")

        print(f"Downloading images to {cache_dir}...")
        cache = blobstore_fetch(urls, cache_dir)
        df["local_media_path"] = urls.map(
            lambda url: cache[url] if os.path.isfile(cache[url]) else ""
        )
        df = df.fillna("")
        self.element_spec.update(
            {"local_media_path": tf.TensorSpec((), dtype=tf.string)}
        )

        def _generator_fn():
            for row_idx, row in df.iterrows():
                yield dict(row)

        dataset = tf.data.Dataset.from_generator(
            _generator_fn, output_signature=self.element_spec
        )

        def _read(features):
            local_media_path = features.pop("local_media_path")
            features["image_byte_string"] = tf.cond(
                tf.strings.length(local_media_path) > 0,
                lambda: tf.io.read_file(local_media_path),
                lambda: "",
            )
            return features

        def _filter_fn(features, image_field):
            return tf.greater(tf.strings.length(features[image_field]), 0)

        if read_images:
            dataset = dataset.map(
                _read, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        if with_media_only:
            image_field = "image_byte_string" if read_images else "local_media_path"
            dataset = dataset.filter(partial(_filter_fn, image_field=image_field))

        if postprocess_fn:
            dataset = dataset.map(
                postprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE
            )
        return dataset
