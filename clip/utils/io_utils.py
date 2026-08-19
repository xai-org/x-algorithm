import os
from urllib.parse import urlparse

from google.cloud import storage
from twitter.ml.common import hadoop


def maybe_download_file(url, local_dir=None, hdfs_config=None):
    if local_dir is None:
        local_dir = os.path.expanduser("~/.cache/clip")

    filename = os.path.basename(url)
    local_path = os.path.join(local_dir, filename)

    if os.path.exists(local_path):
        return local_path

    os.makedirs(local_dir, exist_ok=True)

    parsed_url = urlparse(url)

    print(f"Downloading model {url} -> {local_path}")

    if parsed_url.scheme in ("hdfs", "viewfs"):
        _download_file_from_hadoop(url, local_path, hdfs_config=hdfs_config)
    elif parsed_url.scheme == "gs":
        _download_file_from_gcs(url, local_path)
    else:
        raise ValueError(f"Unsupported scheme: {parsed_url.scheme}")

    return local_path


def _download_file_from_hadoop(url, local_path, hdfs_config=None):
    if hdfs_config is None:
        hdfs_config = hadoop.HdfsConfig(cluster="procatla")
    hdfs_util = hadoop.HdfsUtil(hdfs_config)

    hdfs_util.copy_to_local(url, local_path)


def _download_file_from_gcs(url, local_path):
    parsed_url = urlparse(url)

    bucket_name = parsed_url.netloc
    source_blob_name = parsed_url.path

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(local_path)
