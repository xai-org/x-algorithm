import os
import shutil
import socket
import subprocess
import tempfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""

from twitter.clip import keras_model
from twitter.clip.model import TwitterCLIP


def _rsync_to_packer(role, package_name, file_path):
    nest = "example.invalid"
    upload_dir = ".clip_keras"
    remote_path = os.path.join(upload_dir, os.path.basename(file_path))
    subprocess.check_call(
        ["rsync", "-aPze", "ssh", file_path, f"{nest}:{upload_dir}/"]
    )

    for dc in ["atla", "pdxa"]:
        subprocess.check_call(
            [
                "ssh",
                nest,
                f"packer add_version --cluster={dc} {role} {package_name} {remote_path}",
            ]
        )


def _check_packer():
    hostname = socket.gethostname()

    if not ("example.invalid" in hostname or "tw-mbp" in hostname):
        raise ValueError("Packer add is only supported from on prem or laptop.")


def export_keras_models(model_names):
    _check_packer()

    for model_name in model_names:
        print(model_name)
        if model_name in TwitterCLIP.TWITTER_MODELS:
            top_feedforward = True
            final_embedding_dim = int(model_name.split("-")[-1])
        else:
            top_feedforward = False
            final_embedding_dim = None

        model = TwitterCLIP(
            clip_model_type=model_name,
            top_feedforward=top_feedforward,
            final_embedding_dim=final_embedding_dim,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            keras_model_dir = os.path.join(tmp_dir, "keras_model")
            model.export_to_keras_saved_model(keras_model_dir)

            package_name = keras_model.get_packer_package_name(model_name)
            zip_path = os.path.join(tmp_dir, package_name)
            zip_target = keras_model_dir
            shutil.make_archive(zip_path, "zip", zip_target)

            _rsync_to_packer("embeddings-category", package_name, zip_path + ".zip")

            print("Done")


if __name__ == "__main__":
    export_keras_models(keras_model.MODELS)
