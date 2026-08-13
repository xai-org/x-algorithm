import os
from datetime import datetime


class ClipConfig:
    DEFAULT_FINAL_EMBEDDING_DIM = 256

    def __init__(
        self,
        dataset_path,
        input_resolution=224,
        use_data_augmentation=False,
        clip_model_type="ViT-B/32",
        top_feedforward=True,
        final_embedding_dim=None,
        logit_scale=4.5065,
        model_dir=None,
        save_interval=5000,
        save_last_ckpt_only=True,
        batch_size=128,
        learning_rate=2e-5,
        num_training_steps=100000,
        num_warmup_steps=500,
        num_decay_steps=None,
        mixed_precision=False,
    ):
        if model_dir is None:
            model_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        if num_decay_steps is None:
            num_decay_steps = num_training_steps
        if top_feedforward and final_embedding_dim is None:
            final_embedding_dim = self.DEFAULT_FINAL_EMBEDDING_DIM
        for argname, value in locals().items():
            if argname != "self":
                setattr(self, argname, value)
