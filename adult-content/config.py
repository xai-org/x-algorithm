import os
from datetime import datetime


class TrainingConfig:
    INPUT_EMBEDDING_DIM = 256

    def __init__(
        self,
        train_dataset_path,
        eval_dataset_path,
        resample=False,
        repeat=True,
        input_embedding_dim=None,
        model_dir=None,
        dense_layer_units=128,
        batch_size=256,
        learning_rate=0.001,
        num_epochs=100,
        num_steps_per_epoch=400,
    ):
        if model_dir is None:
            model_dir = os.path.join("runs", datetime.now().strftime("%Y%m%d_%H%M%S"))
        if input_embedding_dim is None:
            input_embedding_dim = self.INPUT_EMBEDDING_DIM
        for argname, value in locals().items():
            if argname != "self":
                setattr(self, argname, value)


class CalibrationConfig:
    def __init__(
        self,
        dataset_path,
        output_dir=None,
        num_thresholds=200,
    ):
        if output_dir is None:
            output_dir = os.path.join(
                "calibration", datetime.now().strftime("%Y%m%d_%H%M%S")
            )
        for argname, value in locals().items():
            if argname != "self":
                setattr(self, argname, value)
