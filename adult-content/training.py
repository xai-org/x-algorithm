import logging

import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from twitter.adult_content.dataset_utils import build_dataset

logger = logging.getLogger(__name__)


def build_model(embedding_length, dense_layer_units, learning_rate):
    model = Sequential()

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07,
        amsgrad=False,
        name="Adam",
    )

    model.add(tf.keras.layers.BatchNormalization())
    model.add(Dense(units=dense_layer_units, activation="gelu"))
    model.add(Dense(units=1, activation="sigmoid"))
    model.build(input_shape=(None, embedding_length))

    metrics = [
        tf.keras.metrics.PrecisionAtRecall(
            recall=0.9, num_thresholds=200, class_id=None, name=None, dtype=None
        ),
        tf.keras.metrics.AUC(
            num_thresholds=200,
            curve="PR",
        ),
    ]

    model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=metrics)

    return model


def train_model(config):
    model = build_model(
        embedding_length=config.input_embedding_dim,
        dense_layer_units=config.dense_layer_units,
        learning_rate=config.learning_rate,
    )

    train_ds = build_dataset(
        dataset_path=config.train_dataset_path,
        embedding_dim=config.input_embedding_dim,
        batch_size=config.batch_size,
        do_resample=config.resample,
        do_repeat=config.repeat,
    )

    eval_ds = build_dataset(
        dataset_path=config.eval_dataset_path,
        embedding_dim=config.input_embedding_dim,
        batch_size=config.batch_size,
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=0,
            patience=5,
            verbose=0,
            mode="auto",
            baseline=None,
            restore_best_weights=True,
        )
    ]

    model.fit(
        train_ds,
        validation_data=eval_ds,
        epochs=config.num_epochs,
        steps_per_epoch=config.num_steps_per_epoch,
        callbacks=callbacks,
    )
    tf.keras.models.save_model(model, config.model_dir)
