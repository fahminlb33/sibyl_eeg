import datetime

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Any, List, Tuple

# this is required for deterministic result
tf.random.set_seed(42)
np.random.seed(42)

def build_model(kind: str, units: List[int], input_shape: Any, num_classes: int, normalize: bool=False, dropout: float=None) -> tf.keras.models.Sequential:
    model = tf.keras.models.Sequential()

    if kind == "dense":
        for i, unit in enumerate(units):
            # add top layer with input shape or subsequent layer
            if i == 0:
                model.add(tf.keras.layers.Dense(unit, input_shape=input_shape))
            else:
                model.add(tf.keras.layers.Dense(unit))

            # add normalization
            if normalize:
                model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Activation("relu"))

            # add regularization
            if dropout is not None:
                model.add(tf.keras.layers.Dropout(dropout))

    elif kind == "cnn":
        for i, unit in enumerate(units):
            # add top layer with input shape or subsequent layer
            if i == 0:
                model.add(tf.keras.layers.Conv1D(unit, input_shape=input_shape, padding="same", kernel_size=2))
            else:
                model.add(tf.keras.layers.Conv1D(unit, padding="same", kernel_size=2))

            # add normalization
            if normalize:
                model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Activation("relu"))
            model.add(tf.keras.layers.MaxPooling1D(pool_size=2))

            # add regularization
            if dropout is not None:
                model.add(tf.keras.layers.Dropout(dropout))

        model.add(tf.keras.layers.Flatten())

    elif kind == "lstm":
        for i, unit in enumerate(units):
            return_sequences = not len(units) == i

            # add top layer with input shape or subsequent layer
            if unit == 0:
                model.add(tf.keras.layers.LSTM(unit, input_shape=input_shape, return_sequences=True))
            else:
                model.add(tf.keras.layers.LSTM(unit, return_sequences=return_sequences))

            # add normalization
            if normalize:
                model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Activation("relu"))

            # add regularization
            if dropout is not None:
                model.add(tf.keras.layers.Dropout(dropout))

    # add output layer
    model.add(tf.keras.layers.Dense(num_classes, activation="sigmoid"))

    return model

def finalize_model(model: tf.keras.models.Sequential) -> tf.keras.models.Sequential:
    optimizer = tf.keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])

def train_model(model: tf.keras.models.Sequential, X: any, y: any, epochs: int, validation_split: float, callbacks: List[tf.keras.callbacks.Callback]=None):
    model.fit(X, y, epochs=epochs, validation_split=validation_split, callbacks=callbacks)

def reshape_data(df: pd.DataFrame) -> Tuple[Any, Any]:
    X = df.iloc[:, 5:69].values.reshape((-1, 256, 64))
    y = df.iloc[:, 4].values.reshape((-1, 256))[:, 0].reshape((-1, 1))

    return (X, y)

def create_tensorboard(prefix: str, logdir: str) -> tf.keras.callbacks.TensorBoard:
    log_dir = "logs/fit/" + prefix + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
