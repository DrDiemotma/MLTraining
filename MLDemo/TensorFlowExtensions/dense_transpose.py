
import tensorflow as tf
import numpy as np
import keras

class DenseTranspose(keras.layers.Layer):
    def __init__(self, dense: keras.layers.Layer, activation=None, **kwargs):
        super().__init__(**kwargs)

        self._dense: keras.layers.Layer = dense
        self._activation = keras.activations.get(activation)
        self._biases = None

    def build(self, batch_input_shape):
        self._biases = self.add_weight(name="bias", shape=self._dense.weights[-1].shape, initializer="zeros")
        super().build(batch_input_shape)

    def call(self, inputs):
        dense_weights = self._dense.weights[0]
        z = tf.matmul(inputs, dense_weights, transpose_b=True)
        return self._activation(z + self._biases)


if __name__ == '__main__':
    fashion_mnist = keras.datasets.fashion_mnist.load_data()
    (X_train_full, y_train_full), (X_test, y_test) = fashion_mnist
    X_train_full = X_train_full.astype(np.float32) / 255
    X_test = X_test.astype(np.float32) / 255
    X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
    y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

    dense_1 = keras.layers.Dense(100, activation="relu")
    dense_2 = keras.layers.Dense(30, "relu")

    tied_encoder = keras.Sequential([keras.layers.Flatten(), dense_1, dense_2])
    tied_decoder = keras.Sequential([
        DenseTranspose(dense_2, activation="relu"),
        DenseTranspose(dense_1),
        keras.layers.Reshape([28, 28])
    ])

    tied_ae = keras.Sequential([tied_encoder, tied_decoder])
    tied_ae.compile(loss=keras.losses.MeanSquaredError, optimizer="nadam")
    tied_ae.fit(X_train, X_train, epochs=5, validation_data=(X_valid, X_valid))
