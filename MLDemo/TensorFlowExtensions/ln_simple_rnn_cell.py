
import keras

class LNSimpleRNNCell(keras.layers.Layer):
    def __init__(self, units: int, activation: str = "tanh", **kwargs):
        super().__init__(**kwargs)
        self.state_size = units  # public for the RNN which embeds this
        self.output_size = units  # public for the RNN which embeds this
        self._simple_rnn_cell: keras.layers.Layer = keras.layers.SimpleRNNCell(units, activation=None)
        self._layer_normalization: keras.layers.Layer = keras.layers.LayerNormalization()
        self._activation: keras.layers.Layer = keras.activations.get(activation)

    def call(self, inputs, states):
        outputs, _ = self._simple_rnn_cell(inputs, states)  # second output is just [outputs] because the cell has no activation
        norm_outputs = self._layer_normalization(outputs)  # normalize BEFORE activation
        active_norm_outputs = self._activation(norm_outputs)
        return active_norm_outputs, [active_norm_outputs]  # like the cell, output the activated outputs twice, once in a list as hidden states
