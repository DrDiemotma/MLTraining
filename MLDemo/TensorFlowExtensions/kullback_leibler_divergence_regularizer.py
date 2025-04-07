
import tensorflow as tf
import keras


class KLDivergenceRegularizer(keras.regularizers.Regularizer):
    def __init__(self, weight, target):
        self._weight = weight
        self._target = target

        self.kl_divergence = keras.losses.kl_divergence

    def call(self, inputs):
        mean_activities = tf.reduce_mean(inputs, axis=0)
        kl_sum = self.kl_divergence(self._target, mean_activities) \
                 + self.kl_divergence(1 - self._target, 1 - mean_activities)
        return self._weight * kl_sum
