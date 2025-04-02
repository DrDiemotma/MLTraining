import tensorflow as tf

def huber_fn(y_true: tf.Tensor, y_pred: tf.Tensor):
    error: tf.Tensor = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss: tf.Tensor = tf.square(error) / 2
    linear_loss: tf.Tensor = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

