from typing import Iterable

import tensorflow as tf
import keras

def train_gan(gan: keras.Model, dataset: Iterable, batch_size: int, codings_size: int, n_epochs: int):
    generator, discriminator = gan.layers

    def train_discriminator_on_batch(current_batch):
        noise = tf.random.normal(shape=[batch_size, codings_size])
        generated_images = generator(noise)
        x_fake_and_real = tf.concat([generated_images, current_batch], axis=0)
        y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
        discriminator.train_on_batch(x_fake_and_real, y1)

    def train_generator():
        noise = tf.random.normal(shape=[batch_size, codings_size])
        y2 = tf.constant([[1.]] * batch_size)
        gan.train_on_batch(noise, y2)

    for epoch in range(n_epochs):
        for x_batch in dataset:
            train_discriminator_on_batch(x_batch)
            train_generator()
