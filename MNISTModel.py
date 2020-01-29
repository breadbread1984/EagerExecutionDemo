#!/usr/bin/python3

import tensorflow as tf;

def MNISTModel():

  inputs = tf.keras.Input(shape = (28,28,1));
  results = tf.keras.layers.Conv2D(20, kernel_size = [5,5], padding = 'valid')(inputs);
  results = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid')(results);
  results = tf.keras.layers.Conv2D(50, kernel_size = [5,5], padding = 'valid')(results);
  results = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid')(results);
  results = tf.keras.layers.Flatten()(results);
  results = tf.keras.layers.Dense(units = 500)(results);
  results = tf.keras.layers.ReLU()(results);
  results = tf.keras.layers.Dense(units = 10)(results);
  return tf.keras.Model(inputs = inputs, outputs = results);

if __name__ == "__main__":

  assert tf.executing_eagerly();
  # print network structure
  lenet = MNISTModel();
  tf.keras.utils.plot_model(lenet, show_shapes = True, dpi = 64);
