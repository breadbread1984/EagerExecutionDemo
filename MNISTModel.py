#!/usr/bin/python3

import tensorflow as tf;

class MNISTModel(tf.keras.Model):
	def __init__(self):
		super(MNISTModel,self).__init__();
		self.conv1 = tf.keras.layers.Conv2D(20, kernel_size = [5,5], padding = 'valid');
		self.maxpool1 = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid');
		self.conv2 = tf.keras.layers.Conv2D(50, kernel_size = [5,5], padding = 'valid');
		self.maxpool2 = tf.keras.layers.MaxPool2D(pool_size = [2,2], strides = [2,2], padding = 'valid');
		self.flatten = tf.keras.layers.Flatten();
		self.dense1 = tf.keras.layers.Dense(units = 500);
		self.relu1 = tf.keras.layers.ReLU();
		self.dense2 = tf.keras.layers.Dense(units = 10);

	def call(self,input):
		result = self.conv1(input);
		result = self.maxpool1(result);
		result = self.conv2(result);
		result = self.maxpool2(result);
		result = self.flatten(result);
		result = self.dense1(result);
		result = self.relu1(result);
		result = self.dense2(result);
		return result;

