#!/usr/bin/python3

import numpy as np;
import cv2;
import tensorflow as tf;
from MNISTModel import MNISTModel;

def main():
	tf.enable_eager_execution();
	#NOTE: change the img yourself to test through images intuitively
	img = cv2.imread('imgs/1.bmp',cv2.IMREAD_GRAYSCALE);
	assert img is not None;
	img = img[np.newaxis,...,np.newaxis].astype(np.float32);
	model = MNISTModel();
	model.load_weights('./model/mnist_model');
	output = tf.nn.softmax(model.predict(img,batch_size = 1)).numpy();
	print(np.argmax(output));

if __name__ == "__main__":
	main();
