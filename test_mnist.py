#!/usr/bin/python3

import os;
import numpy as np;
import tensorflow as tf;
from MNISTModel import MNISTModel;

def parse_function(serialized_example):
        feature = tf.io.parse_single_example(
                serialized_example,
                features = {
                        'data':tf.io.FixedLenFeature((),dtype = tf.string,default_value = ''),
                        'label':tf.io.FixedLenFeature((),dtype = tf.int64,default_value = 0)
                }
        );
        data = tf.io.decode_raw(feature['data'],out_type = tf.uint8);
        data = tf.reshape(data,[28,28,1]);
        data = tf.cast(data,dtype = tf.float32);
        label = tf.cast(feature['label'],dtype = tf.int32);
        return data,label;

def main():

        # create model and load trained model
        model = MNISTModel();
        model.load_weights('./model/mnist_model');
        # load dataset
        testset = tf.data.TFRecordDataset(os.path.join('dataset','testset.tfrecord')).map(parse_function).batch(100);
        # test accuracy
        accuracy = tf.keras.metrics.Accuracy(name = 'accuracy', dtype = tf.float32);
        for (images, labels) in testset:
                onehot_labels = tf.one_hot(labels,10);
                logits = model(images);
                accuracy.update_state(tf.argmax(logits, axis = -1, output_type = tf.int64), tf.cast(labels, tf.int64));
	# print accuracy
        print('Accuracy: %.6f' % (accuracy.result()));

if __name__ == "__main__":

        assert tf.executing_eagerly();
        main();
