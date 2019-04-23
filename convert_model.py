#!/usr/bin/python3

import os;
import tensorflow as tf;

def main():
    tf.keras.backend.set_learning_phase(0); # predict mode
    model = tf.keras.models.load_model('./model/mnist_model.h5'); # load structure with weights
    tf.saved_model.save(model,'./mnist/1/');
    loaded = tf.saved_model.load('./mnist/1/');
    infer = loaded.signatures['serving_default'];
    print('====================NOTE==================');
    print('output tensor name is',infer.structured_outputs);

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();

