#!/usr/bin/python3

import os;
import tensorflow as tf;

def main():
    tf.keras.backend.set_learning_phase(0); # predict mode
    model = tf.keras.models.load_model('./model/mnist_model.h5'); # load structure with weights
    if False == os.path.exists('serving_model'): os.mkdir('serving_model');
    tf.saved_model.save(model,'./serving_model');

if __name__ == "__main__":

    assert tf.executing_eagerly();
    main();

