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

        model = tf.keras.models.load_model('./model/mnist_model.h5');
        import sys;
        import cv2;
        assert len(sys.argv) == 2;
        img_path = sys.argv[1];
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE);
        assert img is not None;
        img = img[np.newaxis,...,np.newaxis].astype(np.float32);
        output = tf.nn.softmax(model.predict(img, batch_size = 1)).numpy();
        print(np.argmax(output));

