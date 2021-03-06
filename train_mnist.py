#!/usr/bin/python3
# tensorflow 2.0 compatible
import os;
import numpy as np;
import tensorflow as tf;
from MNISTModel import MNISTModel;

batch_size = 100;

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
    #create model and optimizer
    model = MNISTModel();
    optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(1e-3, decay_steps = 60000, decay_rate = 0.5));
    #load dataset
    trainset = iter(tf.data.TFRecordDataset(os.path.join('dataset','trainset.tfrecord')).repeat(-1).map(parse_function).shuffle(batch_size).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE));
    #restore from existing checkpoint
    if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
    checkpoint = tf.train.Checkpoint(model = model, optimizer = optimizer);
    checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
    #create log
    log = tf.summary.create_file_writer('checkpoints');
    #train model
    print("training");
    avg_loss = tf.keras.metrics.Mean(name = 'loss', dtype = tf.float32);
    while True:
        (images,labels) = next(trainset);
        with tf.GradientTape() as tape:
            onehot_labels = tf.one_hot(labels,10);
            logits = model(images);
            loss = tf.math.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(onehot_labels,logits));
        avg_loss.update_state(loss);
        #write log
        if tf.equal(optimizer.iterations % 100, 0):
            with log.as_default():
                tf.summary.scalar('loss',avg_loss.result(), step = optimizer.iterations);
            print('Step #%d Loss: %.6f lr: %.6f' % (optimizer.iterations,avg_loss.result(), optimizer._hyper['learning_rate'](optimizer.iterations)));
            if avg_loss.result() < 0.01: break;
            avg_loss.reset_states();
        grads = tape.gradient(loss,model.trainable_variables);
        optimizer.apply_gradients(zip(grads,model.trainable_variables));
        #save model
        if tf.equal(optimizer.iterations % 100, 0):
            checkpoint.save(os.path.join('checkpoints','ckpt'));
    #save the network structure with weights
    if False == os.path.exists('model'): os.mkdir('model');
    model.save('./model/mnist_model.h5');

if __name__ == "__main__":
    
    assert tf.executing_eagerly();
    main();
