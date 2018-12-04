#!/usr/bin/python
# -*- coding: utf-8 -*-
import os;
import numpy as np;
import tensorflow as tf;
import tensorflow.contrib.eager as tfe;
from MNISTModel import MNISTModel;

def parse_function(serialized_example):
	feature = tf.parse_single_example(
		serialized_example,
		features = {
			'data':tf.FixedLenFeature((),dtype = tf.string,default_value = ''),
			'label':tf.FixedLenFeature((),dtype = tf.int64,default_value = 0)
		}
	);
	data = tf.decode_raw(feature['data'],out_type = tf.uint8);
	data = tf.reshape(data,[28,28,1]);
	data = tf.cast(data,dtype = tf.float32);
	label = tf.cast(feature['label'],dtype = tf.int32);
	return data,label;

def main(unused_argv):
	tf.enable_eager_execution();
	#create model and optimizer
	model = MNISTModel();
	optimizer = tf.train.AdamOptimizer(1e-3);
	#load dataset
	trainset = tf.data.TFRecordDataset(os.path.join('dataset','trainset.tfrecord')).map(parse_function).shuffle(100).batch(100);
	testset = tf.data.TFRecordDataset(os.path.join('dataset','testset.tfrecord')).map(parse_function).batch(100);
	#restore from existing checkpoint
	if False == os.path.exists('checkpoints'): os.mkdir('checkpoints');
	checkpoint = tf.train.Checkpoint(model = model,optimizer = optimizer, optimizer_step = tf.train.get_or_create_global_step());
	checkpoint.restore(tf.train.latest_checkpoint('checkpoints'));
	#create log
	log = tf.contrib.summary.create_file_writer('checkpoints');
	log.set_as_default();
	#train model
	print("training");
	while True:
		for (batch,(images,labels)) in enumerate(trainset):
			with tf.GradientTape() as tape:
				onehot_labels = tf.one_hot(labels,10);
				logits = model(images);
				loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels,logits));
			#write log
			with tf.contrib.summary.record_summaries_every_n_global_steps(2,global_step = tf.train.get_global_step()):
				tf.contrib.summary.scalar('loss',loss);
			grads = tape.gradient(loss,model.variables);
			optimizer.apply_gradients(zip(grads,model.variables),global_step = tf.train.get_global_step());
			if batch % 100 == 0: print('Step #%d\tLoss: %.6f' % (batch,loss));
		#save model once every epoch
		checkpoint.save(os.path.join('checkpoints','ckpt'));
		if loss < 0.01: break;
	#eager execution mode has no graph, therefore we can only save model weights but the whole model
	if False == os.path.exists('model'): os.mkdir('model');
	model.save_weights('./model/mnist_model');
	#test model
	print("testing");
	accuracy = tfe.metrics.Accuracy('accuracy',dtype = tf.float32);
	for (batch,(images,labels)) in enumerate(testset):
		onehot_labels = tf.one_hot(labels,10);
		logits = model(images);
		loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels,logits));
		#NOTE:tf.metrics.accuracy is not available in eager execution mode
		#accuracy = tf.metrics.accuracy(labels = labels,predictions = tf.argmax(logits,axis = 1));
		accuracy(tf.argmax(logits,axis = 1,output_type = tf.int64),tf.cast(labels,tf.int64));
		print('Accuracy: %.6f' % (accuracy.result()));

if __name__ == "__main__":
	tf.app.run();
