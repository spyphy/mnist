#!/usr/bin/env python
# -*- coding: utf-8 -*-
""" A one-hidden-layer-MLP MNIST-classifier. """
from __future__ import absolute_import,  division, print_function
# Import the training data (MNIST)
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
# Possibly download and extract the MNIST data set.
# Retrieve the labels as one-hot-encoded vectors.
mnist = input_data.read_data_sets("/tmp/mnist", one_hot=True)
# Create a new graph
graph = tf.Graph()
# Set our graph as the one to add nodes to

BATCH_SIZE = 10
NUM_STEPS = 1000

with graph.as_default():

	# 1. Construct a graph representing the model.
	x = tf.placeholder(tf.float32, [BATCH_SIZE, 784]) # Placeholder for input.
	y = tf.placeholder(tf.float32, [BATCH_SIZE, 10])  # Placeholder for labels.
	
	W_1 = tf.Variable(tf.random_uniform([784, 100])) # 784x100 weight matrix.
	b_1 = tf.Variable(tf.zeros([100]))				# 100-element bias vector.
	layer_1 = tf.nn.relu(tf.matmul(x, W_1) + b_1)   # Output of hidden layer.
	
	W_2 = tf.Variable(tf.random_uniform([100, 10]))	# 100x10 weight matrix.
	b_2 = tf.Variable(tf.zeros([10])) 				# 10-element bias vector.
	layer_2 = tf.matmul(layer_1, W_2) + b_2 		# Output of linear layer.

	output = layer_2

	# 2. Add nodes that represent the optimization algorithm.
	#loss = tf.nn.softmax_cross_entropy_with_logits_v2(layer_2, y)
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits = output, labels = y)
	train_op = tf.train.AdagradOptimizer(0.01).minimize(loss)

	correct_prediction = tf.equal(tf.argmax(output,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	# 3. Execute the graph on batches of input data.
	with tf.Session() as sess:						# Connect to the TF runtime.
		init = tf.global_variables_initializer()
		sess.run(init)		# Randomly initialize weights.
		for step in range(NUM_STEPS):				# Train iteratively for NUM_STEPS.			
			x_data, y_data = mnist.train.next_batch(BATCH_SIZE) # Load one batch of input data
			sess.run(train_op, {x: x_data, y: y_data}) 	 # Perform one training step.
			if step % 100 == 0:
				print('step {0}: b_2={1}'.format(step, sess.run(b_2)))
				print(accuracy.eval(feed_dict = {x : mnist.test.images[0:10], y : mnist.test.labels[0:10]}))

"""
Possible errors:

ValueError: Only call `softmax_cross_entropy_with_logits` with named arguments
- 

Use `tf.global_variables_initializer` instead.

"""