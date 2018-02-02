#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np
import math

import inits
import graph
import utils
from aggregators import WeightedAggregator, MeanAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

class Model(object):
	def __init__(self, **kwargs):
		allowed_kwargs = {'name', 'logging'}
		for kwarg in kwargs.keys():
			assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
		name = kwargs.get('name')
		if not name:
			name = self.__class__.__name__.lower()
		self.name = name

		logging = kwargs.get('logging', False)
		self.logging = logging

		self.vars = {}
		self.placeholders = {}

		self.layers = []
		self.activations = []

		self.inputs = None
		self.outputs = None

		self.loss = 0
		self.accuracy = 0
		self.optimizer = None
		self.opt_op = None

	def _build(self):
		raise NotImplementedError

	def build(self):
		""" Wrapper for _build() """
		with tf.variable_scope(self.name):
			self._build()

		# Build sequential layer model
		self.activations.append(self.inputs)
		for layer in self.layers:
			hidden = layer(self.activations[-1])
			self.activations.append(hidden)
		self.outputs = self.activations[-1]

		# Store model variables for easy access
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

		# Build metrics
		self._loss()
		self._accuracy()

		self.opt_op = self.optimizer.minimize(self.loss)

	def predict(self):
		pass

	def _loss(self):
		raise NotImplementedError

	def _accuracy(self):
		raise NotImplementedError

	def save(self, sess=None):
		if not sess:
			raise AttributeError("TensorFlow session not provided.")
		saver = tf.train.Saver(self.vars)
		save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
		print("Model saved in file: %s" % save_path)

	def load(self, sess=None):
		if not sess:
			raise AttributeError("TensorFlow session not provided.")
		saver = tf.train.Saver(self.vars)
		save_path = "tmp/%s.ckpt" % self.name
		saver.restore(sess, save_path)
		print("Model restored from file: %s" % save_path)

class GeneralizedModel(Model):
	"""
	Base class for models that aren't constructed from traditional, sequential layers.
	Subclasses must set self.outputs in _build method

	(Removes the layers idiom from build method of the Model class)
	"""

	def __init__(self, **kwargs):
		super(GeneralizedModel, self).__init__(**kwargs)
		

	def build(self):
		""" Wrapper for _build() """
		with tf.variable_scope(self.name):
			self._build()

		# Store model variables for easy access
		variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
		self.vars = {var.name: var for var in variables}

		# Build metrics
		self._loss()
		self._accuracy()

		self.opt_op = self.optimizer.minimize(self.loss)

class PretrainModel(GeneralizedModel):
	def __init__(self, placeholders, features, pretrained, dict_size, degree_permuted, rpr_matrix,
					rpr_arg, dropout = 0., nodevec_dim = 200, lr = 0.001, only_f = False, **kwargs):
		"""
		W_mlp is pretrained by node2vec
		"""

		super(PretrainModel, self).__init__(**kwargs)

		self.placeholders = placeholders
		self.degrees = degree_permuted
		self.only_f = only_f
		self.rpr_arg = tf.Variable(tf.constant(rpr_arg, dtype = tf.int64), trainable = False)
		self.rpr_matrix = tf.Variable(tf.constant(rpr_matrix, dtype = tf.float32), trainable = False)

		self.dropout = dropout
		self.feature_dim = features.shape[1]
		self.features = tf.Variable(tf.constant(features, dtype = tf.float32), trainable = False)
		self.train_inputs = placeholders["train_inputs"]
		self.train_labels = placeholders["train_labels"]
		self.batchsize = placeholders["batchsize"]
		self.dim = dict_size
		self.nodevec_dim = nodevec_dim
		self.lr = lr

		self.embeddings =  tf.Variable(tf.constant(pretrained, dtype = tf.float32), 
			trainable = True, name = "embeddings")
		self.nce_weights = tf.Variable(tf.constant(pretrained, dtype = tf.float32), 
			trainable = True, name = "nce_weights")

		self.aggregator_t = WeightedAggregator(self.feature_dim, self.nodevec_dim, dropout = self.dropout,
								name = 'true_agg')
		
		self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)
		
		self.build()

	def sample_aggregate(self, input_args, bs, aggregator):
		samples_arg = tf.nn.embedding_lookup(self.rpr_arg, input_args)
		samples_weights = tf.nn.embedding_lookup(self.rpr_matrix, input_args)
		samples_features = tf.nn.embedding_lookup(self.features, samples_arg) 

		batch_out = aggregator((samples_features, samples_weights, bs, FLAGS.k_RPR))

		# out should be bs * d
		return batch_out

	def _build(self):
		labels = tf.reshape(tf.cast(self.train_labels, dtype = tf.int64),
					[self.batchsize, 1])
		self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
			true_classes = labels,
			num_true = 1,
			num_sampled = FLAGS.neg_sample_size,
			unique = True,
			range_max = len(self.degrees),
			distortion = 0.75,
			unigrams = self.degrees.tolist()))

		self.train_inputs_f = self.sample_aggregate(self.train_inputs, self.batchsize, self.aggregator_t)
		self.train_labels_f = self.sample_aggregate(self.train_labels, self.batchsize, self.aggregator_t)
		self.neg_samples_f = self.sample_aggregate(self.neg_samples, FLAGS.neg_sample_size, self.aggregator_t)

		self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
		self.true_w = tf.nn.embedding_lookup(self.nce_weights, self.train_labels)
		self.false_w = tf.nn.embedding_lookup(self.nce_weights, self.neg_samples)

		self.train_inputs_all = tf.add(self.train_inputs_f, self.embed)
		self.train_labels_all = tf.add(self.train_labels_f, self.true_w)
		self.neg_samples_all = tf.add(self.neg_samples_f, self.false_w)

	def build(self):
		self._build()
		self._loss()

		self._minimize_2()

	def _minimize(self):
		self.opt_op = self.optimizer.minimize(self.loss)

	def _minimize_2(self):
		var_list1 = [var for var in tf.trainable_variables() 
			if var.name == "embeddings:0" or var.name == "nce_weights:0"]
		var_list2 = [var for var in tf.trainable_variables() if var not in var_list1]
		opt2 = tf.train.AdamOptimizer(learning_rate = self.lr)
		opt1 = tf.train.AdamOptimizer(learning_rate = 1e-5)
		grads = tf.gradients(self.loss, var_list1 + var_list2)
		grads1 = grads[:len(var_list1)]
		grads2 = grads[len(var_list1):]
		train_op1 = opt1.apply_gradients(zip(grads1, var_list1))
		train_op2 = opt2.apply_gradients(zip(grads2, var_list2))
		self.opt_op = tf.group(train_op1, train_op2)

	def _loss(self):
		p1 = tf.reduce_sum(tf.multiply(self.train_inputs_f, self.train_labels_f), 1)
		p1 = tf.log(tf.sigmoid(p1) + 0.001)

		p2 = tf.reduce_sum(tf.matmul(self.train_inputs_f, tf.transpose(self.neg_samples_f)))
		p2 = tf.log(tf.sigmoid(-p2) + 0.001)

		p3 = tf.reduce_sum(tf.multiply(self.embed, self.true_w), 1)
		p3 = tf.log(tf.sigmoid(p3) + 0.001)

		p4 = tf.reduce_sum(tf.matmul(self.embed, tf.transpose(self.false_w)))
		p4 = tf.log(tf.sigmoid(-p4) + 0.001)

		p5 = tf.reduce_sum(tf.multiply(self.embed, self.train_labels_f), 1)
		p5 = tf.log(tf.sigmoid(p5) + 0.001)

		p6 = tf.reduce_sum(tf.matmul(self.embed, tf.transpose(self.neg_samples_f)))
		p6 = tf.log(tf.sigmoid(-p6) + 0.001)

		p7 = tf.reduce_sum(tf.multiply(self.true_w, self.train_inputs_f), 1)
		p7 = tf.log(tf.sigmoid(p7) + 0.001)

		p8 = tf.reduce_sum(tf.matmul(self.true_w, tf.transpose(self.neg_samples_f)))
		p8 = tf.log(tf.sigmoid(-p8) + 0.001)

		rho1 = 1.5
		rho2 = 0.75
		rho3 = 1.5
		temp_loss = rho1*(p1+p2)+rho2*(p3+p4)+rho3*(p5+p6)+rho3*(p7+p8)
		self.loss += -tf.reduce_sum(temp_loss) / tf.cast(self.batchsize, tf.float32)
		tf.summary.scalar('loss', self.loss)

class AggregateModel(GeneralizedModel):
	def __init__(self, placeholders, features, dict_size, degree_permuted, rpr_matrix,
					rpr_arg, dropout = 0., nodevec_dim = 200, lr = 0.001, only_f = False, **kwargs):
		"""
		Aggregate feature informations of the neighbors of the current node,
		weighted by Rooted PageRank vector of the current node.
		"""

		super(AggregateModel, self).__init__(**kwargs)

		self.placeholders = placeholders
		self.degrees = degree_permuted
		self.only_f = only_f
		self.rpr_arg = tf.Variable(tf.constant(rpr_arg, dtype = tf.int64), trainable = False)
		self.rpr_matrix = tf.Variable(tf.constant(rpr_matrix, dtype = tf.float32), trainable = False)
		self.dropout = dropout
		self.feature_dim = features.shape[1]
		self.features = tf.Variable(tf.constant(features, dtype = tf.float32), trainable = False)
		self.train_inputs = placeholders["train_inputs"]
		self.train_labels = placeholders["train_labels"]
		self.batchsize = placeholders["batchsize"]
		self.dim = dict_size
		self.nodevec_dim = nodevec_dim

		self.embeddings = inits.glorot([dict_size, nodevec_dim], name = "embeddings")
		self.nce_weights = inits.glorot([dict_size, nodevec_dim], name = "nce_weights")

		self.aggregator_t = WeightedAggregator(self.feature_dim, self.nodevec_dim, dropout = self.dropout,
								name = 'true_agg')
		self.optimizer = tf.train.AdamOptimizer(learning_rate = lr)

		self.build()

	def sample_aggregate(self, input_args, bs, aggregator):
		samples_arg = tf.nn.embedding_lookup(self.rpr_arg, input_args)
		samples_weights = tf.nn.embedding_lookup(self.rpr_matrix, input_args)
		samples_features = tf.nn.embedding_lookup(self.features, samples_arg) 

		batch_out = aggregator((samples_features, samples_weights, bs, FLAGS.k_RPR))

		# out should be bs * d
		return batch_out

	def _build(self):
		labels = tf.reshape(tf.cast(self.train_labels, dtype = tf.int64),
					[self.batchsize, 1])
		self.neg_samples, _, _ = (tf.nn.fixed_unigram_candidate_sampler(
			true_classes = labels,
			num_true = 1,
			num_sampled = FLAGS.neg_sample_size,
			unique = True,
			range_max = len(self.degrees),
			distortion = 0.75,
			unigrams = self.degrees.tolist()))

		self.train_inputs_f = self.sample_aggregate(self.train_inputs, self.batchsize, self.aggregator_t)
		self.train_labels_f = self.sample_aggregate(self.train_labels, self.batchsize, self.aggregator_t)
		self.neg_samples_f = self.sample_aggregate(self.neg_samples, FLAGS.neg_sample_size, self.aggregator_t)

		self.embed = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
		self.true_w = tf.nn.embedding_lookup(self.nce_weights, self.train_labels)
		self.false_w = tf.nn.embedding_lookup(self.nce_weights, self.neg_samples)

		self.train_inputs_all = tf.add(self.train_inputs_f, self.embed)
		self.train_labels_all = tf.add(self.train_labels_f, self.true_w)
		self.neg_samples_all = tf.add(self.neg_samples_f, self.false_w)

	def build(self):
		self._build()
		if self.only_f:
			self._f_loss()
		else:
			self._loss()
		self._minimize()

	def _minimize(self):
		self.opt_op = self.optimizer.minimize(self.loss)

	def _f_loss(self):
		p1 = tf.reduce_sum(tf.multiply(self.train_inputs_f, self.train_labels_f), 1)
		p1 = tf.log(tf.sigmoid(p1) + 0.001)

		p2 = tf.reduce_sum(tf.matmul(self.train_inputs_f, tf.transpose(self.neg_samples_f)))
		p2 = tf.log(tf.sigmoid(-p2) + 0.001) 

		temp_loss = p1 + p2
		self.loss=-tf.reduce_sum(temp_loss) / tf.cast(self.batchsize, tf.float32)
		tf.summary.scalar('loss', self.loss)

	def _loss(self):
		p1 = tf.reduce_sum(tf.multiply(self.train_inputs_f, self.train_labels_f), 1)
		p1 = tf.log(tf.sigmoid(p1) + 0.001)

		p2 = tf.reduce_sum(tf.matmul(self.train_inputs_f, tf.transpose(self.neg_samples_f)))
		p2 = tf.log(tf.sigmoid(-p2) + 0.001)

		p3 = tf.reduce_sum(tf.multiply(self.embed, self.true_w), 1)
		p3 = tf.log(tf.sigmoid(p3) + 0.001)

		p4 = tf.reduce_sum(tf.matmul(self.embed, tf.transpose(self.false_w)))
		p4 = tf.log(tf.sigmoid(-p4) + 0.001)

		p5 = tf.reduce_sum(tf.multiply(self.embed, self.train_labels_f), 1)
		p5 = tf.log(tf.sigmoid(p5) + 0.001)

		p6 = tf.reduce_sum(tf.matmul(self.embed, tf.transpose(self.neg_samples_f)))
		p6 = tf.log(tf.sigmoid(-p6) + 0.001)

		p7 = tf.reduce_sum(tf.multiply(self.true_w, self.train_inputs_f), 1)
		p7 = tf.log(tf.sigmoid(p7) + 0.001)

		p8 = tf.reduce_sum(tf.matmul(self.true_w, tf.transpose(self.neg_samples_f)))
		p8 = tf.log(tf.sigmoid(-p8) + 0.001)

		rho1 = 1.5
		rho2 = 0.75
		rho3 = 1.5
		temp_loss = rho1*(p1+p2)+rho2*(p3+p4)+rho3*(p5+p6)+rho3*(p7+p8)
		self.loss += -tf.reduce_sum(temp_loss) / tf.cast(self.batchsize, tf.float32)
		tf.summary.scalar('loss', self.loss)