#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tensorflow as tf

from layers import Layer, Dense
from inits import glorot, zeros, uniform

class WeightedAggregator(Layer):
	"""
	An multi-layer perceptron based feature aggregator
	"""

	def __init__(self, input_dim, output_dim, dropout = 0., bias = False,
				hidden_dim = 512, act = tf.nn.relu, name = None, **kwargs):
		super(WeightedAggregator, self).__init__(**kwargs)

		self.dropout = dropout
		self.bias = bias
		self.act = act
		self.hidden_dim = hidden_dim

		if name is not None:
			name = '/' + name
		else:
			name  = ''

		with tf.variable_scope(self.name + name + '_vars'):
			self.vars['mlp_weights'] = glorot([input_dim, output_dim], name = 'mlp_weights')
			tf.summary.histogram("mlp_weights", self.vars['mlp_weights'])
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name = 'bias')
				tf.summary.histogram("bias", self.vars['bias'])
		if self.logging:
			self._log_vars()

		self.input_dim = input_dim
		self.output_dim = output_dim

	def _call(self, inputs):
		# vec: bs * k * f, weights: bs * k
		node_vecs, node_weights, bs, k = inputs
		
		# weighted summation
		node_weights = tf.reshape(node_weights, [bs, k, 1])
		node_vecs = tf.reduce_sum(tf.multiply(node_vecs, tf.cast(node_weights, dtype = tf.float32)), 1)
		out_k = tf.matmul(node_vecs, self.vars['mlp_weights'])
		if self.bias:
			out_k += self.vars['bias']

		# out_k: bs * k * d	
		out_k = self.act(out_k)
		return out_k

class MeanAggregator(Layer):
	"""
	An multi-layer perceptron based feature aggregator
	"""

	def __init__(self, input_dim, output_dim, dropout = 0., bias = False,
				hidden_dim = 512, act = tf.nn.relu, name = None, **kwargs):
		super(MeanAggregator, self).__init__(**kwargs)

		self.dropout = dropout
		self.bias = bias
		self.act = act
		self.hidden_dim = hidden_dim

		if name is not None:
			name = '/' + name
		else:
			name  = ''

		self.mlp_layers = []
		self.mlp_layers.append(Dense(input_dim=input_dim,
								output_dim=hidden_dim,
								act=tf.nn.relu,
								dropout=dropout,
								sparse_inputs=False,
								bias = bias,
								logging=self.logging))

		with tf.variable_scope(self.name + name + '_vars'):
			self.vars['mlp_weights'] = glorot([input_dim, output_dim], name = 'mlp_weights')
			tf.summary.histogram("mlp_weights", self.vars['mlp_weights'])
			if self.bias:
				self.vars['bias'] = zeros([output_dim], name = 'bias')
				tf.summary.histogram("bias", self.vars['bias'])
		if self.logging:
			self._log_vars()

		self.input_dim = input_dim
		self.output_dim = output_dim

	def _call(self, inputs):
		node_vecs, node_weights, bs, k = inputs

		node_means = tf.reduce_mean(node_vecs, 1)
		out_k = tf.matmul(node_means, self.vars['mlp_weights'])
		if self.bias:
			out_k += self.vars['bias']
		return self.act(out_k)
