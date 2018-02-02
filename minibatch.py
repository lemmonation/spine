from __future__ import division
from __future__ import print_function

import numpy as np
import math
import random
from scipy.stats import pearsonr, spearmanr
from fastdtw import fastdtw

import graph
import utils

class MinibatchIterator(object):
	"""
	Minibatch Iterator to sample random pairs of positive nodes
	"""
	def __init__(self, G, placeholders, degrees, rpr_matrix, context_pair, 
				batchsize = 128, stru_rate = 0.5, dataset = 'cora', **kwargs):
		self.G = G
		self.placeholders = placeholders
		self.node_num = len(G.nodes())
		self.nodes = np.random.permutation(G.nodes())
		self.edges = np.random.permutation(context_pair)
		self.batch_num = 0
		self.batchsize = batchsize
		self.degrees = degrees
		self.rpr_matrix = rpr_matrix
		self.stru_rate =stru_rate
		self.dataset = dataset

	def batch_feed_dict(self, batch_edges):
		train_inputs = []
		train_labels = []
		for node1, node2 in batch_edges:
			train_inputs.append(node1)
			if random.random() <= self.stru_rate:
				degree_neighbors = self.get_vertices(node1)
				rpr_sample_node = self.rpr_sample(node1, degree_neighbors)
				train_labels.append(rpr_sample_node)
			else:
				train_labels.append(node2)

		feed_dict = dict()
		feed_dict.update({self.placeholders['batchsize'] : len(batch_edges)})
		feed_dict.update({self.placeholders['train_inputs'] : train_inputs})
		feed_dict.update({self.placeholders['train_labels'] : train_labels})

		return feed_dict

	def node_feed_dict(self, batch_nodes):
		train_inputs = train_labels = batch_nodes
		# train_labels is not important, thus equals them

		feed_dict = dict()
		feed_dict.update({self.placeholders['batchsize'] : len(batch_nodes)})
		feed_dict.update({self.placeholders['train_inputs'] : train_inputs})
		feed_dict.update({self.placeholders['train_labels'] : train_labels})

		return feed_dict

	def all_feed_dict(self):
		id_range = range(self.node_num)
		return self.node_feed_dict(id_range)

	def test_feed_dict(self):
		_, _, test_train, test_test = utils.load_pdata(self.dataset)
		test_train = test_train[:, 0]
		test_test = test_test[:, 0]
		t_train_feed = self.node_feed_dict(test_train)
		t_test_feed = self.node_feed_dict(test_test)
		return t_train_feed, t_test_feed

	def next_minibatch_feed_dict(self):
		start  = self.batch_num * self.batchsize
		self.batch_num += 1
		batch_edges = self.edges[start : start + self.batchsize]
		return self.batch_feed_dict(batch_edges)

	def rpr_sample(self, node, neighbors):
		node_rpr_v = self.rpr_matrix[node]
		sim_list = []
		for _neighbor in neighbors:
			neighbor_rpr_v = self.rpr_matrix[_neighbor]
			dits_dtw, _ = fastdtw(node_rpr_v, neighbor_rpr_v, radius = 1, dist = utils.cost)
			sim_list.append(np.exp(-1.0 * dits_dtw))
			
		norm_weight = [float(i) / sum(sim_list) for i in sim_list]
		sampled_neighbor = np.random.choice(neighbors, p = norm_weight)
		return sampled_neighbor

	def get_vertices(self, v):
		num_seleted = 2 * math.log(self.node_num, 2)
		vertices = []

		degree_v = self.G.degree(v)

		try:
			c_v = 0  
	
			for v2 in self.degrees[degree_v]['vertices']:
				if(v != v2):
					vertices.append(v2)
					c_v += 1
					if(c_v > num_seleted):
						raise StopIteration
	
			if('before' not in self.degrees[degree_v]):
				degree_b = -1
			else:
				degree_b = self.degrees[degree_v]['before']
			if('after' not in self.degrees[degree_v]):
				degree_a = -1
			else:
				degree_a = self.degrees[degree_v]['after']
			if(degree_b == -1 and degree_a == -1):
				raise StopIteration
			degree_now = utils.verifyDegrees(degree_v,degree_a,degree_b)
	
			while True:
				for v2 in self.degrees[degree_now]['vertices']:
					if(v != v2):
						vertices.append(v2)
						c_v += 1
						if(c_v > num_seleted):
							raise StopIteration
	
				if(degree_now == degree_b):
					if('before' not in self.degrees[degree_b]):
						degree_b = -1
					else:
						degree_b = self.degrees[degree_b]['before']
				else:
					if('after' not in self.degrees[degree_a]):
						degree_a = -1
					else:
						degree_a = self.degrees[degree_a]['after']
				
				if(degree_b == -1 and degree_a == -1):
					raise StopIteration
	
				degree_now = utils.verifyDegrees(degree_v,degree_a,degree_b)
	
		except StopIteration:
			return list(vertices)
	
		return list(vertices)

	def end(self):
		return self.batch_num * self.batchsize > len(self.edges) - self.batchsize + 1

	def shuffle(self):
		"""
		Re-shuffle the training set. 
		And the batch number
		"""
		self.nodes = np.random.permutation(self.nodes)
		self.edges = np.random.permutation(self.edges)
		self.batch_num = 0