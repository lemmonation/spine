#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import numpy as np
import scipy.io as sio
from scipy import spatial
import networkx as nx
import random
from tqdm import tqdm
from collections import Counter

import graph
import utils
from models import AggregateModel, PretrainModel
from minibatch import MinibatchIterator
import test

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
							"""Whether to log device placement.""")

flags.DEFINE_string('input', '', "the input graph edge list file. must be specified")
flags.DEFINE_string('train_prefix', 'cora', "dataset name.")
flags.DEFINE_boolean('preprocess', False, "if have processed once then could set False")
flags.DEFINE_integer('dim', 200, "embedding dimension.")
flags.DEFINE_integer('batchsize', 512, "batch size")
flags.DEFINE_integer('epoch', 5, "number of training epoches")
flags.DEFINE_float('learning_rate', 0.001, "learning rate")
flags.DEFINE_float('stru_rate', 0.2, "rate between structure sampling and neighbor sampling")
flags.DEFINE_integer('walk_times', 10, "random walk times started at every node.")
flags.DEFINE_integer('walk_length', 40, "random walk length at each node.")
flags.DEFINE_integer('k_RPR', 20, "top-k Rooted PageRank nodes of current node.")
flags.DEFINE_float('alpha', 0.5, "restart rate when random Walking.")
flags.DEFINE_integer('neg_sample_size', 50, "negative sampling size.")
flags.DEFINE_integer('verbose', 20, "how often to print information")
flags.DEFINE_float('dropout', 0., "dropout rate in MLP")
flags.DEFINE_float('weight_decay', 0.001, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('save_per_epoch', 200, 'how often to save the model by epoch')
flags.DEFINE_integer('seed', 123, "seed when random walk.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_boolean('PRETRAIN', True, 'W_mlp pretrained by node2vec')
flags.DEFINE_boolean('CORR', False, "use pearson and spearman correlation to evaluate")

seed = FLAGS.seed
np.random.seed(seed)
tf.set_random_seed(seed)

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir():
	log_dir = "./log/" + FLAGS.train_prefix
	log_dir += "/{lr:0.3f}_{stru_rate:0.1f}_{rpr_k:d}/".format(
			lr = FLAGS.learning_rate,
			stru_rate = FLAGS.stru_rate,
			rpr_k = FLAGS.k_RPR
			)
	if not os.path.exists(log_dir):
		os.makedirs(log_dir)
	return log_dir

def read_graph():
	'''
	Reads the input network.
	'''
	print (" - Loading graph...")
	G = graph.load_edgelist(FLAGS.input,undirected=True)
	print (" - Graph loaded.")
	return G

def construct_rpr_matrix(G, INDUCTIVE = False):
	'''
	Construct Rooted PageRank matrix
	'''
	print ("Number of nodes: {}".format(len(G.nodes())))
	num_walks = len(G.nodes()) * FLAGS.walk_times
	num_nodes = len(G.nodes())

  	print("Number of walks: {}".format(num_walks))
 	print("Walking...")
  	walks = graph.build_deepwalk_corpus(G, num_paths=FLAGS.walk_times, path_length=FLAGS.walk_length, 
  							alpha=FLAGS.alpha, rand=random.Random(FLAGS.seed))
  	all_counts = {}
  	for node in walks.keys():
  		walks_n = walks[node]
  		all_counts[node] = Counter()
  		for walk in walks_n:
  			all_counts[node].update(walk)

  	print("Normal random walks started...")
  	pairs = graph.write_normal_randomwalks(G, 
  		file_= './var/' + FLAGS.train_prefix + '_normal_walks.txt',rand=random.Random(FLAGS.seed))
  
  	print("Normal random walks dumped.")
 
	rpr_matrix = []
	rpr_arg = []
	for node in tqdm(xrange(num_nodes)):
		if node not in all_counts.keys():
			raise NotImplementedError
		temp = all_counts[node].most_common(FLAGS.k_RPR)
		temp_arg = [i[0] for i in temp]
		temp_value = [i[1] for i in temp]
		if len(temp) < FLAGS.k_RPR:
			for _ in xrange(FLAGS.k_RPR - len(temp)):
				temp_value.append(0.0)
				temp_arg.append(node)
		temp_value = np.asarray(temp_value, dtype = 'double')
		temp_value = temp_value / sum(temp_value)
		rpr_matrix.append(temp_value)
		rpr_arg.append(temp_arg)
	rpr_matrix = np.asarray(rpr_matrix, dtype = 'double')
	rpr_arg = np.asarray(rpr_arg, dtype = 'double')
	rpr_file = './var/' + FLAGS.train_prefix + '_rpr.mat'

	sio.savemat(rpr_file, {'rpr_matrix':rpr_matrix})
	return rpr_matrix, pairs, rpr_arg

def construct_placeholders():
	placeholders = {
		'train_inputs' : tf.placeholder(tf.int32, shape = (None), name = 'train_inputs'),
		'train_labels' : tf.placeholder(tf.int32, shape = (None), name = 'train_labels'),
		'batchsize' : tf.placeholder(tf.int32, name = 'batchsize')
	}
	return placeholders

def main():
	G = read_graph()
	if FLAGS.preprocess:
		print (" - Computing Rooted PageRank matrix...")
		rpr_matrix, pairs, rpr_arg = construct_rpr_matrix(G)
		utils.dump_to_disk(rpr_arg, './var/' + FLAGS.train_prefix + '_rpr_arg')
		print (" - RPR matrix completed.")
		degrees, degree_permuted = utils.create_degree(G)
		print (" - Dumping degree vectors to disk...")
		utils.dump_to_disk(degrees, './var/' + FLAGS.train_prefix + '_degrees')
		utils.dump_to_disk(degree_permuted, './var/' + FLAGS.train_prefix + '_degree_permuted')
		print (" - Degree vectors dumped.")
	else:
		print (" - Loading precomputed Rooted PageRank matrix...")
		rpr_file = './var/' + FLAGS.train_prefix + '_rpr.mat'
		rpr_matrix = sio.loadmat(rpr_file)['rpr_matrix']
		rpr_arg = utils.load_pkl('./var/' + FLAGS.train_prefix + '_rpr_arg')
		print (" - RPR matrix loaded.")
		print (" - Loading Degree vectors...")
		degrees = utils.load_pkl('./var/' + FLAGS.train_prefix + '_degrees')
		degree_permuted = utils.load_pkl('./var/' + FLAGS.train_prefix + '_degree_permuted')
		print (" - Degree vectors loaded.")
		pairs = []
		with open('./var/' + FLAGS.train_prefix + '_normal_walks.txt', 'r') as fp:
			for line in fp:
				n_pair = line.split()
				pairs.append((int(n_pair[0]), int(n_pair[1])))
		print (" - Training pairs loaded")

	placeholders = construct_placeholders()

	minibatch = MinibatchIterator(G, placeholders, degrees, rpr_matrix, pairs, 
		batchsize = FLAGS.batchsize, stru_rate = FLAGS.stru_rate, dataset = FLAGS.train_prefix)

	_, features, _, _ = utils.load_pdata(FLAGS.train_prefix)
	# TODO: maybe can be more efficiently written by sparse multipications
	features = np.asarray(features.todense())
	
	if FLAGS.PRETRAIN:
		from gensim.models.keyedvectors import KeyedVectors
		n2v_embedding = './baselines/{}_{}.embeddings'.format('node2vec', FLAGS.train_prefix)
		n_model = KeyedVectors.load_word2vec_format(n2v_embedding, binary=False)
		pretrained = np.asarray([n_model[str(node)] for node in xrange(rpr_matrix.shape[0])])
		model = PretrainModel(placeholders, features, pretrained, len(G.nodes()),
			degree_permuted, rpr_matrix, rpr_arg, 
			dropout = FLAGS.dropout,
			nodevec_dim = FLAGS.dim,
			lr = FLAGS.learning_rate,
			logging = True)
	else:
		model = AggregateModel(placeholders, features, len(G.nodes()),
			degree_permuted, rpr_matrix, rpr_arg, 
			dropout = FLAGS.dropout,
			nodevec_dim = FLAGS.dim,
			lr = FLAGS.learning_rate,
			logging = True)
	
	config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	sess = tf.Session(config = config)
	saver = tf.train.Saver(max_to_keep = 5)
	merged = tf.summary.merge_all()
	summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
	
	# Init variables
	sess.run(tf.global_variables_initializer())
	
	# Train model
	total_steps = 0
	average_time = 0.0
	average_test = 0.0
	test_steps = 0
	epoch_test_acc = [0.0]

	for epoch in xrange(FLAGS.epoch):
		minibatch.shuffle()
		_iter = 0
		print ("Epoch : %02d" % (epoch + 1), "Batchs per epoch : %04d" % (len(pairs) / FLAGS.batchsize))

		while not minibatch.end():
			feed_dict = minibatch.next_minibatch_feed_dict()
			t = time.time()
			# training step
			outs = sess.run([merged, model.opt_op, 
						model.loss, model.embeddings], feed_dict = feed_dict)
			train_cost = outs[2]

			average_time = (average_time * total_steps + time.time() - t) / (total_steps + 1)

			if _iter % FLAGS.verbose == 0:
				if FLAGS.CORR:
					all_feed = minibatch.all_feed_dict()
					out = sess.run([model.train_inputs_all,
								model.train_inputs_f, model.embed, model.loss], feed_dict = all_feed)
					str_corr = test.compute_correlation(FLAGS.train_prefix, out[1], rpr_matrix)
					print ("Epoch: ", '%02d' % (epoch + 1),
						"iter: ", '%03d' % _iter,
						"loss: ", "{:.3f}".format(train_cost),
						"corr: ", str_corr,
						"train time: ", "{:.3f}".format(average_time))
				else:
					train_feed, test_feed = minibatch.test_feed_dict()
					out_train = sess.run([model.train_inputs_all,
									model.train_inputs_f, model.embed], feed_dict = train_feed)
					t1 = time.time()
					out_test = sess.run([model.train_inputs_all,
									model.train_inputs_f, model.embed], feed_dict = test_feed)
					average_test = (average_test * test_steps + time.time() - t1) / (test_steps + 1)
					test_steps += 1

					acc_f = test.feature_test(FLAGS.train_prefix, out_train[1], out_test[1])
					epoch_test_acc.append(acc_f)
					print ("Epoch: ", '%02d' % (epoch + 1),
						"iter: ", '%03d' % _iter,
						"loss: ", "{:.3f}".format(train_cost),
						"now acc: ", "{:.3f}".format(epoch_test_acc[-1]),
						"best acc: ", "{:.3f}".format(max(epoch_test_acc)),
						"train time: ", "{:.3f}".format(average_time),
						"test time: ", "{:.3f}".format(average_test))

			_iter += 1
			total_steps += 1
		if epoch % FLAGS.save_per_epoch:
			saver.save(sess, os.path.join(log_dir(), 'model.ckpt'), epoch)
	print ("Optimization finished !")

if __name__ == '__main__':
	main()
	