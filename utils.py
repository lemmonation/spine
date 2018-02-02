#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import io
import time
import numpy as np
import scipy.io as sio
from scipy import sparse as sp
from scipy import spatial
import cPickle as pkl
import networkx as nx
import random
import math

from collections import Counter

import graph

def cost(a,b):
    ep = 0.001
    m = max(a,b) + ep
    mi = min(a,b) + ep
    return ((m/mi) - 1)

def cos_sim(node_vec, neb_vec):
    cos_ = 1 - spatial.distance.cosine(node_vec, neb_vec)
    return cos_

def create_degree(G):
	print (" - Creating degree vectors...")
	degrees = {}
	degrees_sorted = set()
	degree_permuted = np.zeros((len(G.keys()), ))
	for v in G.keys():
		degree = len(G[v])
		degrees_sorted.add(degree)
		degree_permuted[v] = degree
		if(degree not in degrees):
			degrees[degree] = {}
			degrees[degree]['vertices'] = []
		degrees[degree]['vertices'].append(v)
	degrees_sorted = np.array(list(degrees_sorted),dtype='int')
	#degree_permuted = degrees_sorted
	degrees_sorted = np.sort(degrees_sorted)
	l = len(degrees_sorted)
	for index, degree in enumerate(degrees_sorted):
		if(index > 0):
			degrees[degree]['before'] = degrees_sorted[index - 1]
		if(index < (l - 1)):
			degrees[degree]['after'] = degrees_sorted[index + 1]
	print ("- Degree vectors created.")
	return degrees, degree_permuted

def verifyDegrees(degree_v_root,degree_a,degree_b):

    if(degree_b == -1):
        degree_now = degree_a
    elif(degree_a == -1):
        degree_now = degree_b
    elif(abs(degree_b - degree_v_root) < abs(degree_a - degree_v_root)):
        degree_now = degree_b
    else:
        degree_now = degree_a

    return degree_now 

def dump_to_disk(f, file_name):
	with open(file_name + '.pkl', 'wb') as handle:
		pkl.dump(f, handle, protocol = pkl.HIGHEST_PROTOCOL)

def load_pkl(file_name):
	with open(file_name + '.pkl', 'rb') as handle:
		val = pkl.load(handle)
	return val

def sparse_to_tuple(sparse_mx):
    if not sp.isspmatrix_coo(sparse_mx):
        sparse_mx = sparse_mx.tocoo()
    coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
    values = sparse_mx.data
    shape = sparse_mx.shape
    return coords, values, shape

def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def load_pdata(dataset_str):
    if dataset_str != 'cora' and dataset_str != 'citeseer' and dataset_str != 'pubmed':
        print ('Use datasets other than Planetoid, change load functions')
        pass
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in xrange(len(names)):
        objects.append(pkl.load(open("./data/ind.{}.{}".format(dataset_str, names[i]))))
    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("./data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    if dataset_str == 'citeseer':
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    
    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))

    train_mask = sample_mask(idx_train, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    train_out = []
    for i in idx_train:
        ll = y_train[i].tolist()
        ll = ll.index(1) + 1
        train_out.append([i, ll])
    train_out = np.array(train_out)

    test_out = []
    for i in idx_test:
        ll = y_test[i].tolist()
        ll = ll.index(1) + 1
        test_out.append([i, ll])
    test_out = np.array(test_out)
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
    adj.eliminate_zeros()
    # Check that diag is zero:
    assert np.diag(adj.todense()).sum() == 0

    adj_triu = sp.triu(adj)
    adj_tuple = sparse_to_tuple(adj_triu)
    edges = adj_tuple[0]
    edges_all = sparse_to_tuple(adj)[0]

    num_mask = int(np.floor(edges.shape[0] / 10.))

    return graph, features, train_out, test_out
