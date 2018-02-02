#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import sparse
from io import open
import json
import random
import tensorflow as tf
import sys
from utils import load_pdata, cos_sim, cost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.multiclass import OneVsRestClassifier
from itertools import izip
from sklearn.utils import shuffle as skshuffle
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr
from fastdtw import fastdtw


def compute_correlation(dataset, embeddings, rpr_matrix):
	graph, _, _, _ = load_pdata(dataset)
	eu_dists = []
	stru_dists = []
	for node in graph:
		for nei in graph[node]:
			if node == nei:
				continue
			dist_eu = np.linalg.norm(embeddings[node] - embeddings[nei])
			dist_stru, _ = fastdtw(embeddings[node], embeddings[nei], radius = 1, dist = cost)
			eu_dists.append(dist_eu)
			stru_dists.append(dist_stru)
	pear_rho, pear_p = pearsonr(stru_dists, eu_dists)
	spea_rho, spea_p = spearmanr(stru_dists, eu_dists)
	return "P ratio and p: {:.2f} + {:.2f}, S ratio and p: {:.2f} + {:.2f}".format(pear_rho, pear_p, spea_rho, spea_p)

class TopKRanker(OneVsRestClassifier):
	def predict(self, X, top_k_list):
		assert X.shape[0] == len(top_k_list)
		probs = np.asarray(super(TopKRanker, self).predict_proba(X))
		all_labels = []
		for i, k in enumerate(top_k_list):
			probs_ = probs[i, :]
			labels = self.classes_[probs_.argsort()[-k:]].tolist()
			all_labels.append(labels)
		return all_labels

def feature_test(dataset, train_embeddings, test_embeddings):
	if dataset == 'cora':
		classes = 7
	elif dataset == 'citeseer':
		classes = 6
	elif dataset == 'pubmed':
		classes = 3
	else:
		raise Exception('Error : wrong dataset name')

	_, _, train_data, test_data = load_pdata(dataset)

	test_l = test_data[:, 1]
	test_label = []
	for i in xrange(test_data.shape[0]):
		temp = [0] * classes
		temp[test_data[i][1] - 1] += 1
		test_label.append(temp)
	test_label = np.array(test_label)	 #1000 * 6

	train_l = train_data[:, 1]
	train_label = []
	for i in xrange(train_data.shape[0]):
		temp = [0] * classes
		temp[train_data[i][1] - 1] += 1
		train_label.append(temp)
	train_label = np.array(train_label)	 #120 * 6

	test_in = np.asarray(test_embeddings)
	train_in = np.asarray(train_embeddings)
	
	y_train_ = sparse.coo_matrix(train_label)
	y_train = [[] for x in xrange(y_train_.shape[0])]
	cy =	y_train_.tocoo()
	for i, j in izip(cy.row, cy.col):
		y_train[i].append(j)
	
	assert sum(len(l) for l in y_train) == y_train_.nnz
	
	y_test_ = sparse.coo_matrix(test_label)
	
	y_test = [[] for x in xrange(y_test_.shape[0])]
	cy =	y_test_.tocoo()
	for i, j in izip(cy.row, cy.col):
		y_test[i].append(j)
	y_train = np.array(y_train)
	#y_test = np.array(y_test)

	clf = TopKRanker(LogisticRegression())
	clf.fit(train_in, y_train)
	
	top_k_list = [len(l) for l in y_test]
	preds = clf.predict(test_in, top_k_list)
	acc = accuracy_score(y_test, preds)
	return acc

def role_test(dataset, role_file, r_class, train_embeddings, test_embeddings):
	_, _, train_data, test_data = load_pdata(dataset)
	train_index = train_data[:, 0]
	test_index = test_data[:, 0]

	role_dic = load_roles(role_file)

	test_label = []
	for i in test_index:
		temp = [0] * r_class
		if i not in role_dic.keys():
			temp[random.randint(0, r_class - 1)] += 1
		else:
			temp[role_dic[i]] += 1
		test_label.append(temp)
	test_label = np.array(test_label)

	train_label = []
	for i in train_index:
		temp = [0] * r_class
		if i not in role_dic.keys():
			temp[random.randint(0, r_class - 1)] += 1
		else:
			temp[role_dic[i]] += 1
		train_label.append(temp)
	train_label = np.array(train_label)

	test_in = np.asarray(test_embeddings)
	train_in = np.asarray(train_embeddings)
	
	y_train_ = sparse.coo_matrix(train_label)
	y_train = [[] for x in xrange(y_train_.shape[0])]
	cy =	y_train_.tocoo()
	for i, j in izip(cy.row, cy.col):
		y_train[i].append(j)
	
	assert sum(len(l) for l in y_train) == y_train_.nnz
	
	y_test_ = sparse.coo_matrix(test_label)
	
	y_test = [[] for x in xrange(y_test_.shape[0])]
	cy =	y_test_.tocoo()
	for i, j in izip(cy.row, cy.col):
		y_test[i].append(j)
	y_train = np.array(y_train)
	#y_test = np.array(y_test)

	clf = TopKRanker(LogisticRegression())
	clf.fit(train_in, y_train)
	
	top_k_list = [len(l) for l in y_test]
	preds = clf.predict(test_in, top_k_list)
	acc = accuracy_score(y_test, preds)
	return acc

if __name__ == '__main__':
	prefix = sys.argv[1]
	mask_rate = float(sys.argv[2])
	G, feats, train_data, test_data = load_pdata(prefix)
	features = np.asarray(feats.todense())

	test_id = test_data[:, 0]
	train_id = train_data[:, 0]
	
	feat_train = []
	feat_test = []
	for id_ in train_id:
		feat_train.append(features[id_])
	for id_ in test_id:
		feat_test.append(features[id_])

	acc_f = feature_test(prefix, feat_train, feat_test)
	print ("feats: {:.3f}".format(acc_f))