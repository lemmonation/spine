#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Graph utilities."""

import sys
import math
from io import open
from os import path
from glob import glob
from six.moves import range, zip, zip_longest
from six import iterkeys
from time import time
import random
from random import shuffle
from itertools import product,permutations
from scipy.sparse import issparse
from collections import defaultdict, Iterable
from multiprocessing import cpu_count
import logging

import numpy as np
from concurrent.futures import ProcessPoolExecutor

from multiprocessing import Pool
from multiprocessing import cpu_count

logger = logging.getLogger("structural_embedding")

class Graph(defaultdict):
  """Efficient basic implementation of nx `Graph' “ Undirected graphs with self loops"""  
  def __init__(self):
    super(Graph, self).__init__(list)

  def nodes(self):
    return self.keys()

  def remove_node(self, n):
    nbrs = self[n]
    for u in nbrs:
      self[u].remove(n)
    del self[n]

  def node(self):
    node = {}
    nodes = self.keys()
    for _id in nodes:
      node[_id] = {}
    return node

  def adjacency_iter(self):
    return self.iteritems()

  def subgraph(self, nodes={}):
    subgraph = Graph()
    
    for n in nodes:
      if n in self:
        subgraph[n] = [x for x in self[n] if x in nodes]
        
    return subgraph

  def make_undirected(self):
  
    t0 = time()

    for v in self.keys():
      for other in self[v]:
        if v != other:
          self[other].append(v)
    
    t1 = time()
    #logger.info('make_directed: added missing edges {}s'.format(t1-t0))

    self.make_consistent()
    return self

  def make_consistent(self):
    t0 = time()
    for k in iterkeys(self):
      self[k] = list(sorted(set(self[k])))
    
    t1 = time()
    #logger.info('make_consistent: made consistent in {}s'.format(t1-t0))

    #self.remove_self_loops()

    return self

  def remove_self_loops(self):

    removed = 0
    t0 = time()

    for x in self:
      if x in self[x]: 
        self[x].remove(x)
        removed += 1
    
    t1 = time()

    #logger.info('remove_self_loops: removed {} loops in {}s'.format(removed, (t1-t0)))
    return self

  def check_self_loops(self):
    for x in self:
      for y in self[x]:
        if x == y:
          return True
    
    return False

  def has_edge(self, v1, v2):
    if v2 in self[v1] or v1 in self[v2]:
      return True
    return False

  def degree(self, nodes=None):
    if isinstance(nodes, Iterable):
      return {v:len(self[v]) for v in nodes}
    else:
      return len(self[nodes])

  def order(self):
    "Returns the number of nodes in the graph"
    return len(self)    

  def number_of_edges(self):
    "Returns the number of nodes in the graph"
    return sum([self.degree(x) for x in self.keys()])/2

  def number_of_nodes(self):
    "Returns the number of nodes in the graph"
    return self.order() 

  def gToDict(self):
    d = {}
    for k,v in self.iteritems():
      d[k] = v
    return d

  def printAdjList(self):
    for key,value in self.iteritems():
      print (key,":",value)

  def random_walk(self, path_length, alpha=0, rand=random.Random(), start=None):
   """ Returns a truncated random walk.
       path_length: Length of the random walk.
       alpha: probability of restarts.
       start: the start node of the random walk.
   """
   G = self
   if start:
     path = [start]
   else:
     # Sampling is uniform w.r.t V, and not w.r.t E
     path = [rand.choice(G.keys())]
   while len(path) < path_length:
     cur = path[-1]
     if len(G[cur]) > 0:
       if rand.random() >= alpha:
         path.append(rand.choice(G[cur]))
       else:
         path.append(path[0])
     else:
       break
   return path

  def normal_random_walk(self, path_length, rand = random.Random(), start = None):
    """
    Define a normal random walk without restart to generate positive training pairs
    """
    G = self
    pairs = []
    if start:
      path = [start]
    else:
      # Sampling is uniform w.r.t V, and not w.r.t E
      path = [rand.choice(G.keys())]
    while len(path) < path_length:
      cur = path[-1]
      if len(G[cur]) > 0:
        next_node = rand.choice(G[cur])
        path.append(next_node)
        if path[0] != next_node:
          pairs.append((path[0], next_node))
      else:
        break
    return pairs

def mask_nodes(G, mask_rate, rand = random.Random()):
  num_nodes = len(G.nodes())
  train_num = int(mask_rate * num_nodes)
  test_num = num_nodes - train_num

  test_id = random.sample(G.nodes(), test_num)
  train_id = [i for i in G.nodes() if not i in test_id]

  train_id = sorted(train_id)
  test_id = sorted(test_id)

  G = G.subgraph(train_id)  # a subgraph constructed only by training nodes
  # use a id-map to map the nodes id into (0, len(G))
  sub_id_map = {}
  map_sub_id = {}
  i = 0
  for node in G.keys():
    if i == len(G.keys()):
      break
    sub_id_map[node] = i
    map_sub_id[i] = node
    i += 1

  assert len(sub_id_map.keys()) == len(G.keys()), 'nodes in subgraph are not consisitent with in id map'

  map_G = Graph()
  for node in G.keys():
    map_G[sub_id_map[node]] = [sub_id_map[i] for i in G[node] if i in G.keys()]

  return map_G, test_id, train_id, map_sub_id

def write_normal_randomwalks(G, file_, num_paths = 50, path_length = 5,
                      rand=random.Random(0)):
  nodes = list(G.nodes())
  pairs = []
  with open(file_, 'w') as fp:
    for cnt in range(num_paths):
      rand.shuffle(nodes)
      for node in nodes:
        pair = G.normal_random_walk(path_length, rand = rand, start = node)
        pairs.extend(pair)
        for p in pair:
          fp.write(unicode("{}\t{}\n".format(p[0], p[1])))
  return pairs

def build_deepwalk_corpus(G, num_paths, path_length, alpha=0,
                      rand=random.Random(0)):
  walks = {}
  #walks = []

  nodes = list(G.nodes())
  
  for cnt in range(num_paths):
    rand.shuffle(nodes)
    for node in nodes:
      #walks.append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
      if node in walks.keys():
        walks[node].append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
      else:
        walks[node] = []
        walks[node].append(G.random_walk(path_length, rand=rand, alpha=alpha, start=node))
      
  return walks

def grouper(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)

def parse_adjacencylist(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      introw = [int(x) for x in l.strip().split()]
      row = [introw[0]]
      row.extend(set(sorted(introw[1:])))
      adjlist.extend([row])
  
  return adjlist

def parse_adjacencylist_unchecked(f):
  adjlist = []
  for l in f:
    if l and l[0] != "#":
      adjlist.extend([[int(x) for x in l.strip().split()]])
  
  return adjlist

def load_adjacencylist(file_, undirected=False, chunksize=10000, unchecked=True):

  if unchecked:
    parse_func = parse_adjacencylist_unchecked
    convert_func = from_adjlist_unchecked
  else:
    parse_func = parse_adjacencylist
    convert_func = from_adjlist

  adjlist = []

  t0 = time()

  with open(file_) as f:
    with ProcessPoolExecutor(max_workers=cpu_count()) as executor:
      total = 0 
      for idx, adj_chunk in enumerate(executor.map(parse_func, grouper(int(chunksize), f))):
          adjlist.extend(adj_chunk)
          total += len(adj_chunk)
  
  t1 = time()

  logger.info('Parsed {} edges with {} chunks in {}s'.format(total, idx, t1-t0))

  t0 = time()
  G = convert_func(adjlist)
  t1 = time()

  logger.info('Converted edges to graph in {}s'.format(t1-t0))

  if undirected:
    t0 = time()
    G = G.make_undirected()
    t1 = time()
    logger.info('Made graph undirected in {}s'.format(t1-t0))

  return G 

def load_edgelist(file_, undirected=True):
  G = Graph()
  with open(file_) as f:
    for l in f:
      if(len(l.strip().split()[:2]) > 1):
        x, y = l.strip().split()[:2]
        x = int(x)
        y = int(y)
        G[x].append(y)
        if undirected:
          G[y].append(x)
      else:
        x = l.strip().split()[:2]
        x = int(x[0])
        G[x] = []  
  
  G.make_consistent()
  return G

def from_adjlist(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = list(sorted(set(neighbors)))

    return G


def from_adjlist_unchecked(adjlist):
    G = Graph()
    
    for row in adjlist:
        node = row[0]
        neighbors = row[1:]
        G[node] = neighbors

    return G