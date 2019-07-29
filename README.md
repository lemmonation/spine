# SPINE
This repository provides a reference implementation of the paper "SPINE: Structural Identity Preserved Inductive Network Embedding".
SPINE is an inductive embedding method which can simultaneously preserve the local proximity and the structural identity of nodes. Details can be found [here](http://arxiv.org/abs/1802.03984).

### Requirements

* tensorflow > 1.2.1
* networkx == 1.11
* gensim > 2.3.0
* fastdtw

### Usage

To run SPINE on Cora dataset, execute as:
```
python main.py --input data/cora_id_edge.txt --train_prefix cora --preprocess True
```

### Options

To evaluate the performance by Pearson and Spearman correlation instead of classification accuracy, set ``--CORR True``.

To run SPINE and SPINE-p, deactive and active ``--PRETRAIN`` respectively.

For more options, please check ``main.py``.

### Acknowledgements

We refer to [GraphSAGE](https://github.com/williamleif/GraphSAGE) and [GCN](https://github.com/tkipf/gcn) while constructing code framework and preprocessing datasets. Many thanks to the authors for making their code available.

### Miscellaneous

Please cite our paper if you find SPINE useful in your research.
```
@inproceedings{guo2019spine,
  title={SPINE: Structural Identity Preserved Inductive Network Embedding},
  author={Guo, Junliang and Xu, Linli and Liu, Jingchang},
  booktitle={Twenty-Eighth International Joint Conference on Artificial Intelligence},
  year={2019}
}
```

This is only a reference implementation of SPINE, feel free to ask any question by opening an issue or email me at <leoguojl@gmail.com>.
