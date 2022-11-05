import numpy as np
import scipy.sparse as sp
import os.path as osp
import os
import urllib.request
import sys
import networkx as nx
from utils import get_train_val_test
import torch
import pickle as pkl
class Dataset():
    def __init__(self, root, name, seed=None, norm_feature=False, is_self_loop=False):
        self.name = name.lower()

        assert self.name in ['cora', 'citeseer', 'blogcatalog']

        self.seed = seed
        self.norm_feature = norm_feature
        self.is_self_loop = is_self_loop
        self.url =  'https://raw.githubusercontent.com/danielzuegner/gnn-meta-attack/master/data/%s.npz' % self.name
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'

        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_tvt_ids()
    def get_tvt_ids(self):
        all_ids = pkl.load(open(f'data/{self.name}_split.pkl', 'rb'))
        idx_train, idx_val, idx_test = all_ids[0], all_ids[1], all_ids[2]
        return idx_train, idx_val, idx_test
         
   
    def load_data(self):
        print('Loading {} dataset...'.format(self.name))
       
        if self.name=='blogcatalog':
            return self.load_social_network(self.name)
        elif not osp.exists(self.data_filename):
            self.download_npz()

        adj, features, labels = self.get_adj()
        return adj, features, labels

    def normalize(self,mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = sp.diags(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx

    def load_social_network(self,dataset):
        adj_orig = pkl.load(open(f'data/{dataset}_adj.pkl', 'rb'))
        features = pkl.load(open(f'data/{dataset}_features.pkl', 'rb'))
        labels = pkl.load(open(f'data/{dataset}_labels.pkl', 'rb'))
        if sp.issparse(features):
            features = torch.FloatTensor(features.toarray())

        return adj_orig,features,labels
   

    def sparse_mx_to_torch_sparse_tensor(self,sparse_mx):
        """Convert a scipy sparse matrix to a torch sparse tensor."""
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)

    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)

        adj = adj + adj.T
        adj = adj.tolil()

        adj[adj > 1] = 1

        lcc = self.largest_connected_components(adj)
        adj = adj[lcc][:, lcc]
        if not self.is_self_loop:
            adj.setdiag(0)
        else:
            adj.setdiag(1)

        features = features[lcc]
        if self.norm_feature == True:
            features = self.normalize(features)
        features = torch.FloatTensor(np.array(features.todense()))

        labels = labels[lcc]
        return adj, features, labels



    def load_npz(self, file_name, is_sparse=True):
        with np.load(file_name) as loader:
            if is_sparse:
                adj = sp.csr_matrix((loader['adj_data'], loader['adj_indices'],
                                            loader['adj_indptr']), shape=loader['adj_shape'])
                if 'attr_data' in loader:
                    features = sp.csr_matrix((loader['attr_data'], loader['attr_indices'],
                                                 loader['attr_indptr']), shape=loader['attr_shape'])
                else:
                    features = None
                labels = loader.get('labels')
            else:
                adj = loader['adj_data']
                if 'attr_data' in loader:
                    features = loader['attr_data']
                else:
                    features = None
                labels = loader.get('labels')
        if features is None:
            features = np.eye(adj.shape[0])
        features = sp.csr_matrix(features, dtype=np.float32)
        return adj, features, labels

    def largest_connected_components(self, adj, n_components=1):
        """Select k largest connected components.

		Parameters
		----------
		adj : scipy.sparse.csr_matrix
			input adjacency matrix
		n_components : int
			n largest connected components we want to select
		"""

        _, component_indices = sp.csgraph.connected_components(adj)
        component_sizes = np.bincount(component_indices)
        components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
        nodes_to_keep = [
            idx for (idx, component) in enumerate(component_indices) if component in components_to_keep]
        print("Selecting {0} largest connected components".format(n_components))
        return nodes_to_keep

    def __repr__(self):
        return '{0}(adj_shape={1}, feature_shape={2})'.format(self.name, self.adj.shape, self.features.shape)

    def onehot(self, labels):
        eye = np.identity(labels.max() + 1)
        onehot_mx = eye[labels]
        return onehot_mx


