import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from torch_geometric.nn import GCNConv
import numpy as np
import scipy.sparse as sp
from torch_geometric.utils import from_scipy_sparse_matrix
import utils

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.01, weight_decay=5e-4, with_relu=True, with_bias=True, self_loop=True ,device=None):

        super(GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1 = GCNConv(nfeat, nhid, bias=with_bias,add_self_loops=self_loop)
        self.gc2 = GCNConv(nhid, nclass, bias=with_bias,add_self_loops=self_loop)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        

    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            x = F.relu(self.gc1(x, edge_index,edge_weight))
        else:
            x = self.gc1(x, edge_index,edge_weight)

        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, edge_index,edge_weight)
        return x

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False, **kwargs):
        if initialize:
            self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)


        self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)

  
    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


   
class Dual_GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5, lr=0.001, weight_decay=5e-4, with_relu=True, with_bias=True,
                 self_loop=True, device=None):

        super(Dual_GCN, self).__init__()

        assert device is not None, "Please specify 'device'!"
        self.device = device
        self.nfeat = nfeat
        self.hidden_sizes = [nhid]
        self.nclass = nclass
        self.gc1_1 = GCNConv(nfeat, nhid, bias=with_bias, add_self_loops=self_loop)
        self.gc2_1 = GCNConv(nhid, nclass, bias=with_bias, add_self_loops=self_loop)
        self.gc1_2 = GCNConv(nfeat, nhid, bias=with_bias, add_self_loops=self_loop)
        self.gc2_2 = GCNConv(nhid, nclass, bias=with_bias, add_self_loops=self_loop)
        self.dropout = dropout
        self.lr = lr
        if not with_relu:
            self.weight_decay = 0
        else:
            self.weight_decay = weight_decay
        self.with_relu = with_relu
        self.with_bias = with_bias
        self.output = None
        self.best_model = None
        self.best_output = None
        self.edge_index = None
        self.edge_weight = None
        self.features = None
        self.idx_train_1 = None
        self.idx_train_2 = None



    def forward(self, x, edge_index, edge_weight):
        if self.with_relu:
            x1 = F.relu(self.gc1_1(x, edge_index, edge_weight))
        else:
            x1 = self.gc1_1(x, edge_index, edge_weight)

        x1 = F.dropout(x1, self.dropout, training=self.training)
        x1 = self.gc2_1(x1, edge_index, edge_weight)

        if self.with_relu:
            x2 = F.relu(self.gc1_2(x, edge_index, edge_weight))
        else:
            x2 = self.gc1_2(x, edge_index, edge_weight)

        x2 = F.dropout(x2, self.dropout, training=self.training)
        x2 = self.gc2_2(x2, edge_index, edge_weight)


        return x1,x2

    def initialize(self):
        """Initialize parameters of GCN.
        """
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val=None, train_iters=200, initialize=True, verbose=False,
            **kwargs):

        if initialize:
            self.initialize()

        self.edge_index, self.edge_weight = from_scipy_sparse_matrix(adj)
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.float().to(self.device)

        if sp.issparse(features):
            features = utils.sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        self.features = features.to(self.device)
        self.labels = torch.LongTensor(np.array(labels)).to(self.device)

        self._train_with_val(self.labels, idx_train, idx_val, train_iters, verbose)


    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        if verbose:
            print('=== training gcn model ===')
        optimizer = optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_acc_val = 0

        for i in range(train_iters):
            self.train()
            optimizer.zero_grad()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_train = F.cross_entropy(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            if verbose and i % 10 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

            self.eval()
            output = self.forward(self.features, self.edge_index, self.edge_weight)
            loss_val = F.cross_entropy(output[idx_val], labels[idx_val])
            acc_val = utils.accuracy(output[idx_val], labels[idx_val])

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                self.output = output
                weights = deepcopy(self.state_dict())

        if verbose:
            print('=== picking the best model according to the performance on validation ===')
        self.load_state_dict(weights)


# %%