import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric.utils as utils
import scipy.sparse as sp
from models.Dual_GCN import GCN, Dual_GCN
from utils import accuracy, sparse_mx_to_torch_sparse_tensor

def kl_loss_compute(pred, soft_targets, reduce=True, tempature=1):
    pred = pred / tempature
    soft_targets = soft_targets / tempature
    kl = F.kl_div(F.log_softmax(pred, dim=1), F.softmax(soft_targets, dim=1), reduce=False)
    if reduce:
        return torch.mean(torch.sum(kl, dim=1))
    else:
        return torch.sum(kl, 1)

class LabeledDividedLoss(nn.Module):
    def __init__(self, args):
        super(LabeledDividedLoss, self).__init__()
        self.args = args
        self.epochs = args.epochs
        self.increment = 0.5/self.epochs
        self.decay_w = args.decay_w

    def forward(self, y_1, y_2, t,  co_lambda=0.1, epoch=-1): 
        loss_pick_1 = F.cross_entropy(y_1, t, reduce=False)
        loss_pick_2 = F.cross_entropy(y_2, t, reduce=False)
        loss_pick = loss_pick_1  + loss_pick_2

        ind_sorted = torch.argsort(loss_pick)
        loss_sorted = loss_pick[ind_sorted]
        forget_rate = self.increment*epoch
        remember_rate = 1 - forget_rate
        mean_v = loss_sorted.mean()
        idx_small = torch.where(loss_sorted<mean_v)[0]
      
        remember_rate_small = idx_small.shape[0]/t.shape[0]
       
        remember_rate = max(remember_rate,remember_rate_small)
        num_remember = int(remember_rate * len(loss_sorted))
        ind_update = ind_sorted[:num_remember]
    
        loss_clean = torch.sum(loss_pick[ind_update])/y_1.shape[0]
        ind_all = torch.arange(1, t.shape[0]).long()
        ind_update_1 = torch.LongTensor(list(set(ind_all.detach().cpu().numpy())-set(ind_update.detach().cpu().numpy())))
        p_1 = F.softmax(y_1,dim=-1)
        p_2 = F.softmax(y_2,dim=-1)
        
        filter_condition = ((y_1.max(dim=1)[1][ind_update_1] != t[ind_update_1]) &
                            (y_1.max(dim=1)[1][ind_update_1] == y_2.max(dim=1)[1][ind_update_1]) &
                            (p_1.max(dim=1)[0][ind_update_1] * p_2.max(dim=1)[0][ind_update_1] > (1-(1-min(0.5,1/y_1.shape[0]))*epoch/self.args.epochs)))
        dc_idx = ind_update_1[filter_condition]
        
        adpative_weight = (p_1.max(dim=1)[0][dc_idx]*p_2.max(dim=1)[0][dc_idx])**(0.5-0.5*epoch/self.args.epochs)
        loss_dc = adpative_weight*(F.cross_entropy(y_1[dc_idx],y_1.max(dim=1)[1][dc_idx], reduce=False)+ \
                                   F.cross_entropy(y_2[dc_idx], y_1.max(dim=1)[1][dc_idx], reduce=False))
        loss_dc = loss_dc.sum()/y_1.shape[0]
    
        remain_idx = torch.LongTensor(list(set(ind_update_1.detach().cpu().numpy())-set(dc_idx.detach().cpu().numpy())))
        
        loss1 = torch.sum(loss_pick[remain_idx])/y_1.shape[0]
        decay_w = self.decay_w

        inter_view_loss = kl_loss_compute(y_1, y_2).mean() +  kl_loss_compute(y_2, y_1).mean()

        return loss_clean + loss_dc+decay_w*loss1+co_lambda*inter_view_loss


class PseudoLoss(nn.Module):
    def __init__(self):
        super(PseudoLoss, self).__init__()

    def forward(self, y_1, y_2, idx_add,co_lambda=0.1):
        pseudo_label = y_1.max(dim=1)[1]
        loss_pick_1 = F.cross_entropy(y_1[idx_add], pseudo_label[idx_add], reduce=False)
        loss_pick_2 = F.cross_entropy(y_2[idx_add], pseudo_label[idx_add], reduce=False)
        loss_pick = loss_pick_1.mean() + loss_pick_2.mean()
        inter_view_loss = kl_loss_compute(y_1[idx_add], y_2[idx_add]).mean() + kl_loss_compute(y_2[idx_add], y_1[idx_add]).mean()
        loss = torch.mean(loss_pick)+co_lambda*inter_view_loss

        return loss

class IntraviewReg(nn.Module):
    def __init__(self,device='cuda'):
        super(IntraviewReg, self).__init__()
        self.device = device
    def index_to_mask(self, index, size=None):
        index = index.view(-1)
        size = int(index.max()) + 1 if size is None else size
        mask = index.new_zeros(size, dtype=torch.bool)
        mask[index] = True
        return mask

    def bipartite_subgraph(self,subset, edge_index, max_size):

        subset = (self.index_to_mask(subset[0], size=max_size), self.index_to_mask(subset[1], size=max_size))
        node_mask = subset
        edge_mask = node_mask[0][edge_index[0]] & node_mask[1][edge_index[1]]
        return torch.where(edge_mask == True)[0]

    def neighbor_cons(self,y_1,y_2,edge_index,edge_weight,idx):
        if idx.shape[0]==0:
            return torch.Tensor([0]).to(self.device)
        weighted_adj = utils.to_scipy_sparse_matrix(edge_index, edge_weight.detach())
        colsum = np.array(weighted_adj.sum(0))
        r_inv = np.power(colsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0
        r_mat_inv = sp.diags(r_inv)
        norm_adj = weighted_adj.dot(r_mat_inv)

        norm_idx, norm_weight = utils.from_scipy_sparse_matrix(norm_adj)
        idx_all = torch.arange(0, y_1.shape[0]).to(self.device)

        filter_idx = self.bipartite_subgraph((idx_all,idx), norm_idx.to(self.device),max_size=int(y_1.shape[0]))
        edge_index,edge_weight = norm_idx[:,filter_idx], norm_weight[filter_idx]
        edge_index, edge_weight = edge_index.to(self.device), edge_weight.to(self.device)

        intra_view_loss = (edge_weight*kl_loss_compute(y_1[edge_index[1]], y_1[edge_index[0]].detach())).sum()+ \
                        (edge_weight*kl_loss_compute(y_2[edge_index[1]], y_2[edge_index[0]].detach())).sum()
        intra_view_loss = intra_view_loss/idx.shape[0]
        return intra_view_loss

    def forward(self,y_1,y_2,idx_label,edge_index,edge_weight):
        neighbor_kl_loss = self.neighbor_cons(y_1, y_2, edge_index, edge_weight, idx_label)
        return neighbor_kl_loss


class RTGNN(nn.Module):
    def __init__(self, args, device):
        super(RTGNN, self).__init__()
        self.device = device
        self.args = args
        self.best_val_acc = 0
        self.best_acc_pred_val = 0
        self.best_pred = None
        self.best_graph = None
        self.weights = None
        self.estimator = None
        self.model = None
        self.pred_edge_index = None
        self.criterion = LabeledDividedLoss(args)
        self.criterion_pse = PseudoLoss()
        self.intra_reg = IntraviewReg()


    def fit(self, features, adj, labels, true_labels, idx_train, idx_val, noise_idx, clean_idx):
        self.features_diff = torch.cdist(features, features, 2)
        args = self.args

        edge_index, _ = utils.from_scipy_sparse_matrix(adj)
        edge_index = edge_index.to(self.device)

        if sp.issparse(features):
            features = sparse_mx_to_torch_sparse_tensor(features).to_dense().float()
        else:
            features = torch.FloatTensor(np.array(features))
        features = features.to(self.device)
        
        self.edge_index = edge_index
        self.features = features
        labels = torch.LongTensor(np.array(labels)).to(self.device)
        self.labels = labels
        self.true_labels = true_labels
        self.idx_unlabel = torch.LongTensor(list(set(range(features.shape[0])) - set(idx_train))).to(self.device)
        self.idx_train = torch.LongTensor(idx_train).to(self.device)
        self.predictor = Dual_GCN(nfeat=features.shape[1],
                                  nhid=self.args.hidden,
                                  nclass=labels.max().item() + 1,
                                  self_loop=True,
                                  dropout=self.args.dropout, device=self.device).to(self.device)

        self.estimator = EstimateAdj(features, features.shape[1], args, device=self.device).to(self.device)

        self.pred_edge_index = self.KNN(edge_index, features, self.args.K, idx_train)
        self.optimizer = optim.Adam(list(self.estimator.parameters()) + list(self.predictor.parameters()),
                                    lr=args.lr, weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            self.train(epoch, features, edge_index, idx_train, idx_val, noise_idx, clean_idx)

        print("Optimization Finished!")
      
        print("picking the best model according to validation performance")

        self.predictor.load_state_dict(self.predictor_model_weigths)


    def sparse_mx_to_torch_sparse_tensor(self, sparse_mx):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)


    def train(self, epoch, features, edge_index, idx_train, idx_val, noise_idx, clean_idx):
        args = self.args
        self.predictor.train()
        self.optimizer.zero_grad()
        representations, rec_loss = self.estimator(edge_index, features)
        pred_edge_index = torch.cat([edge_index, self.pred_edge_index], dim=1)
        origin_w = torch.cat([torch.ones(edge_index.shape[1]), torch.zeros(self.pred_edge_index.shape[1])]).to(
            self.device)

        predictor_weights,_ = self.estimator.get_estimated_weigths(pred_edge_index, representations, origin_w)
        edge_remain_idx = torch.where(predictor_weights!=0)[0].detach()
        predictor_weights = predictor_weights[edge_remain_idx]
        pred_edge_index = pred_edge_index[:,edge_remain_idx]

        log_pred, log_pred_1 = self.predictor(features, pred_edge_index, predictor_weights)
        acc_pred_train0 = accuracy(log_pred[idx_train], self.labels[idx_train])
        acc_pred_train1 = accuracy(log_pred_1[idx_train], self.labels[idx_train])
        
        print("=====Train Accuray=====")
        print("Epoch %d: #1 = %f, #2= %f"%(epoch,acc_pred_train0.item(),acc_pred_train1.item()))


        pred = F.softmax(log_pred, dim=1).detach()
        pred1 = F.softmax(log_pred_1, dim=1).detach()

        self.idx_add= self.get_pseudo_label(pred, pred1)
        if epoch==0:
            loss_pred = F.cross_entropy(log_pred[idx_train],self.labels[idx_train])+F.cross_entropy(log_pred_1[idx_train],self.labels[idx_train])
        else:
            loss_pred = self.criterion(log_pred[idx_train], log_pred_1[idx_train], self.labels[idx_train],
                                                    co_lambda=self.args.co_lambda, epoch=epoch)

        if len(self.idx_add) != 0:
            loss_add = self.criterion_pse(log_pred, log_pred_1, self.idx_add, co_lambda=self.args.co_lambda)
        else:
            loss_add = torch.Tensor([0]).to(self.device)

        neighbor_kl_loss = self.intra_reg(log_pred,log_pred_1,self.idx_train, pred_edge_index,predictor_weights)
        total_loss = loss_pred + self.args.alpha * rec_loss + loss_add + self.args.co_lambda*(neighbor_kl_loss)
        total_loss.backward()
        self.optimizer.step()

        self.predictor.eval()
        output0, output1 = self.predictor(features, pred_edge_index, predictor_weights)
        acc_pred_val0 = accuracy(output0[idx_val], self.labels[idx_val])
        acc_pred_val1 = accuracy(output1[idx_val], self.labels[idx_val])
        acc_pred_val = 0.5*(acc_pred_val0+acc_pred_val1)

        if acc_pred_val >= self.best_acc_pred_val:
            self.best_acc_pred_val = acc_pred_val
            self.best_pred_graph = predictor_weights.detach()
            self.best_edge_idx = pred_edge_index.detach()
            self.best_pred = pred.detach()
            self.predictor_model_weigths = deepcopy(self.predictor.state_dict())
        print("=====Validation Accuray=====")
        print("Epoch %d: #1 = %f, #2= %f"%(epoch,acc_pred_val0.item(),acc_pred_val1.item()))


    def test(self, idx_test):
        features = self.features
        self.predictor.eval()
        estimated_weights = self.best_pred_graph
        pred_edge_index = self.best_edge_idx
        output0, output1 = self.predictor(features, pred_edge_index, estimated_weights)
        acc_pred_test0 = accuracy(output0[idx_test], self.labels[idx_test])
        acc_pred_test1 = accuracy(output1[idx_test], self.labels[idx_test])
        
        print("Test Accuray: #1 = %f, #2= %f"%(acc_pred_test0.item(),acc_pred_test1.item()))
        return (acc_pred_test0+acc_pred_test1)/2

    def KNN(self, edge_index, features, K, idx_train):
        if K == 0:
            return torch.LongTensor([])

        poten_edges = []
        if K > len(idx_train):
            for i in range(len(features)):
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        else:
            for i in idx_train:
                sim = torch.div(torch.matmul(features[i], features[self.idx_unlabel].T),
                                features[i].norm() * features[self.idx_unlabel].norm(dim=1))
                _, rank = sim.topk(K)
                indices = self.idx_unlabel[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
            for i in self.idx_unlabel:
                sim = torch.div(torch.matmul(features[i], features[idx_train].T),
                                features[i].norm() * features[idx_train].norm(dim=1))
                _, rank = sim.topk(K)
                indices = idx_train[rank.cpu().numpy()]
                for j in indices:
                    pair = [i, j]
                    poten_edges.append(pair)
        edge_index = list(edge_index.T)
        poten_edges = set([tuple(t) for t in poten_edges])-set([tuple(t) for t in edge_index])
        poten_edges = [list(s) for s in poten_edges]
        poten_edges = torch.as_tensor(poten_edges).T.to(self.device)
      
        return poten_edges

    def get_pseudo_label(self, pred0, pred1):
        filter_condition = ((pred0.max(dim=1)[1][self.idx_unlabel] == pred1.max(dim=1)[1][self.idx_unlabel])&
                            (pred0.max(dim=1)[0][self.idx_unlabel]*pred1.max(dim=1)[0][self.idx_unlabel] > self.args.th**2))
        idx_add = self.idx_unlabel[filter_condition]

        return idx_add.detach()



class EstimateAdj(nn.Module):

    def __init__(self, features, nfea, args, device='cuda'):
        super(EstimateAdj, self).__init__()

        self.estimator = GCN(nfea, args.edge_hidden, args.edge_hidden, dropout=0.0, device=device)
        self.device = device
        self.args = args
        self.representations = 0
        self.sigmoid = nn.Sigmoid()

    def forward(self, edge_index, features):
        representations = self.estimator(features, edge_index, \
                                         torch.ones([edge_index.shape[1]]).to(self.device).float())
        representations =F.normalize(representations,dim=-1)
        rec_loss = self.reconstruct_loss(edge_index, representations)
        return representations, rec_loss

    def get_estimated_weigths(self, edge_index, representations, origin_w=None):
        x0 = representations[edge_index[0]]
        x1 = representations[edge_index[1]]
        output = torch.sum(torch.mul(x0, x1), dim=1)
        estimated_weights = F.relu(output)
        if estimated_weights.shape[0] != 0:
            estimated_weights[estimated_weights < self.args.tau] = 0
            if origin_w != None:
                estimated_weights = origin_w+estimated_weights*(1-origin_w)

        return estimated_weights,None

    def reconstruct_loss(self, edge_index, representations):
        num_nodes = representations.shape[0]
        randn = utils.negative_sampling(edge_index, num_nodes=num_nodes,
                                        num_neg_samples=self.args.n_neg * num_nodes)

        randn = randn[:, randn[0] < randn[1]]
        edge_index = edge_index[:, edge_index[0] < edge_index[1]]
        neg0 = representations[randn[0]]
        neg1 = representations[randn[1]]
        neg = torch.sum(torch.mul(neg0, neg1), dim=1)

        pos0 = representations[edge_index[0]]
        pos1 = representations[edge_index[1]]
        pos = torch.sum(torch.mul(pos0, pos1), dim=1)
        rec_loss = (F.mse_loss(neg, torch.zeros_like(neg), reduction='sum') \
                    + F.mse_loss(pos, torch.ones_like(pos), reduction='sum')) \
                   * num_nodes / (randn.shape[1] + edge_index.shape[1])

        return rec_loss


