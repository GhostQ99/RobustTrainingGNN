import os
import argparse
import numpy as np
import torch
from utils import noisify_with_P
from dataset import Dataset
from models.RTGNN import RTGNN

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=12, help='Random seed.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=128,
                    help='Number of hidden units.')
parser.add_argument('--edge_hidden', type=int, default=64,
                    help='Number of hidden units of MLP graph constructor')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--dataset', type=str, default="blogcatalog",
                    choices=['cora', 'citeseer','blogcatalog'], help='dataset')
parser.add_argument('--ptb_rate', type=float, default=0.3,
                    help="noise ptb_rate")
parser.add_argument('--epochs', type=int,  default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--alpha', type=float, default=1,
                    help='loss weight of graph reconstruction')
parser.add_argument('--tau',type=float, default=0.05,
                    help='threshold of filtering noisy edges')
parser.add_argument('--th',type=float, default=0.95,
                    help='threshold of adding pseudo labels')
parser.add_argument("--K", type=int, default=100,
                    help='number of KNN search for each node')
parser.add_argument("--n_neg", type=int, default=100,
                    help='number of negitive sampling for each node')
parser.add_argument('--noise', type=str, default='uniform', choices=['uniform', 'pair'],
                    help='type of noises')
parser.add_argument('--decay_w', type=float, default=0.1,
                    help='down-weighted factor')
parser.add_argument('--co_lambda',type=float,default=0.1,
                     help='weight for consistency regularization term')

args = parser.parse_known_args()[0]
print(args)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = Dataset(root='./data', name=args.dataset)
adj, features, labels = data.adj, data.features, data.labels
idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
ptb = args.ptb_rate
nclass = labels.max() + 1
args.class_num=nclass
train_labels = labels[idx_train]
val_labels = labels[idx_val]
train_val_labels = np.concatenate([train_labels,val_labels],axis=0)
idx = np.concatenate([idx_train,idx_val],axis=0)
noise_y, P, noise_idx, clean_idx = noisify_with_P(train_val_labels,idx_train.shape[0],nclass, ptb, 10, args.noise)
args.noise_idx, args.clean_idx = noise_idx, clean_idx
noise_labels = labels.copy()
noise_labels[idx] = noise_y



np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
model = RTGNN(args, device)
model.fit(features, adj, noise_labels, labels, idx_train, idx_val, noise_idx, clean_idx)
print("===================")
model.test(idx_test)


