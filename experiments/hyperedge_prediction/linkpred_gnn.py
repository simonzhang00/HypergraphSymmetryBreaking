import time
from copy import deepcopy
from tqdm import tqdm as tqdm
import torch.nn as nn
import sys

sys.path.append("../../")
import pickle as pkl
import glob
import argparse
from hsb.random import set_seed
from hsb.utils import utils_simplicial, utils_hypergraph, networkx_cycles, utils_graph, WL1
from hsb import Hypergraph
from hsb.models import HGNN, HGNNP, HNHN, HyperGCN, UniGCN, UniGIN, UniSAGE, UniGAT
import networkx as nx
import numpy as np
import igraph as ig
import torch_geometric
from torch.distributions import Categorical
import os.path as osp
import time

import torch
import torch.nn.functional as F
from torch.nn import Parameter
from tqdm import tqdm

from torch_geometric.nn import GAE, GCNConv, SAGEConv, GATConv, GINConv, APPNP, GCN2Conv, LEConv
from torch_geometric.nn.models import GIN, GraphSAGE, GAT, GCN

def invert_index(list):
    inverted_index = {}
    for i, item in enumerate(list):
        inverted_index[item] = i
    return inverted_index

def load_data(dataset='FB15k-237', pos_embedding="laplacian_emap", tuple_size=2):
    data_train, data_val, data_test = utils_graph.make_train_test_data(dataset)
    tuple_size = 2

    all_nodes = (list(set([n for s in data_train + data_val + data_test for n in s if len(s) > 1])))

    data_train= [list(sorted([int(v) for v in s])) for s in data_train if len(s) > 1]
    data_train= sorted(data_train, key = lambda i: (len(i), i))
    data_train_ = [",".join([str(u) for u in s]) for s in data_train]
    train_sizes = [len(list(d)) for d in data_train]
    assert (len(train_sizes) == len(data_train))

    data_val= [list(sorted([int(v) for v in s])) for s in data_val if len(s) > 1]
    data_val= sorted(data_val, key = lambda i: (len(i), i))
    data_val_ = [",".join([str(u) for u in s]) for s in data_val]
    val_sizes = [len(list(d)) for d in data_val]
    assert (len(val_sizes) == len(data_val))

    data_test= [list(sorted([int(v) for v in s])) for s in data_test if len(s) > 1]
    data_test= sorted(data_test, key = lambda i: (len(i), i))
    data_test_ = [",".join([str(u) for u in s]) for s in data_test]
    test_sizes = [len(list(d)) for d in data_test]
    assert (len(test_sizes) == len(data_test))
    train_verts = sorted(list(set(([int(w) for e in data_train for w in e]))))

    num_nodes = int(np.max(np.array(all_nodes, dtype=int))) + 1
    train_mask = np.zeros((num_nodes, 1))
    train_mask[[int(u) for u in train_verts]] = 1

    val_verts = sorted(list(set(([int(w) for e in data_val for w in e]))))
    val_mask = np.zeros((num_nodes, 1))
    val_mask[[int(u) for u in val_verts]] = 1

    test_verts = sorted(list(set(([int(w) for e in data_test for w in e]))))
    test_mask = np.zeros((num_nodes, 1))
    test_mask[[int(u) for u in test_verts]] = 1

    train_percentage = 0.2
    pos_train_percentage = 0.5
    neg_proportion= 1.2

    all_train_tuples = list([d for d in data_train if len(d) == tuple_size])
    all_train_tuples = all_train_tuples[:int(train_percentage * len(all_train_tuples))]
    pos_train = np.array(all_train_tuples[:int(len(all_train_tuples) * pos_train_percentage)])
    pos_train_= [",".join([str(v) for v in p]) for p in pos_train]

    all_val_tuples = list([d for d in data_val if len(d) == tuple_size])
    pos_val = np.array(all_val_tuples[:int(len(all_val_tuples) * pos_train_percentage)])
    pos_val_= [",".join([str(v) for v in p]) for p in pos_val]

    all_test_tuples = list([d for d in data_test if len(d) == tuple_size])
    pos_test = np.array(all_test_tuples[:int(len(all_test_tuples) * pos_train_percentage)])
    pos_test_= [",".join([str(v) for v in p]) for p in pos_test]

    tr_extra_neg_samples = int(
        (int(len(pos_train))* neg_proportion))
    val_extra_neg_samples = int(
        (int(len(pos_val))* neg_proportion))
    te_extra_neg_samples = int(
        (int(len(pos_test))* neg_proportion))

    tr_hard_neg_samples= []
    val_hard_neg_samples= []
    test_hard_neg_samples= []
    neg_tuple_train = utils_hypergraph.tuple_negative_sampling(
        tr_hard_neg_samples, train_verts,
        sample_size=tr_extra_neg_samples, tuple_size=tuple_size)
    neg_tuple_val = utils_hypergraph.tuple_negative_sampling(
        val_hard_neg_samples, val_verts,
        sample_size=val_extra_neg_samples, tuple_size=tuple_size)
    neg_tuple_test = utils_hypergraph.tuple_negative_sampling(
        test_hard_neg_samples, test_verts,
        sample_size=te_extra_neg_samples, tuple_size=tuple_size)
    pos_tuples_ = (set((pos_train_)) | set((pos_val_)) | set((pos_test_)))
    pos_tuples= [tuple([int(u) for u in p.split(",")]) for p in pos_tuples_]
    neg_train = np.array(list(set(neg_tuple_train) - set(pos_tuples)), dtype=object)

    neg_val = np.array(list(set(neg_tuple_val) -
                            set(pos_tuples)
                            ), dtype=object)

    neg_test = np.array(list(set(neg_tuple_test) -
                             set(pos_tuples)
                             ), dtype=object)

    positive_ex_tr = np.array([[int(v) for v in x] for x in pos_train]).astype(int)
    negative_ex_tr = np.array([[int(v) for v in x] for x in neg_train]).astype(int)

    positive_ex_val = np.array([[int(v) for v in x] for x in pos_val]).astype(int)
    negative_ex_val = np.array([[int(v) for v in x] for x in neg_val]).astype(int)

    positive_ex_te = np.array([[int(v) for v in x] for x in pos_test]).astype(int)
    negative_ex_te = np.array([[int(v) for v in x] for x in neg_test]).astype(int)

    open_train__ = sorted(list(set(data_train_) - set(list(pos_train_))))
    open_train_ = [[int(s) for s in t.split(",")] for t in open_train__]

    open_val__ = sorted(list(set(data_val_) - set(list(pos_val_))))
    open_val_ = [[int(s) for s in t.split(",")] for t in open_val__]

    open_test__ = sorted(list(set(data_test_) - set(list(pos_test_))))
    open_test_ = [[int(s) for s in t.split(",")] for t in open_test__]

    open_train_verts = sorted(list(set([int(v) for s in open_train_ for v in s])))
    max_open_train_verts= np.max(open_train_verts)+1

    open_val_verts = sorted(list(set([int(v) for s in open_val_ for v in s])))
    max_open_val_verts= np.max(open_val_verts)+1

    open_test_verts = sorted(list(set([int(v) for s in open_test_ for v in s])))
    max_open_test_verts= np.max(open_test_verts)+1
    
    edge_list = [(int(u), int(v)) for s in open_train_+open_val_+open_test_ for u in s for v in s if int(u) < int(v)]
    edge_index = torch.transpose(torch.tensor(edge_list), 0, 1)

    if pos_embedding == "id":
        Q = torch.eye(num_nodes).to(device)
    elif pos_embedding == "laplacian_emap":
        lap_edge_index, lap_edge_attr = torch_geometric.utils.get_laplacian(edge_index, normalization='rw',
                                                                            num_nodes=num_nodes)

        laplacian = torch_geometric.utils.to_dense_adj(edge_index=lap_edge_index, edge_attr=lap_edge_attr).to('cpu')

        L, Q = torch.linalg.eigh(laplacian)
        Q = Q[L != 0]
        Q = torch.transpose(Q, 0, 1).to(device)

    return {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask, 
            "neg_examples_tr": negative_ex_tr, "pos_examples_tr": positive_ex_tr, "neg_examples_val": negative_ex_val,
            "pos_examples_val": positive_ex_val, "neg_examples_te": negative_ex_te, "pos_examples_te": positive_ex_te,
            "edge_index": edge_index,
            "pos_embedding": Q}

class EdgeAgg(nn.Module):
    def __init__(self, hid_channels, weight_decay):
        super().__init__()
        self.lin1 = torch.nn.Linear(hid_channels, hid_channels)
        self.lin2 = torch.nn.Linear(hid_channels, 2)
        self.relu = torch.nn.ReLU()

    def compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.tensor([[1, 0]] * (pos_score.shape[0])), torch.tensor([[0, 1]] * (neg_score.shape[0]))]).to(device).type(torch.float)
        return F.binary_cross_entropy_with_logits(scores, labels)
    
    def forward(self, H, pos_ex, neg_ex):
        pos_scores= (H[pos_ex[:,0]] * H[pos_ex[:,1]])
        neg_scores = (H[neg_ex[:,0]] * H[neg_ex[:,1]])
        loss = self.compute_loss(pos_scores, neg_scores)

        return loss
    
class GNN(torch.nn.Module):
    def __init__(self, in_channels, channels, type= 'GCN', layers=5):
        super().__init__()
        self.type= type
        self.convs= []
        if type=='GCN':
            self.gnn= GCN(in_channels= in_channels, hidden_channels= channels, num_layers= layers, out_channels= 2)
        elif type=='GraphSAGE':
            self.gnn= GraphSAGE(in_channels= in_channels, hidden_channels= channels, num_layers= layers, out_channels= 2)
        elif type=='GAT':
            self.gnn= GAT(in_channels= in_channels, hidden_channels= channels, num_layers= layers, out_channels= 2)
        elif type=='GIN':
            self.gnn= GIN(in_channels= in_channels, hidden_channels= channels, num_layers= layers, out_channels= 2)
        elif type=='APPNP':
            self.mlp1= torch.nn.Linear(in_channels,2)
            self.appnp = APPNP(K=layers, alpha=0.5)
        elif type=='GCN2':
            self.mlp1= torch.nn.Linear(in_channels,2)
            for _ in range(layers):
                self.convs.append(GCN2Conv(2, 0.5))

    def encode(self, x, edge_index):
        if self.type=='GCN2':
            x= self.mlp1(x)
            z= self.convs[0](x,x,edge_index).relu()
            for c in self.convs[1:-1]:
               z=c(z,x,edge_index)
            x= self.convs[-1](z,x,edge_index)
        elif self.type=='APPNP':
            x= self.mlp1(x)
            x=self.appnp(x,edge_index)
        else:
            x= self.gnn(x,edge_index)
        return x

    def forward(self, x, edge_index):
        z = self.encode(x, edge_index)
        return z

def train(net, X, G, pos_ex, neg_ex, optimizer, criterion, epoch):
    net.train()

    loss_mean, st = 0, time.time()
    optimizer.zero_grad()
    H = net(X, G)

    loss = criterion(
        H, pos_ex, neg_ex
    )

    loss.backward()
    optimizer.step()
    loss_mean += loss.item()
    loss_mean /= (pos_ex.shape[0] + pos_ex.shape[0])
    print(f"Epoch: {epoch}, Time: {time.time() - st:.5f}s, Loss: {loss_mean:.5f}")


@torch.no_grad()
def validate(net, X, G, pos_ex_val, neg_ex_val, criterion):
    net.eval()
    with torch.no_grad():
        H = net(X, G)
        pos_score= (H[pos_ex_val[:,0]] * H[pos_ex_val[:,1]])
        neg_score = (H[neg_ex_val[:,0]] * H[neg_ex_val[:,1]])
        scores = torch.cat([F.softmax(pos_score, dim=1), F.softmax(neg_score, dim=1)])
        labels = torch.cat(
            [torch.tensor([[1, 0]] * (pos_score.shape[0])), torch.tensor([[0, 1]] * (neg_score.shape[0]))]).to(
            device).type(torch.float)

        return utils_hypergraph.classification_score_from_y_full(labels.detach().cpu().numpy(),
                                                                 scores.detach().cpu().numpy(), nruns=50)


@torch.no_grad()
def test(net, X, G, pos_ex_te, neg_ex_te, criterion):
    net.eval()
    with torch.no_grad():
        H = net(X, G)
        pos_score= (H[pos_ex_te[:,0]] * H[pos_ex_te[:,1]])
        neg_score = (H[neg_ex_te[:,0]] * H[neg_ex_te[:,1]])

        scores = torch.cat([F.softmax(pos_score, dim=1), F.softmax(neg_score, dim=1)])
        labels = torch.cat(
            [torch.tensor([[1, 0]] * (pos_score.shape[0])), torch.tensor([[0, 1]] * (neg_score.shape[0]))]).to(
            device).type(torch.float)

        return utils_hypergraph.classification_score_from_y_full(labels.detach().cpu().numpy(),
                                                                 scores.detach().cpu().numpy(), nruns=50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SymmetryBreaker')
    parser.add_argument('-gnn', '--gnn', type=str,
                        choices=['GCN', 'GraphSAGE', 'GIN', 'GAT', 'APPNP', 'GCN2'],
                        default='GCN')
    parser.add_argument('-dataset', '--dataset', type=str,
                        choices=['FB15k-237',"penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius", "AIFB", "MUTAG"],
                        help='Dataset to run on.', default='FB15k-237')
    parser.add_argument('-pos_embedding', '--pos_embedding', type=str,
                        choices=['laplacian_emap', 'id'], default='laplacian_emap')
    parser.add_argument('-epochs', '--epochs', type=int, default=2000)
    parser.add_argument('-max_iter', '--max_iter', type=int, default=2)
    parser.add_argument('-tuple_size', '--tuple_size', type=int, default=2)
    parser.add_argument('-embdim', '--embdim', type=int, default=1024)
    parser.add_argument('-layers', '--layers', type=int, default=5)
    parser.add_argument('-drop_rate', '--drop_rate', type=float, default=0.0)
    
    args = (parser.parse_args())

    dim_emb = args.embdim
    lr = 1e-2
    num_workers = 0
    batch_sz = 2048
    val_freq = 20
    epoch_max = args.epochs
    weight_decay = 1e-4
    set_seed(2022)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_time = time.time()
    data = load_data(dataset=args.dataset, pos_embedding=args.pos_embedding, tuple_size=2)
    start_train= time.time()
    X = data["pos_embedding"].to(device)
    train_mask = torch.tensor(data['train_mask']).to(device).to(torch.float).expand(-1, X.shape[1])
    val_mask = torch.tensor(data['val_mask']).to(device).to(torch.float).expand(-1, X.shape[1])
    test_mask = torch.tensor(data['test_mask']).to(device).to(torch.float).expand(-1, X.shape[1])

    edge_index= data['edge_index'].to(device)

    edge_index= torch_geometric.utils.dropout.dropout_edge(edge_index, p= args.drop_rate)[0]

    pos_ex_tr = torch.tensor(data['pos_examples_tr']).to(device)
    neg_ex_tr = torch.tensor(data['neg_examples_tr']).to(device)
    pos_ex_val = torch.tensor(data['pos_examples_val']).to(device)
    neg_ex_val = torch.tensor(data['neg_examples_val']).to(device)
    pos_ex_te = torch.tensor(data['pos_examples_te']).to(device)
    neg_ex_te = torch.tensor(data['neg_examples_te']).to(device)
    net = GNN(X.size(1), args.embdim, args.gnn, layers=args.layers).to(device)

    net = net.to(device)
    criterion = EdgeAgg(args.embdim, weight_decay).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    best_state, best_val, best_epoch = None, 0, -1
    for epoch in range(epoch_max):
        train(net, X, edge_index, pos_ex_tr, neg_ex_tr, optimizer, criterion, epoch)
        if epoch % val_freq == 0:
            val_res = validate(net, X, edge_index, pos_ex_val, neg_ex_val, criterion)

            print(f"Validation: PR-AUC,ROC-AUC -> ", val_res)
            if val_res[0] > best_val:
                best_epoch = epoch
                best_val = val_res[0]
                best_state = deepcopy(net.state_dict())
    print("train finished")
    print(f"best val: ", best_val)
    print(f"best epoch: {best_epoch}")
    print("testing...")
    net.load_state_dict(best_state)
    test_res = test(net, X, edge_index, pos_ex_te, neg_ex_te, criterion)
    print(f"Test: PR-AUC,ROC-AUC -> ", test_res)
    print(start_train - start_time, ", ", time.time() - start_train)
    print(test_res[0])