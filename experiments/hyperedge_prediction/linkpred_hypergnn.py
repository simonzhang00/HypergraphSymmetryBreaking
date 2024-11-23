import time
from copy import deepcopy
from tqdm import tqdm as tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def _construct_graph_from_networkx(g, nx_nodes, nx_edges, vertex_attr_hashable : str = "_nx_name"):
    """Converts the graph from networkx

    Vertex names will be stored as a vertex_attr_hashable attribute (usually
    "_nx_name", but see below). Because igraph stored vertices in an
    ordered manner, vertices will get new IDs from 0 up. In case of
    multigraphs, each edge will have an "_nx_multiedge_key" attribute, to
    distinguish edges that connect the same two vertices.

    @param g: networkx Graph or DiGraph
    @param vertex_attr_hashable: attribute used to store the Python
        hashable used by networkx to identify each vertex. The default value
        '_nx_name' ensures lossless round trip conversions to/from networkx. An
        alternative choice is 'name': in that case, using strings for vertex
        names is recommended and, if the graph is re-exported to networkx,
        Graph.to_networkx(vertex_attr_hashable = "name") must be used to recover
        the correct vertex nomenclature in the exported network.

    """

    # Graph attributes
    gattr = dict(g.graph)

    # Nodes
    vnames = nx_nodes
    vattr = {vertex_attr_hashable: vnames}
    vcount = len(vnames)

    # Dictionary connecting networkx hashables with igraph indices
    vd = {v: i for i, v in enumerate(vnames)}

    # NOTE: we do not need a special class for multigraphs, it is taken
    # care for at the edge level rather than at the graph level.
    graph = ig.Graph(
        n = vcount, directed = False, graph_attrs = gattr, vertex_attrs = vattr
    )

    # Vertex attributes

    # Edges and edge attributes
    eattr_names = {name for (_, _, data) in g.edges.data() for name in data}
    eattr = {name: [] for name in eattr_names}
    edges = []
    # Multigraphs need a hidden attribute for multiedges
    for (u,v) in nx_edges:
        edges.append((vd[u], vd[v]))

    graph.add_edges(edges)

    return graph

class VertexSetAgg(nn.Module):
    """
    Class that implements the set learner on a vertex set.
    """
    def __init__(self, hid_channels, weight_decay):
        super().__init__()
        self.lin1 = torch.nn.Linear(hid_channels, hid_channels)
        self.lin2 = torch.nn.Linear(hid_channels, 2)
        self.relu = torch.nn.ReLU()

    def compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.tensor([[1, 0]] * (pos_score.shape[0])), torch.tensor([[0, 1]] * (neg_score.shape[0]))]).to(
            device).type(torch.float)
        return F.binary_cross_entropy_with_logits(scores, labels)

    def forward(self, H, pos_ex, neg_ex):
        pos_scores = self.lin2(self.relu(self.lin1(torch.sum(H[pos_ex], dim = 1))))
        neg_scores = self.lin2(self.relu(self.lin1(torch.sum(H[neg_ex], dim = 1))))

        loss = self.compute_loss(pos_scores, neg_scores)

        return loss

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
        pos_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[pos_ex_val], dim = 1))))
        neg_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[neg_ex_val], dim = 1))))
        scores = torch.cat([F.softmax(pos_score, dim = 1), F.softmax(neg_score, dim = 1)])
        labels = torch.cat(
            [torch.tensor([[1, 0]] * (pos_score.shape[0])), torch.tensor([[0, 1]] * (neg_score.shape[0]))]).to(
            device).type(torch.float)

        return utils_hypergraph.classification_score_from_y_full(labels.detach().cpu().numpy(),
                                                                 scores.detach().cpu().numpy(), nruns = 50)


@torch.no_grad()
def test(net, X, G, pos_ex_te, neg_ex_te, criterion):
    net.eval()
    with torch.no_grad():
        H = net(X, G)
        pos_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[pos_ex_te], dim = 1))))
        neg_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[neg_ex_te], dim = 1))))

        scores = torch.cat([F.softmax(pos_score, dim = 1), F.softmax(neg_score, dim = 1)])
        labels = torch.cat(
            [torch.tensor([[1, 0]] * (pos_score.shape[0])), torch.tensor([[0, 1]] * (neg_score.shape[0]))]).to(
            device).type(torch.float)

        return utils_hypergraph.classification_score_from_y_full(labels.detach().cpu().numpy(),
                                                                 scores.detach().cpu().numpy(), nruns = 50)


def find_open_triangles(G):
    r"""
    Finds all the 3-cycles of a networkx G
    """
    tris = []
    for t in nx.enumerate_all_cliques(G):
        if len(t) > 3:
            break
        elif len(t) == 3:
            tris.append(t)
    return tris


def power_iteration(
        P: torch.sparse_coo_tensor,
        alpha: float = 0.05,
        max_iter: int = 1000,
        use_tqdm: bool = False,
        epsilon: float = 1.0e-03,
        device = None,
) -> torch.Tensor:
    r"""
    Perform the power iteration.
    .. math::
        \mathbf{x}^{(i+1)} = (1 - \alpha) \cdot \mathbf{A} \mathbf{x}^{(i)} + \alpha \mathbf{x}^{(0)}
    :param adj: shape: ``(n, n)``
        the (sparse) adjacency matrix
    :param x0: shape: ``(n,)``, or ``(n, batch_size)``
        the initial value for ``x``.
    :param alpha: ``0 < alpha < 1``
        the smoothing value / teleport probability
    :param max_iter: ``0 < max_iter``
        the maximum number of iterations
    :param epsilon: ``epsilon > 0``
        a (small) constant to check for convergence
    :param use_tqdm:
        whether to use a tqdm progress bar
    :param device:
        the device to use, or a hint thereof, cf. :func:`resolve_device`
    :return: shape: ``(n,)`` or ``(n, batch_size)``
        the ``x`` value after convergence (or maximum number of iterations).
    """
    P = P.to(device = device).to_dense()

    # power iteration
    P_old = P
    beta = 1.0 - alpha
    progress = tqdm(range(max_iter), unit_scale = True, leave = False, disable = not use_tqdm)
    for i in progress:
        # calculate x = (1 - alpha) * A.dot(x) + alpha * x0

        P = torch.mm(
            # sparse matrix to be multiplied
            P,
            P_old
        )

        diff = torch.max(torch.abs(P - P_old))
        mask = diff > epsilon

        if not mask.any():
            break
        P_old = P

    return P


def construct_hypertransition_matrix(edge_list, num_vertices, device, alpha = 0.0000001):
    """
    .. math::
        P = D^{−1}_v\cdot H\cdot D^{−1}_e \cdot H^T
    """
    r = list(np.hstack(np.array(edge_list, dtype = object)))
    rr = torch.tensor(r).to(device)
    c = list(np.hstack([len(e) * [num_vertices + i] for i, e in enumerate(edge_list)]))
    cc = torch.tensor(c).to(device)
    D_v_inv = torch_geometric.utils.degree(rr).to(device).pow_(-1).to(torch.float)

    D_v_inv = torch.sparse_coo_tensor(
        torch.tensor([list(range((D_v_inv).size(0))), list(range((D_v_inv).size(0)))]).to(device), D_v_inv).to(device)
    D_e_inv = torch_geometric.utils.degree(cc).to(device).pow_(-1).to(torch.float)
    D_e_inv = torch.sparse_coo_tensor(
        torch.tensor([list(range((D_e_inv).size(0))), list(range((D_e_inv).size(0)))]).to(device), D_e_inv).to(device)
    H_sp = torch.sparse_coo_tensor(torch.tensor([r, c]).to(device), torch.ones_like(rr).to(device).to(torch.float))
    H_spT = torch.transpose(H_sp, 0, 1)
    ones = torch.div(torch.ones((num_vertices, num_vertices)).to(torch.float), num_vertices).to(device)

    P = alpha * ones + (1 - alpha) * torch.sparse.mm(torch.sparse.mm(torch.sparse.mm(D_v_inv, H_sp), D_e_inv), H_spT)
    return P.to(device)

def invert_index(list):
    inverted_index = {}
    for i, item in enumerate(list):
        inverted_index[item] = i
    return inverted_index

def load_data(dataset = 'FB15k-237', pos_embedding = "laplacian_emap", tuple_size = 2):
    """
    Constructs train-val-test splits from the datasets. Also computes the positional embeddings.
    """
    data_train, data_val, data_test = utils_graph.make_train_test_data(dataset)
    tuple_size = 2
    graph = True
    all_nodes = (list(set([n for s in data_train + data_val + data_test for n in s if len(s) > 1])))

    data_train = [list(sorted([int(v) for v in s])) for s in data_train if len(s) > 1]
    data_train = sorted(data_train, key = lambda i: (len(i), i))
    data_train_ = [",".join([str(u) for u in s]) for s in data_train]

    train_sizes = [len(list(d)) for d in data_train]
    assert (len(train_sizes) == len(data_train))

    data_val = [list(sorted([int(v) for v in s])) for s in data_val if len(s) > 1]
    data_val = sorted(data_val, key = lambda i: (len(i), i))
    data_val_ = [",".join([str(u) for u in s]) for s in data_val]
    val_sizes = [len(list(d)) for d in data_val]
    assert (len(val_sizes) == len(data_val))

    data_test = [list(sorted([int(v) for v in s])) for s in data_test if len(s) > 1]
    data_test = sorted(data_test, key = lambda i: (len(i), i))
    data_test_ = [",".join([str(u) for u in s]) for s in data_test]
    test_sizes = [len(list(d)) for d in data_test]
    assert (len(test_sizes) == len(data_test))

    train_verts = sorted(list(set(([int(w) for e in data_train for w in e]))))

    num_nodes = int(np.max(np.array(all_nodes, dtype = int))) + 1
    train_mask = np.zeros((num_nodes, 1))
    train_mask[[int(u) for u in train_verts]] = 1

    val_verts = sorted(list(set(([int(w) for e in data_val for w in e]))))
    val_mask = np.zeros((num_nodes, 1))
    val_mask[[int(u) for u in val_verts]] = 1

    test_verts = sorted(list(set(([int(w) for e in data_test for w in e]))))
    test_mask = np.zeros((num_nodes, 1))
    test_mask[[int(u) for u in test_verts]] = 1

    if graph:
        train_percentage = 0.2
        pos_train_percentage = 0.5
        neg_proportion = 1.2
    else:
        train_percentage = 1.0
        pos_train_percentage = 0.5
        neg_proportion = 1.2

    all_train_tuples = list([d for d in data_train if len(d) == tuple_size])
    all_train_tuples = all_train_tuples[:int(train_percentage * len(all_train_tuples))]
    pos_train = np.array(all_train_tuples[:int(len(all_train_tuples) * pos_train_percentage)])
    pos_train_ = [",".join([str(v) for v in p]) for p in pos_train]

    all_val_tuples = list([d for d in data_val if len(d) == tuple_size])
    pos_val = np.array(all_val_tuples[:int(len(all_val_tuples) * pos_train_percentage)])
    pos_val_ = [",".join([str(v) for v in p]) for p in pos_val]

    all_test_tuples = list([d for d in data_test if len(d) == tuple_size])
    pos_test = np.array(all_test_tuples[:int(len(all_test_tuples) * pos_train_percentage)])
    pos_test_ = [",".join([str(v) for v in p]) for p in pos_test]

    tr_extra_neg_samples = int(
        (int(len(pos_train))* neg_proportion))
    val_extra_neg_samples = int(
        (int(len(pos_val))* neg_proportion))
    te_extra_neg_samples = int(
        (int(len(pos_test))* neg_proportion))

    tr_hard_neg_samples = []
    val_hard_neg_samples = []
    test_hard_neg_samples = []
    neg_tuple_train = utils_hypergraph.tuple_negative_sampling(
        tr_hard_neg_samples, train_verts,
        sample_size = tr_extra_neg_samples, tuple_size = tuple_size)
    neg_tuple_val = utils_hypergraph.tuple_negative_sampling(
        val_hard_neg_samples, val_verts,
        sample_size = val_extra_neg_samples, tuple_size = tuple_size)
    neg_tuple_test = utils_hypergraph.tuple_negative_sampling(
        test_hard_neg_samples, test_verts,
        sample_size = te_extra_neg_samples, tuple_size = tuple_size)

    pos_tuples_ = (set((pos_train_)) | set((pos_val_)) | set((pos_test_)))
    pos_tuples = [tuple([int(u) for u in p.split(",")]) for p in pos_tuples_]
    neg_train = np.array(list(set(neg_tuple_train) - set(pos_tuples)), dtype = object)
    neg_val = np.array(list(set(neg_tuple_val) -
                            set(pos_tuples)
                            ), dtype = object)

    neg_test = np.array(list(set(neg_tuple_test) -
                             set(pos_tuples)
                             ), dtype = object)
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
    hyperedge_list = ([s for s in open_train_+open_val_+open_test_ if len(s) > 1])

    edge_list = [(int(u), int(v)) for s in data_train for u in s for v in s if int(u) < int(v)]
    edge_index = torch.transpose(torch.tensor(edge_list), 0, 1)

    if pos_embedding == "id":
        Q = torch.eye(num_nodes).to(device)
    elif pos_embedding == "laplacian_emap":
        lap_edge_index, lap_edge_attr = torch_geometric.utils.get_laplacian(edge_index, normalization = 'rw',
                                                                            num_nodes = num_nodes)

        laplacian = torch_geometric.utils.to_dense_adj(edge_index = lap_edge_index, edge_attr = lap_edge_attr).to('cpu')

        L, Q = torch.linalg.eigh(laplacian)
        Q = Q[L != 0]
        Q = torch.transpose(Q, 0, 1).to(device)

    return {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask, "num_vertices": len(train_mask),
            "neg_examples_tr": negative_ex_tr, "pos_examples_tr": positive_ex_tr, "neg_examples_val": negative_ex_val,
            "pos_examples_val": positive_ex_val, "neg_examples_te": negative_ex_te, "pos_examples_te": positive_ex_te,
            "hyperedge_list": hyperedge_list,
            "pos_embedding": Q}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'SymmetryBreaker')
    parser.add_argument('-hgnn', '--hgnn', type = str,
                        choices = ['HGNN', 'HGNNP', 'HNHN', 'HyperGCN', 'UniGCN', 'UniGIN', 'UniSAGE', 'UniGAT'],
                        default = 'HGNNP')
    parser.add_argument('-dataset', '--dataset', type = str,
                        choices = ['FB15k-237',"penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius", "AIFB", "MUTAG"],
                        help = 'Dataset to run on.', default = 'FB15k-237')
    parser.add_argument('-pos_embedding', '--pos_embedding', type = str,
                        choices = ['laplacian_emap', 'id'], default = 'laplacian_emap')
    parser.add_argument('-epochs', '--epochs', type = int, default = 2000)
    parser.add_argument('-max_iter', '--max_iter', type = int, default = 2)
    parser.add_argument('-tuple_size', '--tuple_size', type = int, default = 2)
    parser.add_argument('-embdim', '--embdim', type = int, default = 1024)
    parser.add_argument('-drop_rate', '--drop_rate', type = float, default = 0.0)
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
    data = load_data(dataset = args.dataset, pos_embedding = args.pos_embedding, tuple_size = 2)
    start_train = time.time()
    X = data["pos_embedding"].to(device)
    train_mask = torch.tensor(data['train_mask']).to(device).to(torch.float).expand(-1, X.shape[1])
    val_mask = torch.tensor(data['val_mask']).to(device).to(torch.float).expand(-1, X.shape[1])
    test_mask = torch.tensor(data['test_mask']).to(device).to(torch.float).expand(-1, X.shape[1])

    G = Hypergraph(data["num_vertices"], data["hyperedge_list"]).to(device)
    pos_ex_tr = torch.tensor(data['pos_examples_tr']).to(device)
    neg_ex_tr = torch.tensor(data['neg_examples_tr']).to(device)
    pos_ex_val = torch.tensor(data['pos_examples_val']).to(device)
    neg_ex_val = torch.tensor(data['neg_examples_val']).to(device)
    pos_ex_te = torch.tensor(data['pos_examples_te']).to(device)
    neg_ex_te = torch.tensor(data['neg_examples_te']).to(device)
    
    if args.hgnn == 'HGNN':
        net = HGNN(X.shape[1], args.embdim, use_bn = True, num_classes = 2)
    elif args.hgnn == 'HNHN':
        net = HNHN(X.shape[1], args.embdim, use_bn = True, num_classes = 2)
    elif args.hgnn == 'HyperGCN':
        net = HyperGCN(X.shape[1], args.embdim, use_bn = True, num_classes = 2)
    elif args.hgnn == 'UniGCN':
        net = UniGCN(X.shape[1], args.embdim, use_bn = True, num_classes = 2)
    elif args.hgnn == 'UniGIN':
        net = UniGIN(X.shape[1], args.embdim, use_bn = True, num_classes = 2)
    elif args.hgnn == 'UniSAGE':
        net = UniSAGE(X.shape[1], args.embdim, use_bn = True, num_classes = 2)
    elif args.hgnn == 'UniGAT':
        net = UniGAT(X.shape[1], args.embdim, num_heads = 5, use_bn = True, num_classes = 2)
    else:
        net = HGNNP(X.shape[1], args.embdim, use_bn = True, num_classes = 2)

    net = net.to(device)
    criterion = VertexSetAgg(args.embdim, weight_decay).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr = lr)

    best_state, best_val, best_epoch = None, 0, -1
    for epoch in range(epoch_max):
        train(net, X, G, pos_ex_tr, neg_ex_tr, optimizer, criterion, epoch)
        if epoch % val_freq == 0:
            val_res = validate(net, X, G, pos_ex_val, neg_ex_val, criterion)

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
    test_res = test(net, X, G, pos_ex_te, neg_ex_te, criterion)
    print(f"Test: PR-AUC,ROC-AUC -> ", test_res)
    print(start_train - start_time, ", ", time.time() - start_train)
    print(test_res[0])