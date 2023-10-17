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
from hsb.utils import utils_hypergraph, utils_hypergraph, networkx_cycles, utils_graph, WL1
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
        Graph.to_networkx(vertex_attr_hashable="name") must be used to recover
        the correct vertex nomenclature in the exported network.

    """

    # Graph attributes
    gattr = dict(g.graph)

    # Nodes
    vnames= nx_nodes
    vattr = {vertex_attr_hashable: vnames}
    vcount = len(vnames)

    # Dictionary connecting networkx hashables with igraph indices
    if False and len(g) and "_igraph_index" in next(iter(g.nodes.values())):
        # Collect _igraph_index and fill gaps
        idx = [x["_igraph_index"] for v, x in g.nodes.data()]
        idx.sort()
        idx_dict = {x: i for i, x in enumerate(idx)}

        vd = {}
        for v, datum in g.nodes.data():
            vd[v] = idx_dict[datum["_igraph_index"]]
    else:
        vd = {v: i for i, v in enumerate(vnames)}

    # NOTE: we do not need a special class for multigraphs, it is taken
    # care for at the edge level rather than at the graph level.
    graph = ig.Graph(
        n=vcount, directed=False, graph_attrs=gattr, vertex_attrs=vattr
    )

    # Vertex attributes
    # Edges and edge attributes
    eattr_names = {name for (_, _, data) in g.edges.data() for name in data}
    eattr = {name: [] for name in eattr_names}
    edges = []
    # Multigraphs need a hidden attribute for multiedges
    if False and isinstance(g, (nx.MultiGraph, nx.MultiDiGraph)):
        eattr["_nx_multiedge_key"] = []
        for (u, v, edgekey, data) in g.edges.data(keys=True):
            edges.append((vd[u], vd[v]))
            for name in eattr_names:
                eattr[name].append(data.get(name))
            eattr["_nx_multiedge_key"].append(edgekey)

    else:
        #for (u, v, data) in g.edges.data():
        for (u,v) in nx_edges:
            edges.append((vd[u], vd[v]))
            #for name in eattr_names:
            #    eattr[name].append(data.get(name))

    # Sort edges if there is a trace of a previous igraph ordering
    if False and "_igraph_index" in eattr:
        # Poor man's argsort
        sortd = [(i, x) for i, x in enumerate(eattr["_igraph_index"])]
        sortd.sort(key=lambda x: x[1])
        idx = [i for i, x in sortd]

        # Get rid of the _igraph_index now
        del eattr["_igraph_index"]

        # Sort edges
        edges = [edges[i] for i in idx]
        # Sort each attribute
        #eattr = {key: [val[i] for i in idx] for key, val in eattr.items()}

    graph.add_edges(edges)

    return graph


class VertexSetAgg(nn.Module):
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
        pos_scores = self.lin2(self.relu(self.lin1(torch.sum(H[pos_ex], dim=1))))
        neg_scores = self.lin2(self.relu(self.lin1(torch.sum(H[neg_ex], dim=1))))

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
        pos_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[pos_ex_val], dim=1))))
        neg_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[neg_ex_val], dim=1))))
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
        pos_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[pos_ex_te], dim=1))))
        neg_score = criterion.lin2(criterion.relu(criterion.lin1(torch.sum(H[neg_ex_te], dim=1))))

        scores = torch.cat([F.softmax(pos_score, dim=1), F.softmax(neg_score, dim=1)])
        labels = torch.cat(
            [torch.tensor([[1, 0]] * (pos_score.shape[0])), torch.tensor([[0, 1]] * (neg_score.shape[0]))]).to(
            device).type(torch.float)

        return utils_hypergraph.classification_score_from_y_full(labels.detach().cpu().numpy(),
                                                                 scores.detach().cpu().numpy(), nruns=50)

def find_open_triangles(G):
    # G is a networkx graph
    tris = []
    for t in nx.enumerate_all_cliques(G):
        if len(t) > 3:
            break;
        elif len(t) == 3:
            tris.append(t)
    return tris


def power_iteration(
        P: torch.sparse_coo_tensor,
        alpha: float = 0.05,
        max_iter: int = 1000,
        use_tqdm: bool = False,
        epsilon: float = 1.0e-03,
        device=None,
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
    P = P.to(device=device).to_dense()

    # power iteration
    P_old = P
    beta = 1.0 - alpha
    progress = tqdm(range(max_iter), unit_scale=True, leave=False, disable=not use_tqdm)
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


def construct_hypertransition_matrix(edge_list, num_vertices, device, alpha=0.0000001):
    # P = D^{−1}_v\cdot H\cdot D^{−1}_e \cdot H^T
    r = list(np.hstack(np.array(edge_list, dtype=object)))
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

def regsubreplacer_G(open_verts, open_edges_, max_open_verts, max_iter):
    G = nx.Graph()
    G.add_edges_from(open_edges_)
    open_verts= sorted(list(set([v  for e in open_edges_ for v in e]))) 
    num_open_verts = len(open_verts)
    g= _construct_graph_from_networkx(G, open_verts, open_edges_)
    graphs = [g]

    wl = WL1.WeisfeilerLehman()
    for i in range(1, args.max_iter):
        if wl is not None:
            label_dicts = wl.fit_transform(graphs, i)

        # Each entry in the list represents the label sequence of a single
        # graph. The label sequence contains the vertices in its rows, and
        # the individual iterations in its columns.
        #
        # Hence, (i, j) will contain the label of vertex i at iteration j.

        label_sequences = [
            np.full((num_open_verts, args.max_iter + 2), np.nan) for graph in graphs
        ]

        for iteration in sorted(label_dicts.keys())[0::1]:
            for graph_index, graph in enumerate(graphs):
                labels_raw, labels_compressed = label_dicts[iteration][graph_index]
                # Store label sequence of the current iteration, i.e. *all*
                # of the compressed labels.
                label_sequences[graph_index][:, iteration] = labels_compressed
    
    multisets = np.array([str(l) for i, l in enumerate(label_sequences[0])])

    uniq, uniq_indices, all_subhypergraph_indices, counts = np.unique(multisets, return_counts=True,
                                                                      return_inverse=True, return_index=True)
    #print("uniq num classes vs num nodes: ", len(uniq), len(multisets))
    
    assert (len(open_verts) == len(multisets))

    node_class_indices_search = np.array(open_verts)
    node_classes_search = (uniq_indices[all_subhypergraph_indices])

    Eisom_byDeg = set([str(sorted([G.degree()[int(v)] for v in e])) for e in open_edges_])
    reg_nodes = []
    reg_hyperedges = []
    reg_hyperedges_len= []
    num_cc= 0
    for c in node_classes_search:
        Vc= node_class_indices_search[node_classes_search==c]
        G_search = nx.induced_subgraph(G, list(Vc))
        CC = list(nx.connected_components(G_search))
        num_cc += len(CC)
        for cc in CC:
            if len(cc) >= 3:
                C = G_search.subgraph(cc)
                deg_p_nodes = str(sorted([G.degree()[int(v)] for v in C.nodes()]))

                if deg_p_nodes not in Eisom_byDeg:
                    reg_nodes.append([int(i) for i in C.nodes()])
                    reg_hyperedges += [[int(b) for b in e] for e in C.edges()]
                    reg_hyperedges_len+=[len(reg_nodes)-1]*len(C.edges())
    return reg_nodes, reg_hyperedges, num_cc, reg_hyperedges_len

def regsubreplacer(open_verts, open_hyperedges_, max_open_verts, max_iter):
    num_open_verts= len(open_verts)
    B = nx.Graph(directed= True)

    incidences = []
    B_nodes= []
    for i, e in enumerate(open_hyperedges_):
        for v in sorted(e):
            assert(i + max_open_verts > v)
            u = int(v)
            B_nodes.append(u)
            incidences.append((u, i + max_open_verts))

    B.add_edges_from(incidences)
    assert(len(set(B_nodes))==len(open_verts))
    g= _construct_graph_from_networkx(B, open_verts+list(x+ max_open_verts for x in range(len(open_hyperedges_))), incidences)
    graphs = [g]

    gwl = WL1.WeisfeilerLehman()
    for i in range(1, args.max_iter):
        if gwl is not None:
            label_dicts = gwl.fit_transform(graphs, i)

        # Each entry in the list represents the label sequence of a single
        # graph. The label sequence contains the vertices in its rows, and
        # the individual iterations in its columns.
        #
        # Hence, (i, j) will contain the label of vertex i at iteration j.

        label_sequences = [
            np.full((num_open_verts, max_iter + 2), np.nan) for graph in graphs
        ]

        for iteration in sorted(label_dicts.keys())[0::1]:
            for graph_index, graph in enumerate(graphs):
                labels_raw, labels_compressed = label_dicts[iteration][graph_index]
                # Store label sequence of the current iteration, i.e. *all*
                # of the compressed labels.
                label_sequences[graph_index][:, iteration] = labels_compressed[:num_open_verts]
    
    multisets = np.array([str(l) for i, l in enumerate(label_sequences[0])])

    uniq, uniq_indices, all_subhypergraph_indices, counts = np.unique(multisets, return_counts=True,
                                                                      return_inverse=True, return_index=True)
    #print("uniq num classes vs num nodes, maxopentrain, numopentrain: ", len(uniq), len(multisets), max_open_verts, num_open_verts)

    assert (len(open_verts) == len(multisets))
    node_class_indices_search = np.asarray(open_verts)
    node_classes_search = (uniq_indices[all_subhypergraph_indices])
    assert(len(node_class_indices_search)== len(node_classes_search))
    
    Eisom_byDeg = set([str(sorted([B.degree()[int(v)] for v in e])) for e in open_hyperedges_])
    reg_nodes = []
    reg_hyperedges = []
    reg_hyperedges_len= []
    #print("node_class_indices_search: ", len(node_class_indices_search))

    num_cc= 0
    for c in uniq_indices:
        Vc = node_class_indices_search[node_classes_search == c]
        #print("degree in G: ", B.degree(Vc))
        Ec = sorted(list(set([e for v in Vc for e in B[v] if (np.asarray([u in Vc for u in B[e]]).all())])))
        for e in Ec:
            assert(e>= max_open_verts)
        B_search = nx.induced_subgraph(B, list(Vc) + Ec)
        CC= list(nx.connected_components(B_search))
        num_cc+= len(CC)
        
        for cc in CC:
            if len(cc) >= 4:
                C= B_search.subgraph(cc)
                left, right = nx.bipartite.sets(C)
                if np.min(list(left)) >= max_open_verts:
                    x = right
                    right = left
                    left = x
                assert(np.min(list(right))>=max_open_verts)
                assert(np.max(list(left))<max_open_verts)
                
                deg_p_nodes = str(sorted([B.degree()[int(v)] for v in left]))
                #print("c degrees: ", deg_p_nodes)
                if deg_p_nodes not in Eisom_byDeg and len(left) >= 3:
                    reg_nodes.append([int(i) for i in left])
                    reg_hyperedges += [[int(b) for b in B_search[r]] for r in right]
                    reg_hyperedges_len+=[len(reg_nodes)-1]*len(right)
    return reg_nodes, reg_hyperedges, num_cc, reg_hyperedges_len

def load_data(dataset='contact-high-school', pos_embedding="laplacian_emap", tuple_size=3):
    if dataset in ['FB15k-237',"penn94", "reed98", 
                                 "amherst41", "cornell5", "johnshopkins55", "genius", "AIFB", "MUTAG"]:
        data_train, data_val, data_test = utils_graph.make_train_test_data(dataset)
        graph = True
    else:
        data_train, data_val, data_test = utils_hypergraph.make_train_test_data(dataset)
        graph = False
    
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

    num_nodes = int(np.max(np.array(all_nodes, dtype=int))) + 1

    train_verts = sorted(list(set(([int(w) for e in data_train for w in e]))))
    train_mask = np.zeros((num_nodes, 1))
    train_mask[[int(u) for u in train_verts]] = 1

    val_verts = sorted(list(set(([int(w) for e in data_val for w in e]))))
    val_mask = np.zeros((num_nodes, 1))
    val_mask[[int(u) for u in val_verts]] = 1

    test_verts = sorted(list(set(([int(w) for e in data_test for w in e]))))
    test_mask = np.zeros((num_nodes, 1))
    test_mask[[int(u) for u in test_verts]] = 1

    if graph:
        train_percentage = 0.2 # the percentage of training to use for prediction
        pos_percentage = 0.5 # the percentage of a split to use as positive samples for prediction
        neg_proportion= 1.2 # (the number of negative samples)/(the number of positive samples)
    else:
        train_percentage = 1.0 # the percentage of training to use for prediction
        pos_percentage = 0.5 # the percentage of a split to use as positive samples for prediction
        neg_proportion= 1.2 # (the number of negative samples)/(the number of positive samples)

    all_train_tuples = list([d for d in data_train if len(d) == tuple_size])
    all_train_tuples = all_train_tuples[:int(train_percentage * len(all_train_tuples))]
    pos_train = np.array(all_train_tuples[:int(len(all_train_tuples) * pos_percentage)])
    pos_train_= [",".join([str(v) for v in p]) for p in pos_train]

    all_val_tuples = list([d for d in data_val if len(d) == tuple_size])
    pos_val = np.array(all_val_tuples[:int(len(all_val_tuples) * pos_percentage)])
    pos_val_= [",".join([str(v) for v in p]) for p in pos_val]

    all_test_tuples = list([d for d in data_test if len(d) == tuple_size])
    pos_test = np.array(all_test_tuples[:int(len(all_test_tuples) * pos_percentage)])
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
    num_open_train_verts = len(open_train_verts)

    open_val_verts = sorted(list(set([int(v) for s in open_val_ for v in s])))
    max_open_val_verts= np.max(open_val_verts)+1
    num_open_val_verts = len(open_val_verts)

    open_test_verts = sorted(list(set([int(v) for s in open_test_ for v in s])))
    max_open_test_verts= np.max(open_test_verts)+1
    num_open_test_verts = len(open_test_verts)

    hyperedge_list = open_train_+open_val_+open_test_

    edge_list = [(int(u), int(v)) for s in data_train for u in s for v in s if int(u) < int(v)]
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

    if graph:
        reg_nodes_train, reg_hyperedges_train, num_cc_train, reg_hyperedges_train_len = regsubreplacer_G(open_train_verts, open_train_, max_open_train_verts, args.max_iter )
        reg_nodes_val, reg_hyperedges_val, num_cc_val, reg_hyperedges_val_len = regsubreplacer_G(open_val_verts, open_val_, max_open_val_verts, args.max_iter )
        reg_nodes_test, reg_hyperedges_test, num_cc_test, reg_hyperedges_test_len = regsubreplacer_G(open_test_verts, open_test_, max_open_test_verts, args.max_iter )
    else:
        reg_nodes_train, reg_hyperedges_train, num_cc_train, reg_hyperedges_train_len = regsubreplacer(open_train_verts, open_train_, max_open_train_verts, args.max_iter )
        reg_nodes_val, reg_hyperedges_val, num_cc_val, reg_hyperedges_val_len = regsubreplacer(open_val_verts, open_val_, max_open_val_verts, args.max_iter )
        reg_nodes_test, reg_hyperedges_test, num_cc_test, reg_hyperedges_test_len = regsubreplacer(open_test_verts, open_test_, max_open_test_verts, args.max_iter )
    
    #print("sym cc/cc train",len(reg_nodes_train), num_cc_train)
    #print("void: sym cc/cc val",len(reg_nodes_val), num_cc_val)
    #print("void: sym cc/cc test",len(reg_nodes_test), num_cc_test)
    reg_nodes= reg_nodes_train
    reg_hyperedges= reg_hyperedges_train
    reg_hyperedges_len= reg_hyperedges_train_len

    #print("negative_ex_tr: ", negative_ex_tr.shape)
    #print("negative_ex_val: ", negative_ex_val.shape)
    #print("negative_ex_te: ", negative_ex_te.shape)
    
    #print("positive_ex_tr: ", positive_ex_tr.shape)
    #print("positive_ex_val: ", positive_ex_val.shape)
    #print("positive_ex_te: ", positive_ex_te.shape)

    return {"train_mask": train_mask, "val_mask": val_mask, "test_mask": test_mask, "num_vertices": len(train_mask),
            "neg_examples_tr": negative_ex_tr, "pos_examples_tr": positive_ex_tr, "neg_examples_val": negative_ex_val,
            "pos_examples_val": positive_ex_val, "neg_examples_te": negative_ex_te, "pos_examples_te": positive_ex_te,
            "reg_node_sets": reg_nodes, "reg_connecting_hyperedges": reg_hyperedges, "hyperedge_list": hyperedge_list,
            "reg_connecting_hyperedges_len": reg_hyperedges_len, 
            "pos_embedding": Q}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SymmetryBreaker')
    parser.add_argument('-hgnn', '--hgnn', type=str,
                        choices=['HGNN', 'HGNNP', 'HNHN', 'HyperGCN', 'UniGCN', 'UniGIN', 'UniSAGE', 'UniGAT'],
                        default='HGNNP')
    parser.add_argument('-dataset', '--dataset', type=str,
                        choices=['congress-bills', 'email-Enron', 'email-Eu',
                                 'contact-high-school', 'cat-edge-DAWN',
                                 'cat-edge-Brain', 'cat-edge-Cooking', 'cat-edge-geometry-questions',
                                 'cat-edge-madison-restaurant-reviews', 'cat-edge-music-blues-reviews',
                                 'cat-edge-vegas-bars-reviews', 'NDC-classes', 'contact-primary-school',
                                 'rand-regular', 'preferential-attachment', 'FB15k-237',"penn94", "reed98", 
                                 "amherst41", "cornell5", "johnshopkins55", "genius", "AIFB", "MUTAG"],
                        help='Dataset to run on.', default='contact-primary-school')
    parser.add_argument('-pos_embedding', '--pos_embedding', type=str,
                        choices=['laplacian_emap', 'id'], default='laplacian_emap')
    parser.add_argument('-epochs', '--epochs', type=int, default=2000)
    parser.add_argument('-max_iter', '--max_iter', type=int, default=2)
    parser.add_argument('-embdim', '--embdim', type=int, default=1024)
    parser.add_argument('-tuple_size', '--tuple_size', type=int, default=3)
    parser.add_argument('-device', '--device', type=str, default=None)
    parser.add_argument('-p_sym', '--p_sym', type=float, default=0.5)
    args = (parser.parse_args())

    dim_emb = args.embdim
    lr = 1e-2
    num_workers = 0
    batch_sz = 2048
    val_freq = 20
    epoch_max = args.epochs
    weight_decay = 1e-4
    set_seed(2022) 
    if args.device==None or args.device=="gpu" or args.device=="cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device= torch.device("cpu")
        
    start_time = time.time()
    data = load_data(dataset=args.dataset, pos_embedding=args.pos_embedding,
                     tuple_size= args.tuple_size)

    start_train = time.time()
    print("Preprocessing time: ", start_train - start_time,"s")
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
    reg_nodes = data['reg_node_sets']
    reg_connecting_hyperedges = data['reg_connecting_hyperedges']
    reg_connecting_hyperedges_len = data['reg_connecting_hyperedges_len']
    if len(reg_nodes) > 0 and args.p_sym==0.0:
        G.remove_hyperedges(reg_connecting_hyperedges)
        G.add_hyperedges(reg_nodes)
    if args.hgnn == 'HGNN':
        net = HGNN(X.shape[1], args.embdim, use_bn=True, num_classes=2)
    elif args.hgnn == 'HNHN':
        net = HNHN(X.shape[1], args.embdim, use_bn=True, num_classes=2)
    elif args.hgnn == 'HyperGCN':
        net = HyperGCN(X.shape[1], args.embdim, use_bn=True, num_classes=2)
    elif args.hgnn == 'UniGCN':
        net = UniGCN(X.shape[1], args.embdim, use_bn=True, num_classes=2)
    elif args.hgnn == 'UniGIN':
        net = UniGIN(X.shape[1], args.embdim, use_bn=True, num_classes=2)
    elif args.hgnn == 'UniSAGE':
        net = UniSAGE(X.shape[1], args.embdim, use_bn=True, num_classes=2)
    elif args.hgnn == 'UniGAT':
        net = UniGAT(X.shape[1], args.embdim, num_heads=5, use_bn=True, num_classes=2)
    else:
        net = HGNNP(X.shape[1], args.embdim, use_bn=True, num_classes=2)

    net = net.to(device)
    criterion = VertexSetAgg(args.embdim, weight_decay).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    Gorig= G.clone()
    best_state, best_val, best_epoch = None, 0, -1
    for epoch in range(epoch_max):
        if len(reg_nodes) > 0 and args.p_sym>0.0:
            selection_indices= np.random.binomial(n=1, p=args.p_sym, size=len(reg_nodes))
            if np.sum(selection_indices)>0:
                G.remove_hyperedges([h for i,h in enumerate(reg_connecting_hyperedges) if selection_indices[reg_connecting_hyperedges_len[i]]>0])
                G.add_hyperedges([h for i,h in enumerate(reg_nodes) if selection_indices[i]>0])
                train(net, X, G, pos_ex_tr, neg_ex_tr, optimizer, criterion, epoch)
                G.add_hyperedges([h for i,h in enumerate(reg_connecting_hyperedges) if selection_indices[reg_connecting_hyperedges_len[i]]>0])
                G.remove_hyperedges([h for i,h in enumerate(reg_nodes) if selection_indices[i]>0])
            else:
                train(net, X, G, pos_ex_tr, neg_ex_tr, optimizer, criterion, epoch)
        else:
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