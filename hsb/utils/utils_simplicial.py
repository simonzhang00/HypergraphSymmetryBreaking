import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
from functools import reduce
import random
import copy
import sys

# code modified from simplex2pred https://github.com/simonepiaggesi/simplex2pred

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


flattened = lambda l: [item for subl in l for item in subl]

###### Data Utilities ######

MAX_ORDER = 3
RANDOM_SEED = 314

def neg_extra_sample_sizes(num_pos_train, num_train_hard_negatives, num_pos_val, num_val_hard_negatives, num_pos_test, num_test_hard_negatives):
    a = num_pos_train
    b = num_train_hard_negatives
    c = num_pos_val
    d = num_val_hard_negatives
    e = num_pos_test
    f = num_test_hard_negatives

    # https://www.wolframalpha.com/input?i=a%21%3D0+and+b%21%3D0+and+c%21%3D0+and+d%21%3D0+and+e%21%3D0+and+f%21%3D0++and+d*a%2Ba*y%3Dc*b%2Bc*x+and+d*f%2Bc*z%3Dd*e%2By*e
    # a/(b+x)=c/(d+y)= e/(f+z) solve for x,y,z when a!=0, b!=0, c!=0, d!=0, e!=0, f!=0
    # (b c - a d + c x)/a=
    tr_extra_neg_samples= 1000
    val_extra_neg_samples = int((b * c - a * d + c * tr_extra_neg_samples) / tr_extra_neg_samples)
    te_extra_neg_samples = int((b * c * e - a * d * f + c * e * tr_extra_neg_samples) / (a * c))

    return tr_extra_neg_samples, val_extra_neg_samples, te_extra_neg_samples

def make_train_test_data_shuffle(dataset, prctl1 = 80, prctl2 = 85):
    '''Returns lists of train/test cliques (maximal simplices) and train/test simplices from higher-order temporal data.
    Input: name of the dataset, test percentiles (example: 80-100 means that test set starts on 80-percentile and ends on 100-percentile)
    Output: lists cliques_train, cliques_test, simplices_train, simplices_test
    '''
    np.random.seed(RANDOM_SEED)
    nverts_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-nverts.txt' % (dataset, dataset), names = ['nverts'])
    nverts_df['simplex_id'] = nverts_df.index

    simplex_ids = nverts_df.apply(lambda x: [x.simplex_id, ] * int(x.nverts), axis = 1).values
    simplex_ids = [item for l in simplex_ids for item in l]

    simplices_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-simplices.txt' % (dataset, dataset), names = ['simplex']).astype(
        str)
    simplices_df['simplex_id'] = simplex_ids


    times_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-times.txt' % (dataset, dataset), names = ['times'])

    data = simplices_df.groupby('simplex_id').apply(lambda x: frozenset(x.simplex))

    totaltimes= len(times_df.times)
    indices= np.random.shuffle(list(range(totaltimes)))

    times_df.loc[:, 'edge'] = data
    df_train = times_df.loc[indices[:int(totaltimes * prctl1)]]
    df_val = times_df.loc[indices[int(totaltimes * prctl1): int(totaltimes * prctl2)]]
    df_test = times_df.loc[indices[:int(totaltimes * prctl2)]]

    cliques_train = [c for c in df_train.edge.tolist() if len(c) > 1]
    cliques_val = [c for c in df_val.edge.unique() if len(c) > 1]
    cliques_test = [c for c in df_test.edge.unique() if len(c) > 1]
    nodes_train= set(flattened(cliques_train))
    nodes_val = set(flattened(cliques_val))
    nodes_test = set(flattened(cliques_test))

    simplex_train = [combinations(max_clique, size) for max_clique in set(cliques_train) \
                     for size in range(2, min(len(max_clique), MAX_ORDER + 1) + 1)]
    data_train = set(map(frozenset, [simplex for simplices in simplex_train
                                   for simplex in simplices if nodes_train.issuperset(simplex)]))
    simplex_val = [combinations(max_clique, size) for max_clique in cliques_val \
                    for size in range(2, min(len(max_clique), MAX_ORDER + 1) + 1)]
    data_val = set(map(frozenset, [simplex for simplices in simplex_val
                                   for simplex in simplices if nodes_val.issuperset(simplex)]))
    simplex_test = [combinations(max_clique, size) for max_clique in cliques_test \
                    for size in range(2, min(len(max_clique), MAX_ORDER + 1) + 1)]
    data_test = set(map(frozenset, [simplex for simplices in simplex_test
                                    for simplex in simplices if nodes_test.issuperset(simplex)]))

    # print("simplex_train: ", len(simplex_train))
    # print("simplex_val: ", len(simplex_val))
    # print("simplex_test: ", len(simplex_test))
    # print("data_train: ", len(data_train))
    # print("data_val: ", len(data_val))
    # print("data_test: ", len(data_test))

    return cliques_train, cliques_val, cliques_test, list(data_train), list(data_val), list(data_test)

def make_train_test_data(dataset, prctl1 = 80, prctl2 = 85):
    '''Returns lists of train/test cliques (maximal simplices) and train/test simplices from higher-order temporal data.
    Input: name of the dataset, test percentiles (example: 80-100 means that test set starts on 80-percentile and ends on 100-percentile)
    Output: lists cliques_train, cliques_test, simplices_train, simplices_test
    '''

    nverts_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-nverts.txt' % (dataset, dataset), names = ['nverts'])
    nverts_df['simplex_id'] = nverts_df.index

    simplex_ids = nverts_df.apply(lambda x: [x.simplex_id, ] * int(x.nverts), axis = 1).values
    simplex_ids = [item for l in simplex_ids for item in l]

    simplices_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-simplices.txt' % (dataset, dataset), names = ['simplex']).astype(
        str)
    simplices_df['simplex_id'] = simplex_ids


    times_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-times.txt' % (dataset, dataset), names = ['times'])

    cutoff1 = np.percentile(times_df.times, prctl1)
    cutoff2 = np.percentile(times_df.times, prctl2)

    data = simplices_df.groupby('simplex_id').apply(lambda x: frozenset(x.simplex))

    times_df.loc[:, 'edge'] = data

    df_train = times_df.loc[times_df.times <= cutoff1]
    df_val = times_df.loc[(times_df.times > cutoff1) & (times_df.times <= cutoff2)]
    df_test= times_df.loc[(times_df.times > cutoff2)]

    cliques_train = [c for c in df_train.edge.tolist() if len(c) > 1]
    cliques_val = [c for c in df_val.edge.unique() if len(c) > 1]
    cliques_test = [c for c in df_test.edge.unique() if len(c) > 1]
    nodes_train= set(flattened(cliques_train))
    nodes_val = set(flattened(cliques_val))
    nodes_test = set(flattened(cliques_test))

    simplex_train = [combinations(max_clique, size) for max_clique in set(cliques_train) \
                     for size in range(2, min(len(max_clique), MAX_ORDER + 1) + 1)]
    data_train = set(map(frozenset, [simplex for simplices in simplex_train
                                   for simplex in simplices if nodes_train.issuperset(simplex)]))
    simplex_val = [combinations(max_clique, size) for max_clique in cliques_val \
                    for size in range(2, min(len(max_clique), MAX_ORDER + 1) + 1)]
    data_val = set(map(frozenset, [simplex for simplices in simplex_val
                                   for simplex in simplices if nodes_val.issuperset(simplex)]))
    simplex_test = [combinations(max_clique, size) for max_clique in cliques_test \
                    for size in range(2, min(len(max_clique), MAX_ORDER + 1) + 1)]
    data_test = set(map(frozenset, [simplex for simplices in simplex_test
                                    for simplex in simplices if nodes_test.issuperset(simplex)]))

    # print("simplex_train: ", len(simplex_train))
    # print("simplex_val: ", len(simplex_val))
    # print("simplex_test: ", len(simplex_test))
    # print("data_train: ", len(data_train))
    # print("data_val: ", len(data_val))
    # print("data_test: ", len(data_test))

    return cliques_train, cliques_val, cliques_test, list(data_train), list(data_val), list(data_test)


def s2vhasse_to_n2vformat(hasse_nx):
    '''Returns the bigger connected component of the input graph as a networkX graph (directed, weighted).
    Input: networkx Graph object
    Output: networkx DiGraph object
    '''

    # hasse_nx.remove_edges_from(nx.selfloop_edges(hasse_nx))
    # hasse_nx.remove_nodes_from(list(nx.isolates(hasse_nx)))

    if frozenset() in hasse_nx.nodes():
        hasse_nx.remove_node(frozenset())

    nx.relabel_nodes(hasse_nx, {n: ','.join(map(str, sorted(map(int, n)))) for n in hasse_nx.nodes()}, copy = False)

    # max_comp = sorted(nx.connected_components(hasse_nx), key = len, reverse = True)[0]
    # hasse_nx = nx.subgraph(hasse_nx, max_comp)

    map_node_weight = {n: float(w['weight']) for n, w in hasse_nx.nodes(data = True)}
    hasse_nx = hasse_nx.to_directed()
    for u, v, d in hasse_nx.edges(data = True):
        d['weight'] = map_node_weight[v]

    return hasse_nx


def unbias_weights_n2vformat(hasse_nx):
    '''Returns Hasse Diagram with "eqbias" weighting scheme as a networkX graph (directed, weighted).
    Input: networkx DiGraph object
    Output: networkx DiGraph object
    '''

    max_size = max([len(u.split(',')) for u in hasse_nx.nodes()])

    if max_size <= 2:
        return hasse_nx

    hasse_new = copy.deepcopy(hasse_nx)
    for u in hasse_nx.nodes():
        u_size = len(u.split(','))
        if (u_size > 1) and (u_size < max_size):
            w_up = sum([w['weight'] for v, w in hasse_nx[u].items() if len(v.split(',')) == u_size + 1])
            w_down = sum([w['weight'] for v, w in hasse_nx[u].items() if len(v.split(',')) == u_size - 1])

            for v, w in hasse_nx[u].items():
                v_size = len(v.split(','))
                if v_size == u_size + 1:
                    hasse_new[u][v]['weight'] *= (w_up + w_down) / (w_up)
                if v_size == u_size - 1:
                    hasse_new[u][v]['weight'] *= (w_up + w_down) / (w_down)
    return hasse_new


###### Sampling Utilities ######

from multiprocessing import Pool, Barrier
from functools import partial


def map_sample_neighbors(neighs_, size_, tuple_):
    '''Utility to sample node neighbors from a group.
    '''

    np.random.seed(RANDOM_SEED)
    neighbors_set = set(flattened([neighs_[n] for n in tuple_]))
    node_set = set(tuple_)

    candidate_list = list(neighbors_set - node_set)

    if len(candidate_list) < size_:
        candidate_list = candidate_list + list(node_set)
        return list(tuple_) + list(np.random.choice(candidate_list, size = size_, replace = True))

    return list(tuple_) + list(np.random.choice(candidate_list, size = size_, replace = False))


def map_sample_common_neighbors(neighs_, tuple_):
    '''Utility to sample node common neighbors from a group.
    '''

    np.random.seed(RANDOM_SEED)
    tuple_len = len(tuple_)
    new_tuple_ = list(np.random.permutation(list(tuple_))[:tuple_len - 1])
    neighbors_set = [set(neighs_[n]) for n in new_tuple_]
    common_set = neighbors_set[0].intersection(*neighbors_set)
    node_set = set(tuple_)

    candidate_list = list(common_set - node_set)

    if len(candidate_list) == 0:
        candidate_list = list(node_set)

    return new_tuple_ + [np.random.choice(candidate_list)]


def naive_negative_sampling(node_list, sample_size, tuple_size):
    '''Returns naively sampled node groups.
    Input: list of nodes, number of samples, size of groups
    Output: numpy array of strings objects (comma-separated node indices)
    '''
    rs = np.random.RandomState(RANDOM_SEED)
    random_tuple = rs.choice(node_list, size = tuple_size * sample_size).reshape(-1, tuple_size)
    return pd.unique([','.join(map(str, sorted(map(int, u)))) \
                      for u in random_tuple if len(set(u)) == tuple_size])


def motifs_negative_sampling(node_list, neighs_dict, sample_size, tuple_size):
    '''Returns node groups sampled with motif sampling (inspired by Algorithm 1 from Patil et al. 2020, ref. [39] of the paper).
    Input: list of nodes, projected graph dictionary, number of samples, size of groups
    Output: numpy array of strings objects (comma-separated node indices)
    '''
    rs = np.random.RandomState(RANDOM_SEED)
    motifs_tuple = rs.choice(node_list, size = sample_size)[:, np.newaxis]

    for _ in range(tuple_size - 1):
        with Pool(processes = 20) as pool:
            motifs_tuple = np.array(pool.map(partial(map_sample_neighbors, neighs_dict, 1), motifs_tuple))
            pool.close()
            pool.join()
    return pd.unique([','.join(map(str, sorted(map(int, u)))) \
                      for u in motifs_tuple if len(set(u)) == tuple_size])


def stars_negative_sampling(node_list, neighs_dict, sample_size, tuple_size):
    '''Returns node groups sampled with star sampling.
    Input: list of nodes, projected graph dictionary, number of samples, size of groups
    Output: numpy array of strings objects (comma-separated node indices)
    '''
    rs = np.random.RandomState(RANDOM_SEED)
    stars_tuple = rs.choice(node_list, size = sample_size)[:, np.newaxis]

    with Pool(processes = 20) as pool:
        stars_tuple = np.array(pool.map(partial(map_sample_neighbors, neighs_dict, tuple_size - 1), stars_tuple))
        pool.close()
        pool.join()
    return pd.unique([','.join(map(str, sorted(map(int, u)))) \
                      for u in stars_tuple if len(set(u)) == tuple_size])


def cliques_negative_sampling(simplex_list, neighs_dict, nodes_train, sample_size, tuple_size):
    '''Returns node groups sampled with clique sampling (inspired by Algorithm 2 from Patil et al. 2020, ref. [39] of the paper).
    Input: list of nodes, projected graph dictionary, number of samples, size of groups
    Output: numpy array of strings objects (comma-separated node indices)
    '''
    rs = np.random.RandomState(RANDOM_SEED)
    cliques_tuple = rs.choice([s for s in simplex_list if len(s) == tuple_size], size = sample_size)
    # barrier = multiprocessing.Barrier(10, timeout = 10)
    with Pool(processes = 20) as pool:
        cliques_tuple = pool.map(partial(map_sample_common_neighbors, neighs_dict), cliques_tuple)
        pool.close()
        pool.join()
    return pd.unique([','.join(map(str, sorted(map(int, u)))) \
                      for u in cliques_tuple if len(set(u)) == tuple_size])


###### Classification utilities ######

from .calibrated_metrics_hypergraph import *

def classification_score_from_x(positive_test, negative_test, embedding_array, embedding_vocab, norder, nruns = 50,
                                max_test_size = 5000):
    '''Returns positive/negative examples with corresponding auc-pr scores, computed from the model embedding matrix.
    Input: numpy arrays of positive and negative tuples (comma-separated node indices), numpy array of simplex embedding,  dictionary, order of similarity (s_0, s_1, etc.), number of realizations, max number of samples per realization.
    Output: list of dictionaries with (y_test, y_pred, auc-pr, node_idx) as items
    '''

    rs = np.random.RandomState(RANDOM_SEED)

    none_scores = {'y_test': None, 'y_pred': None, 'idx_test': None}

    if positive_test.shape[0] == 0 or negative_test.shape[0] == 0:
        return [none_scores for _ in range(nruns)]

    scores = []

    positive_range = np.arange(1, min(max_test_size, len(positive_test)) + 1)
    negative_range = np.arange(1, min(max_test_size, len(negative_test)) + 1)

    n_positive_samples = rs.choice(positive_range, nruns)
    n_negative_samples = rs.choice(negative_range, nruns)
    negative_test_splits = np.split(rs.choice(negative_test, np.sum(n_negative_samples)), np.cumsum(n_negative_samples))

    for run in range(nruns):
        positive_test_x = rs.choice(positive_test, n_positive_samples[run], False)

        negative_test_x = np.unique(negative_test_splits[run])

        all_test_x = np.concatenate([positive_test_x, negative_test_x])

        tf_arrays = np.array(list(map(lambda a: [(embedding_vocab[h], embedding_vocab[k])
                                                 for h, k in combinations([','.join(map(str, sorted(map(int, face))))
                                                                           for face in
                                                                           combinations(a.split(','), norder + 1)], 2)],
                                      all_test_x)))

        y_pred = embedding_array[tf_arrays].prod(axis = 2).sum(axis = -1).mean(axis = -1)
        y_test = np.array([1, ] * len(positive_test_x) + [0, ] * len(negative_test_x))

        scores.append({'y_test': y_test, 'y_pred': y_pred, 'auc_pr': average_precision(y_test, y_pred, pi0 = 0.5),
                       'idx_test': np.array([s.split(',') for s in all_test_x])})

    return scores

def classification_score_from_y_full(y_test, y_score, nruns = 50):
    if len(np.asarray(y_test).shape) > 1 and len(np.asarray(y_score).shape) > 1:
        y_test_01= y_test[:,0]
        y_score_p= y_score[:,0]
        # Data to plot precision - recall curve
        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test_01, y_score_p)
    else:
        # Data to plot precision - recall curve

        precision, recall, thresholds = sklearn.metrics.precision_recall_curve(y_test, y_score)
    # Use AUC function to calculate the area under the curve of precision recall curve
    auc_precision_recall = sklearn.metrics.auc(recall, precision)
    return (auc_precision_recall, sklearn.metrics.roc_auc_score(y_test, y_score))


def classification_score_from_y(y_test, y_pred, nruns = 50):
    '''Returns classification scores for positive/negative examples opportunely sampled.
    Input: numpy arrays with classification labels and predicted scores, number of realizations.
    Output: average auc-pr, std. dev. auc-pr
    '''

    rs = np.random.RandomState(RANDOM_SEED)

    idx = np.arange(y_test.shape[0])
    pos_ = idx[y_test == 1]
    neg_ = idx[y_test == 0]

    pos_sizes = rs.randint(1, 1 + pos_.shape[0], size = nruns)
    pos_samples = np.split(rs.choice(pos_, size = np.sum(pos_sizes)), np.cumsum(pos_sizes))
    neg_sizes = rs.randint(1, 1 + neg_.shape[0], size = nruns)
    neg_samples = np.split(rs.choice(neg_, size = np.sum(neg_sizes)), np.cumsum(neg_sizes))

    aucs = []
    for run in range(nruns):
        pos_sample = np.unique(pos_samples[run])
        neg_sample = np.unique(neg_samples[run])
        y_test_sample = np.concatenate((y_test[pos_sample], y_test[neg_sample]))
        y_pred_sample = np.concatenate((y_pred[pos_sample], y_pred[neg_sample]))
        aucs.append(average_precision(y_test_sample, y_pred_sample, pi0 = 0.5))

    return np.mean(aucs), np.std(aucs)


def classification_score_from_y4(y_test, y_pred, nruns = 50):
    '''Returns classification scores for positive/negative examples opportunely sampled in case of extremely high class imbalance.
    Input: numpy arrays with classification labels and predicted scores, number of realizations.
    Output: average auc-pr, std. dev. auc-pr
    '''

    rs = np.random.RandomState(RANDOM_SEED)

    idx = np.arange(y_test.shape[0])
    pos_ = idx[y_test == 1]
    neg_ = idx[y_test == 0]

    if len(pos_) > len(neg_):
        pos_ = idx[y_test == 0]
        neg_ = idx[y_test == 1]

    neg_sizes = rs.randint(1, 1 + neg_.shape[0], size = nruns)
    neg_samples = np.split(rs.choice(neg_, size = np.sum(neg_sizes)), np.cumsum(neg_sizes))

    aucs = []
    for run in range(nruns):
        pos_sample = pos_
        neg_sample = np.unique(neg_samples[run])
        y_test_sample = np.concatenate((y_test[pos_sample], y_test[neg_sample]))
        y_pred_sample = np.concatenate((y_pred[pos_sample], y_pred[neg_sample]))
        aucs.append(average_precision(y_test_sample, y_pred_sample, pi0 = 0.5))

    return np.mean(aucs), np.std(aucs)

def find_chordless_cycles(A, K):
    r"""
    Finds all the chordless cycles up to size K of an adjacency matrix A
    """
    N= A.shape[0]
    cycles= []
    for i in range(N-2):
        for j in np.arange(i+1, N-1):
            if (not A[i,j]):
                continue
            candidates= [[]]
            for k in np.arange(j+1,N):
                if (not A[i,k]):
                    continue
                if (A[j,k]):
                    cycles.append(3)
                    continue
                v= [j,i,k]
                candidates.append(v)
                while(len(candidates) > 0):
                    v= candidates.pop(-1)
                    if len(v)==0:
                        continue

                    k= v[-1]
                    for m in np.arange(i+1, N):
                        if not A[m,k]:
                            continue
                        if not (m in v):
                            continue
                        if np.sum(A[m][v]) > 0:
                            continue

                        if (adj[m][j]):
                            cycles.append(v.size() + 1)
                        else:
                            v.append(m)
                            candidates.append(v)
    return cycles

if __name__ == "__main__":
    A = np.ones((10,10)) - np.diag([1]*10)
    print(find_chordless_cycles(A, 5))