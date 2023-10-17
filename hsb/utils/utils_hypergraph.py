import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
from functools import reduce
import random
import copy
import sys

MAX_ORDER = 3
RANDOM_SEED = 314

from multiprocessing import Pool, Barrier
from functools import partial

from .calibrated_metrics_hypergraph import *

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

def make_train_test_data(dataset, prctl1=70, prctl2=85):
    '''Returns lists of train/test cliques (maximal hyperedges) and train/test hyperedges from higher-order temporal data.
    Input: name of the dataset, test percentiles (example: 80-100 means that test set starts on 80-percentile and ends on 100-percentile)
    Output: lists cliques_train, cliques_test, hyperedges_train, hyperedges_test
    '''

    nverts_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-nverts.txt' % (dataset, dataset), names=['nverts'])
    nverts_df['hyperedge_id'] = nverts_df.index

    hyperedge_ids = nverts_df.apply(lambda x: [x.hyperedge_id, ] * int(x.nverts), axis=1).values
    hyperedge_ids = [item for l in hyperedge_ids for item in l]

    hyperedges_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-simplices.txt' % (dataset, dataset), names=['hyperedge']).astype(
        str)
    hyperedges_df['hyperedge_id'] = hyperedge_ids


    times_df = pd.read_csv('../../hsb/data/processed-data/%s/%s-times.txt' % (dataset, dataset), names=['times'])

    cutoff1 = np.percentile(times_df.times, prctl1)
    cutoff2 = np.percentile(times_df.times, prctl2)
    # cutoff3 = np.percentile(times_df.times, 100)

    data = hyperedges_df.groupby('hyperedge_id').apply(lambda x: frozenset(x.hyperedge))

    times_df.loc[:, 'edge'] = data

    df_train = times_df.loc[times_df.times <= cutoff1]
    df_val = times_df.loc[(times_df.times > cutoff1) & (times_df.times <= cutoff2)]
    df_test= times_df.loc[(times_df.times > cutoff2)]

    hyperedges_train = [c for c in df_train.edge.tolist() if len(c) > 1]
    hyperedges_val = [c for c in df_val.edge.unique() if len(c) > 1]
    hyperedges_test = [c for c in df_test.edge.unique() if len(c) > 1]

    return hyperedges_train, hyperedges_val, hyperedges_test #cliques_train, cliques_val, cliques_test, list(data_train), list(data_val), list(data_test)
def tuple_negative_sampling(hyperedge_list, nodes, sample_size, tuple_size):
    '''Returns node groups sampled with clique sampling (inspired by Algorithm 2 from Patil et al. 2020, ref. [39] of the paper).
    Input: list of nodes, projected graph dictionary, number of samples, size of groups
    Output: numpy array of strings objects (comma-separated node indices)
    '''
    rs = np.random.RandomState(RANDOM_SEED)
    cliques_tuple= hyperedge_list
    # target_tuples= [s for s in hyperedge_list if len(s) == tuple_size]
    # if len(target_tuples)>0:
    #     cliques_tuple = rs.choice(target_tuples, size=sample_size)
    # else:
    #     cliques_tuple= []

    cliques_tuple+= ([tuple(list(rs.choice(nodes, size= tuple_size))) for _ in range(sample_size)])
    return [tuple(np.array(s.split(","), dtype=int)) for s in pd.unique(np.asarray([','.join(map(str, sorted(map(int, u)))) \
                      for u in cliques_tuple if len(set(u)) == tuple_size]))]

def cliques_negative_sampling(simplex_list, neighs_dict, nodes, sample_size, tuple_size):
    '''Returns node groups sampled with clique sampling (inspired by Algorithm 2 from Patil et al. 2020, ref. [39] of the paper).
    Input: list of nodes, projected graph dictionary, number of samples, size of groups
    Output: numpy array of strings objects (comma-separated node indices)
    '''
    rs = np.random.RandomState(RANDOM_SEED)
    triangles= [s for s in simplex_list if len(s) == tuple_size]
    if len(triangles)>0:
        cliques_tuple = rs.choice(triangles, size=sample_size)
    else:
        cliques_tuple= []
    if len(cliques_tuple)>0:
        # barrier = multiprocessing.Barrier(10, timeout=10)
        with Pool(processes=20) as pool:
            cliques_tuple = pool.map(partial(map_sample_common_neighbors, neighs_dict), cliques_tuple)
            pool.close()
            pool.join()

    cliques_tuple+= ([tuple(list(rs.choice(nodes, size= tuple_size))) for _ in range(sample_size)])
    return pd.unique([','.join(map(str, sorted(map(int, u)))) \
                      for u in cliques_tuple if len(set(u)) == tuple_size])

def classification_score_from_y_full(y_test, y_score, nruns=50):
    if len(np.asarray(y_test).shape)>1 and len(np.asarray(y_score).shape)>1:
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
