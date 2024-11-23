import pandas as pd
import networkx as nx
import numpy as np
from itertools import combinations
from functools import reduce
import random
import copy
import sys
import torch_geometric
import torch
from torch_geometric.datasets import ZINC, AQSOL, GNNBenchmarkDataset
from torch_geometric.loader import DataLoader

MAX_ORDER = 3
RANDOM_SEED = 314

from multiprocessing import Pool, Barrier
from functools import partial

def graph_classification_make_train_test_data(name = "ZINC"):
    if name == 'ZINC':
        path = '../../hsb/data/'
        train_dataset = ZINC(path, subset = True, split = 'train')
        val_dataset = ZINC(path, subset = True, split = 'val')
        test_dataset = ZINC(path, subset = True, split = 'test')

        train_loader = DataLoader(train_dataset, batch_size = 1, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = 1)
        test_loader = DataLoader(test_dataset, batch_size = 1)
    elif name == 'AQSOL':
        path = '../../hsb/data/'
        train_dataset = AQSOL(path, split = 'train')
        val_dataset = AQSOL(path, split = 'val')
        test_dataset = AQSOL(path, split = 'test')

        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = 128)
        test_loader = DataLoader(test_dataset, batch_size = 128)
    elif name in ['PATTERN', 'CLUSTER', 'MNIST', 'CIFAR10', 'TSP', 'CSL']:
        path = '../../hsb/data/'
        train_dataset = GNNBenchmarkDataset(path, name, split = 'train')
        val_dataset = GNNBenchmarkDataset(path, name, split = 'val')
        test_dataset = GNNBenchmarkDataset(path, name, split = 'test')

        train_loader = DataLoader(train_dataset, batch_size = 128, shuffle = True)
        val_loader = DataLoader(val_dataset, batch_size = 128)
        test_loader = DataLoader(test_dataset, batch_size = 128)
    return train_loader,val_loader,test_loader
    
def make_train_test_data(name = 'FB15k-237'):
    if name == 'FB15k-237':
        dataset = torch_geometric.datasets.RelLinkPredDataset('../../hsb/data/', name)
        data = dataset[0]
        edges_train = [list(c) for c in list(torch.transpose(data.train_edge_index,0,1).numpy()) if len(c) > 1]
        edges_val = [list(c) for c in list(torch.transpose(data.valid_edge_index,0,1).numpy()) if len(c) > 1]
        edges_test = [list(c) for c in list(torch.transpose(data.test_edge_index,0,1).numpy()) if len(c) > 1]
    elif name in ["penn94", "reed98", "amherst41", "cornell5", "johnshopkins55", "genius"]:
        dataset = torch_geometric.datasets.LINKXDataset('../../hsb/data/', name)
        data = dataset[0]
        train_edge_index = data.edge_index[:,:int((0.80)*data.edge_index.size(1))]
        valid_edge_index = data.edge_index[:,int((0.80)*data.edge_index.size(1)):int((0.85)*data.edge_index.size(1))]
        test_edge_index = data.edge_index[:,int((0.85)*data.edge_index.size(1)):]
        edges_train = [list(c) for c in list(torch.transpose(train_edge_index,0,1).numpy()) if len(c) > 1]
        edges_val = [list(c) for c in list(torch.transpose(valid_edge_index,0,1).numpy()) if len(c) > 1]
        edges_test = [list(c) for c in list(torch.transpose(test_edge_index,0,1).numpy()) if len(c) > 1]
    elif name in ["AIFB", "MUTAG", "BGS", "AM"]:
        dataset = torch_geometric.datasets.Entities('../../hsb/data/', name)
        data = dataset[0]
        train_edge_index = data.edge_index[:,:int((0.80)*data.edge_index.size(1))]
        valid_edge_index = data.edge_index[:,int((0.80)*data.edge_index.size(1)):int((0.85)*data.edge_index.size(1))]
        test_edge_index = data.edge_index[:,int((0.85)*data.edge_index.size(1)):]
        edges_train = [list(c) for c in list(torch.transpose(train_edge_index,0,1).numpy()) if len(c) > 1]
        edges_val = [list(c) for c in list(torch.transpose(valid_edge_index,0,1).numpy()) if len(c) > 1]
        edges_test = [list(c) for c in list(torch.transpose(test_edge_index,0,1).numpy()) if len(c) > 1]

    return edges_train, edges_val, edges_test

