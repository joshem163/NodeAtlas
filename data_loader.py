import torch
from torch_geometric.datasets import Planetoid, WebKB, WikipediaNetwork,HeterophilousGraphDataset,Actor
import warnings
from torch_geometric.data import Data
#from torch_sparse import SparseTensor
import numpy as np
import os
#warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric.data.dataset")

warnings.filterwarnings("ignore", category=FutureWarning, module="torch_geometric")

def load_data(dataset_Name):
    if dataset_Name=='cora':
        data_loaded = Planetoid(root='/tmp/cora', name='Cora', split='geom-gcn')
        #data_loaded = Planetoid(root='/tmp/cora', name='Cora', split='random',num_train_per_class=40,num_val=100,num_test=540)
    elif dataset_Name=='citeseer':
        data_loaded = Planetoid(root='/tmp/citeseer', name='citeseer', split='geom-gcn')
    elif dataset_Name=='pubmed':
        data_loaded = Planetoid(root='/tmp/pubmed', name='pubmed', split='geom-gcn')
    elif dataset_Name=='texas':
        data_loaded = WebKB(root='/tmp/texas', name='texas')
    elif dataset_Name=='cornell':
        data_loaded = WebKB(root='/tmp/cornell', name='cornell')
    elif dataset_Name=='wisconsin':
        data_loaded = WebKB(root='/tmp/wisconsin', name='wisconsin')
    elif dataset_Name=='chameleon':
        data_loaded = WikipediaNetwork(root='/tmp/chameleon', name='chameleon')
    elif dataset_Name=='squirrel':
        data_loaded = WikipediaNetwork(root='/tmp/squirrel', name='squirrel')
    elif dataset_Name=='actor':
        data_loaded = Actor(root='/tmp/actor')
    elif dataset_Name=='roman-empire':
        data_loaded =HeterophilousGraphDataset(root='/tmp/roman-empire', name='roman-empire')
    elif dataset_Name=='amazon-ratings':
        data_loaded =HeterophilousGraphDataset(root='/tmp/amazon-ratings', name='amazon-ratings')
    elif dataset_Name=='minesweeper':
        data_loaded =HeterophilousGraphDataset(root='/tmp/minesweeper', name='minesweeper')
    elif dataset_Name=='questions':
        data_loaded =HeterophilousGraphDataset(root='/tmp/questions', name='questions')
    elif dataset_Name=='tolokers':
        data_loaded =HeterophilousGraphDataset(root='/tmp/tolokers', name='tolokers')
    else:
        raise NotImplementedError
    return data_loaded

def load_Sq_Cha_filterred(name):
    data = np.load(os.path.join('data/', f'{name}_filtered.npz'))

    node_features = torch.tensor(data['node_features'], dtype=torch.float)
    labels = torch.tensor(data['node_labels'], dtype=torch.long)
    num_nodes=len(labels)
    edges = torch.tensor(data['edges'], dtype=torch.long)
    train_masks = torch.tensor(data['train_masks'])
    val_masks = torch.tensor(data['val_masks'])
    test_masks = torch.tensor(data['test_masks'])


    edge_index = edges.t().contiguous()
    #adj_t = SparseTensor.from_edge_index(edge_index, sparse_sizes=(num_nodes, num_nodes))

    data = Data(x=node_features, edge_index=edge_index, y=labels,train_mask=train_masks,val_mask=val_masks,test_mask=test_masks)

    return data