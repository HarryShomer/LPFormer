import numpy as np
from tqdm import tqdm 
import scipy.sparse as ssp

import torch 
from torch.nn import functional as F
from torch_geometric.data import DataLoader


def calc_CN(data, edges, split="test", batch_size=16384):
    """
    # CNs per sample
    """
    edges = edges.cpu()
    edge_index = data['full_edge_index'].cpu()
    edge_weight = torch.ones(edge_index.size(1)).to(torch.float)
    adj = ssp.csr_matrix((edge_weight, (edge_index[0], edge_index[1])), shape=(data['num_nodes'], data['num_nodes']))

    num_cns = []
    for ind in tqdm(DataLoader(range(edges.shape[0]), batch_size), f"Calc CN - {split}"):
        src, dst = edges[ind, 0], edges[ind, 1]
        cur_scores = np.array(np.sum(adj[src].multiply(adj[dst]), 1)).flatten()
        num_cns.append(cur_scores)

    return torch.from_numpy(np.concatenate(num_cns, 0))


def calc_FS(data, edges, split="test", batch_size=16384):
    """
    Cos-Sim of Features per sample
    """
    edges = edges.cpu()

    mean_x = data['x'].mean(axis=0)
    norm_x = data['x'] - mean_x

    num_fs = []
    for ind in tqdm(DataLoader(range(edges.shape[0]), batch_size), f"Calc FS - {split}"):
        src, dst = edges[ind, 0], edges[ind, 1]
        cur_scores = F.cosine_similarity(norm_x[src], norm_x[dst]).flatten().cpu()
        num_fs.append(cur_scores.numpy())

    return torch.from_numpy(np.concatenate(num_fs, 0))