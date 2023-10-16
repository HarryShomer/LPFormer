"""
Precompute the PPR scores for a dataset.

Scores are used to select nodes for attention
"""
import os
import torch
import argparse
import numpy as np
import pandas as pd
from time import time
from torch_sparse import SparseTensor 
from torch_geometric.utils import coalesce, to_undirected
import joblib  # Make ogb loads faster
from ogb.linkproppred import PygLinkPropPredDataset




def read_data_planetoid(args):
    """
    Read all data for the fixed split. Returns as dict
    """
    data_name = args.data_name
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "dataset")

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:
        path = os.path.join(data_dir, data_name, f"{split}_pos.txt")
  
        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)
            
            node_set.add(sub)
            node_set.add(obj)
            
            if sub == obj:
                continue

            if split == 'train': 
                train_pos.append((sub, obj))
                
            if split == 'valid': valid_pos.append((sub, obj))  
            if split == 'test': test_pos.append((sub, obj))
    
    num_nodes = len(node_set)
    print('# of nodes in ' + data_name + ' is: ', num_nodes)

    for split in ['test', 'valid']:
        path = os.path.join(data_dir, data_name, f"{split}_neg.txt")

        for line in open(path, 'r'):
            sub, obj = line.strip().split('\t')
            sub, obj = int(sub), int(obj)

            if split == 'valid': 
                valid_neg.append((sub, obj))               
            if split == 'test': 
                test_neg.append((sub, obj))

    train_edge = torch.transpose(torch.tensor(train_pos), 1, 0)
    edge_index = torch.cat((train_edge,  train_edge[[1,0]]), dim=1)
    edge_weight = torch.ones(edge_index.size(1))

    adj = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes])
          
    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(os.path.join(data_dir, data_name, "gnn_feature"))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {}
    data['adj'] = adj
    data['edge_index'] = edge_index
    data['num_nodes'] = num_nodes

    data['train_pos'] = train_pos_tensor
    data['train_val'] = train_val

    data['valid_pos'] = valid_pos
    data['valid_neg'] = valid_neg
    data['test_pos'] = test_pos
    data['test_neg'] = test_neg

    return data



def get_ppr_matrix(edge_index, num_nodes, alpha=0.15, eps=5e-5):
    """
    Calc PPR data

    Returns scores and the corresponding nodes

    Adapted from https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/gdc.py
    """
    edge_index = coalesce(edge_index, num_nodes=num_nodes)
    edge_index_np = edge_index.cpu().numpy()

    # Assumes sorted and coalesced edge indices (NOTE: coalesce also sorts edges)
    indptr = torch._convert_indices_from_coo_to_csr(edge_index[0], num_nodes).cpu().numpy()
    
    out_degree = indptr[1:] - indptr[:-1]
    
    start = time()
    print("Calculating PPR...", flush=True)
    neighbors, neighbor_weights = get_calc_ppr()(indptr, edge_index_np[1], out_degree, alpha, eps)
    print(f"Time: {time()-start:.2f} seconds")

    print("\n# Nodes with 0 PPR scores:", sum([len(x) == 1 for x in neighbors]))  # 1 bec. itself
    print(f"Mean # of scores per Node: {np.mean([len(x) for x in neighbors]):.1f}")

    return neighbors, neighbor_weights


def get_calc_ppr():
    """
    Courtesy of https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/transforms/gdc.py
    """
    import numba

    @numba.jit(nopython=True, parallel=True)
    def calc_ppr(
        indptr: np.ndarray,
        indices: np.ndarray,
        out_degree: np.ndarray,
        alpha: float,
        eps: float,
    ):
        r"""Calculate the personalized PageRank vector for all nodes
        using a variant of the Andersen algorithm
        (see Andersen et al. :Local Graph Partitioning using PageRank Vectors.)

        Args:
            indptr (np.ndarray): Index pointer for the sparse matrix
                (CSR-format).
            indices (np.ndarray): Indices of the sparse matrix entries
                (CSR-format).
            out_degree (np.ndarray): Out-degree of each node.
            alpha (float): Alpha of the PageRank to calculate.
            eps (float): Threshold for PPR calculation stopping criterion
                (:obj:`edge_weight >= eps * out_degree`).

        :rtype: (:class:`List[List[int]]`, :class:`List[List[float]]`)
        """
        alpha_eps = alpha * eps
        js = [[0]] * len(out_degree)
        vals = [[0.]] * len(out_degree)
        for inode_uint in numba.prange(len(out_degree)):
            inode = numba.int64(inode_uint)
            p = {inode: 0.0}
            r = {}
            r[inode] = alpha
            q = [inode]
            while len(q) > 0:
                unode = q.pop()

                res = r[unode] if unode in r else 0
                if unode in p:
                    p[unode] += res
                else:
                    p[unode] = res
                r[unode] = 0
                for vnode in indices[indptr[unode]:indptr[unode + 1]]:
                    _val = (1 - alpha) * res / out_degree[unode]
                    if vnode in r:
                        r[vnode] += _val
                    else:
                        r[vnode] = _val

                    res_vnode = r[vnode] if vnode in r else 0
                    if res_vnode >= alpha_eps * out_degree[vnode]:
                        if vnode not in q:
                            q.append(vnode)
            js[inode] = list(p.keys())
            vals[inode] = list(p.values())

        return js, vals

    return calc_ppr


def save_results(dataset, sparse_adj, alpha, eps, split=None, val=False):
    """
    Save results corresponding to values of `alpha` and `eps`

    Saved in `node_subsets/ppr/{dataset}/` folder
    """
    root_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..")
    dataset_dir = os.path.join(root_dir, "node_subsets", "ppr", dataset)
    os.makedirs(dataset_dir, exist_ok=True)

    is_val = "_val" if val else ""

    if split is None:
        file_name = f"sparse_adj-{alpha}_eps-{eps}".replace(".", "") + is_val + ".pt"
    else:
        file_name = f"sparse_adj-{alpha}_eps-{eps}".replace(".", "") + f"_Split-{split}" + is_val + ".pt"

    sparse_adj_file_path = os.path.join(dataset_dir, file_name)

    print(f"Saving data to {sparse_adj_file_path}")
    torch.save(sparse_adj, sparse_adj_file_path)



def create_sparse_ppr_matrix(neighbors, neighbor_weights):
    """
    For all calculated pairs, we can arrange in a NxN sparse weighted Adj matrix 
    """
    ppr_scores = []
    source_edge_ix, target_edge_ix = [], []
    for source_ix, (source_neighbors, source_weights) in enumerate(zip(neighbors, neighbor_weights)):
        source_edge_ix.extend([source_ix] * len(source_neighbors))
        target_edge_ix.extend(source_neighbors)
        ppr_scores.extend(source_weights)

    source_edge_ix = torch.Tensor(source_edge_ix).unsqueeze(0)
    target_edge_ix = torch.Tensor(target_edge_ix).unsqueeze(0)

    ppr_scores = torch.Tensor(ppr_scores)
    edge_ix = torch.cat((source_edge_ix, target_edge_ix), dim=0).long()

    num_nodes = len(neighbors)
    sparse_adj = SparseTensor.from_edge_index(edge_ix, ppr_scores, [num_nodes, num_nodes])

    return sparse_adj



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str)
    parser.add_argument("--eps", help="Stopping criterion threshold", type=float, default=5e-5)
    parser.add_argument("--alpha", help="Teleportation probability", type=float, default=0.15)
    parser.add_argument("--use-val-in-test", action='store_true', default=False)
    args = parser.parse_args()

    if args.data_name.lower() in ['cora', 'citeseer', 'pubmed']:
        data = read_data_planetoid(args)
        neighbors, neighbor_weights = get_ppr_matrix(data['edge_index'], data['num_nodes'], args.alpha, args.eps)
        sparse_adj = create_sparse_ppr_matrix(neighbors, neighbor_weights)
        save_results(args.data_name, sparse_adj, args.alpha, args.eps)
    else:
        dataset = PygLinkPropPredDataset(name=args.data_name)
        data = dataset[0]

        neighbors, neighbor_weights = get_ppr_matrix(data['edge_index'], data['num_nodes'], args.alpha, args.eps)
        sparse_adj = create_sparse_ppr_matrix(neighbors, neighbor_weights)
        save_results(args.data_name, sparse_adj, args.alpha, args.eps)

        if args.use_val_in_test:
            print("Running for Test...")
            split_edge = dataset.get_edge_split()
            val_edge_index = to_undirected(split_edge['valid']['edge'].t())
            full_edge_index = torch.cat([data['edge_index'], val_edge_index], dim=-1)

            neighbors, neighbor_weights = get_ppr_matrix(full_edge_index, data['num_nodes'], args.alpha, args.eps)
            sparse_adj = create_sparse_ppr_matrix(neighbors, neighbor_weights)
            save_results(args.data_name, sparse_adj, args.alpha, args.eps, val=True)


if __name__ == "__main__":
    main()
