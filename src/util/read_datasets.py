import os 
import torch
import numpy as np
from torch_sparse import SparseTensor
from torch.nn.init import xavier_uniform_

import torch_geometric.transforms as T
from torch_geometric.utils import to_undirected, degree

import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset

from util.calc_ppr_scores import get_ppr


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "..", "dataset")
HEART_DIR = os.path.join(DATA_DIR, "heart")


def read_data_ogb(args, device):
    """
    Read data for OGB datasets
    """
    data_obj = {
        "dataset": args.data_name,
    }

    print("Loading all data...")

    dataset = PygLinkPropPredDataset(name=args.data_name)
    data = dataset[0].to(device)
    split_edge = dataset.get_edge_split()

    if "collab" in args.data_name:
        data, split_edge = filter_by_year(data, split_edge)
        data = data.to(device)

    data_obj['num_nodes'] = data.num_nodes
    edge_index = data.edge_index

    if args.data_name != 'ogbl-citation2':
        data_obj['train_pos'] = split_edge['train']['edge'].to(device)
        data_obj['valid_pos'] = split_edge['valid']['edge'].to(device)
        data_obj['valid_neg'] = split_edge['valid']['edge_neg'].to(device)
        data_obj['test_pos'] = split_edge['test']['edge'].to(device)
        data_obj['test_neg'] = split_edge['test']['edge_neg'].to(device)
    else:
        source_edge, target_edge = split_edge['train']['source_node'], split_edge['train']['target_node']
        data_obj['train_pos'] = torch.cat([source_edge.unsqueeze(1), target_edge.unsqueeze(1)], dim=-1).to(device)

        source, target = split_edge['valid']['source_node'],  split_edge['valid']['target_node']
        data_obj['valid_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
        data_obj['valid_neg'] = split_edge['valid']['target_node_neg'].to(device) 

        source, target = split_edge['test']['source_node'],  split_edge['test']['target_node']
        data_obj['test_pos'] = torch.cat([source.unsqueeze(1), target.unsqueeze(1)], dim=-1).to(device)
        data_obj['test_neg'] = split_edge['test']['target_node_neg'].to(device)

    # Overwrite Val/Test pos sample for ogbl-ppa under HeaRT
    if args.heart and "ppa" in args.data_name:
        with open(f'{HEART_DIR}/{args.data_name}/valid_samples_index.pt', "rb") as f:
            val_pos_ix = torch.load(f)
        with open(f'{HEART_DIR}/{args.data_name}/test_samples_index.pt', "rb") as f:
            test_pos_ix = torch.load(f)

        data_obj['valid_pos'] = data_obj['valid_pos'][val_pos_ix, :]
        data_obj['test_pos'] = data_obj['test_pos'][test_pos_ix, :]

    # Test train performance without evaluating all test samples
    idx = torch.randperm(data_obj['train_pos'].size(0))[:data_obj['valid_pos'].size(0)]
    data_obj['train_pos_val'] = data_obj['train_pos'][idx]

    if hasattr(data, 'x') and data.x is not None:
        data_obj['x'] = data.x.to(device).to(torch.float)
    else:
        data_obj['x'] =  torch.nn.Parameter(torch.zeros(data_obj['num_nodes'], args.dim).to(device))
        xavier_uniform_(data_obj['x'])

    if hasattr(data, 'edge_weight') and data.edge_weight is not None:
        edge_weight = data.edge_weight.to(torch.float)
        data.edge_weight = data.edge_weight.view(-1).to(torch.float)
    else:
        edge_weight = torch.ones(edge_index.size(1)).to(device).float()
        
    data_obj['adj_t'] = SparseTensor.from_edge_index(edge_index, edge_weight.squeeze(-1), [data.num_nodes, data.num_nodes]).to(device)

    # TODO: Needed since directed graph
    if args.data_name == 'ogbl-citation2': 
        data_obj['adj_t'] = data_obj['adj_t'].to_symmetric().coalesce()
        data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()
    else:
        data_obj['adj_mask'] = data_obj['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()        
    
    # Don't use edge weight. Only 0/1. Not needed for masking
    data_obj['adj_mask'] = data_obj['adj_mask'].coalesce().bool().int()

    if args.use_val_in_test:
        val_edge_index = split_edge['valid']['edge'].t()
        val_edge_index = to_undirected(val_edge_index).to(device)

        full_edge_index = torch.cat([edge_index, val_edge_index], dim=-1)
        data['full_edge_index'] = full_edge_index.to(device)

        val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float, device=device)
        val_edge_weight = torch.cat([edge_weight, val_edge_weight], 0).view(-1)
        data_obj['full_adj_t'] = SparseTensor.from_edge_index(full_edge_index, val_edge_weight, [data.num_nodes, data.num_nodes]).to(device)

        # Don't use edge weight. Only 0/1. Not needed for masking
        data_obj['full_adj_mask'] = data_obj['full_adj_t'].to_torch_sparse_coo_tensor()
        data_obj['full_adj_mask'] = data_obj['full_adj_mask'].coalesce().bool().int()
    else:
        data_obj['full_adj_t'] = data_obj['adj_t']
        data_obj['full_adj_mask'] = data_obj['adj_mask']

    data_obj['degree'] = degree(edge_index[0], num_nodes=data_obj['num_nodes']).to(device)
    if args.use_val_in_test:
        data_obj['degree_test'] = degree(full_edge_index[0], num_nodes=data_obj['num_nodes']).to(device)

    ### Load PPR matrix
    print("Reading PPR...", flush=True)
    data_obj['ppr'] = get_ppr(args.data_name, edge_index, data['num_nodes'],
                              0.15, args.eps, False).to(device)  

    if args.use_val_in_test:
        data_obj['ppr'] = get_ppr(args.data_name, data['full_edge_index'], data['num_nodes'],
                                0.15, args.eps, True).to(device)  
    else:
        data_obj['ppr_test'] = data_obj['ppr']

    # Overwrite standard negatives
    if args.heart:
        with open(f'{HEART_DIR}/{args.data_name}/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            data_obj['valid_neg'] = torch.from_numpy(neg_valid_edge).to(device)
        with open(f'{HEART_DIR}/{args.data_name}/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            data_obj['test_neg'] = torch.from_numpy(neg_test_edge).to(device)

        # For DDI, val/test takes a long time so only use a subset of val
        if "ddi" in args.data_name:
            num_sample = data_obj['valid_pos'].size(0) // 4
            idx = torch.randperm(data_obj['valid_pos'].size(0))[:num_sample].to(device)
            data_obj['valid_pos'] = data_obj['valid_pos'][idx]
            data_obj['valid_neg'] = data_obj['valid_neg'][idx]
            data_obj['train_pos_val'] = data_obj['train_pos_val'][idx]

    return data_obj




def read_data_planetoid(args, device):
    """
    Read all data for the fixed split. Returns as dict
    """
    data_name = args.data_name

    node_set = set()
    train_pos, valid_pos, test_pos = [], [], []
    train_neg, valid_neg, test_neg = [], [], []

    for split in ['train', 'test', 'valid']:
        path = os.path.join(DATA_DIR, data_name, f"{split}_pos.txt")

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
        path = os.path.join(DATA_DIR, data_name, f"{split}_neg.txt")

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
          
    train_pos_tensor = torch.tensor(train_pos)

    valid_pos = torch.tensor(valid_pos)
    valid_neg =  torch.tensor(valid_neg)

    test_pos =  torch.tensor(test_pos)
    test_neg =  torch.tensor(test_neg)

    idx = torch.randperm(train_pos_tensor.size(0))
    idx = idx[:valid_pos.size(0)]
    train_val = train_pos_tensor[idx]

    feature_embeddings = torch.load(os.path.join(DATA_DIR, data_name, "gnn_feature"))
    feature_embeddings = feature_embeddings['entity_embedding']

    data = {"dataset": args.data_name}
    data['edge_index'] = edge_index.to(device)
    data['num_nodes'] = num_nodes

    data['train_pos'] = train_pos_tensor.to(device)
    data['train_pos_val'] = train_val.to(device)

    data['valid_pos'] = valid_pos.to(device)
    data['valid_neg'] = valid_neg.to(device)
    data['test_pos'] = test_pos.to(device)
    data['test_neg'] = test_neg.to(device)

    data['x'] = feature_embeddings.to(device)

    data['adj_t'] = SparseTensor.from_edge_index(edge_index, edge_weight, [num_nodes, num_nodes]).to(device)
    data['full_adj_t'] = data['adj_t'].to(device)

    data['adj_mask'] = data['adj_t'].to_torch_sparse_coo_tensor()
    data['full_adj_mask'] = data['adj_mask'] = data['adj_mask'].coalesce()

    ### Degree of nodes
    data['degree'] = degree(data['edge_index'][0], num_nodes=data['num_nodes']).to(device)

    ### Load PPR Matrix
    data['ppr'] = get_ppr(args.data_name, data['edge_index'], data['num_nodes'],
                          0.15, args.eps, False).to(device)
    data['ppr_test'] = data['ppr']

    # Overwrite standard negative
    if args.heart:
        with open(f'{HEART_DIR}/{args.data_name}/heart_valid_samples.npy', "rb") as f:
            neg_valid_edge = np.load(f)
            data['valid_neg'] = torch.from_numpy(neg_valid_edge)
        with open(f'{HEART_DIR}/{args.data_name}/heart_test_samples.npy', "rb") as f:
            neg_test_edge = np.load(f)
            data['test_neg'] = torch.from_numpy(neg_test_edge)

    return data



    
def filter_by_year(data, split_edge, year=2007):
    """
    From BUDDY code

    remove edges before year from data and split edge
    @param data: pyg Data, pyg SplitEdge
    @param split_edges:
    @param year: int first year to use
    @return: pyg Data, pyg SplitEdge
    """
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= year).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    return data, split_edge


