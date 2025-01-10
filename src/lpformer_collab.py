import random
import numpy as np
from tqdm import tqdm 
from argparse import ArgumentParser
from collections import defaultdict

import torch
from torch.utils.data import DataLoader

import joblib
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
# from torch_geometric.nn.models import LPFormer, calc_ppr_matrix

from models.other_models import mlp_score
from models.link_transformer import LinkTransformer

from torch_geometric.nn.models import MLP as MLP2


parser = ArgumentParser()
parser.add_argument('--data_name', type=str, default='ogbl-collab')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--runs', help="# runs to run over", type=int, default=10)
parser.add_argument('--batch_size', type=int, default=24000)
parser.add_argument('--hidden_channels', type=int, default=128)
parser.add_argument('--gnn_layers', type=int, default=3)
parser.add_argument('--dropout', help="Applies to GNN and Transformer", type=float, default=0.1)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--eps', help="PPR precision", type=float, default=5e-5)
parser.add_argument('--thresholds', help="List of cn, 1-hop, >1-hop (in that order)", 
                    nargs="+", default=[0, 1e-4, 1e-2])
args = parser.parse_args()

seed=42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



### TODO: Remove!!!!!!!!!!!!!!
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

device = torch.device(args.device)

dataset = PygLinkPropPredDataset(name=args.data_name)
data = dataset[0].to(device)
data.edge_index = data.edge_index.to(device)

if hasattr(data, 'x') and data.x is not None:
    data.x = data.x.to(device).to(torch.float)

split_edge = dataset.get_edge_split()

# Common preprocessing step for ogbl-collab
# Taken from BUDDY Code
# See - https://github.com/melifluos/subgraph-sketching/blob/3562d94a07d1166faa0949030824bf75ad9bb2c4/src/data.py#L122
if "collab" in args.data_name:
    selected_year_index = torch.reshape((split_edge['train']['year'] >= 2007).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()
    # create adjacency matrix
    new_edges = to_undirected(train_edge_index, split_edge['train']['weight'], reduce='add')
    new_edge_index, new_edge_weight = new_edges[0], new_edges[1]
    data.edge_index = new_edge_index
    data.edge_weight = new_edge_weight.unsqueeze(-1)
    
train_pos = split_edge['train']['edge'].to(device)
valid_pos = split_edge['valid']['edge'].to(device)
valid_neg = split_edge['valid']['edge_neg'].to(device)
test_pos = split_edge['test']['edge'].to(device)
test_neg = split_edge['test']['edge_neg'].to(device)
split_data = {
    "train_pos": split_edge['train']['edge'].to(device),
    "valid_pos": split_edge['valid']['edge'].to(device),
    "valid_neg": split_edge['valid']['edge_neg'].to(device),
    "test_pos": split_edge['test']['edge'].to(device),
    "test_neg": split_edge['test']['edge_neg'].to(device)
}   

if hasattr(data, 'edge_weight') and data.edge_weight is not None:
    edge_weight = data.edge_weight.to(torch.float)
    data.edge_weight = data.edge_weight.view(-1).to(torch.float)
else:
    edge_weight = torch.ones(data.edge_index.size(1)).to(device).float()

# Convert edge_index to SparseTensor
adj_prop = SparseTensor.from_edge_index(data.edge_index, edge_weight.squeeze(-1), 
                                        [data.num_nodes, data.num_nodes]).to(device)
# adj_prop = adj_prop.to_symmetric().coalesce()
    
# Collab uses val edges during testing
# So we create additional adjacency that contains them
# If not collab, assign original adjacency
if "collab" in args.data_name:
    val_edge_index = split_edge['valid']['edge'].t()
    val_edge_index = to_undirected(val_edge_index)
    train_val_ei = torch.cat([data.edge_index, val_edge_index], dim=-1).to(device)

    val_edge_weight = torch.ones([val_edge_index.size(1), 1], dtype=torch.float, device=device)
    val_edge_weight = torch.cat([edge_weight.to(device), val_edge_weight.to(device)], 0).view(-1)
    train_val_adj_prop = SparseTensor.from_edge_index(train_val_ei, val_edge_weight, 
                                                   [data.num_nodes, data.num_nodes]).to(device)
else:
    train_val_adj_prop = adj_prop

# ppr_matrix = calc_ppr_matrix(data.edge_index, eps=args.eps)
a = f"/egr/research-dselab/shomerha/lpformer/node_subsets/ppr/{args.data_name}/sparse_adj-015_eps-5e-05.pt"
ppr_matrix = torch.load(a).to_torch_sparse_coo_tensor().to(device)
a = f"/egr/research-dselab/shomerha/lpformer/node_subsets/ppr/{args.data_name}/sparse_adj-015_eps-5e-05_val.pt"
ppr_matrix_val = torch.load(a).to_torch_sparse_coo_tensor().to(device)


data['adj_t'] = adj_prop
data['full_adj_t'] = train_val_adj_prop
data['adj_mask'] = data['adj_t'].to_symmetric().to_torch_sparse_coo_tensor()
data['adj_mask'] = data['adj_mask'].coalesce().bool().int()
data['full_adj_mask'] = data['full_adj_t'].to_symmetric().to_torch_sparse_coo_tensor()
data['full_adj_mask'] = data['full_adj_mask'].coalesce().bool().int()
data['ppr'] = ppr_matrix
data['ppr_test'] = ppr_matrix_val

margs = {
    "dim": 128,
    "num_heads": 1,
    "gnn_layers": 3,
    "trans_layers": 1,
    "residual": False,
    "layer_norm": True,
    "relu": True,
    "mask_input": True,
    "thresh_1hop": args.thresholds[1],
    "thresh_cn": args.thresholds[0],
    "thresh_non1hop": args.thresholds[-1],
    'dropout': 0.1,
    'gnn_drop': 0.1,
    'pred_dropout': 0.1,
    'att_drop': 0.1,
    "feat_drop": 0.1,
}  
model = LinkTransformer(margs, data, device=device).to(device)
# score_func = mlp_score(model.out_dim//2, model.out_dim//2, 1, 2).to(device)
score_func = MLP2([model.out_dim, model.out_dim, 1], norm=None).to(device)

optimizer = torch.optim.Adam(list(model.parameters()) + list(score_func.parameters()), lr=args.lr)

evaluator_hit = Evaluator(name=args.data_name)
# Eval metric differs by dataset
if "collab" in args.data_name.lower():
    test_metric = 50 
elif "ppa" in args.data_name.lower():
    test_metric = 100
elif "ddi" in args.data_name.lower():
    test_metric = 20 
evaluator_hit.K = test_metric


def train_epoch():
    model.train()
    score_func.train()
    train_pos = split_data['train_pos'].to(device)
    adjt_mask = torch.ones(train_pos.size(0), dtype=torch.bool, device=device)

    total_loss = total_examples = 0
    d = DataLoader(range(train_pos.size(0)), args.batch_size, shuffle=False)
    
    for perm in tqdm(d, "Epoch"):
        edges = train_pos[perm].t()

        # Mask positive input samples - Common strategy during training
        adjt_mask[perm] = 0
        edge2keep = train_pos[adjt_mask, :]
        masked_adj_prop = SparseTensor.from_edge_index(edge2keep.t(), sparse_sizes=(data['num_nodes'], data['num_nodes'])).to_device(device)
        masked_adj_prop = masked_adj_prop.to_symmetric()
        # For next batch
        adjt_mask[perm] = 1

        # print(adj_prop.sum().item(), masked_adj_prop.sum().item())

        pos_out = model(edges, masked_adj_prop, adj_mask=adjt_mask)
        pos_out = torch.sigmoid(score_func(pos_out).squeeze(-1))
        pos_loss = -torch.log(pos_out + 1e-6).mean()

        # Trivial random sampling
        neg_edges = torch.randint(0, data['num_nodes'], (edges.size(0), edges.size(1)), 
                                  dtype=torch.long, device=edges.device)
      
        neg_out = model(neg_edges)
        neg_out = torch.sigmoid(score_func(neg_out).squeeze(-1))
        neg_loss = -torch.log(1 - neg_out + 1e-6).mean()

        # exit()
        # print(pos_out.mean().item(), neg_out.mean().item())

        loss = pos_loss + neg_loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples   

    return total_loss / total_examples


@torch.no_grad()
def test():
    # NOTE: Eval for ogbl-citation2 is different
    # See `train.py` in https://github.com/HarryShomer/LPFormer/ for more
    # Also see there for how to eval under the HeaRT setting

    model.eval()
    score_func.eval()
    all_preds = defaultdict(list)

    for split_key, split_vals in split_data.items():
        if "train" not in split_key:
        # if "test" in split_key:            
            preds = []
            for perm in DataLoader(range(split_vals.size(0)), args.batch_size):
                edges = split_vals[perm].t()
                perm_logits = model(edges, test_set=True)
                preds += [torch.sigmoid(score_func(perm_logits)).cpu().squeeze(-1)]

            all_preds[split_key] = torch.cat(preds, dim=0)
    

    print("\nEval:")
    print("-------------------------------------------")
    print(torch.mean(all_preds['valid_pos']), torch.std(all_preds['valid_pos']))
    print(torch.mean(all_preds['valid_neg']), torch.std(all_preds['valid_neg']))
    print(torch.mean(all_preds['test_pos']), torch.std(all_preds['test_pos']))
    print(torch.mean(all_preds['test_neg']), torch.std(all_preds['test_neg']))


    val_hits = evaluator_hit.eval({
                    'y_pred_pos': all_preds['valid_pos'], 
                    'y_pred_neg': all_preds['valid_neg']
                })[f'hits@{test_metric}']
    test_hits = evaluator_hit.eval({
                    'y_pred_pos': all_preds['test_pos'], 
                    'y_pred_neg': all_preds['test_neg']
                })[f'hits@{test_metric}']

    return val_hits, test_hits



val_perf_runs = []
test_perf_runs = []

for run in range(1, args.runs+1):
    print("=" * 75)
    print(f"RUNNING run={run}")
    print("=" * 75)

    best_valid = 0
    best_valid_test = 0

    for epoch in range(1, 1 + args.epochs):
        loss = train_epoch()
        print(f"Epoch {epoch} Loss: {loss:.4f}\n")
                    
        if epoch % 1 == 0:
            print("Evaluating model...\n", flush=True)
            eval_val, eval_test = test()
    
            print(f"Valid Hits@{test_metric} = {eval_val}")
            print(f"Test Hits@{test_metric} = {eval_test}")

            if eval_val > best_valid:
                best_valid = eval_val
                best_valid_test = eval_test
        
    print(f"\nBest Performance is:\n  Valid={best_valid}\n  Test={best_valid_test}")
    val_perf_runs.append(best_valid)
    test_perf_runs.append(best_valid_test)

if args.runs > 1:
    print("\n\n")
    print(f"Results over {args.runs} runs:")
    print(f"  Valid = {np.mean(val_perf_runs)} +/- {np.std(val_perf_runs)}")
    print(f"  Test = {np.mean(test_perf_runs)} +/- {np.std(test_perf_runs)}")

