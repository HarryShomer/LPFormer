import os
import torch
from tqdm import tqdm
from datetime import datetime   
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor

import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from util.utils import *
from train.testing import *

from models.other_models import mlp_score
from models.link_transformer import LinkTransformer, NUM_NODES_CHOSEN


DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "dataset")



def train_epoch(model, score_func, data, optimizer, args, device):
    model.train()
    score_func.train()
    train_pos = data['train_pos'].to(device)

    adjmask = torch.ones(train_pos.size(0), dtype=torch.bool, device=device)
    adjt_mask = torch.ones(train_pos.size(0), dtype=torch.bool, device=device)

    total_loss = total_examples = 0
    d = DataLoader(range(train_pos.size(0)), args.batch_size, shuffle=True)
    d = tqdm(d, "Epoch") #if args.verbose else d
    
    for perm in d:
        edges = train_pos[perm].t()

        # Remove positive samples from adj_mask used in calculating pairwise info
        # Only needed for positive bec. otherwise don't exist
        adjmask[perm] = 0
        edge2keep = train_pos[adjmask, :]
        masked_adj = SparseTensor.from_edge_index(edge2keep.t(), sparse_sizes=(data['num_nodes'], data['num_nodes'])).to_device(device)
        masked_adj = masked_adj.to_symmetric()
        masked_adj = masked_adj.to_torch_sparse_coo_tensor().coalesce().bool().int()
        adjmask[perm] = 1  # For next batch + negatives

        if args.mask_input:
            adjt_mask[perm] = 0
            edge2keep = train_pos[adjt_mask, :]
            
            masked_adjt = SparseTensor.from_edge_index(edge2keep.t(), sparse_sizes=(data['num_nodes'], data['num_nodes'])).to_device(device)
            masked_adjt = masked_adjt.to_symmetric()
            
            # For next batch
            adjt_mask[perm] = 1
        else:
            masked_adjt = None

        h = model(edges, adj_prop=masked_adjt, adj_mask=masked_adj)
        pos_out = score_func(h)
        pos_loss = -torch.log(pos_out + 1e-6).mean()

        # Just do some trivial random sampling for negative samples
        neg_edges = torch.randint(0, data['num_nodes'], (edges.size(0), edges.size(1) * args.num_negative), dtype=torch.long, device=h.device)
        
        h = model(neg_edges)
        neg_out = score_func(h)
        neg_loss = -torch.log(1 - neg_out + 1e-6).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(score_func.parameters(), 1.0)

        optimizer.step()
        optimizer.zero_grad()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples   

    # for k, v in NUM_NODES_CHOSEN.items():
    #     print(k, np.mean(v))
    # exit()

    return total_loss / total_examples



def train_loop(args, train_args, data, device, loggers, seed, model_save_name, verbose):
    """
    Train over N epochs
    """
    k_list = [20, 50, 100]
    eval_metric = args.metric
    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = Evaluator(name='ogbl-citation2') if 'MRR' in loggers else None

    model = LinkTransformer(train_args, data, device=device).to(device)
    score_func = mlp_score(model.out_dim, model.out_dim, 1, args.pred_layers, train_args['pred_dropout']).to(device)
                        
    optimizer = torch.optim.Adam(list(model.parameters()) + list(score_func.parameters()), lr=train_args['lr'], weight_decay=train_args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: train_args['decay'] ** e)
    
    kill_cnt = 0
    best_valid = 0

    for epoch in range(1, 1 + args.epochs):
        print(f">>> Epoch {epoch} - {datetime.now().strftime('%H:%M:%S')}\n" if verbose else "", flush=True, end="")

        loss = train_epoch(model, score_func, data, optimizer, args, device)
        print(f"Epoch {epoch} Loss: {loss:.4f}\n"  if verbose else "", end="")
                    
        if epoch % args.eval_steps == 0:
            print("Evaluating model...\n" if verbose else "", flush=True, end="")
            
            if "citation" not in args.data_name.lower() or args.heart:
                results_rank = test(model, score_func, data, evaluator_hit, evaluator_mrr, args.test_batch_size, k_list, heart=args.heart)
            else:
                results_rank = test_citation2(model, score_func, data, evaluator_hit, evaluator_mrr, args.test_batch_size)

            print(f"Epoch {epoch} Results:\n-----------------\n"  if verbose else "", end="", flush=True)
            for key, result in results_rank.items():
                loggers[key].add_result(seed, result)
                if args.metric == key:
                    print(f"  {key} = {result}\n"  if verbose else "", end="", flush=True)

            best_valid_current = torch.tensor(loggers[eval_metric].results[seed])[:, 1].max()

            if best_valid_current > best_valid:
                kill_cnt = 0
                best_valid = best_valid_current
                if model_save_name is not None:
                    save_model(model, score_func, optimizer, model_save_name + ".pt")
            else:
                kill_cnt += 1
                
                if kill_cnt > args.kill_cnt: 
                    break
                    
        scheduler.step()
    
    return best_valid


def train_data(args, train_args, data, device, verbose=True):
    """
    Run over n random seeds
    """
    init_seed(args.seed)

    if args.save_as is not None:
        model_save_name = os.path.join("checkpoints", args.data_name, args.save_as)
    else:
        model_save_name = None

    loggers = {
        'Hits@20': Logger(args.runs),
        'Hits@50': Logger(args.runs),
        'Hits@100': Logger(args.runs),
    }
    if "citation" in data['dataset'] or data['dataset'] in ['cora', 'citeseer', 'pubmed',  'chameleon', 'squirrel'] or args.heart:
        loggers['MRR'] = Logger(args.runs)

    # Over N splits
    best_valid_results = []

    for seed in tqdm(range(0, args.runs), f"Training over {args.runs} seeds"):
        if args.runs > 1:
            init_seed(seed)

        run_save_name = model_save_name
        if model_save_name is not None and args.runs > 1:
            run_save_name = model_save_name + f"_seed-{seed+1}"

        best_valid = train_loop(args, train_args, data, device, loggers, seed, run_save_name, verbose)
        best_valid_results.append(best_valid)

    for key in loggers.keys():     
        if key == args.metric:
            print(key + "\n" + "-" * len(key))  
            # Both lists. [0] = Train, [1] = Valid, [2] = Test
            best_mean, best_var = loggers[key].print_statistics()
    
    return best_mean[1], f"{best_mean[1]} ± {best_var[1]}", f"{best_mean[2]} ± {best_var[2]}"




