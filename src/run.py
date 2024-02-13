import torch
import argparse
from collections import defaultdict

import joblib  # Make ogb loads faster...idk
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator

from util.utils import *
from util.read_datasets import read_data_ogb, read_data_planetoid
from train.train_model import train_data, test, compute_all_ppr, test_by_all, test_with_att

from models.other_models import mlp_score
from models.link_transformer import LinkTransformer


def save_test_att(cmd_args):
    """
    Save test attention by for each seed
    """
    device = torch.device(f'cuda:{cmd_args.device}' if torch.cuda.is_available() else 'cpu')

    if cmd_args.data_name.lower() in ['cora', 'citeseer', 'pubmed']:
        data = read_data_planetoid(cmd_args, device)
    else:
        data = read_data_ogb(cmd_args, device)

    args = {
        "dim": cmd_args.dim,
        "num_heads": cmd_args.num_heads,
        "gnn_layers": cmd_args.gnn_layers,
        "trans_layers": cmd_args.tlayers,
        "residual": cmd_args.residual,
        "layer_norm": not cmd_args.no_layer_norm,
    }  

    fields = ['thresh_1hop', "mask_input", "count_ra", "filter_1hop", 
              "filter_cn", "thresh_cn", "thresh_non1hop", "ablate_att", 
              "ablate_pe", "ablate_feats", "ablate_ppr", "ablate_counts",
              ]
    for f in fields:
        args[f] = getattr(cmd_args, f)

    model = LinkTransformer(args, data, device=device)
    score_func = mlp_score(model.out_dim, model.out_dim, 1, cmd_args.pred_layers)

    cn_nodes, onehop_nodes, non1hop_nodes = [], [], []

    for run in range(1, cmd_args.runs+1):
        print(f"\n>>> Seed={run}")
        file_seed = os.path.join("checkpoints", cmd_args.data_name, f"{cmd_args.checkpoint}_seed-{run}.pt")
        model, score_func = load_model(model, score_func, file_seed, device)

        att_scores = test_with_att(model, score_func, data, cmd_args.batch_size)
        
        cn_nodes += [att_scores[0]]
        onehop_nodes += [att_scores[1]]
        non1hop_nodes += [att_scores[2]]
    
    cn_nodes = torch.cat(cn_nodes, dim=0)
    onehop_nodes = torch.cat(onehop_nodes, dim=0)
    non1hop_nodes = torch.cat(non1hop_nodes, dim=0)

    att_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "att_scores")
    torch.save(cn_nodes, os.path.join(att_dir, f"{cmd_args.data_name}_CN_att_scores.pt"))
    torch.save(onehop_nodes, os.path.join(att_dir, f"{cmd_args.data_name}_1Hop_att_scores.pt"))
    torch.save(non1hop_nodes, os.path.join(att_dir, f"{cmd_args.data_name}_Non1Hop_att_scores.pt"))



def eval_model(cmd_args):
    """
    """
    k_list = [20, 50, 100]
    device = torch.device(f'cuda:{cmd_args.device}' if torch.cuda.is_available() else 'cpu')

    if cmd_args.data_name.lower() in ['cora', 'citeseer', 'pubmed']:
        data = read_data_planetoid(cmd_args, device)
    else:
        data = read_data_ogb(cmd_args, device)

    args = {
        "dim": cmd_args.dim,
        "num_heads": cmd_args.num_heads,
        "gnn_layers": cmd_args.gnn_layers,
        "trans_layers": cmd_args.tlayers,
        "residual": cmd_args.residual,
        "layer_norm": not cmd_args.no_layer_norm,
        "relu": not cmd_args.no_relu
    }  

    fields = ['thresh_1hop', "mask_input", "count_ra",  
              "filter_1hop", "filter_cn", "thresh_cn", "thresh_non1hop", 
              "ablate_att", "ablate_pe", "ablate_feats", "ablate_ppr", 
              "ablate_counts", "ablate_ppr_type"
              ]
    for f in fields:
        args[f] = getattr(cmd_args, f)

    model = LinkTransformer(args, data, device=device)
    score_func = mlp_score(model.out_dim, model.out_dim, 1, cmd_args.pred_layers)

    evaluator_hit = Evaluator(name='ogbl-collab')
    evaluator_mrr = None
    if "citation" in data['dataset'] or data['dataset'] in ['cora', 'citeseer', 'pubmed'] or cmd_args.heart:
        evaluator_mrr = Evaluator(name='ogbl-citation2')

    if cmd_args.heart:
        cmd_args.metric = 'MRR'
    elif cmd_args.data_name =='ogbl-collab':
        cmd_args.metric = 'Hits@50'
    elif cmd_args.data_name =='ogbl-ddi':
        cmd_args.metric = 'Hits@20'
    elif cmd_args.data_name =='ogbl-ppa':
        cmd_args.metric = 'Hits@100'
    elif cmd_args.data_name =='ogbl-citation2':
        cmd_args.metric = 'MRR'
    else:
        cmd_args.metric = 'MRR'


    if "ogb" in data['dataset']:
        ppr_scores = torch.load(f"heuristic_data/{data['dataset']}_ppr_scores.pt")
    else:
        ppr_scores = compute_all_ppr(data, data['test_pos'], test_set=True)
    print(">>>", ppr_scores.shape)

    # Results by CN range
    # TODO: Rand split
    if cmd_args.runs > 1:
        cn_all_seed_results = defaultdict(list)
        all_seed_results = []

        for run in range(1, cmd_args.runs+1):
            print(f"\n>>> Seed={run}")
            file_seed = os.path.join("checkpoints", cmd_args.data_name, f"{cmd_args.checkpoint}_seed-{run}.pt")
            model, score_func = load_model(model, score_func, file_seed, device)

            # Cumulative Results
            # TODO: Citation2
            # q = test(model, score_func, data, evaluator_hit, evaluator_mrr, cmd_args.batch_size, k_list=k_list)
            # all_seed_results.append(q[cmd_args.metric][-1])

            seed_results = test_by_all(model, score_func, data, evaluator_hit, 
                                       evaluator_mrr, cmd_args, ppr_scores, k_list=k_list)

            # print(seed_results)
            for k, v in seed_results.items():
                cn_all_seed_results[k].append(v[cmd_args.metric])
        
        # By Range
        print("\n\n")
        for k, v in cn_all_seed_results.items():
            print(f"{k} => {np.mean(v)}")

        # Cumulative
        print("\nMean Performance:")
        print(f"    {cmd_args.metric} -->", np.mean(all_seed_results))
    else:
        file = os.path.join("checkpoints", cmd_args.data_name, cmd_args.checkpoint + ".pt")
        model, score_func = load_model(model, score_func, file, device)

        # TODO: Citation2
        results_rank = test(model, score_func, data, evaluator_hit, evaluator_mrr, cmd_args.batch_size, k_list=k_list)
        for key, result in results_rank.items():
            print(f"  {key} = {result}")
        
        # test_with_att(model, score_func, data, cmd_args.batch_size)



def run_model(cmd_args):
    """
    Run model using args
    """
    device = torch.device(f'cuda:{cmd_args.device}' if torch.cuda.is_available() else 'cpu')
    # device = "cpu"  # DEBUG

    if cmd_args.data_name.lower() in ['cora', 'citeseer', 'pubmed']:
        data = read_data_planetoid(cmd_args, device)
    else:
        data = read_data_ogb(cmd_args, device)

    if cmd_args.data_name =='ogbl-collab':
        cmd_args.metric = 'Hits@50'
        gcn_cache = False
    elif cmd_args.data_name =='ogbl-ddi':
        cmd_args.metric = 'Hits@20'
        gcn_cache = True
    elif cmd_args.data_name =='ogbl-ppa':
        cmd_args.metric = 'Hits@100'
        gcn_cache = True
    elif cmd_args.data_name =='ogbl-citation2':
        cmd_args.metric = 'MRR'
        gcn_cache = True
    else:
        cmd_args.metric = 'MRR'
        gcn_cache = False

    # Overwrite
    if cmd_args.heart:
        cmd_args.metric = 'MRR'

    args = {
        'gcn_cache': gcn_cache,
        'gnn_layers': cmd_args.gnn_layers,
        'trans_layers': cmd_args.tlayers,
        'dim': cmd_args.dim,
        'num_heads': cmd_args.num_heads,
        'lr': cmd_args.lr,
        'weight_decay': cmd_args.l2,
        'decay': cmd_args.decay,
        'dropout': cmd_args.dropout,
        'gnn_drop': cmd_args.gnn_drop,
        'pred_dropout': cmd_args.pred_drop,
        'att_drop': cmd_args.att_drop,
        "feat_drop": cmd_args.feat_drop,
        "residual": cmd_args.residual,
        "layer_norm": not cmd_args.no_layer_norm,
        "relu": not cmd_args.no_relu
    }

    # Important shit from argparse
    fields = ['thresh_1hop', "mask_input", "count_ra",  
              "filter_1hop", "filter_cn", "thresh_cn", "thresh_non1hop", 
              "ablate_att", "ablate_pe", "ablate_feats", "ablate_ppr", 
              "ablate_counts", "ablate_ppr_type"
              ]
    for f in fields:
        args[f] = getattr(cmd_args, f)

    train_data(cmd_args, args, data, device, verbose = not cmd_args.non_verbose)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='ogb-collab')
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument("--mask-input", action='store_true', default=False)
    parser.add_argument("--non-verbose", action='store_true', default=False)

    # Model Settings
    parser.add_argument('--dim', type=int, default=256)
    parser.add_argument('--tlayers', type=int, default=1)
    parser.add_argument('--num-heads', type=int, default=1)
    parser.add_argument('--gnn-layers', type=int, default=2)
    parser.add_argument('--pred-layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--gnn-drop', type=float, default=0.2)
    parser.add_argument('--att-drop', type=float, default=0.1)
    parser.add_argument('--pred-drop', type=float, default=0)
    parser.add_argument('--feat-drop', type=float, default=0)
    parser.add_argument("--residual", action='store_true', default=False)
    parser.add_argument("--no-layer-norm", action='store_true', default=False)
    parser.add_argument("--no-relu", action='store_true', default=False)

    # Train Settings
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--decay', type=float, default=1)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--test-batch-size', type=int, default=32768)
    parser.add_argument('--num-negative', type=int, default=1)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--kill_cnt', dest='kill_cnt', default=100, type=int, help='early stopping')
    parser.add_argument('--output_dir', type=str, default='output_test')
    parser.add_argument('--l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--use-val-in-test", action='store_true', default=False)
    parser.add_argument("--remove-pos-edges", action='store_true', default=False)
    
    parser.add_argument("--heart", action='store_true', default=False)
    parser.add_argument('--save-as', type=str, default=None)
    parser.add_argument('--metric', type=str, default='Hits@100')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)

    parser.add_argument("--dump-att", action='store_true', default=False)
    parser.add_argument("--eval", action='store_true', default=False)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--bymetric", type=str, default="cn")
    parser.add_argument('--percentile', type=float, default=75)

    parser.add_argument('--eps', type=float, default=1e-4)
    parser.add_argument('--thresh-cn', type=float, default=1e-2)
    parser.add_argument('--thresh-1hop', type=float, default=1e-2)
    parser.add_argument('--thresh-non1hop', type=float, default=1e-2)

    parser.add_argument("--filter-cn", action='store_true', default=False)
    parser.add_argument("--filter-1hop", action='store_true', default=False)
    parser.add_argument("--count-ra", action='store_true', default=False)

    # Ablation study params
    parser.add_argument("--ablate-att", action='store_true', default=False)
    parser.add_argument("--ablate-pe", action='store_true', default=False)
    parser.add_argument("--ablate-feats", action='store_true', default=False)
    parser.add_argument("--ablate-ppr", action='store_true', default=False)
    parser.add_argument("--ablate-counts", action='store_true', default=False)
    parser.add_argument("--ablate-ppr-type", action='store_true', default=False)

    args = parser.parse_args()

    init_seed(args.seed)
    args.test_batch_size = args.batch_size if args.test_batch_size is None else args.test_batch_size

    if args.eval:
        eval_model(args)
    elif args.dump_att:
        save_test_att(args)
    else:
        run_model(args)


if __name__ == "__main__":
    main()