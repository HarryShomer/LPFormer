import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_hits(evaluator, pos_pred, neg_pred, k_list):
    results = {}
    for K in k_list:
        evaluator.K = K
        hits = evaluator.eval({
            'y_pred_pos': pos_pred,
            'y_pred_neg': neg_pred,
        })[f'hits@{K}']

        results[f'Hits@{K}'] = hits

    return results        



def compute_edge_cn(
        input_data, 
        batch_size, 
        adj,
    ):
    all_cns = []

    for perm in tqdm(DataLoader(range(input_data.size(0)), batch_size), "Getting Local/Global"):
        edge = input_data[perm].t()

        # Define masks
        source_row = adj[edge[0]].to_dense().bool()
        target_row = adj[edge[1].tolist()].to_dense().bool()

        num_cns = local_mask = source_row & target_row

        # NOTE: Counts
        all_cns  += [num_cns.sum(axis=1).cpu()] 

    all_cns = torch.cat(all_cns, dim=0).float()

    return all_cns


def test_by_metric(
        model, 
        score_func, 
        data, 
        evaluator_hit, 
        evaluator_mrr, 
        cmd_args, 
        k_list=[100],
    ):
    """
    Performance by metric range.
    """
    model.eval()
    score_func.eval()

    cn_vals = [(0, 1), (1, 3), (3, 10), (10, 1000000)]

    with torch.no_grad():
        print("Getting Positive Preds ...")
        pos_test_pred = ...
        print("Getting Negative Preds ...")
        neg_test_pred = ...

        cn_vals = compute_edge_cn(data['test_pos'], 1024, data['adj'])

        all_results = {}
        for cn_bin in cn_vals:
            cn_ix = (cn_vals >= cn_bin[0]) & (cn_vals < cn_bin[1])
            pos_test_pred_bin = pos_test_pred[cn_ix]
            bin_result = evaluate_hits(evaluator_hit, pos_test_pred_bin, neg_test_pred, k_list)

            print(cn_bin, bin_result)

    return all_results      