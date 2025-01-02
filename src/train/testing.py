import torch
from torch_scatter import scatter
from tqdm import tqdm
from torch.utils.data import DataLoader
from scipy import stats

from time import perf_counter

from util.utils import *
from train.evaluation import get_metric_score, get_metric_score_citation2, evaluate_hits, evaluate_mrr, sample_level_hits



def test_edge_citation2(model, score_func, input_data, h, batch_size, mrr_mode=False, negative_data=None, test=False):
    """
    Evaluate performance on val/test for citation2
    """
    preds = []

    if mrr_mode:
        source = input_data.t()[0]
        source = source.view(-1, 1).repeat(1, 1000).view(-1)
        target_neg = negative_data.view(-1)

        for perm in tqdm(DataLoader(range(source.size(0)), batch_size), "In Circles..."):
            src, dst_neg = source[perm], target_neg[perm]
            edge = torch.stack((src, dst_neg), dim=-1).t()
            
            elementwise_feats = model.elementwise_lin(h[src] * h[dst_neg])
            pairwise_feats, _ = model.calc_pairwise(edge, h, test_set=test)
            combined_feats = torch.cat((elementwise_feats, pairwise_feats), dim=-1)
            preds += [score_func(combined_feats).squeeze().cpu()]

        pred_all = torch.cat(preds, dim=0).view(-1, 1000)
    else:
        for perm in DataLoader(range(input_data.size(0)), batch_size):
            edge = input_data[perm].t()
            src, dst_neg = edge[0], edge[1]

            elementwise_feats = model.elementwise_lin(h[src] * h[dst_neg])
            pairwise_feats, _ = model.calc_pairwise(edge, h, test_set=test)
            combined_feats = torch.cat((elementwise_feats, pairwise_feats), dim=-1)
            preds += [score_func(combined_feats).squeeze().cpu()]

        pred_all = torch.cat(preds, dim=0)

    return pred_all


def test_citation2(model, score_func, data, evaluator_hit, evaluator_mrr, batch_size):
    """
    Specific to Citation2

    Prop only once to save time 
    """
    model.eval()
    score_func.eval()

    with torch.no_grad():
        h = model.propagate()

        neg_valid_pred = test_edge_citation2(model, score_func, data['valid_pos'], h, batch_size, mrr_mode=True, negative_data=data['valid_neg'])
        pos_valid_pred = test_edge_citation2(model, score_func, data['valid_pos'], h, batch_size)
        pos_test_pred  = test_edge_citation2(model, score_func, data['test_pos'], h, batch_size, test=True)
        neg_test_pred  = test_edge_citation2(model, score_func, data['test_pos'], h, batch_size, mrr_mode=True, negative_data=data['test_neg'], test=True)
        pos_train_pred = test_edge_citation2(model, score_func, data['train_pos_val'], h, batch_size)
            
        pos_valid_pred = pos_valid_pred.view(-1)
        pos_test_pred = pos_test_pred.view(-1)
        pos_train_pred = pos_valid_pred.view(-1)
            
        result = get_metric_score_citation2(evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
        
    return result


@torch.no_grad()
def test_edge(model, score_func, input_data, batch_size, test_set=False, dump_att=False):
    """
    Evaluate performance on val/test set
    """
    preds = []

    # for perm in tqdm(DataLoader(range(input_data.size(0)), batch_size), "Evaluating"):
    for perm in DataLoader(range(input_data.size(0)), batch_size):
        edge = input_data[perm].t()
        h = model(edge, test_set=test_set)
        preds += [score_func(h).cpu()]

    pred_all = torch.cat(preds, dim=0)

    return pred_all


@torch.no_grad()
def test_heart_negatives(negative_data, model, score_func, batch_size=32768, test_set=False):
    """
    For HeaRT setting
    """    
    neg_preds = []
    num_negative = negative_data.size(1)
    negative_data = torch.permute(negative_data, (2, 0, 1)).reshape(2, -1).t()

    # TODO: Move to parent function so only run once
    h = model.propagate()

    qqq = DataLoader(range(negative_data.size(0)),  batch_size)
    # qqq = tqdm(qqq, "Testing")

    for perm in qqq:
        neg_edges = negative_data[perm].t().to(h.device)

        elementwise_feats = model.elementwise_lin(h[neg_edges[0]] * h[neg_edges[1]])
        pairwise_feats, _ = model.calc_pairwise(neg_edges, h, test_set=test_set)
        combined_feats = torch.cat((elementwise_feats, pairwise_feats), dim=-1)

        neg_preds += [score_func(combined_feats).squeeze().cpu()]

    neg_preds = torch.cat(neg_preds, dim=0).view(-1, num_negative)

    return neg_preds


def test(
        model, 
        score_func, 
        data, 
        evaluator_hit, 
        evaluator_mrr, 
        batch_size, 
        k_list=[100],
        heart=False,
        dump_att=False,
        dump_test=False,  # Get performance on sample level
        metric="Hits@100"
    ):
    model.eval()
    score_func.eval()

    with torch.no_grad():
        pos_train_pred = test_edge(model, score_func, data['train_pos_val'], batch_size)
        pos_valid_pred = test_edge(model, score_func, data['valid_pos'], batch_size)
        pos_test_pred = test_edge(model, score_func, data['test_pos'], batch_size, test_set=True, dump_att=dump_att)

        if heart:
            neg_valid_pred = test_heart_negatives(data['valid_neg'], model, score_func, batch_size=batch_size)
            neg_test_pred = test_heart_negatives(data['test_neg'], model, score_func, batch_size=batch_size, test_set=True)

            pos_valid_pred = pos_valid_pred.view(-1)
            pos_test_pred = pos_test_pred.view(-1)
            pos_train_pred = pos_train_pred.view(-1)
            
            result = get_metric_score_citation2(evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred)
        else:
            neg_valid_pred = test_edge(model, score_func, data['valid_neg'], batch_size)
            neg_test_pred = test_edge(model, score_func, data['test_neg'], batch_size, test_set=True, dump_att=dump_att)

            neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
            
            result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, 
                                      neg_valid_pred, pos_test_pred, neg_test_pred, k_list)

    if dump_test:
        all_sample_hits = []

        for perm in tqdm(DataLoader(range(pos_test_pred.size(0)),  4096), "Sample-Level Hits"):
            a = sample_level_hits(pos_test_pred[perm], neg_test_pred)
            all_sample_hits += [a[metric]]
        
        all_sample_hits = torch.cat(all_sample_hits)
        return result, all_sample_hits

    return result


