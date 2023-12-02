import torch
from torch_scatter import scatter
from tqdm import tqdm
from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from collections import defaultdict

from util.utils import *
from train.evaluation import get_metric_score, get_metric_score_citation2, evaluate_hits, evaluate_mrr, get_ranking_list



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
        # h = None

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
def test_edge_heart(score_func, input_data, h, batch_size,  negative_data):
    """
    For HeaRT setting
    """
    pos_preds = []
    neg_preds = []
        
    for perm in DataLoader(range(input_data.size(0)),  batch_size):
        pos_edges = input_data[perm].t()
        neg_edges = torch.permute(negative_data[perm], (2, 0, 1))

        pos_scores = score_func(h[pos_edges[0]], h[pos_edges[1]]).cpu()
        neg_scores = score_func(h[neg_edges[0]], h[neg_edges[1]]).cpu()

        pos_preds += [pos_scores]
        neg_preds += [neg_scores]
    
    neg_preds = torch.cat(neg_preds, dim=0)
    pos_preds = torch.cat(pos_preds, dim=0)

    return pos_preds, neg_preds


def test(
        model, 
        score_func, 
        data, 
        evaluator_hit, 
        evaluator_mrr, 
        batch_size, 
        k_list=[100],
        heart=False,
        dump_att=False
    ):
    model.eval()
    score_func.eval()

    test_edge_func = test_edge_heart if heart else test_edge
    
    with torch.no_grad():
        pos_train_pred = test_edge_func(model, score_func, data['train_pos_val'], batch_size)

        pos_valid_pred = test_edge_func(model, score_func, data['valid_pos'], batch_size)
        neg_valid_pred = test_edge_func(model, score_func, data['valid_neg'], batch_size)

        pos_test_pred = test_edge_func(model, score_func, data['test_pos'], batch_size, test_set=True, dump_att=dump_att)
        neg_test_pred = test_edge_func(model, score_func, data['test_neg'], batch_size, test_set=True, dump_att=dump_att)

        neg_valid_pred, pos_valid_pred = torch.flatten(neg_valid_pred),  torch.flatten(pos_valid_pred)
        pos_test_pred, neg_test_pred = torch.flatten(pos_test_pred), torch.flatten(neg_test_pred)

        result = get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, k_list)

    return result


################################################################################
################################################################################
################################################################################


def test_edge_att(model, score_func, input_data, batch_size, test_set=False):
    """
    Evaluate performance on val/test set
    """
    atts = []
    masks = []
    node_indicators = []

    for ix, perm in enumerate(DataLoader(range(input_data.size(0)), batch_size)):
        edge = input_data[perm].t()
        _, a = model(edge, test_set=test_set, return_weights=True)

        # Need to make batch size relative to each other
        a[0] = a[0] + (batch_size * ix)     
        atts += [a.cpu()]

        cn_info, onehop_info, non1hop_info = model.compute_node_mask(edge, test_set, None)
        all_mask = torch.cat((cn_info[0], onehop_info[0], non1hop_info[0]), dim=-1)
        masks += [all_mask.cpu()]

        node_indicator = torch.zeros(all_mask.size(1))
        node_indicator[:cn_info[0].size(1)] = 0
        node_indicator[cn_info[0].size(1) : cn_info[0].size(1) + onehop_info[0].size(1)] = 1
        node_indicator[cn_info[0].size(1) + onehop_info[0].size(1) :] = 2

        node_indicators += [node_indicator.cpu()]


    atts = torch.cat(atts, dim=1)
    # pred_all = torch.cat(preds, dim=0)
    masks = torch.cat(masks, dim=1)
    node_indicators = torch.cat(node_indicators, dim=0)

    return atts, masks, node_indicator


def test_with_att(
        model, 
        score_func, 
        data, 
        batch_size, 
    ):
    """
    Include Att Distribution
    """
    model.eval()
    score_func.eval()

    print("Getting Att Scores...")

    with torch.no_grad():
        pos_test_att, pos_test_masks, node_indicator = test_edge_att(model, score_func, data['test_pos'], batch_size, test_set=True)

    batch_ix = pos_test_att[0].long()

    # Number of each type of node for each test sample
    cns = node_indicator == 0
    num_cns_attend = scatter(cns.long(), batch_ix, dim=0, dim_size=data['test_pos'].size(0), reduce="sum")
    onehop = node_indicator == 1
    num_1hop_attend = scatter(onehop.long(), batch_ix, dim=0, dim_size=data['test_pos'].size(0), reduce="sum")
    non_onehop = node_indicator == 2
    num_non1hop_attend = scatter(non_onehop.long(), batch_ix, dim=0, dim_size=data['test_pos'].size(0), reduce="sum")

    # Determine nodes that correspond to samples with at least 1 node of each type
    at_least_1_each = (num_cns_attend > 0) & (num_1hop_attend > 0) & (num_non1hop_attend > 0)
    at_least_1_ix = torch.nonzero(at_least_1_each).squeeze(0)
    at_least_1_mask = torch.isin(batch_ix, at_least_1_ix)
    at_least_1_mask = at_least_1_mask.long() == 1

    ### NOTE: No Constraint
    at_least_1_mask = torch.ones_like(at_least_1_mask)

    filtered_cn_nodes = pos_test_att[1][(at_least_1_mask) & (node_indicator == 0)]
    filtered_1hop_nodes = pos_test_att[1][(at_least_1_mask) & (node_indicator == 1)]
    filtered_non1hop_nodes = pos_test_att[1][(at_least_1_mask) & (node_indicator == 2)]

    return filtered_cn_nodes, filtered_1hop_nodes, filtered_non1hop_nodes


################################################################################
################################################################################
################################################################################

def test_by_all(
        model, 
        score_func, 
        data, 
        evaluator_hit, 
        evaluator_mrr, 
        cmd_args, 
        global_metric,
        k_list=[100],
    ):
    """
    """
    all_results = {}
    score_func.eval()
    lower_b, upper_b = cmd_args.percentile, cmd_args.percentile

    with torch.no_grad():
        if "citation" in data['dataset']:
            h = model.propagate()
            print("Getting Positive Preds ...")
            pos_test_pred  = test_edge_citation2(model, score_func, data['test_pos'], h, cmd_args.batch_size, test=True).view(-1)
            print("Getting Negative Preds ...")
            neg_test_pred  = test_edge_citation2(model, score_func, data['test_pos'], h, cmd_args.batch_size, mrr_mode=True, negative_data=data['test_neg'], test=True)
        else:
            print("Getting Positive Preds ...")
            pos_test_pred = test_edge(model, score_func, data['test_pos'], cmd_args.batch_size, test_set=True).squeeze(-1)
            print("Getting Negative Preds ...")
            neg_test_pred = test_edge(model, score_func, data['test_neg'], cmd_args.batch_size, test_set=True).squeeze(-1)

        lbl_dir = os.path.join(os.path.expanduser("~"), "linktransformer", "heuristic_data")
        num_cns = torch.load(os.path.join(lbl_dir, f"{data['dataset']}_CN.pt"))

        # 1-hot features -- they suck
        if "ppa" not in data['dataset']:
            feat_sim = torch.load(os.path.join(lbl_dir, f"{data['dataset']}_feat_sim.pt"))
        
            threshs1 = [np.percentile(num_cns, upper_b), np.percentile(feat_sim, lower_b), np.percentile(global_metric, lower_b)]
            threshs2 = [np.percentile(num_cns, lower_b), np.percentile(feat_sim, upper_b), np.percentile(global_metric, lower_b)]
            threshs3 = [np.percentile(num_cns, lower_b), np.percentile(feat_sim, lower_b), np.percentile(global_metric, upper_b)]

            only_cn_ix =  (num_cns >= threshs1[0]) & (feat_sim < threshs1[1])  & (global_metric < threshs1[2])
            only_feat_ix =  (num_cns < threshs2[0])  & (feat_sim >= threshs2[1]) & (global_metric < threshs2[2])
            only_global_ix = (num_cns < threshs3[0])  & (feat_sim < threshs3[1])  & (global_metric >= threshs3[2])
        
        else:
            feat_sim = None 
            threshs1 = [np.percentile(num_cns, upper_b), np.percentile(global_metric, lower_b)]
            threshs2 = [np.percentile(num_cns, lower_b), np.percentile(global_metric, upper_b)]

            only_cn_ix =  (num_cns >= threshs1[0])   & (global_metric < threshs1[1])
            only_global_ix = (num_cns < threshs2[0]) & (global_metric >= threshs2[1])

        print("# CN =", only_cn_ix.sum().item())
        print("# Feat =", only_feat_ix.sum().item() if feat_sim is not None else "NA")
        print("# Global =", only_global_ix.sum().item())
        
        if data['dataset'] in ['ogbl-citation2', 'cora', 'citeseer', 'pubmed']:
            # print("---", pos_test_pred[only_cn_ix].shape, neg_test_pred[only_cn_ix].shape)
            # exit()
            all_results[f"CN"] = evaluate_mrr(pos_test_pred[only_cn_ix], neg_test_pred)
            all_results[f"FS"] = evaluate_mrr(pos_test_pred[only_feat_ix], neg_test_pred)
            all_results[f"Katz"] = evaluate_mrr(pos_test_pred[only_global_ix], neg_test_pred)
        else:
            all_results[f"CN"] = evaluate_hits(evaluator_hit, pos_test_pred[only_cn_ix], neg_test_pred, k_list)
            all_results[f"Katz"] = evaluate_hits(evaluator_hit, pos_test_pred[only_global_ix], neg_test_pred, k_list)
            
            if "ppa" not in data['dataset']:
                all_results[f"FS"] = evaluate_hits(evaluator_hit, pos_test_pred[only_feat_ix], neg_test_pred, k_list)

        print(all_results)  

    return all_results        




def compute_all_ppr(ppr_matrix, edges, test_set=False, bs=4096):
    """
    Symmetric score
    """
    all_ppr = []

    for e in tqdm(edges.tolist(), "Slooowly Indexing PPR Matrix"):
        x = (ppr_matrix[e[0]][e[1]] + ppr_matrix[e[1]][e[0]]) / 2
        all_ppr.append(x)    
    
    return torch.Tensor(all_ppr)

