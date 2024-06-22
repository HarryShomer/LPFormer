import torch 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score



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




def evaluate_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''
    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits10_list = (ranking_list <= 10).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)

    mrr_list = 1./ranking_list.to(torch.float)

    results = { "Hits@10": hits10_list.mean().item(),
                "Hits@50":  hits50_list.mean().item(),
                "Hits@100": hits100_list.mean().item(), 
                'MRR': mrr_list.mean().item()}

    return results


def sample_level_hits(y_pred_pos, y_pred_neg):
    """
    Hits on sample level
    """
    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    hits20_list = (ranking_list <= 20).to(torch.float)
    hits50_list = (ranking_list <= 50).to(torch.float)
    hits100_list = (ranking_list <= 100).to(torch.float)

    return {"Hits@20": hits20_list, "Hits@50": hits50_list, "Hits@100": hits100_list}


def get_ranking_list(y_pred_pos, y_pred_neg):
    """
    Just list of ranks for all samples

    Mean of optimistic + pessimistic
    """
    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1

    return ranking_list


def evaluate_auc(val_pred, val_true):
    results = {}
    
    valid_auc = roc_auc_score(val_true, val_pred)
    valid_auc = round(valid_auc, 4)
    results['AUC'] = valid_auc

    valid_ap = average_precision_score(val_true, val_pred)
    valid_ap = round(valid_ap, 4)
    results['AP'] = valid_ap

    return results



def get_metric_score(evaluator_hit, evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred, k_list=[100]):
    """
    Get vals for diff metrics
    """
    result = {}

    result_hit_train = evaluate_hits(evaluator_hit, pos_train_pred, neg_val_pred, k_list)
    result_hit_val = evaluate_hits(evaluator_hit, pos_val_pred, neg_val_pred, k_list)
    result_hit_test = evaluate_hits(evaluator_hit, pos_test_pred, neg_test_pred, k_list)

    for K in k_list:
        result[f'Hits@{K}'] = (result_hit_train[f'Hits@{K}'], result_hit_val[f'Hits@{K}'], result_hit_test[f'Hits@{K}'])

    if evaluator_mrr is not None:
        result_mrr_train = evaluate_mrr(pos_train_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
        result_mrr_val = evaluate_mrr(pos_val_pred, neg_val_pred.repeat(pos_val_pred.size(0), 1) )
        result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred.repeat(pos_test_pred.size(0), 1) )
        result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])

    return result


def get_metric_score_citation2(evaluator_mrr, pos_train_pred, pos_val_pred, neg_val_pred, pos_test_pred, neg_test_pred):
    """
    Specific to Citation2
    """
    k_list = [20, 50, 100]
    result = {}

    # result_mrr_train = evaluate_mrr(evaluator_mrr,  pos_train_pred, neg_val_pred, k_list=k_list)
    # result_mrr_val = evaluate_mrr(evaluator_mrr, pos_val_pred, neg_val_pred, k_list=k_list)
    # result_mrr_test = evaluate_mrr(evaluator_mrr, pos_test_pred, neg_test_pred, k_list=k_list)
    result_mrr_train = evaluate_mrr(pos_train_pred, neg_val_pred)
    result_mrr_val = evaluate_mrr(pos_val_pred, neg_val_pred)
    result_mrr_test = evaluate_mrr(pos_test_pred, neg_test_pred)

    result['MRR'] = (result_mrr_train['MRR'], result_mrr_val['MRR'], result_mrr_test['MRR'])
    # for K in k_list:
    #     result[f'Hits@{K}'] = (result_mrr_train[f'mrr_hit{K}'], result_mrr_val[f'mrr_hit{K}'], result_mrr_test[f'mrr_hit{K}'])

    return result