import os 
import json
import random
import argparse
import numpy as np
from tqdm import tqdm 
from scipy import stats
import matplotlib.pyplot as plt

import scipy.sparse as ssp
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

import torch 
from torch.nn import functional as F
from torch_geometric.data import DataLoader

from util.read_datasets import read_data_ogb, read_data_planetoid


ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
ANALYSIS_DIR = os.path.join(ROOT_DIR, "data")
DATA_DIR = os.path.join(ROOT_DIR, "..", "dataset")
BUDDY_DIR = os.path.join(ROOT_DIR, "..", "buddy_feats")
PPR_DIR = os.path.join(ROOT_DIR, "..", "node_subsets", "ppr")
LBL_DIR = os.path.join(ROOT_DIR, "..", "lbl_data")



def get_scaler(scaler_arg):
    scaler_arg = scaler_arg.lower()

    if "standard" in scaler_arg:
        return StandardScaler()
    elif "robust" in scaler_arg:
        return RobustScaler()
    elif "minmax" in scaler_arg:
        return MinMaxScaler()
    else:
        raise ValueError(f"No scalar with name '{scaler_arg}'")
    


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
        cur_scores = F.cosine_similarity(norm_x[src], norm_x[dst]).flatten()
        # cur_scores = norm_x[src] * norm_x[dst]
        num_fs.append(cur_scores.cpu().numpy())

    return torch.from_numpy(np.concatenate(num_fs, 0))


def compute_corr(data, args):
    """
    This monkey's gone to heaven
    """
    pos_cn = calc_CN(data, data['test_pos'])
    neg_cn = calc_CN(data, data['test_neg'])
    pos_fs = calc_FS(data, data['test_pos'])
    neg_fs = calc_FS(data, data['test_neg'])

    # For ALL, Collab has >2x more negative than positive. Downsample negative so equal
    # When no CNs, we downsample the negatives so the same number of positive/negative sample.
    # Done since typically negatives have many more samples w/o CNs
    if args.data_name == "ogbl-collab":
        perm = torch.randperm(len(neg_fs))
        idx = perm[:len(pos_fs)]
        
        neg_fs = neg_fs[idx]
        neg_cn = neg_cn[idx]

    lbls = torch.cat((torch.ones(len(pos_fs)), torch.zeros(len(neg_fs))))

    cn_feats = torch.cat((pos_cn, neg_cn))
    fs_feats = torch.cat((pos_fs, neg_fs))
    sub_feats = torch.stack((cn_feats, fs_feats)).t()
    # sub_feats = torch.cat((cn_feats.reshape(-1, 1), fs_feats), dim=-1)

    # Scale features
    sub_feats, lbls = sub_feats.cpu().numpy(), lbls.cpu().numpy()
    sub_feats = get_scaler(args.scaler).fit_transform(sub_feats)

    print("Fitting model...")
    model = LogisticRegression(max_iter=250, penalty='none', random_state=42).fit(sub_feats, lbls)
    all_coefs = model.coef_[0].tolist()

    print("\nRaw Coefs:", all_coefs)

    # print(mean_coefs)
    normalized_coefs = np.array(np.abs(all_coefs)) / np.abs(all_coefs).sum()

    print("Normalized Coefs:", normalized_coefs)


def plot_by_cn_fs(data):
    """
    Density of positive links by FS, controlling for CNs
    """
    cn_vals = calc_CN(data, data['test_pos'])
    fs_vals = calc_FS(data, data['test_pos'])

    # Convert to percentiles
    cn_vals = stats.rankdata(cn_vals.numpy(), "average") / len(cn_vals)
    fs_vals = stats.rankdata(fs_vals.numpy(), "average") / len(fs_vals)

    heur_bins = [(0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.01)]

    density_by_bin = {h : [] for h in heur_bins}

    for out_bin in heur_bins:
        cn_mask = (cn_vals >= out_bin[0]) & (cn_vals < out_bin[1])

        for in_bin in heur_bins:
            fs_mask = (fs_vals >= in_bin[0]) & (fs_vals < in_bin[1])
            cn_fs_vals = fs_vals[cn_mask & fs_mask]
            density_by_bin[out_bin].append(len(cn_fs_vals))
        
        bin_sum = sum(density_by_bin[out_bin])
        density_by_bin[out_bin] = [d / bin_sum for d in density_by_bin[out_bin]]

        print("\n>>>", out_bin)
        print(density_by_bin[out_bin])



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", default="ogbl-collab")
    parser.add_argument("--scaler", help="'standard', 'robust', 'minmax'", default="standard")
    parser.add_argument("--sim", default="cos")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--use-val-in-test", action='store_true', default=False)
    parser.add_argument("--heart", action='store_true', default=False)
    parser.add_argument('--eps', type=float, default=1e-4)
    args = parser.parse_args()

    random.seed(100)
    np.random.seed(100)
    torch.manual_seed(100)
    torch.cuda.manual_seed(100)

    if "collab" in args.data_name:
        args.use_val_in_test = True 

    if "ogb" in args.data_name:
        data = read_data_ogb(args, args.device)
    else:
        data = read_data_planetoid(args, args.device) 

    # compute_corr(data, args)
    plot_by_cn_fs(data)



if __name__ == "__main__":
    main()