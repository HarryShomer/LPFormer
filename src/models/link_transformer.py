import math
import torch
import torch.nn as nn
from torch_scatter import scatter 
from torch.nn.init import uniform_
import numpy as np 


from models.other_models import *
from modules.node_encoder import NodeEncoder
from modules.layers import LinkTransformerLayer

import warnings
warnings.filterwarnings("ignore")



class LinkTransformer(nn.Module):
    """
    Transformer for Link Prediction
    """
    def __init__(
        self,
        train_args,
        data,
        device="cuda"
    ):
        super().__init__()
        
        self.train_args = train_args
        self.data = data
        self.device = device

        # PPR Thresholds
        self.thresh_cn = train_args['thresh_cn']
        self.thresh_1hop = train_args['thresh_1hop']
        self.thresh_non1hop = train_args['thresh_non1hop']

        # Allows for easier way to know if we need to ignore some set of nodes
        # Also allows us to be more efficient
        if self.thresh_non1hop == 1 and self.thresh_1hop == 1:
            self.mask = "cn"
        elif self.thresh_non1hop == 1 and self.thresh_1hop < 1:
            self.mask = "1-hop"
        else:
            self.mask = "all"
            
        self.dim = train_args['dim']
        self.att_drop = train_args.get('att_drop', 0)
        self.num_layers = train_args['trans_layers']
        self.num_nodes = data['x'].shape[0]
        self.out_dim = self.dim * 2

        self.gnn_norm = nn.LayerNorm(self.dim)
        self.node_encoder = NodeEncoder(data, train_args, device=device)

        self.att_layers = nn.ModuleList()
        att_inner_dim = self.dim * 2 if self.num_layers > 1 else self.dim

        self.att_layers.append(LinkTransformerLayer(self.dim, train_args, out_dim=att_inner_dim))
        for _ in range(self.num_layers-2):
            self.att_layers.append(LinkTransformerLayer(self.dim, train_args, node_dim=self.dim))
        if self.num_layers > 1:
            self.att_layers.append(LinkTransformerLayer(self.dim, train_args, out_dim=self.dim, node_dim=self.dim))

        self.elementwise_lin = MLP(2, self.dim, self.dim, self.dim)
        
        # Structural info
        self.ppr_encoder_cn = MLP(2, 2, self.dim, self.dim)
        if self.mask == "cn":
            count_dim = 1
        elif self.mask == "1-hop":
            self.ppr_encoder_onehop = MLP(2, 2, self.dim, self.dim)
            count_dim = 3
        else:
            count_dim = 4
            self.ppr_encoder_onehop = MLP(2, 2, self.dim, self.dim)
            self.ppr_encoder_non1hop = MLP(2, 2, self.dim, self.dim)
        
        pairwise_dim = self.dim * train_args['num_heads'] + count_dim
        self.pairwise_lin = MLP(2, pairwise_dim, pairwise_dim, self.dim)  


    def forward(self, batch, adj_prop=None, adj_mask=None, test_set=False, return_weights=False):
        """
        Calculate edge representations

        Parameters:
        ----------
            batch: torch.Tensor
                2 x BS Tensor that hold source and target nodes
            test_set: bool
                Whether evaluating on test set. Needed if using val_edges in agg

        Returns:
        --------
        torch.Tensor
            BS x self.dim
        """
        batch = batch.to(self.device)

        X_node = self.propagate(adj_prop, test_set)
        x_i, x_j = X_node[batch[0]], X_node[batch[1]]
        elementwise_edge_feats = self.elementwise_lin(x_i * x_j)

        pairwise_feats, att_weights = self.calc_pairwise(batch, X_node, test_set, adj_mask=adj_mask, return_weights=return_weights)
        combined_feats = torch.cat((elementwise_edge_feats, pairwise_feats), dim=-1)

        return combined_feats if not return_weights else (combined_feats, att_weights)
    

    def propagate(self, adj=None, test_set=False):
        """
        Propagate via GNN

        Returns:
        -------
        torch.Tensor
            |V| x self.dim
        """
        if adj is None:
            adj = self.get_adj(test_set)
        x = self.data['x']

        if "emb" in self.data:
            x = self.data['emb'](x)

        X_node = self.node_encoder(x, adj, test_set)        
        X_node = self.gnn_norm(X_node)

        return X_node


    def calc_pairwise(self, batch, X_node, test_set=False, adj_mask=None, return_weights=False):
        """
        Calculate the pairwise features for the node pairs

        TODO: Remove duplicate code later!!!

        Returns:
        -------
        torch.Tensor
            BS x self.dim
        """
        k_i, k_j = X_node[batch[0]], X_node[batch[1]]
        pairwise_feats = torch.cat((k_i, k_j), dim=-1)

        if self.mask == "cn":
            cn_info, _, _ = self.compute_node_mask(batch, test_set, adj_mask)
            node_mask = cn_info[0]
            pes = self.get_pos_encodings(cn_info)

            for l in range(self.num_layers):
                pairwise_feats, att_weights = self.att_layers[l](node_mask, pairwise_feats, X_node, pes, return_weights=return_weights)
            
            num_cns = self.get_count(node_mask, batch, test_set)
            pairwise_feats = torch.cat((pairwise_feats, num_cns), dim=-1)

        else:
            cn_info, onehop_info, non1hop_info = self.compute_node_mask(batch, test_set, adj_mask)
            
            if non1hop_info is not None:
                all_mask = torch.cat((cn_info[0], onehop_info[0], non1hop_info[0]), dim=-1)
                pes = self.get_pos_encodings(cn_info, onehop_info, non1hop_info)
            else:
                all_mask = torch.cat((cn_info[0], onehop_info[0]), dim=-1)
                pes = self.get_pos_encodings(cn_info, onehop_info)

            for l in range(self.num_layers):
                pairwise_feats, att_weights = self.att_layers[l](all_mask, pairwise_feats, X_node, pes, None, return_weights)
            
            num_cns, num_1hop, num_non1hop, num_neighbors = self.get_structure_cnts(batch, cn_info, onehop_info, non1hop_info, test_set=test_set) 

            if num_non1hop is not None:
                pairwise_feats = torch.cat((pairwise_feats, num_cns, num_1hop, num_non1hop, num_neighbors), dim=-1)
            else:
                pairwise_feats = torch.cat((pairwise_feats, num_cns, num_1hop, num_neighbors), dim=-1)

        pairwise_feats = self.pairwise_lin(pairwise_feats)
        return pairwise_feats, att_weights

    

    def get_pos_encodings(self, cn_info, onehop_info=None, non1hop_info=None):
        """
        Ensure symmetric by making `enc = g(a, b) + g(b, a)`

        Returns:
        --------
        torch.Tensor
            Concatenated encodings for cn and 1-hop
        """
        cn_a = self.ppr_encoder_cn(torch.stack((cn_info[1], cn_info[2])).t())
        cn_b = self.ppr_encoder_cn(torch.stack((cn_info[2], cn_info[1])).t())
        cn_pe = cn_a + cn_b


        if onehop_info is None:
            return cn_pe

        onehop_a = self.ppr_encoder_onehop(torch.stack((onehop_info[1] , onehop_info[2])).t())
        onehop_b = self.ppr_encoder_onehop(torch.stack((onehop_info[2], onehop_info[1])).t())
        onehop_pe = onehop_a + onehop_b


        if non1hop_info is None:
            return torch.cat((cn_pe, onehop_pe), dim=0)

        non1hop_a = self.ppr_encoder_non1hop(torch.stack((non1hop_info[1] , non1hop_info[2])).t())
        non1hop_b = self.ppr_encoder_non1hop(torch.stack((non1hop_info[2] , non1hop_info[1])).t())
        non1hop_pe = non1hop_a + non1hop_b

        return torch.cat((cn_pe, onehop_pe, non1hop_pe), dim=0)


    def compute_node_mask(self, batch, test_set, adj):
        """
        Get mask based on type of node

        When mask_type != "cn", also return the ppr vals for both the 
        source and target

        NOTE:
            1. Adj used here has no edge weights. Only 0/1!!!
            2. Adj must already be coalesced for this to work correctly!!!
            3. Pos Edges in batch must be masked.
        """
        if adj is None:
            adj = self.get_adj(test_set, mask=True)
        
        src_adj = torch.index_select(adj, 0, batch[0])
        tgt_adj = torch.index_select(adj, 0, batch[1])

        if self.mask == "cn":
            # 1 when CN, 0 otherwise
            pair_adj = src_adj * tgt_adj
        else:
            # Equals: {0: ">1-Hop", 1: "1-Hop (Non-CN)", 2: "CN"}
            pair_adj = src_adj + tgt_adj  
        
        pair_ix, node_type, src_ppr, tgt_ppr = self.get_ppr_vals(batch, pair_adj, test_set)

        cn_filt_cond = (src_ppr >= self.thresh_cn) & (tgt_ppr >= self.thresh_cn)
        onehop_filt_cond = (src_ppr >= self.thresh_1hop) & (tgt_ppr >= self.thresh_1hop)

        if self.mask != "cn":
            filt_cond = torch.where(node_type == 1, onehop_filt_cond, cn_filt_cond)
        else:
            filt_cond = torch.where(node_type == 0, onehop_filt_cond, cn_filt_cond)

        pair_ix, node_type = pair_ix[:, filt_cond], node_type[filt_cond]
        src_ppr, tgt_ppr = src_ppr[filt_cond], tgt_ppr[filt_cond]

        # >1-Hop mask is gotten separately
        if self.mask == "all":
            non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr = self.get_non_1hop_ppr(batch, test_set=test_set)

        # Dropout
        if self.training and self.att_drop > 0:
            pair_ix, src_ppr, tgt_ppr, node_type = self.drop_pairwise(pair_ix, src_ppr, tgt_ppr, node_type)
            if self.mask == "all":
                non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr, _ = self.drop_pairwise(non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr)

        # Separate out CN and 1-Hop
        if self.mask != "cn":
            cn_ind = node_type == 2
            cn_ix, cn_src_ppr, cn_tgt_ppr = pair_ix[:, cn_ind], src_ppr[cn_ind], tgt_ppr[cn_ind]

            one_hop_ind = node_type == 1
            onehop_ix, onehop_src_ppr, onehop_tgt_ppr = pair_ix[:, one_hop_ind], src_ppr[one_hop_ind], tgt_ppr[one_hop_ind]


        if self.mask == "cn":
            return (pair_ix, src_ppr, tgt_ppr), None, None
        elif self.mask == "1-hop":
            return (cn_ix, cn_src_ppr, cn_tgt_ppr), (onehop_ix, onehop_src_ppr, onehop_tgt_ppr), None
        else:
            return (cn_ix, cn_src_ppr, cn_tgt_ppr), (onehop_ix, onehop_src_ppr, onehop_tgt_ppr), (non1hop_ix, non1hop_src_ppr, non1hop_tgt_ppr)


    def get_ppr_vals(self, batch, pair_diff_adj, test_set=False):
        """
        Get the src and tgt ppr vals

        `pair_diff_adj` specifies type of nodes we select
        """
        # Additional terms for also choosing scores when ppr=0
        # Multiplication removes any values for nodes not in batch
        # Addition then adds offset to ensure we select when ppr=0
        # All selected scores are +1 higher than their true val
        ppr = self.get_ppr(test_set)
        src_ppr_adj = torch.index_select(ppr, 0, batch[0]) * pair_diff_adj + pair_diff_adj
        tgt_ppr_adj = torch.index_select(ppr, 0, batch[1]) * pair_diff_adj + pair_diff_adj

        # Can now convert ppr scores to dense
        ppr_ix  = src_ppr_adj.coalesce().indices()
        src_ppr = src_ppr_adj.coalesce().values()
        tgt_ppr = tgt_ppr_adj.coalesce().values()

        # If one is 0 so is the other
        # NOTE: Should be few to no nodes here
        # src_ppr = src_ppr[src_ppr != 0]
        # tgt_ppr = tgt_ppr[tgt_ppr != 0]
        # ppr_ix = ppr_ix[:, src_ppr != 0]
        
        # TODO: Needed due to a bug in recent torch versions 
        # see here for more - https://github.com/pytorch/pytorch/issues/114529
        # note that if one is 0 so is the other
        zero_vals = (src_ppr != 0) 
        src_ppr = src_ppr[zero_vals]
        tgt_ppr = tgt_ppr[tgt_ppr != 0]
        ppr_ix = ppr_ix[:, zero_vals]

        pair_diff_adj = pair_diff_adj.coalesce().values()
        pair_diff_adj = pair_diff_adj[src_ppr != 0]

        # Remove additional +1 from each ppr val
        src_ppr = (src_ppr - pair_diff_adj) / pair_diff_adj
        tgt_ppr = (tgt_ppr - pair_diff_adj) / pair_diff_adj

        return ppr_ix, pair_diff_adj, src_ppr, tgt_ppr


    def drop_pairwise(self, node_ix, src_ppr=None, tgt_ppr=None, node_indicator=None):
        """
        Drop nodes used in pairwise info
        """
        num_indices = math.ceil(node_ix.size(1) * (1-self.att_drop))
        indices = torch.randperm(node_ix.size(1))[:num_indices]
        node_ix = node_ix[:, indices]

        if src_ppr is not None:
            src_ppr = src_ppr[indices]
        if tgt_ppr is not None:
            tgt_ppr = tgt_ppr[indices]
        if node_indicator is not None:
            node_indicator = node_indicator[indices]

        return node_ix, src_ppr, tgt_ppr, node_indicator
    

    def get_structure_cnts(self, batch, cn_info, onehop_info, non1hop_info=None, test_set=None):
        """
        Counts for CNs, 1-Hop, and >1-Hop
        """
        num_cns = self.get_count(cn_info[0], batch, test_set)            
        num_1hop = self.get_num_ppr_thresh(batch, onehop_info[0], onehop_info[1], 
                                           onehop_info[2], test_set=test_set)

        num_ppr_ones = self.get_num_ppr_thresh(batch, onehop_info[0], onehop_info[1], 
                                               onehop_info[2], thresh=0, test_set=test_set)
        num_neighbors = num_cns + num_ppr_ones

        if non1hop_info is None:
            return num_cns, num_1hop, None, num_neighbors
        else:
            num_non1hop = self.get_count(non1hop_info[0], batch, test_set)
            return num_cns, num_1hop, num_non1hop, num_neighbors


    def get_num_ppr_thresh(self, batch, onehop_mask, src_ppr, tgt_ppr, test_set=False, thresh=None):
        """
        Get # of nodes where ppr(a, v) >= thresh & ppr(b, v) >= thresh

        When src_ppr is None just get srabda
        """
        if thresh is None:
            thresh = self.thresh_1hop

        weight = torch.ones(onehop_mask.size(1), device=onehop_mask.device)

        ppr_above_thresh = (src_ppr >= thresh) & (tgt_ppr >= thresh)
        num_ppr = scatter(ppr_above_thresh.float() * weight, onehop_mask[0].long(), dim=0, dim_size=batch.size(1), reduce="sum")
        num_ppr = num_ppr.unsqueeze(-1)

        return num_ppr


    def get_count(self, node_mask, batch, test_set):
        """
        # of CNs for each sample in batch 
        """
        weight = torch.ones(node_mask.size(1), device=node_mask.device)

        num_cns = scatter(weight, node_mask[0].long(), dim=0, dim_size=batch.size(1), reduce="sum")
        num_cns = num_cns.unsqueeze(-1)

        return num_cns


    def get_adj(self, test_set=False, mask=False):
        """
        Use val_edges in agg when testing and appropriate Tensor in self.data
        """
        suffix = "mask" if mask else "t"
        if test_set:
            return self.data[f'full_adj_{suffix}']
        
        return self.data[f'adj_{suffix}']

    def get_ppr(self, test_set=False):
        """
        Use val_edges in agg when testing and appropriate Tensor in self.data
        """
        if test_set and 'ppr_test' in self.data:
            return self.data[f'ppr_test']
        
        return self.data[f'ppr']

    def get_degree(self, test_set=False):
        """
        Use val_edges in agg when testing and appropriate Tensor in self.data
        """
        if test_set and 'degree_test' in self.data:
            return self.data[f'degree_test']
        
        return self.data[f'degree']
    

    def agg_by_weight(self, batch, X, weight_ix, weight_vals=None):
        """
        Perform a weighted sum by weights for each node in batch
        """
        batch_num = weight_ix[0]   # Corresponding entry for node

        if weight_vals is not None:
            # weighted_hids = weight_vals.unsqueeze(-1) * X[ppr_ix[1]]
            weighted_hids = weight_vals * X[weight_ix[1]]
        else:
            weighted_hids = X[weight_ix[1]]

        output = scatter(weighted_hids, batch_num, dim=0, dim_size=batch.size(1), reduce="sum")

        return output

    def get_non_1hop_ppr(self, batch, test_set=False):
        """
        Get PPR scores for non-1hop nodes.

        NOTE: Use original adj_mask (in train_model.train_epoch() we remove the batch links)
        Done since removing them converts src/tgt to >1-hop nodes
        Therefore removing CN and 1-hop will also remove the batch links
        Don't bother in testing since we are only given the train graph
        """
        adj = self.get_adj(test_set, mask=True)
        src_adj = torch.index_select(adj, 0, batch[0])
        tgt_adj = torch.index_select(adj, 0, batch[1])

        ppr = self.get_ppr(test_set)
        src_ppr = torch.index_select(ppr, 0, batch[0])
        tgt_ppr = torch.index_select(ppr, 0, batch[1])

        # Remove CN scores
        src_ppr = src_ppr - src_ppr * (src_adj * tgt_adj)
        tgt_ppr = tgt_ppr - tgt_ppr * (src_adj * tgt_adj)
        # Also need to remove CN entries in Adj, otherwise leak into next computation
        src_adj = src_adj - src_adj * (src_adj * tgt_adj)
        tgt_adj = tgt_adj - tgt_adj * (src_adj * tgt_adj)

        # Remove 1-Hop scores
        src_ppr = src_ppr - src_ppr * (src_adj + tgt_adj)
        tgt_ppr = tgt_ppr - tgt_ppr * (src_adj + tgt_adj)

        # Want to make sure we include both when we convert to dense so indices align
        # Do so by adding 1 to each based on the other
        src_ppr_add = src_ppr + torch.sign(tgt_ppr)
        tgt_ppr_add = tgt_ppr + torch.sign(src_ppr)

        src_ix = src_ppr_add.coalesce().indices()
        src_vals = src_ppr_add.coalesce().values()
        # tgt_ix = tgt_ppr_add.coalesce().indices()
        tgt_vals = tgt_ppr_add.coalesce().values()

        # Now we can remove value which is just 1
        # NOTE: This technically creates -1 scores for ppr scores that were 0 for src and tgt
        # Doesn't matter as they'll be filtered out by condition later 
        src_vals = src_vals - 1
        tgt_vals = tgt_vals - 1

        ppr_condition = (src_vals >= self.thresh_non1hop) & (tgt_vals >= self.thresh_non1hop)
        src_ix, src_vals, tgt_vals = src_ix[:, ppr_condition], src_vals[ppr_condition], tgt_vals[ppr_condition]

        return src_ix, src_vals, tgt_vals

