import torch
import torch.nn as nn
import torch.nn.functional as F

from models.other_models import GCN


class NodeEncoder(nn.Module):
    """
    Handles encoding of features & PEs

    Also how to combine them
    """
    def __init__(
        self, 
        data,
        train_args,
        device="cuda"
    ):
        super().__init__()

        self.device = device
        self.dim = train_args['dim'] 
        init_dim = self.dim if 'emb' in data else data['x'].size(1)

        self.feat_drop = train_args.get('feat_drop', 0)
        self.feat_transform = nn.Linear(init_dim, self.dim)

        self.gnn_encoder = GCN(init_dim, self.dim, self.dim, train_args['gnn_layers'], 
                               train_args.get('gnn_drop', 0), cached=train_args.get('gcn_cache'), 
                               residual=train_args['residual'], layer_norm=train_args['layer_norm'],
                               relu=train_args['relu'])


    def forward(self, features, adj_t, test_set=False):
        """
        1. Transform all PEs
        2. Transform all node features
        3. Nodes + PEs
        """
        features = F.dropout(features, p=self.feat_drop, training=self.training)        
        X_gnn = self.gnn_encoder(features, adj_t)

        return X_gnn

