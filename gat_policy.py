#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  3 10:45:22 2025

@author: sven
"""

torch, nn = try_import_torch()
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as gn
from torch_geometric.data import Data, Batch


class SlimFC(nn.Module):
    """Simple PyTorch version of `linear` function"""

    def __init__(self,
                 in_size: int,
                 out_size: int,
                 initializer: Any = None,
                 activation_fn: Any = None,
                 use_bias: bool = True,
                 bias_init: float = 0.0,
                 cust_norm: bool = False):
        """Creates a standard FC layer, similar to torch.nn.Linear

        Args:
            in_size(int): Input size for FC Layer
            out_size (int): Output size for FC Layer
            initializer (Any): Initializer function for FC layer weights
            activation_fn (Any): Activation function at the end of layer
            use_bias (bool): Whether to add bias weights or not
            bias_init (float): Initalize bias weights to bias_init const
        """
        super(SlimFC, self).__init__()
        layers = []
        self.cust_norm = cust_norm
        # Actual nn.Linear layer (including correct initialization logic).
        linear = nn.Linear(in_size, out_size, bias=use_bias)
        if initializer is None:
            initializer = nn.init.xavier_uniform_
        initializer(linear.weight)
        if use_bias is True:
            nn.init.constant_(linear.bias, bias_init)
        layers.append(linear)
        # Activation function (if any; default=None (linear)).
        if isinstance(activation_fn, str):
            activation_fn = get_activation_fn(activation_fn, "torch")
        if activation_fn is not None:
            layers.append(activation_fn())
        # Put everything in sequence.
        self._model = nn.Sequential(*layers)

    # Described in "Striving for simplicity and performance in off-policy DRL" (https://arxiv.org/pdf/1910.02208.pdf)
    def simple_norm(self, x):
        K_v = torch.tensor(x.size()[1])
        Gs_v = torch.sum(torch.abs(x), dim=1).view(-1, 1)
        Gs_v = Gs_v / K_v
        ones_v = torch.ones(Gs_v.size())
        if Gs_v.get_device() != -1:
            ones_v = ones_v.to("cuda:0")
        Gs_mod1_v = torch.where(Gs_v >= 1, Gs_v, ones_v)
        return x / Gs_mod1_v

    def forward(self, x: TensorType) -> TensorType:
        x = self._model(x)
        if self.cust_norm:
            x = self.simple_norm(x)
        return torch.tanh(x)


class GraphEncoder(nn.Module):
    def __init__(
            self,
            model_config,
            local
    ):
        nn.Module.__init__(self)

        # Encoder settings.
        self.local = local
        self.output_dim = 256
        self.num_node_feat = 17
        self.num_node_emb = 128
        self.graph_emb = 32
        self.gat_heads = 3
        self.num_actions = 4
        
        # Aggregations.
        self.aggr = [gn.SoftmaxAggregation(t=1, learn=True)]
        self.pool = gn.MultiAggregation(self.aggr)

        # Layers.
        self.fc_node      = nn.Linear(self.num_node_feat, self.num_node_emb)
        self.conv1        = gn.GATv2Conv(self.num_node_emb, self.graph_emb, heads=self.gat_heads, add_self_loops=True)
        self.conv2        = gn.GATv2Conv(self.graph_emb*self.gat_heads, self.graph_emb, heads=self.gat_heads, add_self_loops=True)
        self.fc_out       = nn.Linear(self.graph_emb*self.gat_heads*len(self.aggr), self.output_dim)
        
        self.nonlin       = nn.GELU()
        self.fc_node_norm = nn.Sequential(self.nonlin, nn.LayerNorm(self.num_node_emb))
        self.conv1_norm   = nn.Sequential(self.nonlin, nn.LayerNorm(self.graph_emb*self.gat_heads))
        self.conv2_norm   = nn.Sequential(self.nonlin, nn.LayerNorm(self.graph_emb*self.gat_heads))
        self.fc_out_norm  = nn.Sequential(self.nonlin, nn.LayerNorm(self.output_dim))
        
        self.p_branch     = SlimFC(in_size=self.output_dim,
                                   out_size=self.num_actions,
                                   initializer=normc_initializer(0.01),
                                   activation_fn=None,
                                   cust_norm=True)

    def forward(self, inputs) -> (TensorType, List[TensorType]):
        # Preprocess inputs into a batch of graphs, with each graph representing:
        # - the observations of a policy agent (if this is an actor encoder dealing with local observations)
        # - the scenario state centred on a policy agent (if this is a centralised critic)
        inputs = inputs.view(inputs.shape[0], -1, self.num_node_feat)  # Shape: [B, N, F]
        
        # If self.local, zero out nodes that aren't visible to observing agents.
        if self.local:
            inputs = inputs * inputs[:, :, -1].unsqueeze(-1)
        
        # Embed observations (entity nodes) with a FC layer.
        x = self.fc_node_norm(self.fc_node(inputs))
            
        # Construct batched edge indices defining star graphs.
        # Source nodes: All. Targets = central/zeroeth nodes.
        batch_size, num_nodes, num_features = x.size()
        node_offset = torch.arange(batch_size).unsqueeze(1) * num_nodes
        source_nodes = torch.arange(num_nodes).unsqueeze(0).repeat(batch_size, 1) + node_offset
        target_nodes = node_offset.repeat(1, num_nodes)
        edge_index = torch.stack([source_nodes.flatten(), target_nodes.flatten()], dim=0)

        # Batch inputs to be processed by graph layers.
        x_flat = x.view(-1, num_features)
        batch = Batch.from_data_list([Data(x=x_flat, edge_index=edge_index)])
        batch.batch = torch.arange(batch_size).repeat_interleave(num_nodes)
        
        # Move to GPU if available.
        if inputs.device.index is not None:
            batch = batch.cuda()
    
        # Graph conv layers.
        x = self.conv1_norm(self.conv1(batch.x, batch.edge_index))
        x = self.conv2_norm(self.conv2(x, batch.edge_index))
        
        # Global pool.
        x = self.pool(x, batch.batch)
        
        # FC out.
        x = self.fc_out_norm(self.fc_out(x))
    
        return x