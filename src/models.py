import dgl
import torch
import torch.nn as nn
from dgl.nn import GATConv, GATv2Conv, GraphConv, GCN2Conv
from dgl.nn.pytorch import GATConv, GATv2Conv, GCN2Conv, SAGEConv


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(dgl.function.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = SAGEConv(
            in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = SAGEConv(
            in_feats=hid_feats, out_feats=1, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)

        h = torch.relu(h)
        h = self.conv2(graph, h)
        # print(h.size())
        # exit()
        return h


class SAGENodeRegression(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, "mean"))
        self.layers.append(SAGEConv(n_hidden, n_hidden, "mean"))
        self.layers.append(SAGEConv(n_hidden, 1, "mean"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, sg, x):
        h = x
        for l, layer in enumerate(self.layers):
            h = layer(sg, h)
            if l != len(self.layers) - 1:
                h = nn.functional.relu(h)
                h = self.dropout(h)
        return h


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.sage = SAGE(in_features, hidden_features, out_features)
        self.pred = DotProductPredictor()

    def forward(self, g, x):
        h = self.sage(g, x)
        return self.pred(g, h)


# Define the GCN model
class GCNNodeRegression(nn.Module):
    def __init__(self, in_feats, hidden_feats, output_dim):
        super(GCNNodeRegression, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_feats)
        # self.conv2 = GraphConv(hidden_feats, output_dim)    # output_dim? hidden_feats
        self.conv2 = GraphConv(hidden_feats, hidden_feats)  # output_dim? hidden_feats
        # self.fc = nn.Linear(hidden_feats, output_dim)
        # self.fc = nn.Linear(output_dim, 1)
        self.fc = nn.Linear(hidden_feats, 1)

    def forward(self, g, features):
        # Apply GCN convolution layers
        x = torch.relu(self.conv1(g, features).flatten(1))
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.conv2(g, x).flatten(1))

        # Global average pooling
        # g.ndata['h'] = x
        # hg = dgl.mean_nodes(g, 'h')
        # output = hg

        # Fully connected layer for regression
        # output = self.fc(hg)
        return self.fc(x)


class GCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, output_dim)
        self.linear1 = torch.nn.Linear(output_dim, 1)

    def forward(self, graph, x):
        x = self.conv1(graph, x)
        x = x.relu()
        x = nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.conv2(graph, x)
        x = self.linear1(x)
        return x


# class GCNNodeRegression(torch.nn.Module):
#     def __init__(self, in_dim, hidden_dim, output_dim):
#         super(GCNNodeRegression, self).__init__()
#         torch.manual_seed(12345)
#         self.conv1 = GraphConv(in_dim, hidden_dim)
#         self.conv2 = GraphConv(hidden_dim, 1)  # 1 output channel for regression
#
#     def forward(self, x, edge_index):
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = nn.functional.dropout(x, p=0.5, training=self.training)
#         x = self.conv2(x, edge_index)
#         return x


# Define the GATv2 model
class GATv2(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, num_heads):
        super(GATv2, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATv2Conv(in_dim, hidden_dim, num_heads, feat_drop=0.6, attn_drop=0.6))
        self.layers.append(GATv2Conv(hidden_dim * num_heads, output_dim, 1, feat_drop=0.6, attn_drop=0.6))

        # self.layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop=0.06, attn_drop=0.06))
        # self.layers.append(GATConv(hidden_dim * num_heads, num_classes, 1, feat_drop=0.06, attn_drop=0.06))

        # self.layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop=0.6, attn_drop=0.6,  allow_zero_in_degree=True))
        # self.layers.append(GATConv(hidden_dim * num_heads, num_classes, 1, feat_drop=0.6, attn_drop=0.6,  allow_zero_in_degree=True))

        # self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        # self.conv2 = GATConv(hidden_dim * num_heads, num_classes, 1)

        # self.layers.append(GATv2Conv(in_dim, hidden_dim, num_heads, feat_drop=0.06, attn_drop=0.06))
        # self.layers.append(GATv2Conv(hidden_dim * num_heads, num_classes, 1, feat_drop=0.06, attn_drop=0.06))

    def forward(self, graph, features):
        x = features
        for layer in self.layers:
            x = layer(graph, x).flatten(1)
        return x


class GATv2NodeRegression(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, output_dim):
        super(GATv2NodeRegression, self).__init__()
        self.conv1 = GATv2Conv(in_feats, hidden_feats, num_heads)  # feat_drop=0.6, attn_drop=0.6 --> works bad
        self.conv2 = GATv2Conv(hidden_feats * num_heads, hidden_feats, num_heads)  # feat_drop=0.6, attn_drop=0.6
        self.conv3 = GATv2Conv(hidden_feats * num_heads, 1, 1)  # feat_drop=0.6, attn_drop=0.6
        # self.fc = nn.Linear(hidden_feats * num_heads, output_dim)

        # self.conv1 = GATv2Conv(in_feats, hidden_feats, num_heads, feat_drop=0.6, attn_drop=0.6)
        # self.conv2 = GATv2Conv(hidden_feats * num_heads, hidden_feats, num_heads, feat_drop=0.6, attn_drop=0.6)
        self.fc = nn.Linear(hidden_feats * num_heads, 1)

    def forward(self, g, features):
        # Apply GAT convolutional layers
        x = self.conv1(g, features).flatten(1)
        x = torch.relu(x)
        x = self.conv2(g, x).flatten(1)
        x = torch.relu(x)
        out = self.conv3(g, x)

        # Global average pooling
        g.ndata['h'] = x
        hg = dgl.mean_nodes(g, 'h')
        # print(x)
        # print(hg)
        # out = self.fc(hg)
        # print(out)
        # out = self.fc(x)

        # Fully connected layer for regression
        return out


# Define the GAT model
class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, num_heads):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop=0.6, attn_drop=0.6))
        self.layers.append(GATConv(hidden_dim * num_heads, output_dim, 1, feat_drop=0.6, attn_drop=0.6))

        # self.layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop=0.06, attn_drop=0.06))
        # self.layers.append(GATConv(hidden_dim * num_heads, num_classes, 1, feat_drop=0.06, attn_drop=0.06))

        # self.layers.append(GATConv(in_dim, hidden_dim, num_heads, feat_drop=0.6, attn_drop=0.6,  allow_zero_in_degree=True))
        # self.layers.append(GATConv(hidden_dim * num_heads, num_classes, 1, feat_drop=0.6, attn_drop=0.6,  allow_zero_in_degree=True))

        # self.conv1 = GATConv(in_dim, hidden_dim, num_heads)
        # self.conv2 = GATConv(hidden_dim * num_heads, num_classes, 1)

        # self.layers.append(GATv2Conv(in_dim, hidden_dim, num_heads, feat_drop=0.06, attn_drop=0.06))
        # self.layers.append(GATv2Conv(hidden_dim * num_heads, num_classes, 1, feat_drop=0.06, attn_drop=0.06))

    def forward(self, graph, features):
        x = features
        for layer in self.layers:
            x = layer(graph, x).flatten(1)
        return x

    # def forward(self, graph, features):
    #     x = features
    #     for layer in self.layers:
    #         x = (nn.functional.relu(layer(graph, x)).flatten(1))
    #     return x

    # def forward(self, graph, x):
    #     # for layer in self.layers:
    #     #     x = layer(graph, x).flatten(1)
    #     # return x
    #     x = nn.functional.relu(self.conv1(graph=graph, feat=x))
    #     return nn.functional.relu(self.conv2(graph=graph, feat=x))

    # def forward(self, g, features):
    #     h = features
    #     for l in range(len(self.layers) - 1):
    #         h = self.layers[l](g, h).flatten(1)
    #     h = (self.layers[-1](g, h)).mean(1)
    #     return h


# class GATNodeRegression(nn.Module):
#     def __init__(self, in_feats, hidden_feats, output_dim, num_heads):
#         super(GATNodeRegression, self).__init__()
#         self.conv1 = GATConv(in_feats, hidden_feats, num_heads)
#         self.conv2 = GATConv(hidden_feats * num_heads, output_dim, 1)
#         self.fc = nn.Linear(output_dim * num_heads, 1)
#
#     def forward(self, g, features):
#         # Apply GAT convolutional layers
#         x = self.conv1(g, features).flatten(1)
#         x = torch.relu(x)
#         x = self.conv2(g, x).flatten(1)
#         x = torch.relu(x)
#
#         # Global average pooling
#         g.ndata['h'] = x
#         hg = dgl.mean_nodes(g, 'h')
#
#         # Fully connected layer for regression
#         return self.fc(hg)

class GATNodeRegression(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_heads, output_dim):
        super(GATNodeRegression, self).__init__()
        self.conv1 = GATConv(in_feats, hidden_feats, num_heads)  # feat_drop=0.6, attn_drop=0.6 --> works bad
        self.conv2 = GATConv(hidden_feats * num_heads, hidden_feats, num_heads)  # feat_drop=0.6, attn_drop=0.6
        # self.fc = nn.Linear(hidden_feats * num_heads, output_dim)
        self.fc = nn.Linear(hidden_feats * num_heads, 1)

    def forward(self, g, features):
        # Apply GAT convolutional layers
        x = self.conv1(g, features).flatten(1)
        x = torch.relu(x)
        x = self.conv2(g, x).flatten(1)
        x = torch.relu(x)

        # Global average pooling
        g.ndata['h'] = x
        hg = dgl.mean_nodes(g, 'h')
        # print(x)
        # print(hg)

        # Fully connected layer for regression
        out = self.fc(hg)
        out = self.fc(x)

        return out
