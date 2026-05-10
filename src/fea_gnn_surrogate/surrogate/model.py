import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GPSConv, GINConv


class SharedMPNN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_message_passing_steps):
        super(SharedMPNN, self).__init__()

        self.conv0 = nn.Linear(num_features, hidden_dim)
        self.conv1 = GraphConv(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.num_message_passing_steps = num_message_passing_steps

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv0(x)
        x = F.relu(x)

        for _ in range(self.num_message_passing_steps):
            x = self.conv1(x, edge_index)
            x = F.relu(x)

        x = self.fc2(x)
        return x


class GPSModel(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim, num_layers,
                 heads=4, dropout=0.2, attn_dropout=0.2):
        super(GPSModel, self).__init__()

        self.proj = nn.Linear(num_features, hidden_dim)

        self.gps_layers = nn.ModuleList()
        for _ in range(num_layers):
            local_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            local_conv = GINConv(local_nn)
            gps_layer = GPSConv(
                channels=hidden_dim,
                conv=local_conv,
                heads=heads,
                dropout=dropout,
                act='relu',
                attn_kwargs={'dropout': attn_dropout},
            )
            self.gps_layers.append(gps_layer)

        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None else None

        x = self.proj(x)

        for gps_layer in self.gps_layers:
            x = gps_layer(x, edge_index, batch=batch)

        x = self.fc_out(x)
        return x


class FC(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(num_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x
