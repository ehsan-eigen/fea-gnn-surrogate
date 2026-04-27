import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv


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
