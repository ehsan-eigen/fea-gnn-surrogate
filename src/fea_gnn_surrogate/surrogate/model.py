import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, GPSConv, GINConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, degree


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


def _compute_spd_dense(adj, max_spd):
    """Batched BFS to get shortest-path distances on a dense adjacency.

    adj: (B, N, N) float (0/1), mask handled by caller — padding rows/cols are zero.
    Returns: (B, N, N) long, where entry == max_spd + 1 means unreachable.
    """
    B, N, _ = adj.shape
    device = adj.device
    adj_bool = adj > 0
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0).expand(B, -1, -1)

    spd = torch.full((B, N, N), max_spd + 1, dtype=torch.long, device=device)
    spd = torch.where(eye, torch.zeros_like(spd), spd)

    reachable = eye.clone()
    frontier = eye.clone()
    for k in range(1, max_spd + 1):
        # one step out from current frontier
        next_frontier = (frontier.float() @ adj_bool.float()) > 0
        new = next_frontier & ~reachable
        if not new.any():
            break
        spd = torch.where(new, torch.full_like(spd, k), spd)
        reachable = reachable | new
        frontier = new
    return spd


class ShortestPathBias(nn.Module):
    """Vanilla Graphormer spatial encoding: learnable per-head bias indexed by SPD.

    Designed to be swapped later for physics-informed kernels (diffusion,
    random-walk / PageRank, effective resistance on a stiffness-weighted graph).
    """

    def __init__(self, heads, max_spd=20):
        super().__init__()
        self.heads = heads
        self.max_spd = max_spd
        # +2: indices 0..max_spd are real distances; max_spd+1 is unreachable.
        self.embed = nn.Embedding(max_spd + 2, heads)

    def forward(self, adj, mask):
        spd = _compute_spd_dense(adj, self.max_spd)  # (B, N, N)
        bias = self.embed(spd)  # (B, N, N, H)
        return bias.permute(0, 3, 1, 2).contiguous()  # (B, H, N, N)


class GraphormerLayer(nn.Module):
    def __init__(self, hidden_dim, heads, dropout, attn_dropout, ffn_ratio=4):
        super().__init__()
        assert hidden_dim % heads == 0
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.o = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop = nn.Dropout(attn_dropout)

        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        ffn_dim = hidden_dim * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x, bias, mask):
        # x: (B, N, D); bias: (B, H, N, N); mask: (B, N) bool
        B, N, D = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).reshape(B, N, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, H, N, hd)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, N, N)
        attn = attn + bias
        if mask is not None:
            key_mask = (~mask).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
            attn = attn.masked_fill(key_mask, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = x + self.o(out)
        x = x + self.ffn(self.ln2(x))
        return x


class Graphormer(nn.Module):
    """Standard Graphormer (Ying et al., NeurIPS 2021) — global-attention baseline.

    Implements the two structural encodings applicable to this dataset:
      * Centrality encoding — separate in/out-degree embeddings added to input
        (Eq. 5 in the paper).
      * Spatial encoding — learnable per-head bias indexed by shortest-path
        distance, added to attention scores (Eq. 6 in the paper).

    The third encoding from the paper (edge encoding along shortest paths,
    Eq. 7) is omitted because the line-graph dataset has no `edge_attr`. The
    spatial-encoding module is pluggable (`attn_bias=...`) to support later
    replacement with physics-informed kernels (diffusion, random walk,
    effective resistance on a stiffness-weighted graph).
    """

    def __init__(self, num_features, hidden_dim, output_dim, num_layers,
                 heads=3, dropout=0.2, attn_dropout=0.2, ffn_ratio=4,
                 max_degree=128, max_spd=20, attn_bias=None):
        super().__init__()
        assert hidden_dim % heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by heads ({heads})"
        )
        self.proj = nn.Linear(num_features, hidden_dim)
        self.max_degree = max_degree
        self.in_degree_emb = nn.Embedding(max_degree + 1, hidden_dim)
        self.out_degree_emb = nn.Embedding(max_degree + 1, hidden_dim)
        self.attn_bias = attn_bias if attn_bias is not None else ShortestPathBias(heads, max_spd=max_spd)

        self.layers = nn.ModuleList([
            GraphormerLayer(hidden_dim, heads, dropout, attn_dropout, ffn_ratio=ffn_ratio)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') and data.batch is not None \
            else torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        in_deg = degree(edge_index[1], num_nodes=x.size(0)).long().clamp(max=self.max_degree)
        out_deg = degree(edge_index[0], num_nodes=x.size(0)).long().clamp(max=self.max_degree)
        h = self.proj(x) + self.in_degree_emb(in_deg) + self.out_degree_emb(out_deg)

        h_dense, mask = to_dense_batch(h, batch)  # (B, N, D), (B, N)
        adj = to_dense_adj(edge_index, batch, max_num_nodes=h_dense.size(1))  # (B, N, N)

        bias = self.attn_bias(adj, mask)  # (B, H, N, N)

        for layer in self.layers:
            h_dense = layer(h_dense, bias, mask)

        h_dense = self.ln_f(h_dense)
        h_out = h_dense[mask]  # (sum_N, D), in original batch order
        return self.fc_out(h_out)


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
