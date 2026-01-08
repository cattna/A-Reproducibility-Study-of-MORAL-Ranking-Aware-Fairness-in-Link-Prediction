import torch
import os
from datasets import Facebook

os.makedirs("data/splits", exist_ok=True)

# 1. Load dataset
dataset = Facebook()

# 2. Adjacency & edge list
adj = dataset.adj_.coalesce()
edge_index = adj.indices()           # [2, E]
num_nodes = adj.size(0)
num_edges = edge_index.size(1)

# 3. Random permutation for positive edges
perm = torch.randperm(num_edges)
n_train = int(0.7 * num_edges)
n_val = int(0.1 * num_edges)

train_idx = perm[:n_train]
val_idx = perm[n_train:n_train + n_val]
test_idx = perm[n_train + n_val:]

def sample_negative_edges(num_samples):
    neg_edges = set()
    pos_edges = set(
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(num_edges)
    )

    while len(neg_edges) < num_samples:
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if u != v and (u, v) not in pos_edges:
            neg_edges.add((u, v))

    return torch.tensor(list(neg_edges), dtype=torch.long)

# 4. Build splits (EDGE FORMAT: [E,2])
splits = {}

for name, idx in [
    ("train", train_idx),
    ("val", val_idx),
    ("test", test_idx),
]:
    pos = edge_index[:, idx].t()                 # [E,2]
    neg = sample_negative_edges(pos.size(0))     # [E,2]

    splits[name] = {
        "edge": pos,
        "edge_neg": neg,
    }

# 5. Save (FORMAT MORAL)
torch.save((dataset, splits), "data/splits/facebook.pt")

print("DONE: data/splits/facebook.pt created (WITH edge_neg)")
