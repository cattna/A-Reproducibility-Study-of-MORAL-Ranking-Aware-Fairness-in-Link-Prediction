import torch
import os
import argparse
import datasets

os.makedirs("data/splits", exist_ok=True)

def sample_negative_edges(edge_index, num_nodes, num_samples):
    pos_edges = set(
        (edge_index[0, i].item(), edge_index[1, i].item())
        for i in range(edge_index.size(1))
    )

    neg_edges = set()
    while len(neg_edges) < num_samples:
        u = torch.randint(0, num_nodes, (1,)).item()
        v = torch.randint(0, num_nodes, (1,)).item()
        if u != v and (u, v) not in pos_edges:
            neg_edges.add((u, v))

    return torch.tensor(list(neg_edges), dtype=torch.long)


def main(dataset_name):
    # 1. ambil class dataset dari datasets.py
    dataset_class = getattr(datasets, dataset_name)
    dataset = dataset_class()

    # 2. adjacency & edge list
    adj = dataset.adj_.coalesce()
    edge_index = adj.indices()  # [2, E]
    num_nodes = adj.size(0)
    num_edges = edge_index.size(1)

    # 3. split edge
    perm = torch.randperm(num_edges)
    n_train = int(0.7 * num_edges)
    n_val = int(0.1 * num_edges)

    splits_idx = {
        "train": perm[:n_train],
        "val": perm[n_train:n_train + n_val],
        "test": perm[n_train + n_val:]
    }

    splits = {}
    for name, idx in splits_idx.items():
        pos = edge_index[:, idx].t()  # [E,2]
        neg = sample_negative_edges(edge_index, num_nodes, pos.size(0))
        splits[name] = {
            "edge": pos,
            "edge_neg": neg
        }

    # 4. simpan sesuai format MORAL
    out_path = f"data/splits/{dataset_name.lower()}.pt"
    torch.save((dataset, splits), out_path)

    print(f"DONE: {out_path} created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="Dataset class name in datasets.py (e.g. Facebook, German, Credit)")
    args = parser.parse_args()

    main(args.dataset)
