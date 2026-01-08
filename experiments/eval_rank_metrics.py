import argparse
from pathlib import Path
import numpy as np
import torch

def generate_array_greedy_dkl(n: int, distribution: np.ndarray) -> torch.Tensor:
    actual_counts = np.zeros_like(distribution, dtype=float)
    result = []
    for i in range(n):
        if i == 0:
            choice = int(np.argmax(distribution))
        else:
            desired = distribution * i
            deficit = desired - actual_counts
            choice = int(np.argmax(deficit))
        result.append(choice)
        actual_counts[choice] += 1
    return torch.tensor(result)

def ndkl_from_group_sequence(groups: torch.Tensor, pi: torch.Tensor) -> float:
    # groups: [K] values 0/1/2
    # pi: [3] target distribution
    K = groups.numel()
    eps = 1e-12
    discounts = 1.0 / torch.log2(torch.arange(2, K + 2, dtype=torch.float32))
    Z = discounts.sum()

    ndkl = 0.0
    counts = torch.zeros_like(pi, dtype=torch.float32)
    for i in range(K):
        g = int(groups[i].item())
        counts[g] += 1.0
        p_hat = counts / (i + 1.0)
        # D_KL(p_hat || pi)
        dkl = (p_hat * (torch.log((p_hat + eps) / (pi + eps)))).sum().item()
        ndkl += discounts[i].item() * dkl

    return ndkl / Z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, type=str)
    ap.add_argument("--runs", default=3, type=int)
    ap.add_argument("--K", default=1000, type=int)
    ap.add_argument("--splits_dir", default="data/splits", type=str)
    args = ap.parse_args()

    ds = args.dataset
    K = args.K

    data_obj, splits = torch.load(Path(args.splits_dir) / f"{ds}.pt")

    sens = data_obj.sens().cpu()  # shape [N]
    if sens.dim() == 2:
        sens = sens.squeeze(1)
    sens = sens.float()

    train_edges = splits["train"]["edge"]  # [E,2]
    train_groups = sens[train_edges].sum(dim=1).long()  # 0/1/2
    counts = torch.bincount(train_groups, minlength=3).float()
    pi = counts / counts.sum()  # [3]

    test_split = splits["test"]
    test_edges = torch.cat([test_split["edge"], test_split["edge_neg"]], dim=0)  # [M,2]
    test_labels = torch.cat(
        [torch.ones(test_split["edge"].size(0)), torch.zeros(test_split["edge_neg"].size(0))],
        dim=0,
    ).float()

    edge_groups = sens[test_edges].sum(dim=1).long()  # [M] in {0,1,2}

    output_positions = generate_array_greedy_dkl(K, pi.numpy())  # [K]

    precs, ndkls = [], []
    for run in range(args.runs):
        run_suffix = f"{ds}_MORAL_GAE_{run}"
        scores = torch.load(f"three_classifiers_{ds}_MORAL_GAE_{run}.pt", map_location="cpu").float()

        if scores.numel() != test_edges.size(0):
            raise RuntimeError(
                f"Score length mismatch for run {run}: "
                f"scores={scores.numel()} vs candidates={test_edges.size(0)}"
            )

        final_labels = torch.zeros(K, dtype=torch.float32)
        final_groups = torch.zeros(K, dtype=torch.long)

        for g in range(3):
            mask_pos = (output_positions == g)
            need = int(mask_pos.sum().item())
            if need == 0:
                continue

            cand_mask = (edge_groups == g)
            if cand_mask.sum().item() == 0:
                continue

            group_scores = scores[cand_mask]
            group_labels = test_labels[cand_mask]
            group_sorted, idx = torch.sort(group_scores, descending=True)
            group_labels_sorted = group_labels[idx]

            take = min(need, group_labels_sorted.numel())
            # fill the required positions
            pos_idx = torch.where(mask_pos)[0][:take]
            final_labels[pos_idx] = group_labels_sorted[:take]
            final_groups[pos_idx] = g

        prec_at_1000 = final_labels.mean().item()
        ndkl = ndkl_from_group_sequence(final_groups, pi)

        print(f"run {run}: prec@{K}={prec_at_1000:.6f}  NDKL={ndkl:.6f}  pi={pi.tolist()}")
        precs.append(prec_at_1000)
        ndkls.append(ndkl)

    precs = np.array(precs)
    ndkls = np.array(ndkls)
    print("\n=== SUMMARY ===")
    print(f"{ds}: prec@{K} mean±std = {precs.mean():.6f} ± {precs.std(ddof=0):.6f}")
    print(f"{ds}: NDKL     mean±std = {ndkls.mean():.6f} ± {ndkls.std(ddof=0):.6f}")

if __name__ == "__main__":
    main()
