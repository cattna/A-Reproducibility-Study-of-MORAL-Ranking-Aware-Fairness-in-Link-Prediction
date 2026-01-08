import os
import glob
import torch
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Facebook, German, Pokec_n, Pokec_z

# =====================================================
# Paper settings
# =====================================================
TOPK = 100
SEEDS = [0, 1, 2]

# Methods in paper order (A..J)
METHODS = [
    ("UGE", "UGE"),
    ("EDITS", "EDITS"),
    ("FairAdj", "FairAdj"),
    ("FairEGM", "FairEGM"),
    ("FairLP", "FairLP"),
    ("GRAPHAIR", "GRAPHAIR"),
    ("FairWalk", "FairWalk"),
    ("DELTR", "DELTR"),
    ("DetConstSort", "DetConstSort"),
    ("MORAL", "MORAL"),
]

# Paper dataset order/layout
DATASETS = [
    ("facebook", "Facebook"),
    ("credit", "Credit"),
    ("german", "German"),
    ("nba", "NBA"),
    ("pokec_n", "Pokec-n"),
    ("pokec_z", "Pokec-z"),
]

# =====================================================
# Graph + sensitive loaders (authoritative)
# =====================================================
def load_graph_and_sensitive(ds_name: str):
    """
    Returns:
      A: adjacency as torch.BoolTensor [N,N] (undirected, no self loops)
      sens: torch.LongTensor [N]
    """
    if ds_name in ["facebook", "german", "pokec_n", "pokec_z"]:
        ds_cls = {
            "facebook": Facebook,
            "german": German,
            "pokec_n": Pokec_n,
            "pokec_z": Pokec_z,
        }[ds_name]
        ds = ds_cls()
        A_sp = ds.adj().coalesce()
        N = A_sp.size(0)
        A = torch.zeros((N, N), dtype=torch.bool)
        r, c = A_sp.indices()
        mask = r != c
        r, c = r[mask], c[mask]
        A[r, c] = True
        A[c, r] = True
        sens = ds.sens().long()
        return A, sens

    if ds_name == "nba":
        df = pd.read_csv("dataset/nba/nba.csv")
        id_col = next(c for c in ["player_id", "id", "user_id"] if c in df.columns)
        raw_ids = df[id_col].astype(int).values
        id_map = {rid: i for i, rid in enumerate(raw_ids)}
        sens_col = next(
            c for c in [
                "nationality", "country", "birth_country",
                "nation", "international", "is_foreign"
            ] if c in df.columns
        )
        sens = torch.tensor(
            df[sens_col].astype("category").cat.codes.values,
            dtype=torch.long
        )
        N = sens.numel()
        A = torch.zeros((N, N), dtype=torch.bool)
        with open("dataset/nba/nba_relationship.txt") as f:
            for line in f:
                u_raw, v_raw = map(int, line.split())
                if u_raw in id_map and v_raw in id_map:
                    u, v = id_map[u_raw], id_map[v_raw]
                    if u != v:
                        A[u, v] = True
                        A[v, u] = True
        return A, sens

    if ds_name == "credit":
        df = pd.read_csv("dataset/credit/credit.csv")
        age_col = next(c for c in ["age", "Age", "AGE"] if c in df.columns)
        sens = torch.tensor(
            (df[age_col] > 25).astype(int).values,
            dtype=torch.long
        )
        N = sens.numel()
        A = torch.zeros((N, N), dtype=torch.bool)
        with open("dataset/credit/credit_edges.txt") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                u = int(float(parts[0]))
                v = int(float(parts[1]))
                if 0 <= u < N and 0 <= v < N and u != v:
                    A[u, v] = True
                    A[v, u] = True
        return A, sens

    raise ValueError(ds_name)

# =====================================================
# Locate method checkpoints (robust)
# =====================================================
def find_pt(ds_name: str, method_key: str, seed: int):
    patterns = [
        f"three_classifiers_{ds_name}_{method_key}_GAE_{seed}.pt",
        f"three_classifiers_{ds_name}_{method_key}_{seed}.pt",
    ]
    for p in patterns:
        if os.path.exists(p):
            return p
    return None

# =====================================================
# Extract score matrix (if possible)
# =====================================================
def extract_score_matrix(obj, N: int):
    """
    Attempts to extract a full pairwise score matrix.
    If impossible, raises RuntimeError.
    """
    if torch.is_tensor(obj):
        if obj.dim() == 2 and obj.size(0) == obj.size(1) == N:
            return obj.float()
        if obj.dim() == 2 and obj.size(0) == N:
            Z = obj.float()
            return torch.sigmoid(Z @ Z.t())
        raise RuntimeError

    if not isinstance(obj, dict):
        raise RuntimeError

    for k in ["adj_pred", "recon_adj", "y_pred", "scores_mat"]:
        if k in obj and torch.is_tensor(obj[k]):
            t = obj[k]
            if t.dim() == 2 and t.size(0) == t.size(1) == N:
                return t.float()

    for v in obj.values():
        if torch.is_tensor(v) and v.dim() == 2 and v.size(0) == v.size(1) == N:
            return v.float()

    for v in obj.values():
        if torch.is_tensor(v) and v.dim() == 2 and v.size(0) == N:
            Z = v.float()
            return torch.sigmoid(Z @ Z.t())

    raise RuntimeError

# =====================================================
# Top-K predicted non-edges
# =====================================================
def topk_predicted_nonedges(scores, A, topk=TOPK):
    N = scores.size(0)
    iu, iv = torch.triu_indices(N, N, offset=1)
    mask = ~A[iu, iv]
    iu, iv = iu[mask], iv[mask]
    s = scores[iu, iv]
    k = min(topk, s.numel())
    _, idx = torch.topk(s, k=k, largest=True)
    return list(zip(iu[idx].tolist(), iv[idx].tolist()))

def pair_proportions(edge_list, sens):
    ss = ds = 0
    for u, v in edge_list:
        if int(sens[u]) == int(sens[v]):
            ss += 1
        else:
            ds += 1
    total = max(1, len(edge_list))
    # IMPORTANT: E_{s'-s'} = 0 because no missing sensitive attributes
    return ss / total, ds / total, 0.0

def average_method(ds_name, method_key, A, sens):
    props = []
    N = A.size(0)
    for seed in SEEDS:
        path = find_pt(ds_name, method_key, seed)
        if path is None:
            return None
        ckpt = torch.load(path, map_location="cpu")
        scores = extract_score_matrix(ckpt, N)
        edges = topk_predicted_nonedges(scores, A, TOPK)
        props.append(pair_proportions(edges, sens))

    ss = sum(p[0] for p in props) / len(props)
    ds = sum(p[1] for p in props) / len(props)
    oo = sum(p[2] for p in props) / len(props)
    return ss, ds, oo

# =====================================================
# References
# =====================================================
def original_edge_distribution(A, sens):
    N = A.size(0)
    iu, iv = torch.triu_indices(N, N, offset=1)
    mask = A[iu, iv]
    iu, iv = iu[mask], iv[mask]
    ss = ds = 0
    for u, v in zip(iu.tolist(), iv.tolist()):
        if int(sens[u]) == int(sens[v]):
            ss += 1
        else:
            ds += 1
    total = max(1, ss + ds)
    return ss / total, ds / total, 0.0

def dyadic_reference_height(orig_props):
    # Paper: purple = E_{s-s} + E_{s'-s'}
    return orig_props[0] + orig_props[2]

# =====================================================
# Plot Figure 3
# =====================================================
def plot_figure3():
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    axes = axes.flatten()

    for ax, (ds_name, title) in zip(axes, DATASETS):
        A, sens = load_graph_and_sensitive(ds_name)

        xlabels = [f"{chr(ord('A')+i)})" for i in range(len(METHODS))] + ["Original", "Dyadic"]
        x = list(range(len(xlabels)))

        blue  = [0.0] * len(xlabels)  # E_{s-s}
        black = [0.0] * len(xlabels)  # E_{s'-s}
        red   = [0.0] * len(xlabels)  # E_{s'-s'}
        missing = [False] * len(xlabels)

        for i, (method_key, _) in enumerate(METHODS):
            try:
                out = average_method(ds_name, method_key, A, sens)
            except Exception:
                out = None
            if out is None:
                missing[i] = True
                continue
            blue[i], black[i], red[i] = out

        orig = original_edge_distribution(A, sens)
        oi = len(METHODS)
        blue[oi], black[oi], red[oi] = orig

        dy_idx = len(METHODS) + 1
        dy_h = dyadic_reference_height(orig)

        ax.bar(x[:oi+1], blue[:oi+1], color="#1f77b4", label=r"$E_{s-s}$")
        ax.bar(
            x[:oi+1],
            black[:oi+1],
            bottom=blue[:oi+1],
            color="black",
            label=r"$E_{s'-s}$",
        )
        ax.bar(
            x[:oi+1],
            red[:oi+1],
            bottom=[blue[j] + black[j] for j in range(oi+1)],
            color="#d62728",
            label=r"$E_{s'-s'}$",
        )

        ax.bar(
            [dy_idx],
            [dy_h],
            color="purple",
            hatch="///",
            label="Dyadic Reference",
        )

        ax.set_title(title)
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8)
        ax.grid(True, axis="y", linestyle=":", alpha=0.7)

        for i_m in range(len(METHODS)):
            if missing[i_m]:
                ax.text(i_m, 0.5, "∅", ha="center", va="center", fontsize=10)

        # Explicit note for red bars (paper-consistent)
        ax.text(
            0.02,
            0.02,
            r"$E_{s'-s'}=0$ (no missing attributes)",
            transform=ax.transAxes,
            fontsize=7,
            color="#d62728",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(
        "Figure 3 (Reproduced): Pair proportions in Top-100 predicted non-edges",
        y=0.98,
    )
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/figure3_reproduced.png"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_figure3()
