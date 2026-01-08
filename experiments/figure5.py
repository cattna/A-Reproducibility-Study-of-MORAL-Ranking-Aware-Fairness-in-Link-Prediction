import os
import math
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from datasets import Facebook, German, Pokec_n, Pokec_z

# =========================
# Settings (paper-like)
# =========================
MAX_K = 2000
STEP = 25

DATASETS = [
    ("facebook", "Facebook"),
    ("credit", "Credit"),
    ("german", "German"),
    ("nba", "Nba"),
    ("pokec_n", "Pokec-n"),
    ("pokec_z", "Pokec-z"),
]

# Candidate non-edges sampled (keeps runtime sane)
# Increase if you want smoother curves; 200k is usually fine on CPU.
NONEDGE_SAMPLE = 200_000
RNG_SEED = 123

# =========================
# Graph + sensitive loaders
# =========================
def load_graph_and_sensitive(ds_name: str):
    """
    Returns:
      A: adjacency as torch.BoolTensor [N,N] (undirected, no self loops)
      sens: torch.LongTensor [N]
    """
    if ds_name in ["facebook", "german", "pokec_n", "pokec_z"]:
        ds_cls = {"facebook": Facebook, "german": German, "pokec_n": Pokec_n, "pokec_z": Pokec_z}[ds_name]
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
        sens = torch.tensor(df[sens_col].astype("category").cat.codes.values, dtype=torch.long)
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
        sens = torch.tensor((df[age_col] > 25).astype(int).values, dtype=torch.long)
        N = sens.numel()
        A = torch.zeros((N, N), dtype=torch.bool)
        with open("dataset/credit/credit_edges.txt") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 2:
                    continue
                u = int(float(parts[0])); v = int(float(parts[1]))
                if 0 <= u < N and 0 <= v < N and u != v:
                    A[u, v] = True
                    A[v, u] = True
        return A, sens

    raise ValueError(ds_name)

# =========================
# Pair type and target proportions
# =========================
def pair_type(u, v, sens):
    return 0 if int(sens[u]) == int(sens[v]) else 1  # 0=ss, 1=ds

def target_from_original_edges(A, sens):
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
    tot = max(1, ss + ds)
    return np.array([ss / tot, ds / tot], dtype=float)

# =========================
# Candidate sampling + proxy scoring (Jaccard)
# =========================
def build_neighbors(A):
    # list of neighbor sets for jaccard (CPU friendly)
    N = A.size(0)
    neigh = []
    for i in range(N):
        neigh.append(set(torch.nonzero(A[i], as_tuple=False).view(-1).tolist()))
    return neigh

def sample_nonedges(A, num_samples, rng):
    N = A.size(0)
    pairs = []
    tries = 0
    while len(pairs) < num_samples and tries < num_samples * 50:
        u = int(rng.integers(0, N))
        v = int(rng.integers(0, N))
        if u == v:
            tries += 1
            continue
        if u > v:
            u, v = v, u
        if not A[u, v]:
            pairs.append((u, v))
        tries += 1
    return pairs

def jaccard_score(u, v, neigh):
    Nu = neigh[u]
    Nv = neigh[v]
    inter = len(Nu & Nv)
    union = len(Nu | Nv)
    if union == 0:
        return 0.0
    return inter / union

def score_candidates(cands, neigh):
    # vector of float scores
    return np.array([jaccard_score(u, v, neigh) for (u, v) in cands], dtype=float)

# =========================
# Greedy vs Worst quota ranking
# =========================
def quota_vector(target_p, K):
    q = np.floor(target_p * K + 1e-9).astype(int)
    while q.sum() < K:
        q[np.argmax(target_p)] += 1
    while q.sum() > K:
        q[np.argmax(q)] -= 1
    return q

def greedy_quota_ranking(cands, scores, sens, target_p, K):
    order = np.argsort(-scores)  # high to low
    need = quota_vector(target_p, K).copy()
    picked = []
    for idx in order:
        u, v = cands[idx]
        t = pair_type(u, v, sens)
        if need[t] > 0:
            picked.append((u, v))
            need[t] -= 1
            if len(picked) >= K:
                break
    if len(picked) < K:
        for idx in order:
            uv = cands[idx]
            if uv not in picked:
                picked.append(uv)
            if len(picked) >= K:
                break
    return picked

def worst_quota_ranking(cands, scores, sens, target_p, K):
    order = np.argsort(scores)  # low to high (adversarial)
    need = quota_vector(target_p, K).copy()
    picked = []
    for idx in order:
        u, v = cands[idx]
        t = pair_type(u, v, sens)
        if need[t] > 0:
            picked.append((u, v))
            need[t] -= 1
            if len(picked) >= K:
                break
    if len(picked) < K:
        for idx in order:
            uv = cands[idx]
            if uv not in picked:
                picked.append(uv)
            if len(picked) >= K:
                break
    return picked

# =========================
# NDKL (discounted running KL)
# =========================
def kl(p, q, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    return float(np.sum(p * np.log(p / q)))

def ndkl_curve(ranking_edges, sens, target_p, max_k, step):
    discounts = np.array([1.0 / math.log2(i + 2) for i in range(max_k)], dtype=float)
    Zprefix = np.cumsum(discounts)

    ss = ds = 0
    running = 0.0
    out_k, out_ndkl = [], []

    for i in range(1, max_k + 1):
        u, v = ranking_edges[i - 1]
        if pair_type(u, v, sens) == 0:
            ss += 1
        else:
            ds += 1
        pref = np.array([ss / i, ds / i], dtype=float)
        running += kl(pref, target_p) * discounts[i - 1]

        if i % step == 0 or i == max_k:
            out_k.append(i)
            out_ndkl.append(running / Zprefix[i - 1])

    return out_k, out_ndkl

# =========================
# Figure 5
# =========================
def plot_figure5():
    rng = np.random.default_rng(RNG_SEED)

    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    axes = axes.flatten()

    for ax, (ds_name, title) in zip(axes, DATASETS):
        A, sens = load_graph_and_sensitive(ds_name)
        target_p = target_from_original_edges(A, sens)

        neigh = build_neighbors(A)

        # sample candidate non-edges + score with proxy
        cands = sample_nonedges(A, NONEDGE_SAMPLE, rng)
        scores = score_candidates(cands, neigh)

        K = min(MAX_K, len(cands))
        greedy_rank = greedy_quota_ranking(cands, scores, sens, target_p, K)
        worst_rank  = worst_quota_ranking(cands, scores, sens, target_p, K)

        ks, g_curve = ndkl_curve(greedy_rank, sens, target_p, K, STEP)
        _,  w_curve = ndkl_curve(worst_rank,  sens, target_p, K, STEP)

        ax.plot(ks, w_curve, "--", label="Simulated Worst Ranking")
        ax.plot(ks, g_curve, "-", label="Greedy Ranking")
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel("NDKL")
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.7)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Figure 5 (Logic-equivalent): NDKL gap vs K (Greedy vs Simulated Worst)", y=0.98)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])

    os.makedirs("figures", exist_ok=True)
    out_path = "figures/figure5_logic.png"
    plt.savefig(out_path, dpi=200)
    print(f"[OK] Saved: {out_path}")
    plt.show()

if __name__ == "__main__":
    plot_figure5()
