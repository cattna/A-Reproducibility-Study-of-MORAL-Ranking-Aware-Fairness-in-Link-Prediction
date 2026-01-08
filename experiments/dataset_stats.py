import torch
import pandas as pd

from datasets import Facebook, German, Pokec_n, Pokec_z

ATTR_MAP = {
    "facebook": "Gen.",
    "german": "Age",
    "nba": "Nat.",
    "pokec_n": "Gen.",
    "pokec_z": "Gen.",
    "credit": "Age",
}

def print_table(headers, rows):
    cols = list(zip(*([headers] + rows)))
    widths = [max(len(str(x)) for x in col) for col in cols]

    def fmt(row):
        return " | ".join(str(x).ljust(w) for x, w in zip(row, widths))

    sep = "-+-".join("-" * w for w in widths)

    print(fmt(headers))
    print(sep)
    for r in rows:
        print(fmt(r))

def compute_stats(edges, X, s):
    num_nodes = X.shape[0]
    num_edges = len(edges)

    same = diff = 0
    for u, v in edges:
        if s[u] == s[v]:
            same += 1
        else:
            diff += 1

    total = same + diff
    es_diff = round(100 * diff / total, 2) if total > 0 else 0.0
    es_same = round(100 * same / total, 2) if total > 0 else 0.0

    es_other = 0.0

    return num_nodes, num_edges, X.shape[1], es_diff, es_same, es_other

def load_moral_dataset(ds):
    A = ds.adj().coalesce()
    X = ds.features()
    s = ds.sens()

    row, col = A.indices()

    # undirected, no self-loops
    mask = (row != col) & (row < col)
    edges = list(zip(row[mask].tolist(), col[mask].tolist()))

    return compute_stats(edges, X, s)

def load_nba():
    df = pd.read_csv("dataset/nba/nba.csv")

    id_col = next(c for c in ["player_id", "id", "user_id"] if c in df.columns)
    id_map = {pid: i for i, pid in enumerate(df[id_col].values)}

    sens_col = next(
        c for c in [
            "nationality", "country", "birth_country",
            "nation", "international", "is_foreign"
        ] if c in df.columns
    )

    s = torch.tensor(
        df[sens_col].astype("category").cat.codes.values,
        dtype=torch.long,
    )

    X = torch.tensor(
        df.drop(columns=[id_col, sens_col]).values,
        dtype=torch.float,
    )

    edges = []
    with open("dataset/nba/nba_relationship.txt") as f:
        for line in f:
            u_raw, v_raw = map(int, line.split())
            if u_raw in id_map and v_raw in id_map:
                edges.append((id_map[u_raw], id_map[v_raw]))

    return compute_stats(edges, X, s)

def load_credit():
    df = pd.read_csv("dataset/credit/credit.csv")

    age_col = next(c for c in ["age", "Age", "AGE"] if c in df.columns)

    s = torch.tensor(
        (df[age_col] > 25).astype(int).values,
        dtype=torch.long,
    )

    X = torch.tensor(
        df.drop(columns=[age_col]).values,
        dtype=torch.float,
    )

    edges = []
    with open("dataset/credit/credit_edges.txt") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            u = int(float(parts[0]))
            v = int(float(parts[1]))
            edges.append((u, v))

    return compute_stats(edges, X, s)

if __name__ == "__main__":

    rows = []

    V, E, F, d, s, o = load_moral_dataset(Facebook())
    rows.append(["facebook", V, E, F, ATTR_MAP["facebook"], d, s, o])

    V, E, F, d, s, o = load_moral_dataset(German())
    rows.append(["german", V, E, F, ATTR_MAP["german"], d, s, o])

    V, E, F, d, s, o = load_nba()
    rows.append(["nba", V, E, F, ATTR_MAP["nba"], d, s, o])

    V, E, F, d, s, o = load_moral_dataset(Pokec_n())
    rows.append(["pokec_n", V, E, F, ATTR_MAP["pokec_n"], d, s, o])

    V, E, F, d, s, o = load_moral_dataset(Pokec_z())
    rows.append(["pokec_z", V, E, F, ATTR_MAP["pokec_z"], d, s, o])

    V, E, F, d, s, o = load_credit()
    rows.append(["credit", V, E, F, ATTR_MAP["credit"], d, s, o])

    headers = [
        "Dataset",
        "|V|",
        "|E|",
        "Feat.",
        "Attr.",
        "E_s'-s (%)",
        "E_s-s (%)",
        "E_s'-s' (%)",
    ]

    print_table(headers, rows)
