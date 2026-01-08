import torch

def score_edges_dot(z: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    # edges: [E,2]
    src, dst = edges[:,0], edges[:,1]
    logits = (z[src] * z[dst]).sum(dim=1)
    return torch.sigmoid(logits)

@torch.no_grad()
def main():
    # 1) load splits
    data, splits = torch.load("data/splits/cora.pt")
    test_pos = splits["test"]["edge"]
    test_neg = splits["test"]["edge_neg"]
    edges = torch.cat([test_pos, test_neg], dim=0)
    y_true = torch.cat([torch.ones(test_pos.size(0)), torch.zeros(test_neg.size(0))], dim=0)

    # 2) load model output embeddings z
    # TODO: sesuaikan: dari training kamu, z disimpan di mana?
    # Contoh: jika kamu punya checkpoint yang menyimpan z:
    ckpt = torch.load("PATH_KE_CHECKPOINT.pt", map_location="cpu")
    z = ckpt["z"]  # <-- sesuaikan key ini

    # 3) score edges
    scores = score_edges_dot(z, edges)

    # 4) ranking & prec@1000
    order = torch.argsort(scores, descending=True)
    topk = order[:1000]
    prec_at_1000 = y_true[topk].mean().item()

    print("prec@1000 =", prec_at_1000)

if __name__ == "__main__":
    main()
