# A Reproducibility Study of MORAL: Ranking-Aware Fairness in Link Prediction

This repository presents a **faithful reproducibility study** of the paper:

> **Breaking the Dyadic Barrier: Rethinking Fairness in Link Prediction Beyond Demographic Parity**  
> (AAAI 2026)

The study focuses on validating the empirical and conceptual claims of **MORAL (Multi-Output Ranking Aggregation for Link fairness)**, with particular emphasis on **ranking-aware fairness evaluation** in link prediction tasks.

---

## 1. Objective

The primary goal of this project is **not** to propose a new method, but to **critically reproduce and validate** the claims of the MORAL framework, specifically:

- Whether dyadic fairness metrics such as Demographic Parity (ΔDP) can hide ranking-level bias.
- Whether **ranking-aware metrics**, particularly **NDKL**, provide a more faithful fairness assessment.
- Whether MORAL achieves a meaningful fairness–utility trade-off compared to prior fair link prediction methods.

This reproduction follows the **official MORAL repository and experimental protocol** as closely as possible, while explicitly documenting all deviations and limitations.

---

## 2. Reproduced Components

This study reproduces and analyzes the following components from the original paper:

### Tables
- **Table 1**: Dataset statistics and sensitive attribute distributions  
- **Table 2**: Ranking-based fairness and utility metrics  
- **Table 3**: Cora benchmark results  

### Figures
- **Figure 3**: Pair-type proportions in top-K predicted non-edges  
- **Figure 5**: Ranking exposure curves under different fairness constraints  

All figures are regenerated using **logic-equivalent evaluation pipelines** derived from the released code and saved model checkpoints.

---

## 3. Datasets

Six real-world graph datasets are used, matching the original paper:

- **Facebook** (Gender)
- **Credit** (Age)
- **German** (Age)
- **NBA** (Nationality)
- **Pokec-n** (Gender)
- **Pokec-z** (Gender)

Dataset statistics (number of nodes, edges, features, and sensitive attributes) were verified to match the original manuscript.

---

## 4. Experimental Setup

- **Runs**: 3 independent runs per dataset (random seeds 0, 1, 2)
- **Model**: Graph Autoencoder (GAE), as used in the MORAL framework
- **Evaluation Metrics**:
  - **Utility**: Precision@1000
  - **Fairness**: Normalized Discounted KL-Divergence (NDKL)

All experiments were executed on **Chameleon Cloud bare-metal infrastructure** to minimize hardware-induced variability.

---

## 5. Key Findings

### Ranking-Level Bias
The reproduction confirms that **ΔDP is insufficient** to capture exposure bias in ranked link prediction outputs. Certain pair types are overrepresented at top ranks despite appearing infrequently in the original graph.

### Ranking-Aware Fairness
NDKL provides a **more faithful representation of fairness** by accounting for exposure across the ranking prefix, validating the paper’s core motivation.

### MORAL Performance
MORAL consistently achieves:
- **High utility** (Precision@1000 ≈ 0.9–1.0)
- **Low NDKL values** (≈ 0.01 in reproduced results)

Although absolute NDKL values differ numerically from the paper, **the qualitative trends and relative comparisons remain consistent**.

### Cora Dataset
A notable numerical gap is observed on the Cora dataset. This discrepancy is attributed to **evaluation protocol differences**, particularly in candidate edge construction and ranking scope, rather than model failure.

---

## 6. Reproducibility Challenges

Several challenges were encountered:

- Evaluation-time artifacts (candidate non-edge lists, full score matrices) were **not released**.
- Ranking evaluation logic was implemented implicitly in scripts.
- NDKL is highly sensitive to normalization constants and candidate selection.

As a result, this study focuses on **logic-faithful reproduction** rather than bit-identical numerical matching.

---

## 7. Repository Structure

```

MORAL/
├── dataset/              
├── data/              
├── experiments/        
├── figures/        
├── datasets.py         
├── moral.py         
├── utils.py            
├── README.md

````

## 8. How to Reproduce

1. Clone the repository
2. Install dependencies (see `requirements.txt` or environment setup in the paper)
3. Generate dataset splits:
   ```bash
   python experiments/generate_splits.py
   
4. Run evaluations:
   ```bash
   python experiments/eval_rank_metrics.py --dataset <dataset>

Example:
   ```bash
   python experiments/eval_rank_metrics.py --dataset facebook

5. Reproduce figures:
   ```bash
   python experiments/figure3.py
   python experiments/figure5.py

## 9. Limitations and Future Work

While the qualitative claims of MORAL are validated, this study highlights important gaps:

* Lack of released evaluation artifacts limits exact numerical reproduction.
* Ranking-aware fairness remains sensitive to deployment-specific ranking protocols.
* Behavior under dynamic graphs and large-scale industrial settings remains unexplored.
