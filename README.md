# BrainMRIAnalysis

A research-grade, modular framework for **hierarchical brain MRI tumor classification**.

This repository integrates four complementary modeling paradigms:

1. **Contrastive Learning** (SimCLR-style self-supervision)  
2. **Attention-Based Multi-Instance Learning (MIL)**  
3. **Graph Neural Networks (GNN)**  
4. **Probabilistic Model Ensemble (MIL + GNN + SSL)**  

The framework is designed for:

- Patient-level modeling  
- Hierarchical tumor type → grade prediction  
- Reproducible experimentation  
- Robust evaluation with ROC / PR / confusion matrices  
- Modular cross-paradigm comparison and fusion  

---

# Executive Architecture Overview

Each paradigm captures a different inductive bias:

| Module | Inductive Bias | Modeling Unit |
|--------|----------------|--------------|
| ContrastiveLearning | Representation invariance | Slice-level |
| MultiInstanceLearning | Attention-weighted aggregation | Bag of slices |
| GraphNeuralNetworks | Relational slice modeling | Slice graph |
| Ensemble | Cross-model fusion | Probability-level |

All supervised models follow hierarchical factorization:

```
P(type, grade | X) = P(type | X) · P(grade | X, type)
```

Grade inference uses probabilistic mixture:

```
P(g) = Σ_t P(t) · P(g | t)
```

---

# Project Structure

```
brainmrianalysis/
│
├── src/
│   ├── ContrastiveLearning/
│   ├── MultiInstanceLearning/
│   ├── GraphNeuralNetworks/
│   └── brainmrianalysis/
│
├── scripts/
├── tests/
├── requirements.txt
└── pyproject.toml
```

---

# Expected Dataset Layout

```
DATA_ROOT/
  actrocytoma-g1/
    P1/
      slice_0001.png
      slice_0002.png
  glioblastoma-g4/
  meningioma-g2/
```

Folder naming convention:

```
<tumor_type>-g<1..4>
```

---

# Installation

```bash
pip install -r requirements.txt
```

If using Graph Neural Networks, follow PyTorch Geometric installation:

https://pytorch-geometric.readthedocs.io/

Optional developer tools:

```bash
pip install pytest ruff black pre-commit
```

---

# Experimental Workflows

## 1. Contrastive Learning

```bash
python src/ContrastiveLearning/pretrain_simclr.py   --data_root /path/to/DATA_ROOT   --epochs 50   --batch_size 64   --encoder mobilenet_v2
```

Output: `simclr_pretrain.pt`

---

## 2. Attention-Based MIL

```bash
python src/MultiInstanceLearning/train.py   --data_root /path/to/DATA_ROOT   --epochs 30   --batch_size 2   --bag_size 8   --emb_dim 256   --attn_dim 128
```

Full evaluation:

```bash
python src/MultiInstanceLearning/e2e_mil_two_step_pipeline.py   --data_root /path/to/DATA_ROOT
```

---

## 3. Graph Neural Networks

```bash
python src/GraphNeuralNetworks/e2e_two_step_pipeline.py   --data_root /path/to/DATA_ROOT   --epochs 40   --bag_size 8   --node_dim 64   --hidden_dim 128
```

---

## 4. Ensemble

```bash
python src/MultiInstanceLearning/ensemble_two_step.py   --data_root /path/to/DATA_ROOT   --ckpt_mil runs/e2e_mil_two_step/best.pt   --ckpt_gnn runs/gnn_two_step_e2e/best.pt   --ckpt_ssl runs/e2e_simclr_two_step/mil_two_step_best.pt   --w_mil 1.0   --w_gnn 1.0   --w_ssl 1.0
```

---

# Evaluation Metrics

Tumor Type:

- Accuracy  
- Balanced Accuracy  
- Macro / Weighted F1  
- ROC-AUC  
- Precision-Recall AP  

Grade:

- Teacher-forced accuracy  
- End-to-end grade accuracy  
- Strict joint correctness  

---

# Sanity Checks

```bash
python scripts/sanity_data_audit.py --data_root /path/to/DATA_ROOT
```

```bash
python scripts/sanity_seed_repro.py --data_root /path/to/DATA_ROOT
```

```bash
python src/MultiInstanceLearning/sanity_overfit_small.py   --data_root /path/to/DATA_ROOT
```

Expected training accuracy → ~1.0

---

# License

MIT License.
