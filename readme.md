# BrainMRIAnalysis

A research-grade, modular framework for hierarchical brain MRI tumor
classification.

This repository integrates four complementary modeling paradigms:

1.  Contrastive Learning (SimCLR-style self-supervision)
2.  Attention-Based Multi-Instance Learning (MIL)
3.  Graph Neural Networks (GNN)
4.  Probabilistic Model Ensemble (MIL, GNN and SSL)

The framework is designed for:

-   Patient-level modeling
-   Hierarchical tumor type → grade prediction
-   Reproducible experimentation
-   Robust evaluation with ROC / PR / confusion matrices
-   Modular cross-paradigm comparison and fusion

# Executive Architecture Overview

Each paradigm captures a different structural bias:

  ---------------------------------------------------------------------------------
  Module                  Inductive Bias                Modeling Unit
  ----------------------- ----------------------------- ---------------------------
  ContrastiveLearning     Representation invariance     Slice-level

  MultiInstanceLearning   Attention-weighted            Bag of slices
                          aggregation                   

  GraphNeuralNetworks     Relational slice modeling     Slice graph

  Ensemble                Cross-model fusion            Probability-level fusion
  ---------------------------------------------------------------------------------

Final hierarchical factorization used across modules:

P(type, grade \| X) = P(type \| X) · P(grade \| X, type)

Grade inference uses mixture:

P(g) = Σ_t P(t) P(g \| t)

# Project Structure

brainmrianalysis/ │ ├── src/ │ ├── ContrastiveLearning/ │ ├──
MultiInstanceLearning/ │ ├── GraphNeuralNetworks/ │ └──
brainmrianalysis/ │ ├── scripts/ ├── tests/ ├── requirements.txt └──
pyproject.toml

# Expected Dataset Layout

The dataset must follow the hierarchical structure:

DATA_ROOT/ actrocytoma-g1/ P1/ slice_0001.png slice_0002.png
glioblastoma-g4/ meningioma-g2/

Folder naming convention:

`<tumor_type>`{=html}-g\<1..4\>

Example: - meningioma-g1 - glioblastoma-g4

Each patient folder contains 2D MRI slices.

# Installation

Create environment:

pip install -r requirements.txt

If using GNN:

Install PyTorch Geometric according to:
https://pytorch-geometric.readthedocs.io/

Optional developer tools:

pip install pytest ruff black pre-commit

# Experimental Workflows

============================================================ 1.
Contrastive Learning (Self-Supervised Pretraining)
============================================================

Learn MRI-specific slice representations without labels:

python src/ContrastiveLearning/pretrain_simclr.py\
--data_root /path/to/DATA_ROOT\
--epochs 50\
--batch_size 64\
--encoder mobilenet_v2

Output: simclr_pretrain.pt

This encoder can be transferred to supervised pipelines.

============================================================ 2.
Attention-Based MIL (Hierarchical Two-Step)
============================================================

python src/MultiInstanceLearning/train.py\
--data_root /path/to/DATA_ROOT\
--epochs 30\
--batch_size 2\
--bag_size 8\
--emb_dim 256\
--attn_dim 128

Full evaluation and plots:

python src/MultiInstanceLearning/e2e_mil_two_step_pipeline.py\
--data_root /path/to/DATA_ROOT

Outputs: - ROC curves - PR curves - Confusion matrices -
metrics_test.json - test_predictions.csv

============================================================ 3. Graph
Neural Network (Slice → Graph → Classification)
============================================================

python src/GraphNeuralNetworks/e2e_two_step_pipeline.py\
--data_root /path/to/DATA_ROOT\
--epochs 40\
--bag_size 8\
--node_dim 64\
--hidden_dim 128

Outputs: - Graph-based hierarchical classification - ROC / PR curves -
Confusion matrices - predictions_test.csv

============================================================ 4. Ensemble
(MIL + GNN + SSL)
============================================================

Combine all trained models probabilistically:

python src/MultiInstanceLearning/ensemble_two_step.py\
--data_root /path/to/DATA_ROOT\
--ckpt_mil runs/e2e_mil_two_step/best.pt\
--ckpt_gnn runs/gnn_two_step_e2e/best.pt\
--ckpt_ssl runs/e2e_simclr_two_step/mil_two_step_best.pt\
--w_mil 1.0\
--w_gnn 1.0\
--w_ssl 1.0

Outputs: - metrics_ensemble.json - predictions_ensemble.csv - Ensemble
ROC/PR plots - Confusion matrices

# Evaluation Metrics

For Tumor Type:

-   Accuracy
-   Balanced Accuracy
-   Macro / Weighted F1
-   ROC-AUC
-   Precision-Recall AP

For Grade:

-   Teacher-forced accuracy
-   End-to-end grade accuracy
-   Strict joint correctness (type AND grade correct)

# Sanity Checks

Dataset validation:

python scripts/sanity_data_audit.py --data_root /path/to/DATA_ROOT

Seed reproducibility:

python scripts/sanity_seed_repro.py --data_root /path/to/DATA_ROOT

MIL overfit test:

python src/MultiInstanceLearning/sanity_overfit_small.py\
--data_root /path/to/DATA_ROOT

Expected: Training accuracy → \~1.0

# Research Contributions

This framework enables:

-   Self-supervised MRI representation learning
-   Attention-based patient-level modeling
-   Graph-based relational slice modeling
-   Hierarchical probabilistic tumor inference
-   Cross-paradigm ensemble fusion
-   Reproducible medical AI experimentation

# Limitations

-   2D slice modeling (no volumetric 3D context)
-   Independent grade heads per tumor type
-   Ensemble weights manually specified
-   Performance sensitive to dataset size and balance

# Future Directions

-   3D volumetric encoders
-   Transformer-based MIL
-   Graph Attention Networks
-   Bayesian model averaging
-   Meta-learned ensemble weighting
-   Cross-institution domain adaptation
-   Uncertainty calibration analysis

# License

MIT License.
