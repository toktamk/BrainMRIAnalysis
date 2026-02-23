# Ensemble Two-Step Brain MRI Tumor Classification

# Executive Overview

This script implements a **model-level probabilistic ensemble** for
hierarchical brain MRI tumor analysis.

It combines predictions from three independently trained models:

1.  Attention-Based Multi-Instance Learning (MIL)
2.  Graph Neural Network (GNN)
3.  SSL-based Two-Step MIL (Contrastive Pretraining + Supervised
    Fine-Tuning)

The ensemble operates at the **probability level** and supports:

-   Tumor type classification
-   Grade prediction via hierarchical decomposition
-   End-to-end strict correctness evaluation
-   Full ROC / PR / Confusion Matrix reporting

Implementation: fileciteturn2file0

# Scientific Motivation

Each model captures complementary inductive biases:

  Model     Structural Bias
  --------- -----------------------------------------
  MIL       Attention-based slice weighting
  GNN       Relational slice modeling
  SSL-MIL   Self-supervised representation learning

Combining them reduces:

-   Variance
-   Overfitting risk
-   Architecture-specific bias
-   Sensitivity to data imbalance

The ensemble approximates:

P(type, grade \| X) ≈ Σ w_i P_i(type, grade \| X)

# Methodological Framework

## Hierarchical Factorization

Each model outputs:

-   type_logits: (B, T)
-   grade_logits: (B, T, G)

Final grade probability is computed via mixture:

P(g) = Σ_t P(t) · P(g \| t)

Implemented in:

grade_probs_mixture()

## Class Alignment

Before averaging probabilities, class order is aligned across models.

Utilities:

-   build_reindex()
-   reindex_probs()

Ensures consistent mapping across: - MIL - GNN - SSL

## Weighted Probability Averaging

User-defined weights:

--w_mil --w_gnn --w_ssl

Normalized internally:

w = w / sum(w)

Final probabilities:

p_type_ens = w1·p1 + w2·p2 + w3·p3 p_grade_ens = w1·p1 + w2·p2 + w3·p3

# Required Inputs

The script requires:

-   Trained MIL checkpoint
-   Trained GNN checkpoint
-   Trained SSL-MIL checkpoint
-   Dataset root
-   Optional split.json (for consistent evaluation)

Default checkpoints:

--ckpt_mil runs/e2e_mil_two_step/best.pt --ckpt_gnn
runs/gnn_two_step_e2e/best.pt --ckpt_ssl
runs/e2e_simclr_two_step/mil_two_step/mil_two_step_best.pt

# Complete Experimental Workflow

============================================================ STEP 1 ---
Train Individual Models
============================================================

Train:

-   MultiInstanceLearning/train.py
-   GraphNeuralNetworks/e2e_two_step_pipeline.py
-   ContrastiveLearning pipeline

Ensure consistent dataset splits.

============================================================ STEP 2 ---
Run Ensemble
============================================================

python ensemble_two_step.py\
--data_root "D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets"\
--ckpt_mil runs/e2e_mil_two_step/best.pt\
--ckpt_gnn runs/gnn_two_step_e2e/best.pt\
--ckpt_ssl runs/e2e_simclr_two_step/mil_two_step/mil_two_step_best.pt\
--w_mil 1.0\
--w_gnn 1.0\
--w_ssl 1.0

# Output Artifacts

Generated files:

-   cm_type_ens.png
-   cm_grade_ens.png
-   roc_type_ens.png
-   pr_type_ens.png
-   roc_grade_ens.png
-   pr_grade_ens.png
-   predictions_ensemble.csv
-   metrics_ensemble.json

Metrics include:

-   Tumor type accuracy
-   Grade accuracy
-   Strict end-to-end correctness
-   Per-class ROC-AUC
-   Per-class Average Precision

# Evaluation Metrics

Type:

-   Accuracy
-   Balanced Accuracy
-   Macro F1
-   ROC-AUC

Grade:

-   Accuracy
-   Macro F1
-   ROC-AUC

Strict Metric:

end2end_all_correct

# Research Contributions

This ensemble provides:

-   Architecture-level uncertainty reduction
-   Cross-paradigm fusion (MIL + GNN + SSL)
-   Probabilistic hierarchical fusion
-   Robust test-time evaluation pipeline
-   Class-order-safe aggregation

# Limitations

-   Assumes identical class universes
-   Requires aligned preprocessing
-   Requires consistent dataset splits
-   Equal weighting may not be optimal

# Future Extensions

-   Validation-set weight optimization
-   Bayesian model averaging
-   Stacked meta-learner (logistic regression on logits)
-   Temperature calibration per model
-   Uncertainty-based weighting
-   Cross-validation ensemble

# Citation Template

Toktam Khatibi, 2026\
Ensemble Learning for Hierarchical Brain MRI Tumor Classification\
Tarbiat Modares University
