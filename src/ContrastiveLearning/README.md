# MRI Tumor Analysis Using Contrastive Learning and Hierarchical Classification


# Executive Overview

This folder provides a research-grade deep learning framework for
brain MRI tumor analysis that integrates:

1.  Self-Supervised Contrastive Pretraining (SimCLR)
2.  Hierarchical Two-Step Supervised Classification (Tumor Type and its Grade Determination)

The framework supports:

-   Domain-specific representation learning
-   Patient-level multi-instance modeling
-   Hierarchical probabilistic grade inference
-   Comprehensive evaluation with ROC and Precision--Recall analysis

# Scientific Motivation

Brain MRI datasets present several methodological challenges:

-   Limited labeled data
-   High intra-class variability
-   Subtype imbalance
-   Multi-center heterogeneity
-   Complex radiological morphology

Contrastive self-supervised learning enables feature extraction without
labels, allowing the model to learn MRI-specific structural invariances
prior to supervised fine-tuning.

# Methodological Framework

## Stage 1 --- Contrastive Pretraining (SimCLR)

We train an encoder fθ using NT-Xent contrastive loss:

L = -log( exp(sim(z_i, z_j)/τ) / Σ_k exp(sim(z_i, z_k)/τ) )

Where:

-   z = g(fθ(x))
-   g = projection head
-   τ = temperature

Properties:

-   No tumor labels required
-   MRI-specific augmentations
-   Learns radiologically meaningful embeddings
-   Projection head discarded after training

Output:

simclr_pretrain.pt

## Stage 2 --- Hierarchical Two-Step Classification

Instead of flat classification, we model:

P(type \| X) P(grade \| X, type)

Final end-to-end grade probability:

P(g) = Σ_t P(t) P(g \| t)

Architecture:

-   Backbone: MobileNetV2 or SmallCNN
-   Slice-level feature extraction
-   Patient-level pooling (Multi-Instance Learning)
-   Tumor type classification head
-   Type-conditioned grade head

# Folder Structure

src/ContrastiveLearning/

-   pretrain_simclr.py
-   e2e_two_step_pipeline_ssl.py
-   models.py
-   losses.py
-   MRIData.py

# Data Organization

Expected structure:

MRI_Mahdieh_Datasets/ actrocytoma-g1/ P1/ P2/ glioblastoma-g3/
meningioma-g2/

Each subtype directory contains patient folders, which contain MRI
slices.

Splitting is performed at patient level with robust fallback
stratification.

# Complete Experimental Workflow

========================================= STEP 1 --- Contrastive
Pretraining (SimCLR) =========================================

Objective: Learn domain-specific MRI encoder weights without using
labels.

Command (Windows one-line):

python src/ContrastiveLearning/pretrain_simclr.py --data_root
"D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets" --outdir
"runs`\simclr`{=tex}\_pretrain_mri" --encoder mobilenet_v2
--projection_dim 128 --temperature 0.5 --epochs 50 --batch_size 64 --lr
1e-3 --pretrained

What happens:

-   Loads MRI slices in contrastive mode
-   Applies stochastic augmentation twice per sample
-   Trains encoder + projection head
-   Saves best checkpoint based on validation contrastive loss

Output checkpoint:

runs/simclr_pretrain_mri/simclr_pretrain.pt

This file contains:

-   simclr_state (encoder + projection weights)
-   encoder configuration
-   training metadata

Only encoder weights will be used for downstream supervised training.

## STEP 2 --- Supervised Two-Step Training Using SSL Encoder

Objective: Fine-tune the contrastively pretrained encoder for tumor type
and grade classification.

Command (Windows one-line):

python src/ContrastiveLearning/e2e_two_step_pipeline_ssl.py --data_root
"D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets" --outdir
"runs`\e`{=tex}2e_two_step_full_ssl" --epochs 40 --batch_size 2
--bag_size 8 --bag_policy uniform --backbone mobilenet_v2 --ssl_ckpt
"runs`\simclr`{=tex}\_pretrain_mri`\simclr`{=tex}\_pretrain.pt"

What happens:

-   Loads SimCLR checkpoint
-   Extracts encoder.\* weights
-   Loads them into TwoStepClassifier backbone
-   Trains tumor type head
-   Trains grade head conditioned on tumor type
-   Performs early stopping
-   Evaluates on test set

Optional: Freeze backbone (recommended for small datasets)

Add:

--freeze_backbone

## STEP 3 --- Evaluation and Output Artifacts

Generated outputs include:

-   best.pt
-   test_metrics.json
-   test_report.json
-   roc_tumor_type_ovr.png
-   pr_tumor_type_ovr.png
-   roc_grade_e2e_ovr.png
-   pr_grade_e2e_ovr.png
-   curves_summary.json

Metrics reported:

-   Tumor Type Accuracy
-   End-to-End Grade Accuracy
-   Conditional Grade Accuracy
-   Macro F1
-   Confusion Matrices
-   Per-class AUC
-   Per-class Average Precision

# Research Contributions

This framework enables:

-   Domain-specific MRI representation learning
-   Hierarchical probabilistic tumor modeling
-   Robust evaluation under subtype imbalance
-   Patient-level clinical alignment
-   Modular experimentation (SSL → supervised transfer)

# Limitations

-   Rare subtypes reduce statistical reliability
-   2D slice modeling does not capture full 3D tumor volume
-   Performance sensitive to augmentation strategy
-   Requires careful class imbalance management

# Future Extensions

-   3D contrastive pretraining
-   Vision Transformer backbone
-   Attention-based MIL pooling
-   Radiomics and deep feature fusion
-   Cross-institution domain adaptation
-   Uncertainty calibration analysis

# Citation Template

Toktam Khatibi, 2026, MRI Tumor Analysis via Contrastive Learning and Hierarchical
Classification, Tarbiat Modares University.
