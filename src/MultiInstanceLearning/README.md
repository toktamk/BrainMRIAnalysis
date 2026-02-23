# Brain MRI Tumor Analysis Using Attention-Based Multi-Instance Learning (MIL)

# Executive Overview

This folder implements a research-grade **hierarchical two-step
Multi-Instance Learning (MIL) framework** for patient-level brain MRI
tumor analysis.

The system models each patient as a bag of slices and performs:

1.  Tumor Type Classification
2.  Grade Prediction Conditioned on Tumor Type
3.  End-to-End Hierarchical Evaluation
4.  Full ROC / PR / Confusion Matrix Reporting

The architecture is built around **attention-based pooling (Ilse et al.,
2018)**, enabling interpretable slice weighting at patient level.

# Scientific Motivation

Brain MRI tumor datasets are naturally multi-instance:

-   Each patient contains multiple slices
-   Labels exist at patient level (not slice level)
-   Slices vary in diagnostic importance

MIL provides:

-   Patient-level supervision
-   Learnable slice importance via attention
-   Robustness to slice ordering
-   Clinical interpretability (attention weights)

# Methodological Framework

## Instance Encoder

Each slice is encoded via CNN:

InstanceEncoder: - Conv → ReLU → Pool - Conv → ReLU → Pool - Conv → ReLU
→ AdaptiveAvgPool - Linear projection

See implementation in fileciteturn1file1

## Attention Pooling

We use attention-based aggregation:

z = Σ a_i h_i

Where:

a_i = softmax( wᵀ tanh(V h_i) )

This allows the model to learn diagnostically relevant slices.

## Two-Step Hierarchical Classification

The model factorizes:

P(type, grade \| X) = P(type \| X) P(grade \| X, type)

Outputs:

-   type_logits: (B, T)
-   grade_logits: (B, T, G)

One grade head per tumor type.

Full model: fileciteturn1file1

# Folder Structure

MultiInstanceLearning/

-   train.py
-   e2e_mil_two_step_pipeline.py
-   models.py
-   MRIDataRead.py
-   sanity_overfit_small.py

Training script: fileciteturn1file4\
Full evaluation pipeline: fileciteturn1file0\
Dataset loader: fileciteturn1file2\
Overfit sanity check: fileciteturn1file3

# Data Organization

Expected dataset structure:

MRI_Mahdieh_Datasets/ - actrocytoma-g1/ - P1/ - P2/ - glioblastoma-g4/ -
meningioma-g2/

Each subtype folder follows:

`<tumor>`{=html}-g\<1..4\>

Each patient folder contains slice images.

# Complete Experimental Workflow

============================================================ STEP 1 ---
Standard Two-Step MIL Training
============================================================

python train.py\
--data_root "D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets"\
--out_dir runs/mil_two_step\
--epochs 30\
--batch_size 2\
--bag_size 8\
--emb_dim 256\
--attn_dim 128

Outputs:

-   best.pt
-   test_metrics.json

============================================================ STEP 2 ---
Full End-to-End MIL + Reporting
============================================================

python e2e_mil_two_step_pipeline.py\
--data_root "D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets"\
--out_dir runs/e2e_mil_two_step\
--epochs 30\
--batch_size 2

Generated artifacts:

-   roc_type.png
-   pr_type.png
-   cm_type.png
-   roc_grade_teacher_forced.png
-   pr_grade_teacher_forced.png
-   cm_grade_teacher_forced.png
-   roc_grade_end2end.png
-   pr_grade_end2end.png
-   cm_grade_end2end.png
-   test_predictions.csv
-   metrics_test.json
-   history.json

============================================================ STEP 3 ---
Overfit Sanity Check
============================================================

python sanity_overfit_small.py\
--data_root "D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets"\
--steps 300

Expected result:

-   Training accuracy → \~1.0
-   Confirms gradient flow & label alignment

# Reported Metrics

Tumor Type:

-   Accuracy
-   Balanced Accuracy
-   Macro F1
-   ROC-AUC
-   Precision-Recall AP

Grade (Teacher Forced):

-   Accuracy
-   Macro F1
-   ROC-AUC

Grade (End-to-End):

-   Accuracy
-   Strict joint correctness

Joint metric:

-   end2end_all_correct

# Research Contributions

This MIL framework enables:

-   Patient-level hierarchical modeling
-   Attention-based slice importance weighting
-   Modular type-conditioned grade heads
-   Full evaluation pipeline with clinical metrics
-   Deterministic reproducibility

# Limitations

-   2D slice-based modeling
-   No explicit 3D volumetric context
-   Independent grade heads (no shared constraints)
-   Attention weights not yet exported for visualization

# Future Extensions

-   Attention weight visualization per slice
-   Transformer-based MIL
-   Multi-scale slice encoding
-   Integration with Graph Neural Networks
-   Cross-dataset domain generalization
-   Uncertainty estimation

# Citation Template

Toktam Khatibi, 2026\
Attention-Based Multi-Instance Learning for Hierarchical Brain MRI Tumor
Classification\
Tarbiat Modares University
