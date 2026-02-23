# Brain MRI Tumor Analysis Using Graph Neural Networks (GNN)

# Executive Overview

This folder implements a research-grade **end-to-end Graph Neural
Network (GNN) framework** for hierarchical brain tumor analysis using
multi-slice MRI data.

The pipeline performs:

1.  Patient-level graph construction from MRI slices\
2.  Tumor Type Classification\
3.  Grade Prediction Conditioned on Tumor Type\
4.  End-to-End Joint Evaluation

The system is designed for:

-   Patient-level modeling
-   Multi-instance learning via graph aggregation
-   Hierarchical probabilistic inference
-   Imbalanced medical datasets
-   Reproducible experimental workflows

# Scientific Motivation

MRI tumor diagnosis presents several structural challenges:

-   Variable number of slices per patient
-   Heterogeneous tumor morphology
-   Strong inter-grade similarity
-   Class imbalance
-   Limited labeled data

Instead of flattening slices or using simple pooling, this framework
models each patient as a **graph of slices**, enabling:

-   Structured relational modeling
-   Local slice-to-slice dependency capture
-   Global patient-level representation learning

# Methodological Framework

## Graph Construction

Each patient is represented as:

-   Nodes → MRI slices
-   Node features → CNN-encoded slice embeddings
-   Edges → Chain adjacency (0--1--2--...--N)

Graph construction is implemented in:

-   `graphs.py`
-   `graph_builder.py`

Chain edges are created using:

``` python
build_chain_graph(num_nodes)
```

Node features are produced by:

``` python
SliceEncoder(in_channels, out_dim)
```

See implementation details in fileciteturn0file2

## Hierarchical Two-Step Classification

The model follows a probabilistic decomposition:

P(type, grade \| X) = P(type \| X) · P(grade \| X, type)

Architecture:

1.  GCN Layer 1
2.  GCN Layer 2
3.  Global Mean Pooling
4.  Tumor Type Head
5.  Per-Type Grade Heads

Forward outputs:

-   type_logits: (B, T)
-   grade_logits: (B, T, G)

Implemented in fileciteturn0file3

## Loss Function

Teacher-forced grade training:

-   CrossEntropy(type)
-   CrossEntropy(grade \| true type)

Total loss:

L = λ_type L_type + λ_grade L_grade

Metrics computed include:

-   Tumor type accuracy
-   Grade accuracy (teacher-forced)
-   End-to-end grade accuracy
-   Joint strict accuracy

# Folder Structure

GraphNeuralNetworks/

-   train.py
-   e2e_two_step_pipeline.py
-   graphs.py
-   graph_builder.py
-   models.py
-   MRIDataRead.py

Main training entry point: fileciteturn0file5\
Full evaluation pipeline: fileciteturn0file0\
Dataset loader: fileciteturn0file4

# Data Organization

Expected dataset structure:

MRI_Mahdieh_Datasets/ - actrocytoma-g1/ - P1/ - P2/ - glioblastoma-g3/ -
meningioma-g2/

Folder format:

`<tumor>`{=html}-g\<1..4\>

Parsing logic implemented in MRIDataRead.py.

Each patient folder contains 2D MRI slices.

# Complete Experimental Workflow

============================================================ STEP 1 ---
Standard Two-Step GNN Training
============================================================

python train.py\
--data_root "D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets"\
--out_dir runs/gnn_two_step\
--epochs 30\
--batch_size 8\
--bag_size 8\
--bag_policy uniform\
--node_dim 64\
--hidden_dim 128

Outputs:

-   best.pt
-   test_metrics.json

============================================================ STEP 2 ---
Full End-to-End Evaluation and Reports
============================================================

python e2e_two_step_pipeline.py\
--data_root "D:`\datasets`{=tex}`\MRI`{=tex}\_Mahdieh_Datasets"\
--out_dir runs/gnn_two_step_e2e\
--epochs 40\
--batch_size 8\
--bag_size 8

Generated artifacts:

-   cm_type.png
-   cm_grade_e2e.png
-   cm_joint_type_grade.png
-   roc_type.png
-   pr_type.png
-   roc_grade_e2e.png
-   pr_grade_e2e.png
-   predictions_test.csv
-   test_reports.json

# Reported Metrics

Tumor Type:

-   Accuracy
-   Balanced Accuracy
-   Macro / Micro F1
-   Per-class AUC
-   Per-class AP

Grade (End-to-End):

-   Accuracy
-   Balanced Accuracy
-   Macro F1
-   ROC-AUC
-   Precision-Recall

Joint (Type and Grade):

-   Strict accuracy
-   Macro F1

# Research Contributions

This framework provides:

-   Structured patient-level modeling
-   Graph-based multi-instance learning
-   Hierarchical tumor-grade modeling
-   Robust evaluation pipeline
-   Class imbalance handling via weighted CE
-   Deterministic reproducibility (seed control)

# Limitations

-   2D slice modeling (no volumetric 3D context)
-   Chain graph topology (no learned adjacency)
-   Limited augmentation strategies
-   GCN scalability constraints for large graphs

# Future Extensions

-   Attention-based graph pooling (GAT)
-   Learnable adjacency matrices
-   3D CNN slice encoding
-   Vision Transformer node encoders
-   Cross-institution domain adaptation
-   Bayesian uncertainty calibration

# Citation Template

Toktam Khatibi, 2026\
Graph-Based Hierarchical Brain MRI Tumor Classification\
Tarbiat Modares University
