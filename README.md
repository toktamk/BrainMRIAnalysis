# BrainMRIAnalysis

Research-grade framework for brain MRI experiments across three modeling
paradigms:

-   **ContrastiveLearning** --- SimCLR-style representation learning
    using NT-Xent loss.
-   **GraphNeuralNetworks** --- Slice-bag → graph construction → graph
    classification (PyTorch Geometric).
-   **MultiInstanceLearning** --- Attention-based MIL for bag-of-slices
    classification.

The repository is structured for reproducibility, modular
experimentation, and CI validation.

------------------------------------------------------------------------

## Project Structure

    brainmrianalysis/
    │
    ├── src/
    │   ├── ContrastiveLearning/
    │   ├── GraphNeuralNetworks/
    │   ├── MultiInstanceLearning/
    │   └── brainmrianalysis/
    │
    ├── scripts/
    ├── tests/
    ├── pyproject.toml
    └── requirements.txt

------------------------------------------------------------------------

## Expected Dataset Layout

The dataset must follow this directory structure:

    DATA_ROOT/
      grade1/
        patient_001/
          slice_0001.png
          slice_0002.png
      grade2/
      grade3/
      grade4/

-   Class labels are inferred from folder names (`grade1`--`grade4`).
-   Images are resized to `--image_size` (default: 224).
-   CrossEntropyLoss expects class indices `0..3`.

------------------------------------------------------------------------

## Installation

Create a virtual environment and install dependencies:

``` bash
pip install -r requirements.txt

```

If using PyTorch Geometric, install it following:
https://pytorch-geometric.readthedocs.io/

------------------------------------------------------------------------

## Running Tests

``` bash
pytest
```

------------------------------------------------------------------------

## Running Experiments

### Contrastive Learning

``` bash
python src/ContrastiveLearning/train.py   --data_root /path/to/DATA_ROOT   --epochs 100   --batch_size 64   --encoder mobilenet_v2   --pretrained
```

------------------------------------------------------------------------

### Graph Neural Network

``` bash
python src/GraphNeuralNetworks/train.py   --data_root /path/to/DATA_ROOT   --epochs 50   --bagsize 5   --bags_num 60
```

------------------------------------------------------------------------

### Multi-Instance Learning

``` bash
python src/MultiInstanceLearning/train.py   --data_root /path/to/DATA_ROOT   --epochs 30   --bagsize 4   --bags_num 80
```

------------------------------------------------------------------------

## Sanity Checks

### Dataset Audit

``` bash
python scripts/sanity_data_audit.py --data_root /path/to/DATA_ROOT
```

### Determinism Check

``` bash
python scripts/sanity_seed_repro.py --data_root /path/to/DATA_ROOT
```

------------------------------------------------------------------------

## Reproducibility Notes

-   All training scripts accept `--seed`.
-   Deterministic CuDNN settings are enabled when possible.
-   Sanity overfit scripts are provided to validate learning capacity.

------------------------------------------------------------------------

## License

MIT License.
