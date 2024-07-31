They are four .py files in this folder:
    - SimCLR.py
    - MRIData.py
    - ContrastiveLossClass.py
    - main_MRI_ContrastiveLearning.py
    
SimCLR.py:
The SimCLR class is a PyTorch implementation of the SimCLR framework, 
which is designed for contrastive learning of visual representations. 
This class defines the architecture of the model, including a base encoder for feature extraction 
and a projection head for mapping the features into a lower-dimensional space.

MRIDATA.py
The MRIData class is a custom PyTorch dataset that loads and preprocesses MRI image pairs for use in a machine learning model. 
This class inherits from torchvision.datasets.VisionDataset and provides a convenient way to load and access MRI data.

ContrastiveLossClass.py
This loss function is commonly used in contrastive learning tasks, particularly in models like SimCLR, to encourage similar embeddings to be close together in the feature space while pushing dissimilar embeddings apart.
The ContrastiveLoss class computes the contrastive loss between pairs of embeddings and their projections. It leverages cosine similarity to measure the similarity between embeddings and employs a temperature scaling factor to control the sharpness of the similarity distribution.

main_MRI_ContrastiveLearning.py
This code sets up the data loader for the MRI dataset and prepares the encoder for use in the subsequent steps of the SimCLR training process.

For running the codes, you should run only main_MRI_ContrastiveLearning.py
Other files and classes are called in main_MRI_ContrastiveLearning.py.