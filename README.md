# BrainMRIAnalysis
 Brain MRI Analysis repository consists of four different methods for analyzing brain MRI images including:
 1- Contrastive learning
 2- Graph neural networks
 3- Multi-instance learning
 
 The python codes for each method saved in its corresponding folder. Moreover, each folder has a separate
 readme.md file for additional necessary details of each method.
 
 1- Contrastive learning
 Contrastive learning in this repository is based on SimCLR as a self-supervised learning framework designed to learn robust visual representations from unlabeled MRI images. 
 In this approach, the model consider MRI image pairs belonging to the same person as positive pair and 
 MrI image pairs of two different persons as a negative pair.
 The image pairs are then encoded into feature vectors. The core idea is to maximize the similarity between the positive pairs while minimizing the similarity between the negative pairs. 
 This is achieved through a contrastive loss function that leverages cosine similarity and a temperature parameter to refine the learning process. 
 By training on large batches of data, SimCLR effectively learns to create meaningful embeddings that can be utilized for various downstream tasks, significantly improving performance over traditional supervised methods.
 The encoder weights are saved to be used further for MRI classification for detecting tumor types and their grades.
 
 2- Graph neural networks
 This section has several python codes. Some of them are written using Keras API and some others use pytorch for designing, training and applying neural network models.
a code using pytorch (main_pytorch.py) implements a Graph Neural Network (GNN) for classifying MRI images using PyTorch and PyTorch Geometric. The workflow involves reading MRI data, generating bags of images, constructing graphs, and training a GNN model to classify the MRI scans.

We have implemented (main_CNN_Keras.py) including a graph-based convolutional neural network (CNN) for classifying MRI images using TensorFlow, Keras and NetworkX. The workflow involves reading MRI data, generating bags of images,constructing graphs, and training a CNN model to classify the MRI scans.
Another code (main_gcn_keras.py) implements a Graph Convolutional Network (GCN) for classifying MRI images using TensorFlow and NetworkX. This project demonstrates how to leverage graph-based representations of MRI images to enhance classification tasks using a Graph Convolutional Network. The integration of graph theory with deep learning allows for more effective feature extraction and improved model performance on medical imaging datasets.
 
 3- Multi-instance learning
This project (main_MultiInstanceLearning.py) demonstrates how to apply Multiple Instance Learning (MIL) for classifying MRI images. By treating each MRI scan as a bag of instances (image slices) and leveraging the MIL approach, the model can effectively learn from the data and make accurate predictions. The integration of MIL with deep learning allows for more robust and effective classification of medical imaging datasets.