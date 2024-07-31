#This project implements a Graph Convolutional Network (GCN) for classifying MRI images using TensorFlow and NetworkX. The workflow involves reading MRI data, generating bags of images, constructing graphs, and training a GCN model to classify the MRI scans.
#This project consists of several key components:
#MRIDataRead: A module for reading and preprocessing MRI data.
#CreateGraph: A module for constructing graphs from image data.
#GCNModel: A custom model for implementing the Graph Convolutional Network.

import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

import MRIDataRead

root = root = "D://datasets//MRI_Mahdieh_Datasets//task1//"
newdim = 256
bagsize = 6
mdr = MRIDataRead.MRIDataRead(root,newdim)
total_mris,targets,channel_num = mdr.ReadData()
data_bags, target_bags = mdr.GenBags(total_mris,targets,bagsize)
graph_list = []
import CreateGraph
cg = CreateGraph.CreateGraph('keras')
adjacencymatrix_list = []
for i in range(len(data_bags)):
    graph, adjacency = cg.construct_graph(data_bags[i])
    graph_list.append(graph)
    adjacencymatrix_list.append(adjacency)
x = []
for i in range(len(graph_list)):
    graph = graph_list[i]
    adjacency = adjacencymatrix_list[i]
    features = np.array([graph.nodes[j]['features'] for j in graph.nodes()])
    x.append(features)

x = np.array(x,dtype=np.float32)
print(x.shape)
num_classes = 4
y = to_categorical(target_bags, num_classes=num_classes)
from sklearn.model_selection import train_test_split
train_indices, test_indices , train_labels, test_labels = train_test_split(range(len(x)), y,test_size=0.2, stratify=y)
x_train, x_test = x[train_indices], x[test_indices]
y_train, y_test = y[train_indices], y[test_indices]
djacency_train = [adjacencymatrix_list[i] for i in train_indices]
adjacency_test = [adjacencymatrix_list[i] for i in test_indices]
adjacency_train = np.array(adjacency_train)
adjacency_test = np.array(adjacency_test)
print(x_train.shape,adjacency_train.shape,train_labels.shape)
import GCNModel
model = GCNModel.GCNModel(num_classes)
learning_rate = 0.01  
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.compile(optimizer=optimizer,loss=tf.keras.losses.CategoricalCrossentropy(),metrics=['accuracy'])
model.fit([x_train, adjacency_train], y_train, epochs=50, batch_size=16, verbose=1)
loss, accuracy = model.evaluate([x_test, adjacency_test], y_test, verbose=0)
print(f'Test loss: {loss:.4f}')
print(f'Test accuracy: {accuracy:.4f}')