#A custom model for implementing the Graph Convolutional Network

import numpy as np
import networkx as nx
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
class GraphConvolution(layers.Layer):
    def __init__(self, units, activation):
        super(GraphConvolution, self).__init__()
        self.units = units
        self.activation = activation

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.kernel = self.add_weight("kernel", shape=(input_dim, self.units))
        #self.kernel = self.add_weight(name='kernel',shape=(input_shape[1], self.units),initializer='glorot_uniform',trainable=True)

    def call(self, inputs):
        features = tf.matmul(inputs, self.kernel)
        if self.activation is not None:
            features = self.activation(features)
        return features


class GCNModel(Model):
    def __init__(self, num_classes):
        super(GCNModel, self).__init__()
        self.gcn1 = GraphConvolution(32, activation=tf.nn.relu)
        self.gcn2 = GraphConvolution(64, activation=tf.nn.relu)
        self.gcn3 = GraphConvolution(128, activation=tf.nn.relu)
        self.fc1 = layers.Dense(512, activation=tf.nn.relu)
        self.fc2 = layers.Dense(256, activation=tf.nn.relu)
        self.fc3 = layers.Dense(num_classes, activation=tf.nn.softmax)
        self.flatten = layers.Flatten()
        
    def call(self, inputs):
        x, adjacency = inputs
        x = self.gcn1(x)
        x = tf.matmul(adjacency, x)
        x = self.gcn2(x)
        x = tf.matmul(adjacency, x)
        x = self.gcn3(x)
        #x = tf.reduce_mean(x, axis=1)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x