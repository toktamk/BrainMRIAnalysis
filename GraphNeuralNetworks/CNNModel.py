#CNNModel: A custom neural network model for implementing the CNN architecture.

from tensorflow.keras import layers, Model
class CNNModel(Model):
    def __init__(self, num_classes):
        super(CNNModel, self).__init__()
        # Define your CNN layers
        self.conv1 = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.maxpool1 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = layers.Conv2D(64, kernel_size=(3, 3), activation='relu')
        self.maxpool2 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')
        self.maxpool3 = layers.MaxPooling2D(pool_size=(2, 2))
        self.conv4 = layers.Conv2D(256, kernel_size=(3, 3), activation='relu')
        self.maxpool4 = layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(512, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.maxpool3(x)
        x = self.conv4(x)
        x = self.maxpool4(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x