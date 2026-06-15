import numpy as np
import pandas as pd
import argparse

class Layers:
    def __init__(self, layer_size, activation="sigmoid", weight_initializer="heUniform"):
        self.layer_size = layer_size
        self.activation = activation               
        self.weight_initializer = weight_initializer
        if self.activation != "sigmoid":
            raise NotImplementedError()
        if self.weight_initializer != "heUniform":
            raise NotImplementedError()

    def initialize(self, prev_layer_size):
        if self.weight_initializer == "heUniform":
            limit = np.sqrt(6 / prev_layer_size)
            self.weights = np.random.uniform(-limit, +limit, size=(prev_layer_size, self.layer_size))

    def compute_z(self, X):
        output = np.dot(X, self.weights)
        if self.activation == "sigmoid":
            output = 1 / (1 + np.exp(-output))
        if self.activation == "relu":
            output = np.max(0, output)
        if self.activation == "softmax":
            output = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)
        return (output)

    def __len__(self):
        return self.layer_size
class DenseLayer(Layers):
    pass


class Model: 
    def __init__(self, network):
        self.network = network

    def fit(self,  data_train, data_valid, loss="categoricalCrossentropy", learning_rate=0.0314, batch_size=8, epochs=84):
        self.X_train = data_train[0]
        self.y_train = data_train[1]
        self.X_val = data_val[0]
        self.y_val = data_val[1]
        self.X_train = np.hstack([np.ones((self.X_train.shape[0], 1)), self.X_train])
        self._initalize_weights()
        z = self._forward_propagation()
        self._backward_propagation(z)

    def _initalize_weights(self):
        prev_layer_size = len(self.X_train)
        for layer in self.network:
            layer.initialize(prev_layer_size)
            prev_layer_size = len(layer)

    def _forward_propagation(self):
        for i in range(network):
            z = self.network[i].compute_z(z)
        return (z)
        

    def _backward_propagation(self, z):
        delta = -np.sum(self.y_train * np.log(z))
        pass


def extract_data(path):
    data = pd.read_csv(path, header=None)
    data[1] = data[1].map({"M":1, "B":0})
    y = data[1].to_numpy()
    X = data.drop(columns=[0, 1]).to_numpy()
    return (X, y)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("val_path")
    args = parser.parse_args()
    data_train = extract_data(args.train_path)
    data_val = extract_data(args.val_path)
    network = [DenseLayer(len(data_val[0][0]), "sigmoid", "heUniform" ),
               DenseLayer(30, "sigmoid", "heUniform" ),
               DenseLayer(2, "softmax", "heUniform" )]
    model = Model(network)
    model.fit(data_train, data_val)




        









