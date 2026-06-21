import numpy as np
import pandas as pd
import argparse

class Layers:
    def __init__(self, layer_size, activation="sigmoid", weight_initializer="heUniform"):
        self.layer_size = layer_size
        self.activation = activation               
        self.weight_initializer = weight_initializer
        if self.weight_initializer != "heUniform":
            raise NotImplementedError()

    def initialize(self, prev_layer_size):
        if self.weight_initializer == "heUniform":
            limit = np.sqrt(6 / prev_layer_size)
            self.weights = np.zeros(1, self.layer_size)
            self.biais = np.random.uniform(-limit, +limit, size=(1, self.layer_size))

    def forward(self, X):
        # z with row = sample and column = z 
        self.z = np.dot(X, self.weights) + self.biais
        if self.activation == "sigmoid":
            self.activ_output = 1 / (1 + np.exp(-self.z))
        if self.activation == "relu":
            self.activ_output = np.maximum(0, self.z)
        if self.activation == "softmax":
            self.activ_output = np.exp(self.z) / np.sum(np.exp(self.z), axis=1, keepdims=True)
        return (self.activ_output)

    def update_weights(self, delta, X):
        # dot because we want to do the sum of all sample
        gradiant_weights = np.dot(X.T, delta)
        gradiant_weights = gradiant_weights / delta.shape[0]
        gradiant_bias = np.sum(delta, axis=0, keepdims=True) 
        gradiant_bias =  gradiant_bias / delta.shape[0]
        self.weights -= gradiant_weights
        self.biais -= gradiant_bias

    def compute_gradiant(self,delta, weights_next, activation_prev):
        # backprop with row = sample and column = neuron
        backprop = np.dot(delta, weights_next.T)
        delta = backprop * self.activ_output * (1 - self.activ_output)
        self.update_weights(delta, activation_prev)
        return (delta)

    def __len__(self):
        return self.layer_size

class DenseLayer(Layers):
    pass


class Model: 
    def __init__(self, network):
        self.network : list[Layers] = network

    def fit(self,  data_train, data_valid, loss="categoricalCrossentropy", learning_rate=0.0314, batch_size=8, epochs=84):
        self.X_train = data_train[0]
        self.y_train = data_train[1]
        self.X_val = data_val[0]
        self.y_val = data_val[1]
        self._initalize_weights()
        z = self._forward_propagation()
        self._backward_propagation(z)

    def _initalize_weights(self):
        prev_layer_size = len(self.X_train[0])
        for layer in self.network:
            layer.initialize(prev_layer_size)
            prev_layer_size = len(layer)

    def _forward_propagation(self):
        z = self.X_train
        for i in range(len(self.network)):
            z = self.network[i].forward(z)
        return (z)


    def _backward_propagation(self, z):
        end_layer : Layers = self.network[-1]
        second_end = self.network[-2]
        delta = z - self.y_train
        end_layer.update_weights(delta, second_end.activ_output)
        for i in range(len(self.network) - 2, 0, -1):
            prev_weights = self.network[i + 1].weights
            prev_activation = self.network[i - 1].activ_output
            delta = self.network[i].compute_gradiant(delta, prev_weights, prev_activation)


def extract_data(path):
    data = pd.read_csv(path, header=None)
    data[1] = data[1].map({"M":1, "B":0})
    y = data[1].to_numpy()
    y = np.eye(2)[y]
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




        









