import numpy as np
import pandas as pd
import argparse
from matplotlib import pyplot as plt

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
            self.bias = np.zeros((1, self.layer_size))
            self.weights = np.random.uniform(-limit, +limit, size=(prev_layer_size, self.layer_size))

    def forward(self, X, should_save):
        # z with row = sample and column = z 
        
        z = np.dot(X, self.weights) + self.bias
        if self.activation == "sigmoid":
            activ_output = 1 / (1 + np.exp(-z))
        if self.activation == "relu":
            activ_output = np.maximum(0, z)
        if self.activation == "softmax":
            activ_output = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        if should_save:
            self.activ_output = activ_output
            self.z = z
        return (activ_output)

    def compute_gradiant(self, delta, X):
        # dot because we want to do the sum of all sample
        gradiant_weights = np.dot(X.T, delta)
        self.gradiant_weights = gradiant_weights / delta.shape[0]
        gradiant_bias = np.sum(delta, axis=0, keepdims=True) 
        self.gradiant_bias =  gradiant_bias / delta.shape[0]

    def apply_gradiants(self, lr):
        self.weights -= lr * self.gradiant_weights
        self.bias -= lr * self.gradiant_bias


    def compute_layer_error(self,delta, weights_next, activation_prev):
        # backprop with row = sample and column = neuron
        backprop = np.dot(delta, weights_next.T)
        if self.activation == "sigmoid":
            delta = backprop * self.activ_output * (1 - self.activ_output)
        if self.activation == "relu":
            delta = backprop * np.where(self.activ_output > 0, 1, 0)
        self.compute_gradiant(delta, activation_prev)
        return (delta)

    def __len__(self):
        return self.layer_size

class DenseLayer(Layers):
    pass

class Model: 
    def __init__(self, network):
        self.network : list[Layers] = network
        self.iters = {}
        self.loss = {}
        self.accuracy = {}
        self.iters["train"] = []
        self.iters["val"] = []
        self.loss["train"] = []
        self.loss["val"] = []
        self.accuracy["train"] = []
        self.accuracy["val"] = []

    def fit(self,  data_train, data_val, loss="categoricalCrossentropy", learning_rate=0.0314, batch_size=8, epochs=84):
        self.X_train = data_train[0]
        self.y_train = data_train[1]
        self.X_val = data_val[0]
        self.y_val = data_val[1]
        self.epochs = epochs 
        self.lr = learning_rate
        self._initalize_weights()
        print("Starting Forward...")
        for i in range(0, epochs):
            z = self._forward_propagation(self.X_train)
            self._save_plot_data(i, z)
            self._backward_propagation(z)
        print(f"validation loss: {self.loss['val'][-1]}")
        print(f"training loss: {self.loss['train'][-1]}")
        print(f"validation accuracy: {self.accuracy['val'][-1]}")
        print(f"training accuracy: {self.accuracy['train'][-1]}")
        self._draw_plot()
        
    def _compute_loss(self, X, y, z):
            loss = -1 / self.X_val.shape[0] * np.sum(y * np.log(z))
            return (loss)

    def _compute_accuracy(self, y, z):
        y_pred_idx = np.argmax(z, axis=1)
        y_true_idx = np.argmax(y, axis=1)
        return (np.mean(y_true_idx == y_pred_idx))

    def _draw_plot(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("Cost function progression during training")
        ax1.plot(self.iters["train"], self.loss["train"], label="train data", color="blue", linestyle="-")
        ax1.plot(self.iters["val"], self.loss["val"], label="validation data", color="orange", linestyle="--")
        ax1.legend(loc="upper right")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("accuracy progression during training")
        ax2.plot(self.iters["train"], self.accuracy["train"], label="train data", color="blue", linestyle="-")
        ax2.plot(self.iters["val"], self.accuracy["val"], label="validation data", color="orange", linestyle="--")
        ax2.legend(loc="upper right")
        fig.tight_layout()
        plt.show(block=True)

    def _save_plot_data(self, i, z):
        z_val = self._forward_propagation(self.X_val, False)
        train_loss = self._compute_loss(self.X_train, self.y_train, z)
        val_loss = self._compute_loss(self.X_val, self.y_val, z_val)
        train_acc = self._compute_accuracy(self.y_train, z)
        val_acc = self._compute_accuracy(self.y_val, z_val)
        self.loss["train"].append(train_loss)
        self.loss["val"].append(val_loss)
        self.iters["train"].append(i)
        self.iters["val"].append(i)
        self.accuracy["train"].append(train_acc)
        self.accuracy["val"].append(val_acc)

    def _initalize_weights(self):
        prev_layer_size = len(self.X_train[0])
        for layer in self.network:
            layer.initialize(prev_layer_size)
            prev_layer_size = len(layer)

    def _forward_propagation(self, X, should_save=True):
        z =  X
        for i in range(len(self.network)):
            z = self.network[i].forward(z, should_save)
        return (z)


    def _backward_propagation(self, z):
        end_layer : Layers = self.network[-1]
        if len(self.network) < 2:
            second_end = self.X_train
        else:
            second_end = self.network[-2].activ_output
        delta = z - self.y_train
        end_layer.compute_gradiant(delta, second_end)
        for i in range(len(self.network) - 2, -1, -1):
            prev_weights = self.network[i + 1].weights
            if i != 0:
                prev_activation = self.network[i - 1].activ_output
            else:
                prev_activation = self.X_train
            delta = self.network[i].compute_layer_error(delta, prev_weights, prev_activation)
        for i in range(0, len(self.network)):
            network[i].apply_gradiants(self.lr)



def extract_data(path):
    data = pd.read_csv(path, header=None)
    data[1] = data[1].map({"M":1, "B":0})
    y = data[1].to_numpy()
    y = np.eye(2)[y]
    X = data.drop(columns=[0, 1]).to_numpy()
    return (X, y)

def standardise_data(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    data = (data - mean) / std
    return(data, mean, std)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("train_path")
    parser.add_argument("val_path")
    parser.add_argument("-l", "--learning-rate")
    args = parser.parse_args()
    data_X, data_y = extract_data(args.train_path)
    data_val_X, data_val_y = extract_data(args.val_path)
    data_X, mean, std = standardise_data(data_X)
    data_val_X, _, _ = standardise_data(data_val_X, mean, std)
    network = [DenseLayer(40, "relu", "heUniform" ),
               DenseLayer(40, "relu", "heUniform" ),
               DenseLayer(2, "softmax", "heUniform" )]
    model = Model(network)
    model.fit((data_X, data_y), (data_val_X, data_val_y), epochs=200, learning_rate=float(args.learning_rate))




        









