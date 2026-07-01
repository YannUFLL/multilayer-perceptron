

import numpy as np
from argparse import ArgumentParser

import pandas as pd


MODEL_PATH = "model.npz"
OUTPUT_PATH = "predict.txt"

class DenseLayer():

    def __init__(self, weights, bias, activation):
        self.weights = weights
        self.bias = bias
        self.activation = activation

    def forward(self, X):
        z = np.dot(X, self.weights) + self.bias
        if self.activation == "sigmoid":
            activ_output = 1 / (1 + np.exp(-z))
        if self.activation == "relu":
            activ_output = np.maximum(0, z)
        if self.activation == "softmax":
            activ_output = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
        return activ_output

class Model():

    def __init__(self, network):
        self.network = network

    def predict(self,  data):
        self.X = data
        z = self._forward_propagation(self.X)
        predict = np.argmax(z, axis=1)
        return (predict)

    def _forward_propagation(self, X):
        z =  X
        for i in range(len(self.network)):
            z = self.network[i].forward(z)
        return (z)

def load_model():
    data = np.load(MODEL_PATH)
    layer_number = sum(1 for cle in data if cle.startswith("w_"))
    network = []
    for i in range(0, layer_number):
        weights = data[f"w_{i}"]
        bias = data[f"b_{i}"]
        activation = str(data[f"a_{i}"])
        network.append(DenseLayer(weights, bias, activation))

    std = data["std"]
    mean = data["mean"]
        
    model = Model(network)

    return (model, mean, std)

def standardise_data(data, mean=None, std=None):
    if mean is None:
        mean = np.mean(data, axis=0)
    if std is None:
        std = np.std(data, axis=0)
    data = (data - mean) / std
    return(data, mean, std)


def load_data(file_path, mean, std, test):
    data = pd.read_csv(file_path, header=None)
    idx = data[0].to_numpy(dtype=int)
    if test:
            X = data.drop(columns=[0, 1]).to_numpy()
    else:
            X = data.drop(columns=[0]).to_numpy()
    X, _, _ = standardise_data(X, mean, std)
    return (X, idx)

def save_prediction(predictions, idx):
    labels = np.where(predictions == 1, "M", "B")

    df = pd.DataFrame({
        "id": idx.astype(int),
        "diagnosis":labels
    })
    df.to_csv(OUTPUT_PATH, sep=",", header=False, index=False)


def main():
    parser = ArgumentParser()
    parser.add_argument("file_path")
    parser.add_argument("-t", "--test-mode", action="store_true", help="Add -t flag when the data provided contains result")
    args = parser.parse_args()
    file_path = args.file_path
    model, mean, std = load_model()
    data, idx = load_data(file_path, mean, std, args.test_mode)
    predictions = model.predict(data)
    save_prediction(predictions, idx)


if __name__ == "__main__":
    main()
