import numpy as np
from layer import Layer_Dense
from activation import Activation_ReLU, Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossEntropy, Activation_Sigmoid
from loss import Loss_CategoricalCrossEntropy, Loss_BinaryCrossEntropy
from optimizer import Optimizer_SGD
import matplotlib.pyplot as plt
from preprocess import get_data
from tools import extract_csv
from neural_network import NeuralNetwork

(X, test_X), (Y, test_Y) = get_data(extract_csv("data.csv"))

Y = np.array(Y).reshape(-1, 1)
X = np.array(X)

test_X = np.array(test_X)
test_Y = np.array(test_Y).reshape(-1, 1)

nn = NeuralNetwork(Loss_BinaryCrossEntropy(), 5001, Optimizer_SGD(decay=1e-3, momentum=0.9))

nn.add_layer(Layer_Dense(X.shape[1], 24), Activation_ReLU())
nn.add_layer(Layer_Dense(24, 24), Activation_ReLU())
nn.add_layer(Layer_Dense(24, 24), Activation_ReLU())
nn.add_layer(Layer_Dense(24, 1), Activation_Sigmoid())

nn.train(X, Y, display=True, plot=False)
predictions = nn.predict(test_X)

accuracy = np.mean(predictions == test_Y)
true_table = np.array([1 if pred == real else 0 for pred, real in zip(predictions, test_Y)]).reshape(1, -1)

print(f'accuracy: {accuracy:.3f}')
print("true table : ", true_table)

