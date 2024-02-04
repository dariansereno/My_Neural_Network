import numpy as np
from layer import Layer_Dense
from activation import Activation_ReLU, Activation_Softmax
from loss import Loss_CategoricalCrossEntropy
import matplotlib.pyplot as plt
from spiral_data import spiral_data

np.random.seed(0)

X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = Loss_CategoricalCrossEntropy()
loss = loss_function.calculate(activation2.output, y)

prediction = np.argmax(activation2.output, axis=1)

accuracy = np.mean(prediction==y)


print("accuracy :" , accuracy)

print("loss : ", loss)
