import numpy as np
from layer import Layer_Dense
from activation import Activation_ReLU, Activation_Softmax, Activation_Softmax_Loss_CategoricalCrossEntropy
from loss import Loss_CategoricalCrossEntropy
from optimizer import Optimizer_SGD
import matplotlib.pyplot as plt
from spiral_data import spiral_data

#np.random.seed(1)

X, y = spiral_data(100, 3)

dense1 = Layer_Dense(2, 64)
dense2 = Layer_Dense(64, 3)
#dense3 = Layer_Dense(64, 3)

activation1 = Activation_ReLU()
#activation2 = Activation_ReLU()
loss_activation = Activation_Softmax_Loss_CategoricalCrossEntropy()

optimizer = Optimizer_SGD(decay=1e-3, momentum=0.9)

epochs = []
losses = []
accuracies = []

for epoch in range(10001):
	dense1.forward(X)
	activation1.forward(dense1.output)
	dense2.forward(activation1.output)
	loss = loss_activation.forward(dense2.output, y)
	#dense3.forward(activation2.output)
	#loss = loss_activation.forward(dense3.output, y)

	prediction = np.argmax(loss_activation.output, axis=1)
	accuracy = np.mean(prediction==y)
	if not epoch % 100:
		print(f'epoch: {epoch}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {optimizer.current_learning_rate}')

	loss_activation.backward(loss_activation.output, y)
	#dense3.backward(loss_activation.dinputs)
	#activation2.backward(dense3.dinputs)
	dense2.backward(loss_activation.dinputs)
	activation1.backward(dense2.dinputs)
	dense1.backward(activation1.dinputs)

	epochs.append(epoch)
	losses.append(loss)
	accuracies.append(accuracy)

	optimizer.pre_update_params()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update_params()

plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, losses, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, accuracies, label='Accuracy', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()