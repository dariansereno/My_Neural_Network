from layer import Layer
from activation import Activation
from loss import Loss
from optimizer import Optimizer, Optimizer_SGD
import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:

	def __init__(self, loss: Loss, epochs=1001, optimizer= Optimizer_SGD()):
		self.optimizer: Optimizer = optimizer
		self.loss: Loss = loss
		self.n_layer = 0
		self.epochs = epochs
		self.layers = []
		self.activations = []
		self.accuracies = []
		self.losses = []
	
	def add_layer(self, layer: Layer, activation: Activation):
		self.n_layer += 1
		self.layers.append(layer)
		self.activations.append(activation)
	
	def train(self, X, Y, display=False, plot=False):
		self.accuracies.clear()
		self.losses.clear()

		for _ in range(self.epochs):
			layer: Layer
			activation: Activation
			feed = X.copy()

			# forward
			for (layer, activation) in zip(self.layers, self.activations):
				layer.forward(feed)
				feed = layer.output
				activation.forward(feed)
				feed = activation.output

			predictions = (feed > 0.5) * 1
			accuracy = np.mean(predictions==Y)

			loss = self.loss.calculate(feed, Y)
			self.loss.backward(feed, Y)
			feed = self.loss.dinputs

			self.losses.append(loss)
			self.accuracies.append(accuracy)
			if (display):
				if not _ % 100:
					print(f'epoch: {_}, ' + f'acc: {accuracy:.3f}, ' + f'loss: {loss:.3f}, ' + f'lr: {self.optimizer.current_learning_rate}')
			
			# backward
			for (layer, activation) in zip(reversed(self.layers), reversed(self.activations)):
				activation.backward(feed)
				feed = activation.dinputs
				layer.backward(feed)
				feed = layer.dinputs
		
			# update
			self.optimizer.pre_update_params()
			for layer in self.layers:
				self.optimizer.update_params(layer)
			self.optimizer.post_update_params()

		if (plot):
			self.plot()

	def predict(self, X):
		feed = X.copy()

		layer: Layer
		activation: Activation
		for (layer, activation) in zip(self.layers, self.activations):
			layer.forward(feed)
			feed =layer.output
			activation.forward(feed)
			feed = activation.output
		return  (feed > 0.5)
	
	def plot(self):
		plt.figure(figsize=(12, 4))

		# loss
		plt.subplot(1, 2, 1)
		plt.plot(range(self.epochs), self.losses, label='Loss')
		plt.title('Training Loss')
		plt.xlabel('Epochs')
		plt.ylabel('Loss')
		plt.legend()

		# accuracy
		plt.subplot(1, 2, 2)
		plt.plot(range(self.epochs), self.accuracies, label='Accuracy', color='orange')
		plt.title('Training Accuracy')
		plt.xlabel('Epochs')
		plt.ylabel('Accuracy')
		plt.legend()

		plt.tight_layout()
		plt.show()





