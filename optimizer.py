from abc import ABC, abstractmethod
import numpy as np
from layer import Layer

class Optimizer(ABC):
	def __init__(self, learning_rate=1.0, decay=.0, momentum=.0):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.momentum = momentum
	@abstractmethod
	def pre_update_params(self):
		pass

	@abstractmethod
	def update_params(self, layer: Layer):
		pass

	@abstractmethod
	def post_update_params(self):
		pass

class Optimizer_SGD(Optimizer):
	def __init__(self, learning_rate=1.0, decay=0.0, momentum=.0):
		super().__init__(learning_rate, decay, momentum)
	def pre_update_params(self):
		if (self.decay):
			self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

	def update_params(self, layer):
		if (self.momentum):
			if not hasattr(layer, 'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				layer.bias_momentums = np.zeros_like(layer.biases)

			weights_update = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
			layer.weight_momentums = weights_update

			biases_update = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
			layer.bias_momentums = biases_update
		else:
			weights_update += -self.learning_rate * layer.dweights
			biases_update += -self.learning_rate * layer.dbiases
		
		layer.weights += weights_update
		layer.biases += biases_update
	
	def post_update_params(self):
		self.iterations += 1
