from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
	def __init__(self, n_inputs, n_neurons):
		self.weigths = 0.10 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

		# exemple : On a 2 inputs, et 3 neurones : weights : [[2.1, 0.3, 0.56], [1.32, 0.56, 3.21]], biais : [0.32, 0.67, 1.56]

	@abstractmethod
	def forward(self):
		pass

class Layer_Dense(Layer):
	def __init__(self, n_inputs, n_neurons):
		super().__init__(n_inputs, n_neurons)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weigths) + self.biases	