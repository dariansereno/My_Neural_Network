from abc import ABC, abstractmethod
import numpy as np

class Layer(ABC):
	def __init__(self, n_inputs, n_neurons):
		self.weitghs = np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))

	@abstractmethod
	def forward(self):
		pass

class Layer_Dense(Layer):
	def __init__(self, n_inputs, n_neurons):
		super().__init__(n_inputs, n_neurons)
	def forward(self, inputs):
		self.output = np.dot(inputs, self.weitghs) + self.biases	
