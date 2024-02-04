from abc import ABC, abstractmethod
import numpy as np

class Activation(ABC):
	@abstractmethod
	def forward(self):
		pass

class Activation_ReLU(Activation):
	def forward(self, inputs):
		self.output = np.maximum(0, inputs)

class Activation_Sigmoid(Activation):
	def forward(self, inputs):
		return 1/(1 + np.exp(-inputs))
	
class Activation_Softmax(Activation):
	def forward(self, inputs):
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)