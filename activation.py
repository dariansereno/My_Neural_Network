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
		# on prend tout les inputs (matrice de vecteur ex : [[1, 2, 3], [2, 1, 3], [3, 1, 2]])
		# pour chaque vecteur on soustrais la valeur maximal ex : [[1 - 3, 2 - 3, 3 - 3], [2 - 3, 1 - 3, 3 - 3], [3 - 3, 1 - 3, 2 - 3]]
		# puis on passe le result en exponentiel. Donc pas de valeur négative (pas logique) et l'exponentiel
		# est monotonique, ce qui permet de garder une cohérence de valeur : + la valeur est petite (negative ou proche de zero)
		# et + elle sera proche de zero, et + la valeur est grande + elle sera grande.
		exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		# Probleme : L'exponentiel grandi bien trop vite, donc on normalize les valeurs entre 0 - 1. En divisant ça par la somme de toutes les valeurs
		# exponentiel.
		self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)