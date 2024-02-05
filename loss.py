from abc import ABC, abstractmethod
import numpy as np

class Loss(ABC):
	@abstractmethod
	def forward(self):
		pass

	def calculate(self, output, y):
		sample_losses = self.forward(output, y)
		data_loss = np.mean(sample_losses) # calcul la moyenne de la loss
		return data_loss

class Loss_CategoricalCrossEntropy(Loss):
	def forward(self, y_pred, y_true):
		samples = len(y_pred)
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

		if len(y_true.shape) == 1: # quand on reçoit un vecteur de label : [0, 2, 0, 1] -> la classe 0, 2, 0, 1 sont vrai.
			# donc on cherche directement a la case des predictions les valeurs
			correct_confidences = y_pred_clipped[range(samples), y_true]
		elif len(y_true.shape) == 2: # One hot encoding, on reçoit une matrice : [[1, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]]
			# donc on a juste a multiplier chaque vecteur par la reponse : [1, 0, 0] * 0 = 0, [0, 0, 1] * 2 = 2.... 
			#(meme resultat que pour la ligne au dessus)
			correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
		
		negative_log_likelihoods = -np.log(correct_confidences)
		# ici on fout au -logarithme (naturel !! base E) les resultat. Ca permet de pouvoir revenir au resultat en 
		# mettant en exponentiel le logarithme. (pratique pour la backpropagation et l'optimisation)
		return negative_log_likelihoods