# -*- coding: utf-8 -*-

import numpy as np
from .neuronio_perceptron import *

class Perceptron:
	
	def __init__(self, learning_rate = 0.1, tol = 0.01, random_state = -np.inf):
		self._weights = None
		self._input_size = 0
		self._output_size = 0
		self._lr = learning_rate
		self._tol = tol
		self._neuronios = None
		self._classes = None
		self.coef_ = None 

		if(random_state == -np.inf):
			self._random_state = np.random.randint(10000)
		else:
			self._random_state = random_state


	def fit(self, X, y):
		self._classes = np.unique(y)
		self._input_size = len(X)
		self._neuronios = []

		neuronio = None
		y_mapeado = None
		for i in range(len(self._classes)):
			neuronio = NeuronioPerceptron(learning_rate = self._lr, 
				tol = self._tol, random_state = self._random_state)
			y_mapeado = y == self._classes[i]
			neuronio.fit(X,y_mapeado, self._input_size)
			self._neuronios.append(neuronio)


	def predict(self,X):
		self._output_size = len(X)

		list_dis_hiperplano_neuronio = []
		self._weights = []

		for i in range(len(self._neuronios)):
			neuronio = self._neuronios[i]
			list_neu = neuronio.predict_dist_hiperplano(X.transpose(),self._output_size)
			list_dis_hiperplano_neuronio.append(list_neu)
			self._weights.append(neuronio.coef_)

		self.coef_ = np.array(self._weights)

		array_dist = np.transpose(np.array(list_dis_hiperplano_neuronio))
		array_indice_classes_previstas = np.argmax(array_dist, axis = 1)
		array_classes_previstas = self._classes[array_indice_classes_previstas]

		return array_classes_previstas
	