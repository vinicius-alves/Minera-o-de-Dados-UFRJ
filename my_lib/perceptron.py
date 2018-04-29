# -*- coding: utf-8 -*-

import numpy as np
from .neuronio_perceptron import *

class Perceptron:
	
	def __init__(self, learning_rate = 0.1, tol = 0.01, random_state = 0):
		self._weights = None
		self._input_size = 0
		self._output_size = 0
		self._lr = learning_rate
		self._tol = tol
		self._random_state = random_state
		self._neuronios = None
		self.coef_ = None 


	def fit(self, X, y):
		classes = np.unique(y)

		#print (classes)

		self._neuronios = []

		neuronio = None
		for i in range(len(classes)):
			neuronio = NeuronioPerceptron(learning_rate = self._lr, tol = self._tol, random_state = self._random_state)
			neuronio.fit(X,y)
			self._neuronios.append(neuronio)


		#aprendizado
		print ("")


	def predict(self,X):
		#prevê uma saída
		return []

	