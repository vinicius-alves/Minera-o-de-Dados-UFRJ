
import numpy as np

matrix_X_linhas = [[1,2,3],
				   [1,2,3]]

matrix_Y_linhas = [2,3,0]

for linha in matrix_X_linhas:
	linha.append(1) # para w0

matrix_X = np.matrix(matrix_X_linhas)

matrix_Y = np.matrix(matrix_Y_linhas).T

num_pontos = matrix_X.shape[0]
num_features = matrix_X.shape[1]

matrix_W = np.matrix(np.ones(num_features))

#print matrix_W

#print "\n" , matrix_X.T, "\n"

#print matrix_W*matrix_X[0].T


def ponto_corretamente_classificado(ponto_X, ponto_Y):

	print ponto_X.T
	
	valor_estimado  =  matrix_W*np.matrix(ponto_X).T

	#return np.sign(ponto_Y) == np.sign(valor_estimado)


for i in range(num_pontos):

	if(not(ponto_corretamente_classificado(matrix_X.A[i],matrix_Y.A[i]))):
		#matrix_W[i] += matrix_Y[i]*matrix_X[i]
		print "b"

	print matrix_W

