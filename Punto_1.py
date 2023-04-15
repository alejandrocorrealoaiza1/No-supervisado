import numpy as np

#Generar matriz
rows = 2
columns = 2
Matrix_1 = np.random.rand(rows, columns)
print(Matrix_1)

#Rank: número de campos o columnas máximas de la matriz para que sea cuadrada
from numpy.linalg import matrix_rank
rank = matrix_rank(Matrix_1)
print(rank)

#Trace: la suma de la diagonal principal
trace = Matrix_1.trace()
print(trace)

#Hallar el determinante de una matriz
if rows == columns:
  determinate = np.linalg.det(Matrix_1)
  print(determinate)
else:
  print("la matriz debe ser cuadrada")

# Hallar la inversa de una matriz
if rows == columns and determinate != 0:
  print("la matriz es inversible")
  inverted = np.linalg.inv(Matrix_1)
  print(inverted)
else:
  print("la matriz no es inversible")

eigen = np.linalg.eig(Matrix_1)
print(eigen)

eigen = np.linalg.eig(inverted)
print(eigen)

"""Los valores propios de una matriz son los mismos
los vectores propios de éstas tienen la misma magnitud con dirección parelala pero sentidos
contrarios, es decir, que al sumarlos nos da un vector nulo"""
