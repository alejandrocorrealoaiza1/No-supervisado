#The Singular-Value Decomposition, or SVD for short, is a matrix decomposition method 
#for reducing a matrix to its constituent parts in order to make certain subsequent matrix calculations simpler.

import numpy as np
from numpy import linalg

class SVD:

    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
    
    
    def fit(self, X):
        # X: Original array

        # Compute the factor by Singular Value 
        # Decomposition
        U, s, V = np.linalg.svd(X, full_matrices=False)

        S = np.zeros((X.shape[0], X.shape[1]))
        S[:X.shape[0], :X.shape[0]] = np.diag(s)

        self.S = S[:, :self.n_components]
        self.U = U
        self.V= V[:self.n_components, :]
      
        # A tuple with one vector unitary (matrix is orthonornal), one matrix singular values, one matriz singular vectors
    
    def transform(self, X):
    #Transform X with SVD components
        X_svd = self.U.dot(self.S.dot(self.V))

        return X_svd


