
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from sklearn.linear_model import LogisticRegression
from Package.PCA import PCA
import matplotlib.pyplot as plt
import numpy as np


# CARGAR EL DATASET
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# DISTINGUIR ENTRE UNOS Y OCHOS
X_train_08 = X_train[(y_train == '0') | (y_train == '8')]/255 
y_train_08 = y_train[(y_train == '0') | (y_train == '8')]

X_test_08 = X_test[(y_test == '0') | (y_test == '8')]/255 
y_test_08 = y_test[(y_test == '0') | (y_test == '8')]

X_train_08 = X_train_08.reshape(X_train_08.shape[0] , X_train_08.shape[1]* X_train_08.shape[2])   
X_test_08 = X_test_08.reshape(X_test_08.shape[0] , X_test_08.shape[1]* X_test_08.shape[2])  

ids1 = np.random.randint(0,60000, (3000))
X_train_s = X_train[ids1]
y_train_s = y_train[ids1]

ids2 = np.random.randint(0,10000, (100))
X_test_s = X_test[ids2]
y_test_s = y_test[ids2]

# UTILIZAR EL MÉTODO PCA
pca = PCA(n_components=2)
pca.fit_transform(X_train_s)
X_08_pca = pca.transform(X_test_s)



# CRAR EL MODELO
#model = LogisticRegression()

# ENTRENAR EL MODELO
#model.fit(X_08_pca, y_train_08)

# VALIDACIÓN DEL MODELO
#accuracy = model.score(X_test_08, y_test_08)
# print(accuracy)