from sklearn.decomposition import PCA
import tensorflow as tf
from keras.datasets import mnist

# CARGAR EL DATASET
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# DISTINGUIR ENTRE UNOS Y OCHOS
X_train_08 = X_train[(y_train == '0') | (y_train == '8')]/255 
y_train_08 = y_train[(y_train == '0') | (y_train == '8')]

X_test_08 = X_test[(y_test == '0') | (y_test == '8')]/255 
y_test_08 = y_test[(y_test == '0') | (y_test == '8')]

X_train_08 = X_train_08.reshape(X_train_08.shape[0] , X_train_08.shape[1]* X_train_08.shape[2])   
X_test_08 = X_test_08.reshape(X_test_08.shape[0] , X_test_08.shape[1]* X_test_08.shape[2])  

# CREAR MODELO
pca = PCA(n_components=2)
# ENTRENAR
pca.fit(X_train_08)
# TRANSFORMAR
reduced_x_train = pca.transform(X_train_08)