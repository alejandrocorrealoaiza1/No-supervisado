
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from Package.SVD import SVD
from Package.PCA import PCA


# CARGAR EL DATASET
mnist = fetch_openml('mnist_784')

X = mnist.data
y = mnist.target

# PARTIR ENTRE DATOS DE ENTRANAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# DISTINGUIR ENTRE UNOS Y OCHOS
X_train_08 = X_train[(y_train == '0') | (y_train == '8')]
#RESHAPE PARA QUE PCA LO PUEDA UTLIZAR
X_train_08 = X_train_08.reshape(X_train_08.shape[0],X_train_08.shape[1]*X_train_08.shape[2])

y_train_08 = y_train[(y_train == '0') | (y_train == '8')]

X_test_08 = X_test[(y_test = '0') | (y_test == '8')]
#RESHAPE PARA QUE PCA LO PUEDA UTLIZAR
X_test_08 = X_test_08.reshape(X_test_08.shape[0],X_test_08.shape[1]*X_test_08.shape[2])

y_test_08 = y_test[(y_test == '0') | (y_test == '8')]


# UTILIZAR EL MÉTODO PCA
pca = PCA(n_components=2)
pca.fit(X_train_08)
X_08_pca = pca.transform(X_train_08)

# CRAR EL MODELO
model = LogisticRegression()

# ENTRENAR EL MODELO
model.fit(X_08_pca, y_train_08)

# VALIDACIÓN DEL MODELO
accuracy = model.score(X_test_08, y_test_08)
print(accuracy)