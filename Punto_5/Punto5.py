# The Logistic Regression is classification algorithm used when the output is categorical. 
# The ideology behind the classification is finding the relationship between the features and probabilities. 

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# CARGAR EL DATASET
mnist = fetch_openml('mnist_784')

X = mnist.data
y = mnist.target

# PARTIR ENTRE DATOS DE ENTRANAMIENTO Y PRUEBA
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

# DISTINGUIR ENTRE UNOS Y OCHOS
X_train_08 = X_train[(y_train == '0') | (y_train == '8')]
y_train_08 = y_train[(y_train == '0') | (y_train == '8')]
X_test_08 = X_test[(y_test == '0') | (y_test == '8')]
y_test_08 = y_test[(y_test == '0') | (y_test == '8')]

# CRAR EL MODELO
model = LogisticRegression()

# ENTRENAR EL MODELO
model.fit(X_train_08, y_train_08)

# VALIDACIÓN DEL MODELO
accuracy = model.score(X_test_08, y_test_08)
print(accuracy)

""" EL MODELO TIENE UN PORCENTAJE DE EXACTITUD POR ENCIMA DEL 95 %, ES UN MUY BUEN ESTIMADOR"""