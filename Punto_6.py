
from keras.datasets import mnist
from sklearn.preprocessing import StandardScaler
from Package.PCA import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# CARGAR EL DATASET
(X_train, y_train), (X_test, y_test) = mnist.load_data()


# DISTINGUIR ENTRE UNOS Y OCHOS
X_train = X_train[(y_train==0)|(y_train==8)]
y_train_ = y_train[(y_train==0)|(y_train==8)]
X_test = X_test[(y_test==0)|(y_test==8)]
y_test = y_test[(y_test==0)|(y_test==8)]


#EL DATASET ES DE TRES DIMENSIONES, SE PASA A 2D
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1]*X_test.shape[2])


#SE ESCALAN LOS DATOS
x_test = StandardScaler().fit_transform(X_test)

#MODELO PCA
pca = PCA(n_components=2)
X_test_PCA = pca.fit_transform(X_test)


#GRAFICAR PCA
fig,ax=plt.subplots(1,1,figsize=(9,9))
sns.scatterplot(x = X_test_PCA[:,0]
                , y = X_test_PCA[:,1]
                , hue = y_test
                , palette = 'Paired'
                , legend = 'full')
plt.show()

