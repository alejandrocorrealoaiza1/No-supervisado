print("ESOOOO ES")


from Package.SVD import SVD
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

#CARGAR FOTO
picture = Image.open("data\AlejandroC.jpeg")

#MODIFCAR FOTO
personal_picture = np.asarray(picture.convert('L').resize((256,256)))

#NUMERO DE COMPONENETES
componentes = np.array([2,4,8])

for index, n in enumerate(componentes):

    #MODELO
    svd = SVD(n_components=n)

    #ENTERNAR MODELO
    picture_svd=svd.fit(personal_picture)

    #TRANSFORMAR
    picture_svd=svd.transform(personal_picture)
    print(picture_svd.shape)

    #PLOTEAR
    plt.figure()

    plt.subplot(2,2,1)
    plt.imshow(personal_picture, cmap='gray')
    plt.axis('off')
    plt.title('Imagen original')

    plt.subplot(2,2,2)
    plt.imshow(picture_svd,cmap='gray')
    plt.axis('off')
    plt.title('SVD')

plt.show()

"ENTRE MÁS COMPONENTES LA IMAGEN ES MÁS PARECIDA A LA ORGINAL, SIN EMBARGO, CON POCOS SE VE QUE ALCANZA A REPRESENTAR BIEN"

#LA DIRENCIA SE PUEDE MEDIR A TRAVÉS DE LA DISTANCIA EUCLIDIANA.
difference = np.linalg.norm(personal_picture - picture_svd)
print(difference)