from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# DESCARAGAR EL DATA SET
mnist = MNIST(root='data/', download=True)

mnist[0]

image, label = mnist[0]
plt.imshow(image, cmap='gray')
print('Label:', label)