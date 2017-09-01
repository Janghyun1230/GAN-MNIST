import numpy as np
import matplotlib.pyplot as plt


def preprocess(x):
    return 2 * x - 1.0


def deprocess(x):
    return (x + 1.0) / 2.0


def noise(batch_size, dim):
    return np.random.uniform(low=-1.0, high=1.0, size=(batch_size, dim))


def image_show(images):
    num = images.shape[0]
    plt.figure(figsize = (10,3))
    for i in range(num):
        plt.subplot(1,num,i+1)
        plt.imshow(deprocess(images[i]).reshape(28,28))
        plt.axis('off')
    plt.show()
        