# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
from pathlib import Path
import numpy as np

# %matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt

repo_directory = Path('D:\Kaggle\MNIST_Repo')
data_path = repo_directory / 'Data'

# -

from mlxtend.data import loadlocal_mnist
import platform

# +
if not platform.system() == 'Windows':
    X, y = loadlocal_mnist(
            images_path=data_path / 'train-images-idx3-ubyte', 
            labels_path= data_path / 'train-labels-idx1-ubyte')

else:
    X, y = loadlocal_mnist(
        images_path=data_path / 'train-images.idx3-ubyte', 
        labels_path=data_path / 'train-labels.idx1-ubyte')
# -

np.savetxt(fname=data_path / 'images.csv', 
           X=X, delimiter=',', fmt='%d')
np.savetxt(fname=data_path / 'labels.csv', 
           X=y, delimiter=',', fmt='%d')

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
