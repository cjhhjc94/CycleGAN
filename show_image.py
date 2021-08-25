import pickle as pkl
import matplotlib.pyplot as plt
import numpy as np

def imshow(img):
    npimg = img.numpy()
    img = np.transpose(npimg, (1, 2, 0))
    plt.imshow(img)