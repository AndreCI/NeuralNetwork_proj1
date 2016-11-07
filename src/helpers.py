"""Python script for Exercise set 6 of the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt

def rewritedata(x):
    data_matrix = np.ones((np.shape(x)[0],28,28))
    for i in range (0,28):
        data_matrix[:,i,:] = data[:,i*28:(i+1)*28]
    return data_matrix

def visualize(number,label):
    assert np.shape(number)==(28,28)
    #plt.title('Image of ',label)
    #plt.annotate(('label is ',label),xy=(2, 1))
    plt.title(("An image of a %s"%label))
    plt.imshow(number,cmap='Greys')
    plt.show()
