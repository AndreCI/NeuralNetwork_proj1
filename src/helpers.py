"""Python script for the Unsupervised and 
Reinforcement Learning.
"""

import numpy as np
import matplotlib.pylab as plb
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def rewritedata(indata):
    data_matrix = np.ones((np.shape(indata)[0],28,28))
    for i in range (0,28):
        data_matrix[:,i,:] = indata[:,i*28:(i+1)*28]
    return data_matrix

def visualize(number,label):
    assert np.shape(number)==(28,28)
    #plt.title('Image of ',label)
    #plt.annotate(('label is ',label),xy=(2, 1))
    plt.title(("An image of a %s"%label))
    plt.imshow(number,cmap='Greys')
    plt.show()
